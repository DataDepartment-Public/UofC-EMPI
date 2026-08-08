"""Train + promote the non-match gate and ML matcher against a SYNTHETIC run.

Neither model has a usable training CLI: `src/models/ml_matcher/train.py` is
an explicit unimplemented scaffold, and the non-match gate has no `train.py`
at all — both were, in practice, only ever trained from the Jupyter
notebooks (`notebooks/ml_model/confident_nonmatch/pair_classifier_lightgbm_nonmatch_gate_v1.ipynb`
and `notebooks/ml_model/confident_match/pair_classifier_lightgbm_confident_match_v5.ipynb`)
against real gold-labeled PHI data that must never exist in this environment.

This script trains the identical model shape — same 12 features via
`FeatureBuilderV5` (ported verbatim from those notebooks already), same
LightGBM hyperparameters, same 60/20/20 split, same export/promote contract
— against a synthetic pipeline run instead, so local dev sees genuine (not
hand-mocked) SHAP explanations end-to-end.

Gold labels come from two sources over the run's `non_matches` pool (the
exact pool the gate/ML matcher score at serving time, so train/serve stay
consistent):
  * planted pairs from `scripts/generate_synthetic_raw.py --emit-labels` —
    positives (confident match or ambiguous), whichever of them actually
    survived into `non_matches` rather than being auto-merged outright.
  * every other `non_matches` pair — a confident non-match (blocking found
    it, but it planted no relationship at all).

USAGE:
    python -m src.pipeline --input data/raw/MDM_Population.csv --run-id synth-train
    python scripts/train_synthetic_models.py --run-id synth-train \\
        --planted-labels data/gold_labels/synthetic_planted_pairs.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier, early_stopping, log_evaluation  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import settings as default_settings, configure_logging  # noqa: E402
from src.contracts import RunManifest  # noqa: E402
from src.models.ml_matcher.lightgbm_v5 import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    DirectMatchAdapter,
    FeatureBuilderV5,
)
from src.models.nonmatch_gate import registry as gate_registry  # noqa: E402
from src.models.ml_matcher import registry as ml_registry  # noqa: E402

logger = logging.getLogger("eMPI.train_synthetic_models")

LGBM_PARAMS = dict(
    objective="binary", n_estimators=1000, learning_rate=0.05, num_leaves=15,
    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, class_weight="balanced", n_jobs=-1, verbose=-1,
)


def _resolve_from_manifest(settings, run_id: str | None) -> tuple[Path, Path]:
    """(cleaned, non_matches) paths from a RunManifest — mirrors
    `fs_matcher/train.py::_resolve_from_manifest`, targeting `non_matches`
    (what the gate/ML matcher actually score) instead of the pre-rules
    `candidate_pairs` pool (what the real notebooks train against instead,
    via a separately-curated gold-labels file we don't have here)."""
    runs_dir = Path(settings.runs_dir)
    if run_id is not None:
        manifest_path = runs_dir / f"run_{run_id}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest for --run-id={run_id!r} at {manifest_path}.")
    else:
        candidates = sorted(runs_dir.glob("run_*.json"))
        if not candidates:
            raise FileNotFoundError(f"No run manifests found in {runs_dir}.")
        manifest_path = candidates[-1]

    manifest = RunManifest.model_validate_json(manifest_path.read_text())
    cleaned_path = settings.project_root / manifest.cleaned.path
    non_matches_path = settings.project_root / manifest.non_matches.path
    for label, p in (("cleaned", cleaned_path), ("non_matches", non_matches_path)):
        if not p.exists():
            raise FileNotFoundError(f"Manifest references {label}={p}, which doesn't exist.")
    logger.info("Resolved inputs from manifest %s (run_id=%s)", manifest_path.name, manifest.run_id)
    return cleaned_path, non_matches_path


def _canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    a = out[["PATID_A", "PATID_B"]].min(axis=1)
    b = out[["PATID_A", "PATID_B"]].max(axis=1)
    out["PATID_A"], out["PATID_B"] = a, b
    return out


def _build_gold_labels(non_matches: pd.DataFrame, planted: pd.DataFrame) -> pd.DataFrame:
    """Every `non_matches` pair, labeled: a planted pair keeps its
    `final_gold_label`/`ambiguous_pair`; anything else is a confident
    non-match (blocking surfaced it, but nothing was planted there)."""
    pool = _canonicalize_pairs(non_matches[["PATID_A", "PATID_B"]]).drop_duplicates()
    planted = _canonicalize_pairs(planted)
    gold = pool.merge(planted, on=["PATID_A", "PATID_B"], how="left")
    gold["final_gold_label"] = gold["final_gold_label"].fillna(False).astype(bool)
    gold["ambiguous_pair"] = gold["ambiguous_pair"].fillna(False).astype(bool)
    return gold.reset_index(drop=True)


def _binary_metrics(y_true: pd.Series, y_pred_pos: pd.Series) -> dict:
    tp = int(((y_true == 1) & y_pred_pos).sum())
    fp = int(((y_true == 0) & y_pred_pos).sum())
    fn = int(((y_true == 1) & ~y_pred_pos).sum())
    tn = int(((y_true == 0) & ~y_pred_pos).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4)}


def _split(X: pd.DataFrame, y: pd.Series, seed: int):
    """60/20/20 stratified train/val/test — same scheme as both notebooks."""
    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.20, random_state=seed, stratify=y,
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.25, random_state=seed, stratify=y.iloc[idx_trainval],
    )
    return (
        X.iloc[idx_train], y.iloc[idx_train],
        X.iloc[idx_val], y.iloc[idx_val],
        X.iloc[idx_test], y.iloc[idx_test],
    )


def _fit(X_train, y_train, X_val, y_val, seed: int) -> LGBMClassifier:
    """Same model/features/hyperparameters as both notebooks. Early stopping
    monitors `binary_logloss` rather than the notebooks' `auc`: this
    synthetic data's planted scenarios are cleanly separable by rank almost
    immediately (AUC plateaus in ~7 rounds), which halts training long
    before the probabilities themselves are well-calibrated — every score
    comes out compressed near 0.5 despite near-perfect ranking. A real,
    noisier dataset doesn't saturate AUC this fast, so the notebooks never
    hit this; logloss keeps training until predictions are genuinely
    confident, not just correctly ordered.
    """
    model = LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)],
    )
    return model


def _git_sha() -> str | None:
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _train_gate(features: pd.DataFrame, gold: pd.DataFrame, settings, seed: int) -> Path:
    y = (gold["final_gold_label"] | gold["ambiguous_pair"]).astype(int)
    X = features[FEATURE_COLS]
    X_train, y_train, X_val, y_val, X_test, y_test = _split(X, y, seed)
    logger.info("[gate] train=%d val=%d test=%d (positives: %d/%d/%d)",
                len(y_train), len(y_val), len(y_test),
                int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))

    model = _fit(X_train, y_train, X_val, y_val, seed)

    proba_test = model.predict_proba(X_test)[:, 1]
    threshold = settings.gate_threshold
    y_pred = pd.Series(proba_test >= threshold, index=y_test.index)
    metrics = _binary_metrics(y_test, y_pred)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc_auc = float(roc_auc_score(y_test, proba_test))
        pr_auc = float(average_precision_score(y_test, proba_test))
    except ValueError:
        roc_auc = pr_auc = float("nan")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gate_dir = Path(settings.gate_model_dir)
    gate_dir.mkdir(parents=True, exist_ok=True)
    artifact = gate_dir / f"nonmatch_gate_{ts}.pkl"
    import joblib
    joblib.dump(model, artifact)

    meta = {
        "model_name": "nonmatch_gate", "model_file": artifact.name,
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_sha": _git_sha(),
        "data_source": "synthetic (scripts/generate_synthetic_raw.py + train_synthetic_models.py)",
        "seed": seed, "feature_cols": list(FEATURE_COLS), "gate_threshold": threshold,
        "test_metrics": {
            "n_test": int(len(y_test)), "roc_auc": roc_auc, "pr_auc": pr_auc,
            "gate_at_threshold": {
                "threshold": threshold,
                "plausible_precision": metrics["precision"],
                "plausible_recall": metrics["recall"],
                **{k: metrics[k] for k in ("tp", "fp", "fn", "tn")},
            },
        },
    }
    gate_registry.meta_path_for(artifact).write_text(json.dumps(meta, indent=2))
    logger.info("[gate] test gate_at_threshold: %s", meta["test_metrics"]["gate_at_threshold"])
    reason = gate_registry.promote(artifact, settings, force=True)
    logger.info("[gate] PROMOTED (forced — synthetic run, not compared to any real baseline): %s", reason)
    return artifact


def _train_ml_matcher(features: pd.DataFrame, gold: pd.DataFrame, settings, seed: int) -> Path:
    """v5 semantics: trained on the WHOLE gold pool (no plausible-only
    filter — that was v3's population choice), target class 1 = confident
    match (`final_gold_label & ~ambiguous_pair`), so `predict_proba(X)[:, 1]`
    is the served score directly (`DirectMatchAdapter`, no inversion)."""
    X = features[FEATURE_COLS]
    y = (gold["final_gold_label"] & ~gold["ambiguous_pair"]).astype(int)
    X_train, y_train, X_val, y_val, X_test, y_test = _split(X, y, seed)
    logger.info("[ml_matcher] train=%d val=%d test=%d (confident_match: %d/%d/%d)",
                len(y_train), len(y_val), len(y_test),
                int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))

    model = _fit(X_train, y_train, X_val, y_val, seed)

    proba_test = model.predict_proba(X_test)[:, 1]  # P(confident match) -- served score, no inversion
    auto_merge_threshold = settings.ml_auto_merge_threshold
    y_pred = pd.Series(proba_test >= auto_merge_threshold, index=y_test.index)
    metrics = _binary_metrics(y_test, y_pred)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ml_dir = Path(settings.ml_model_dir)
    ml_dir.mkdir(parents=True, exist_ok=True)
    artifact = ml_dir / f"ml_model_confident_match_v5_{ts}.pkl"
    import joblib
    joblib.dump(DirectMatchAdapter(model), artifact)

    meta = {
        "model_name": "ml_matcher", "model_file": artifact.name,
        "model_version": "lightgbm_v5",
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_sha": _git_sha(),
        "data_source": "synthetic (scripts/generate_synthetic_raw.py + train_synthetic_models.py)",
        "seed": seed, "feature_cols": list(FEATURE_COLS),
        "thresholds": {"auto_merge": auto_merge_threshold},
        "test_metrics": {"n_test": int(len(y_test)), "metrics_auto_merge": metrics},
    }
    ml_registry.meta_path_for(artifact).write_text(json.dumps(meta, indent=2))
    logger.info("[ml_matcher] test metrics_auto_merge: %s", metrics)
    reason = ml_registry.promote(artifact, settings, force=True)
    logger.info("[ml_matcher] PROMOTED (forced — synthetic run, not compared to any real baseline): %s", reason)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=None,
                        help="Pipeline run to train from (data/runs/run_<id>.json). Default: latest.")
    parser.add_argument("--planted-labels", type=Path, required=True,
                        help="Planted-pair labels CSV from generate_synthetic_raw.py --emit-labels.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    settings = default_settings
    configure_logging(settings, level=args.log_level)
    settings.ensure_dirs()

    cleaned_path, non_matches_path = _resolve_from_manifest(settings, args.run_id)
    df_clean = pd.read_parquet(cleaned_path)
    non_matches = pd.read_parquet(non_matches_path)
    planted = pd.read_csv(args.planted_labels, dtype={"PATID_A": str, "PATID_B": str})

    gold = _build_gold_labels(non_matches, planted)
    logger.info(
        "non_matches pool=%d -> gold labels: confident=%d ambiguous=%d confident_non_match=%d",
        len(gold), int(gold["final_gold_label"].sum()), int(gold["ambiguous_pair"].sum()),
        int((~gold["final_gold_label"] & ~gold["ambiguous_pair"]).sum()),
    )
    if gold["final_gold_label"].sum() < 10 or gold["ambiguous_pair"].sum() < 10:
        raise RuntimeError(
            "Too few positive/ambiguous examples in the non_matches pool to train "
            "meaningfully — regenerate synthetic data with a higher --dup-rate or --n."
        )

    features = FeatureBuilderV5().build_features(gold[["PATID_A", "PATID_B"]], df_clean)
    assert (features["PATID_A"].values == gold["PATID_A"].values).all()
    assert (features["PATID_B"].values == gold["PATID_B"].values).all()

    gate_artifact = _train_gate(features, gold, settings, args.seed)
    ml_artifact = _train_ml_matcher(features, gold, settings, args.seed)

    print(f"Trained + promoted gate:       {gate_artifact}")
    print(f"Trained + promoted ml_matcher: {ml_artifact}")


if __name__ == "__main__":
    main()
