"""fs-train — train + persist the production FS model (offline, VM-only).

    python -m src.models.fs_matcher.train [--promote]

*** RUN ON THE VM ONLY (reads the PHI silver labels + cleaned cohort). ***
Never logs individual PATIDs, names, SSNs, DOBs, or other identifiers — only
aggregate counts, non-PHI score quantiles, and held-out test metrics.

Flow:
  1. Load cleaned records + the full candidate pool + the silver labels.
  2. Stratified 80/20 split of the silver labels (train m; test = held-out P/R).
  3. Train FSMatcher (λ from PRIOR_RULES, m from the train split, u by sampling).
  4. Persist the trained Splink model JSON + a `.meta.json` provenance/metrics
     sidecar into `settings.fs_model_dir`.
  5. Score the full candidate pool, join labels, and write the labeled GBT
     training feature parquet to `settings.fs_output_dir`.
  6. With `--promote`, run the deploy-gate and (if it passes) point `active.json`
     at the new model.

Because the silver labels are Stage-3 deterministic confirmations (removed from
`non_matches`), the model is trained/scored against the FULL pre-rules candidate
pool. Serving (the pipeline) scores only the `non_matches` pool.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import settings as default_settings, configure_logging  # noqa: E402
from src.models.fs_matcher.matcher import (  # noqa: E402
    FSMatcher,
    MODEL_NAME,
    classification_config_from_settings,
)
from src.models.fs_matcher import registry  # noqa: E402
from src.models.fs_matcher.versioning import (  # noqa: E402
    latest_versioned,
    version_tag_from_filename,
)

logger = logging.getLogger("eMPI.fs_train")

# Production default (multi-core VM). FSMatcher's u-guard raises a clear
# RuntimeError if single-CPU; we catch and retry with 1e4.
U_MAX_PAIRS = 1e6

CLEANED_GLOBS = ("MDM_Population_cleaned_v*_*.parquet", "MDM_Population_cleaned_*Z.parquet")
CANDIDATE_PAIRS_GLOBS = ("candidate_pairs_v*_*.parquet", "candidate_pairs_*Z.parquet")


# ── Input resolution ─────────────────────────────────────────────────────────
def _resolve_latest(dir_: Path, globs: tuple[str, ...]) -> Path:
    """Latest versioned file, else newest by mtime, across the given globs."""
    if dir_.is_dir():
        try:
            return latest_versioned(dir_, globs[0])
        except FileNotFoundError:
            pass
        for pattern in globs:
            hits = sorted(dir_.glob(pattern), key=lambda p: p.stat().st_mtime)
            if hits:
                return hits[-1]
    raise FileNotFoundError(
        f"No files matching {globs} in {dir_}. Run the upstream stage or pass "
        "the path explicitly."
    )


def _derive_data_version(path: Path) -> str:
    tag = version_tag_from_filename(path)
    if tag != "unversioned":
        return tag
    import re
    m = re.search(r"(\d{8}T\d{6}Z)", path.name)
    return m.group(1) if m else path.stem


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


# ── Silver-label handling (lifted from run_real_enhanced_3) ──────────────────
_REQUIRED_LABEL_COLS = ("PATID_A", "PATID_B")
_BINARY_MAP = {
    True: 1, False: 0, 1: 1, 0: 0,
    "True": 1, "False": 0, "TRUE": 1, "FALSE": 0, "true": 1, "false": 0,
    "1": 1, "0": 0,
}


def _coerce_binary_label(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    coerced = series.map(_BINARY_MAP)
    if coerced.isna().any():
        bad = sorted(series[coerced.isna()].astype(str).unique())[:5]
        raise ValueError(
            f"Label column has unrecognised values (e.g. {bad}); expected "
            "True/False, 0/1, or their string spellings."
        )
    return coerced.astype(int)


def _load_silver_labels(path: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"PATID_A": str, "PATID_B": str})
    missing = [c for c in (*_REQUIRED_LABEL_COLS, label_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Silver labels {path} missing required column(s) {missing}; "
            f"have {sorted(df.columns)}."
        )
    df[label_col] = _coerce_binary_label(df[label_col])
    return df


def _canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    a = out[["PATID_A", "PATID_B"]].min(axis=1)
    b = out[["PATID_A", "PATID_B"]].max(axis=1)
    out["PATID_A"] = a
    out["PATID_B"] = b
    return out


def _stratified_split(
    labels: pd.DataFrame, label_col: str, test_size: float, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the binary label (no sklearn dependency)."""
    test_parts = [
        grp.sample(frac=test_size, random_state=seed)
        for _, grp in labels.groupby(label_col)
    ]
    test = pd.concat(test_parts)
    train = labels.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ── Metrics ──────────────────────────────────────────────────────────────────
def _binary_metrics(y_true: pd.Series, y_pred_pos: pd.Series) -> dict:
    tp = int(((y_true == 1) & y_pred_pos).sum())
    fp = int(((y_true == 0) & y_pred_pos).sum())
    fn = int(((y_true == 1) & ~y_pred_pos).sum())
    tn = int(((y_true == 0) & ~y_pred_pos).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def _evaluate_test_split(
    model: FSMatcher, df_clean: pd.DataFrame, trained_settings: dict,
    test_labels: pd.DataFrame, label_col: str,
) -> dict:
    """Score the held-out test pairs and compute confusion + precision/recall."""
    canon = _canonicalize_pairs(test_labels)[["PATID_A", "PATID_B", label_col]]
    pairs = canon[["PATID_A", "PATID_B"]].drop_duplicates().reset_index(drop=True)
    logger.info("Scoring %d held-out test pairs for metrics...", len(pairs))
    classified = model.score(pairs, df_clean, trained_settings)
    merged = classified.merge(canon, on=["PATID_A", "PATID_B"], how="inner")
    n_unscored = len(canon) - len(merged)
    if n_unscored:
        logger.warning("%d/%d test pairs did not resolve and were dropped from metrics.",
                       n_unscored, len(canon))
    y_true = merged[label_col]
    tier = merged["classification_tier"]
    confusion = (
        merged.groupby([label_col, "classification_tier"]).size().unstack(fill_value=0)
    )
    logger.info("Test confusion (rows=label, cols=tier):\n%s", confusion)
    return {
        "n_test_pairs": int(len(merged)),
        "n_test_positives": int((y_true == 1).sum()),
        "n_test_negatives": int((y_true == 0).sum()),
        "confusion_tier_by_label": {
            int(lbl): {t: int(confusion.loc[lbl].get(t, 0)) for t in confusion.columns}
            for lbl in confusion.index
        },
        "metrics_auto_merge": _binary_metrics(y_true, tier == "auto_merge"),
        "metrics_merge_or_review": _binary_metrics(
            y_true, tier.isin(["auto_merge", "human_review"])
        ),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    s = default_settings
    default_silver = s.project_root / "data" / "silver_labels" / "silver_labels_v1_2026_06_21.csv"
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cleaned-index", type=Path, default=None,
                   help=f"Cleaned records parquet. Default: latest in {s.processed_dir}.")
    p.add_argument("--candidate-pairs", type=Path, default=None,
                   help=f"Full candidate-pairs parquet. Default: latest in {s.blocking_dir}.")
    p.add_argument("--silver-labels", type=Path, default=default_silver,
                   help=f"Silver-labels CSV (PATID_A, PATID_B, silver_label). Default: {default_silver}.")
    p.add_argument("--label-col", default="silver_label",
                   help="Label column in --silver-labels (default: 'silver_label').")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction of silver labels held out for testing (default: 0.2).")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Seed for the stratified train/test split (default: 42).")
    p.add_argument("--data-version", default=None,
                   help="Tag used in output filenames. Default: parsed from candidate pairs.")
    p.add_argument("--u-max-pairs", type=float, default=U_MAX_PAIRS,
                   help="Splink u-sampling max_pairs (default 1e6).")
    p.add_argument("--promote", action="store_true",
                   help="After training, run the deploy-gate and point active.json at the new model.")
    p.add_argument("--force-promote", action="store_true",
                   help="Promote even if the deploy-gate fails (use with care).")
    p.add_argument("--log-level", default=None, help="Override EMPI_LOG_LEVEL.")
    return p.parse_args()


def _train_once(
    df_clean: pd.DataFrame, scoring_pool: pd.DataFrame, train_labels: pd.DataFrame,
    label_col: str, u_max_pairs: float, settings,
):
    """Train FSMatcher on the train split + score the full pool. Returns
    (model, classified_full, trained_settings)."""
    model = FSMatcher(
        labels_df=train_labels, label_col=label_col,
        classification_config=classification_config_from_settings(settings),
        u_max_pairs=u_max_pairs, include_address=True,
    )
    classified_full, linker = model.run(
        scoring_pool, df_clean, full_output=True, return_linker=True,
    )
    trained_settings = linker.misc.save_model_to_json()
    return model, classified_full, trained_settings


def main() -> None:
    args = parse_args()
    settings = default_settings
    configure_logging(settings, level=args.log_level)
    settings.ensure_dirs()

    # ── Resolve inputs ────────────────────────────────────────────────────────
    cleaned_path = args.cleaned_index or _resolve_latest(settings.processed_dir, CLEANED_GLOBS)
    pool_path = args.candidate_pairs or _resolve_latest(settings.blocking_dir, CANDIDATE_PAIRS_GLOBS)
    if not args.silver_labels.exists():
        raise FileNotFoundError(
            f"Silver labels not found: {args.silver_labels}. VM-only (gitignored PHI)."
        )
    data_version = args.data_version or _derive_data_version(pool_path)
    logger.info("cleaned=%s pool=%s labels=%s version=%s",
                cleaned_path, pool_path, args.silver_labels, data_version)

    # ── Load ──────────────────────────────────────────────────────────────────
    df_clean = pd.read_parquet(cleaned_path)
    scoring_pool = pd.read_parquet(pool_path)
    silver = _load_silver_labels(args.silver_labels, args.label_col)
    train_labels, test_labels = _stratified_split(
        silver, args.label_col, args.test_size, args.split_seed,
    )
    logger.info(
        "records=%d pool=%d silver=%d (pos=%d) -> train=%d test=%d",
        len(df_clean), len(scoring_pool), len(silver),
        int((silver[args.label_col] == 1).sum()), len(train_labels), len(test_labels),
    )

    # ── Train (with single-CPU retry) ─────────────────────────────────────────
    try:
        model, classified_full, trained_settings = _train_once(
            df_clean, scoring_pool, train_labels, args.label_col, args.u_max_pairs, settings,
        )
    except RuntimeError as exc:
        logger.warning("Retrying with u_max_pairs=1e4 due to: %s", exc)
        model, classified_full, trained_settings = _train_once(
            df_clean, scoring_pool, train_labels, args.label_col, 1e4, settings,
        )

    # ── Persist model JSON + meta sidecar ─────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = settings.fs_model_dir / f"fs_model_{ts}.json"
    model_path.write_text(json.dumps(trained_settings, indent=2, default=str))
    logger.info("Wrote trained model -> %s", model_path)

    test_metrics = _evaluate_test_split(
        model, df_clean, trained_settings, test_labels, args.label_col,
    )
    logger.info("Held-out auto_merge metrics: %s", test_metrics["metrics_auto_merge"])

    meta = {
        "model_name": MODEL_NAME,
        "model_file": model_path.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "data_version": data_version,
        "silver_labels_path": str(args.silver_labels),
        "seed": args.split_seed,
        "u_max_pairs": args.u_max_pairs,
        "thresholds": {
            "auto_merge": settings.fs_auto_merge_threshold,
            "review_floor": settings.fs_review_floor,
        },
        "split": {
            "test_size": args.test_size,
            "n_train": int(len(train_labels)),
            "n_train_positives": int((train_labels[args.label_col] == 1).sum()),
            "n_test": int(len(test_labels)),
        },
        "test_metrics": test_metrics,
    }
    registry.meta_path_for(model_path).write_text(json.dumps(meta, indent=2))
    logger.info("Wrote model meta -> %s", registry.meta_path_for(model_path))

    # ── Labeled GBT training feature parquet ──────────────────────────────────
    feats = model.to_fs_features(
        classified_full, labels_df=silver, label_col=args.label_col, candidates_only=True,
    )
    labeled = feats[feats["label"].notna()].reset_index(drop=True)
    feat_path = settings.fs_output_dir / f"fs_features_train_{data_version}.parquet"
    labeled.to_parquet(feat_path, index=False)
    logger.info(
        "Wrote labeled GBT training features -> %s (%d labeled candidate rows of %d candidates)",
        feat_path, len(labeled), len(feats),
    )

    # ── Optional promotion (deploy-gate) ──────────────────────────────────────
    if args.promote or args.force_promote:
        try:
            reason = registry.promote(model_path, settings, force=args.force_promote)
            logger.info("PROMOTED: %s", reason)
        except registry.DeployGateError as exc:
            logger.error("NOT PROMOTED: %s", exc)
            sys.exit(1)
    else:
        logger.info("Model trained but NOT promoted (pass --promote to activate).")


if __name__ == "__main__":
    main()
