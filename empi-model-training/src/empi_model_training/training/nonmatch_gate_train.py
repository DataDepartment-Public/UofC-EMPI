"""Non-match gate training — Azure ML job entry point.

    uv run python -m empi_model_training.training.nonmatch_gate_train \
        --cleaned-index PATH --gold-labels PATH [--promote]

**Independent implementation.** This repo does not import or call into
`empi-service` (a deliberate project decision — see `CLAUDE.md`). Both the
feature engineering and the training loop (previously only a notebook,
`empi-service/notebooks/ml_model/confident_nonmatch/
pair_classifier_lightgbm_nonmatch_gate_v1.ipynb`) are reproduced here as
independent code, faithful to the documented behavior in
`empi-service/docs/Nonmatch-Gate-Guide.md` — not shared or imported.

This is the pipeline's Stage-4.25 non-match gate — NOT the Stage-4.5
`ml_matcher` (`lightgbm_train.py`). It has its own model store
(`models/nonmatch_gate/`), its own registry, and its own target: `1` =
plausible (`final_gold_label | ambiguous_pair`, i.e. match ∪ ambiguous), `0`
= confident non-match. Trained on the WHOLE candidate pool, exactly the
complement of the matcher's population split. Feature engineering
(`build_features`/`FEATURE_COLS`) is identical to `lightgbm_train.py`'s --
the gate and the matcher are meant to see identical inputs, per
`src.models.ml_matcher.lightgbm_v5.FeatureBuilderV5` on the service side --
duplicated here rather than imported across these two training scripts,
consistent with this repo's "independent reimplementation" rule and the
service's own choice to duplicate rather than share across model packages.

Served as a plain `joblib.dump` of the raw fitted classifier -- no adapter
wrapper. `predict_proba(X)[:, 1]` is already `P(plausible)`, the served
score, with no inversion to guard.

The dropped pairs are the costly error here (unrecoverable downstream), so
the operating point favors recall on the positive (plausible) class. Default
threshold: 0.30 -- same as `EMPI_GATE_THRESHOLD`'s default in empi-service.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jellyfish
import mlflow
import numpy as np
import pandas as pd

from empi_model_training.training import registry_utils

logger = logging.getLogger("empi_model_training.nonmatch_gate_train")

MODEL_NAME = "nonmatch_gate"
RANDOM_STATE = 42

# ── Feature roster (identical to lightgbm_train.py's -- see module docstring) ─
MISSING, SAME, DIFFERENT = "missing", "same", "different"
COMPARE_LEVELS = [MISSING, SAME, DIFFERENT]
CATEGORICAL_FEATURES = ["sound_first", "sound_last", "cmp_street_num"]
NUMERIC_FEATURES = [
    "sim_jw_first",
    "sim_jw_last",
    "sim_jw_middle",
    "sim_lev_email",
    "sim_lev_address1",
    "addr_token_jaccard",
    "ssn_digit_frac",
    "sim_dob",
    "sim_phones",
]
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

_CLEAN_COLS = [
    "FirstNM_clean",
    "LastNM_clean",
    "MiddleNM_clean",
    "BirthDT_clean",
    "SSN_clean",
    "Email_clean",
    "AddressLine1_clean",
    "Phones_set",
]

_num_re = re.compile(r"\d+")
_NA_TOKENS = {"nan", "none", "<na>", "nat", "null"}


# ── Scalar helpers (identical to lightgbm_train.py's) ────────────────────────
def _norm(x: Any) -> str | None:
    if isinstance(x, str):
        s = x.strip().lower()
        return s if s and s not in _NA_TOKENS else None
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    s = str(x).strip().lower()
    return s if s and s not in _NA_TOKENS else None


def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(object).map(_norm)


def _parse_set(value: Any) -> set[str]:
    if isinstance(value, (set, frozenset, list, tuple, np.ndarray)):
        return {str(p).strip() for p in value if str(p).strip()}
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return set()
        except (TypeError, ValueError):
            pass
        return set()
    v = value.strip()
    if v in ("", "nan", "None", "set()", "{}", "[]"):
        return set()
    try:
        parsed = ast.literal_eval(v)
        if isinstance(parsed, (set, list, tuple)):
            return {str(p).strip() for p in parsed if str(p).strip()}
    except (ValueError, SyntaxError, TypeError):
        pass
    cleaned = v.strip("{}[]").replace("'", "").replace('"', "")
    return {p.strip() for p in cleaned.split(",") if p.strip()}


def _jw(x: Any, y: Any) -> float:
    if not (isinstance(x, str) and isinstance(y, str)):
        return float("nan")
    return float(jellyfish.jaro_winkler_similarity(x, y))


def _lev_sim(x: Any, y: Any) -> float:
    if not (isinstance(x, str) and isinstance(y, str)):
        return float("nan")
    m = max(len(x), len(y))
    return 1.0 - jellyfish.levenshtein_distance(x, y) / m if m else float("nan")


def _metaphone(x: Any) -> str | None:
    s = _norm(x)
    if not s:
        return None
    return jellyfish.metaphone(s) or None


def _street_num(x: Any) -> str | None:
    s = _norm(x)
    if not s:
        return None
    m = _num_re.search(s)
    return m.group() if m else None


def _tok_jaccard(x: Any, y: Any) -> float:
    xn, yn = _norm(x), _norm(y)
    if not xn or not yn:
        return float("nan")
    sx, sy = set(xn.split()), set(yn.split())
    return len(sx & sy) / len(sx | sy) if sx and sy else float("nan")


def _ssn_frac(x: Any, y: Any) -> float:
    xn, yn = _norm(x), _norm(y)
    if not xn or not yn:
        return float("nan")
    n = min(len(xn), len(yn))
    if n == 0:
        return float("nan")
    return sum(1 for i in range(n) if xn[i] == yn[i]) / max(len(xn), len(yn))


def _phones_best_jw(av: Any, bv: Any) -> float:
    a_set, b_set = _parse_set(av), _parse_set(bv)
    if not a_set or not b_set:
        return float("nan")
    return max(jellyfish.jaro_winkler_similarity(x, y) for x in a_set for y in b_set)


def _cmp_categorical(a_norm: pd.Series, b_norm: pd.Series) -> pd.Categorical:
    missing = a_norm.isna() | b_norm.isna()
    same = (a_norm == b_norm) & ~missing
    out = np.where(missing, MISSING, np.where(same, SAME, DIFFERENT))
    return pd.Categorical(out, categories=COMPARE_LEVELS)


# ── Feature builder (identical to lightgbm_train.py's) ──────────────────────
def _side(pairs: pd.DataFrame, col: str, side: str) -> pd.Series:
    name = f"{col}_{side}"
    if name in pairs.columns:
        return pairs[name]
    return pd.Series(np.nan, index=pairs.index, dtype="object")


def _sim_column(pairs: pd.DataFrame, col: str, fn: Any) -> pd.Series:
    a = _norm_series(_side(pairs, col, "A"))
    b = _norm_series(_side(pairs, col, "B"))
    return pd.Series(
        [fn(x, y) for x, y in zip(a, b, strict=True)], index=pairs.index, dtype="float"
    )


def _pair_apply(pairs: pd.DataFrame, col: str, fn: Any) -> pd.Series:
    return pd.Series(
        [fn(x, y) for x, y in zip(_side(pairs, col, "A"), _side(pairs, col, "B"), strict=True)],
        index=pairs.index,
        dtype="float",
    )


def _sound_cmp(pairs: pd.DataFrame, col: str) -> pd.Categorical:
    a = _side(pairs, col, "A").map(_metaphone)
    b = _side(pairs, col, "B").map(_metaphone)
    return _cmp_categorical(a, b)


def materialize_pairs(pairs: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Join cleaned attributes onto both sides of each pair. `pairs` must have
    a plain 0..n-1 index (reset by the caller) so row order is preserved."""
    clean = df_clean
    if "PATID" in clean.columns:
        clean = clean.set_index("PATID")
    clean = clean[[c for c in _CLEAN_COLS if c in clean.columns]]
    clean = clean[~clean.index.duplicated(keep="first")]
    idx_a = pairs["PATID_A"].astype(str).to_numpy()
    idx_b = pairs["PATID_B"].astype(str).to_numpy()
    a = clean.add_suffix("_A").reindex(idx_a).reset_index(drop=True)
    b = clean.add_suffix("_B").reindex(idx_b).reset_index(drop=True)
    base = pairs[["PATID_A", "PATID_B"]].reset_index(drop=True)
    return pd.concat([base, a, b], axis=1)


def build_features(pairs: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """`pairs` needs `PATID_A`/`PATID_B`. Returns the 12 FEATURE_COLS + PATID_A/B,
    row order preserved. A missing `MiddleNM_clean` degrades gracefully to NaN."""
    joined = materialize_pairs(pairs, df_clean)
    feat = pd.DataFrame(index=joined.index)

    for feat_name, col in [
        ("sim_jw_first", "FirstNM_clean"),
        ("sim_jw_last", "LastNM_clean"),
        ("sim_jw_middle", "MiddleNM_clean"),
    ]:
        feat[feat_name] = _sim_column(joined, col, _jw)

    feat["sim_lev_email"] = _sim_column(joined, "Email_clean", _lev_sim)
    feat["sim_lev_address1"] = _sim_column(joined, "AddressLine1_clean", _lev_sim)
    feat["sound_first"] = _sound_cmp(joined, "FirstNM_clean")
    feat["sound_last"] = _sound_cmp(joined, "LastNM_clean")

    sn_a = _side(joined, "AddressLine1_clean", "A").map(_street_num)
    sn_b = _side(joined, "AddressLine1_clean", "B").map(_street_num)
    feat["cmp_street_num"] = _cmp_categorical(sn_a, sn_b)
    feat["addr_token_jaccard"] = _pair_apply(joined, "AddressLine1_clean", _tok_jaccard)
    feat["ssn_digit_frac"] = _pair_apply(joined, "SSN_clean", _ssn_frac)

    dob_a = pd.to_datetime(_side(joined, "BirthDT_clean", "A"), errors="coerce")
    dob_b = pd.to_datetime(_side(joined, "BirthDT_clean", "B"), errors="coerce")
    sa, sb = dob_a.dt.strftime("%Y%m%d"), dob_b.dt.strftime("%Y%m%d")
    feat["sim_dob"] = pd.Series(
        [_lev_sim(x, y) for x, y in zip(sa, sb, strict=True)], index=joined.index, dtype="float"
    )
    feat["sim_phones"] = pd.Series(
        [
            _phones_best_jw(av, bv)
            for av, bv in zip(
                _side(joined, "Phones_set", "A"), _side(joined, "Phones_set", "B"), strict=True
            )
        ],
        index=joined.index,
        dtype="float",
    )

    for c in CATEGORICAL_FEATURES:
        feat[c] = pd.Categorical(feat[c], categories=COMPARE_LEVELS)
    for c in NUMERIC_FEATURES:
        feat[c] = feat[c].astype("float")

    out = feat[FEATURE_COLS].copy()
    out.insert(0, "PATID_B", joined["PATID_B"].to_numpy())
    out.insert(0, "PATID_A", joined["PATID_A"].to_numpy())
    return out.reset_index(drop=True)


# ── Data ingestion + target ───────────────────────────────────────────────────
def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False})
    )
    return mapped.fillna(False).astype(bool)


def load_gold_pairs(gold_labels_path: Path) -> pd.DataFrame:
    """Gold labels over the WHOLE candidate pool -- no population filter.
    Target: class 1 = plausible (`final_gold_label | ambiguous_pair`, i.e.
    match ∪ ambiguous); class 0 = confident non-match. The exact complement
    of `lightgbm_train.py`'s `target_confident_match` split."""
    gold = pd.read_csv(gold_labels_path, dtype={"PATID_A": str, "PATID_B": str})
    gold["final_gold_label"] = _to_bool(gold["final_gold_label"])
    gold["ambiguous_pair"] = _to_bool(gold["ambiguous_pair"])
    gold["target_plausible"] = (gold["final_gold_label"] | gold["ambiguous_pair"]).astype(int)
    logger.info(
        "gold=%d, plausible rate=%.3f (confident match=%d, ambiguous=%d)",
        len(gold),
        gold["target_plausible"].mean(),
        int((gold["final_gold_label"] & ~gold["ambiguous_pair"]).sum()),
        int(gold["ambiguous_pair"].sum()),
    )
    return gold


def _canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["PATID_A"], out["PATID_B"] = (
        out[["PATID_A", "PATID_B"]].min(axis=1),
        out[["PATID_A", "PATID_B"]].max(axis=1),
    )
    return out


def merge_reviewer_labels(pairs: pd.DataFrame, reviewer_labels_path: Path | None) -> pd.DataFrame:
    """Fold in reviewer-confirmed labels (empi-service's
    scripts/export_reviewer_labels.py output -- PATID_A, PATID_B,
    reviewer_label) into the whole pool built by `load_gold_pairs`.

    A reviewer decision is always definitive -- never ambiguous_pair -- and
    is always `target_plausible=1` regardless of which way `reviewer_label`
    went: a reviewer only ever acts on a pair that reached the review queue,
    meaning something upstream already judged it plausible enough to show a
    human (never a pair the gate would have dropped as a confident
    non-match). Reviewer rows win over gold labels for any pair present in
    both. No-op if not given.
    """
    if reviewer_labels_path is None:
        return pairs
    reviewer = pd.read_csv(reviewer_labels_path, dtype={"PATID_A": str, "PATID_B": str})
    reviewer["final_gold_label"] = reviewer["reviewer_label"].astype(bool)
    reviewer["ambiguous_pair"] = False
    reviewer["target_plausible"] = 1
    reviewer = reviewer[
        ["PATID_A", "PATID_B", "final_gold_label", "ambiguous_pair", "target_plausible"]
    ]

    combined = pd.concat(
        [_canonicalize_pairs(pairs), _canonicalize_pairs(reviewer)], ignore_index=True
    )
    return combined.drop_duplicates(subset=["PATID_A", "PATID_B"], keep="last").reset_index(
        drop=True
    )


# ── Split + training (same hyperparameters as lightgbm_train.py) ────────────
LGBM_PARAMS: dict[str, Any] = dict(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


@dataclass
class SplitData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def split_60_20_20(feat: pd.DataFrame, y: pd.Series, seed: int) -> SplitData:
    from sklearn.model_selection import train_test_split

    x = feat[FEATURE_COLS]
    idx = np.arange(len(x))
    idx_trainval, idx_test = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.25, random_state=seed, stratify=y.iloc[idx_trainval]
    )
    return SplitData(
        x_train=x.iloc[idx_train],
        y_train=y.iloc[idx_train],
        x_val=x.iloc[idx_val],
        y_val=y.iloc[idx_val],
        x_test=x.iloc[idx_test],
        y_test=y.iloc[idx_test],
    )


def train_lightgbm(split: SplitData) -> Any:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        split.x_train,
        split.y_train,
        eval_set=[(split.x_val, split.y_val)],
        eval_metric="auc",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


@dataclass
class TrainResult:
    model_path: Path
    meta: dict[str, Any]
    roc_auc: float
    pr_auc: float


def run(args: argparse.Namespace) -> TrainResult:
    from sklearn.metrics import average_precision_score, roc_auc_score

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_gold_pairs(args.gold_labels)
    if args.reviewer_labels is not None:
        before = len(pairs)
        pairs = merge_reviewer_labels(pairs, args.reviewer_labels)
        logger.info(
            "merged reviewer labels from %s: %d -> %d rows (reviewer wins on overlap)",
            args.reviewer_labels,
            before,
            len(pairs),
        )
    df_clean = pd.read_parquet(args.cleaned_index)
    feat = build_features(pairs[["PATID_A", "PATID_B"]], df_clean)
    y = pairs["target_plausible"].astype(int).reset_index(drop=True)

    split = split_60_20_20(feat, y, args.split_seed)
    for name, yy in [("train", split.y_train), ("val", split.y_val), ("test", split.y_test)]:
        logger.info(
            "%s: n=%d plausible=%d rate=%.3f", name, len(yy), int(yy.sum()), float(yy.mean())
        )

    model = train_lightgbm(split)
    logger.info(
        "best_iteration=%s best_val_auc=%s",
        model.best_iteration_,
        model.best_score_["valid_0"]["auc"],
    )

    proba_test = model.predict_proba(split.x_test)[:, 1]  # P(plausible) -- served score, no adapter
    roc_auc = float(roc_auc_score(split.y_test, proba_test))
    pr_auc = float(average_precision_score(split.y_test, proba_test))

    y_pos = split.y_test.to_numpy()
    keep_pred = (proba_test >= args.gate_threshold).astype(int)
    tp = int(((keep_pred == 1) & (y_pos == 1)).sum())
    fp = int(((keep_pred == 1) & (y_pos == 0)).sum())
    fn = int(((keep_pred == 0) & (y_pos == 1)).sum())
    tn = int(((keep_pred == 0) & (y_pos == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # plausible recall == 1 - miss rate

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_path = model_dir / f"nonmatch_gate_{ts}.pkl"
    import joblib

    joblib.dump(model, model_path)  # raw fitted model -- no adapter wrapper needed
    logger.info("Wrote trained model -> %s", model_path)

    meta = {
        "model_name": MODEL_NAME,
        "model_file": model_path.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "notes": (
            "LightGBM non-match gate -- the pipeline Stage-4.25 gate, NOT the "
            "Stage-4.5 ml_matcher. score = predict_proba[:, 1] = P(plausible) "
            "= P(match or ambiguous). High score -> keep pair; low score -> "
            "confident not-match (discard)."
        ),
        "feature_cols": list(FEATURE_COLS),
        "gate_threshold": args.gate_threshold,
        "test_metrics": {
            "n_test": int(len(split.y_test)),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "gate_at_threshold": {
                "threshold": args.gate_threshold,
                "plausible_precision": round(precision, 4),
                "plausible_recall": round(recall, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
        },
    }
    registry_utils.meta_path_for(model_path).write_text(json.dumps(meta, indent=2))
    logger.info(
        "gate @P(plausible)>=%.2f: P=%.3f R=%.3f (tp=%d fp=%d fn=%d)",
        args.gate_threshold,
        precision,
        recall,
        tp,
        fp,
        fn,
    )

    if args.promote or args.force_promote:
        try:
            reason = registry_utils.promote(
                model_path,
                model_dir,
                args.deploy_gate_margin,
                force=args.force_promote,
                metric_fn=registry_utils.gate_pr,
            )
            logger.info("PROMOTED: %s", reason)
        except registry_utils.DeployGateError as exc:
            logger.error("NOT PROMOTED: %s", exc)
            raise

    return TrainResult(model_path=model_path, meta=meta, roc_auc=roc_auc, pr_auc=pr_auc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cleaned-index", type=Path, required=True, help="Cleaned records parquet.")
    p.add_argument(
        "--gold-labels",
        type=Path,
        required=True,
        help="Gold-labels CSV (PATID_A, PATID_B, final_gold_label, ambiguous_pair).",
    )
    p.add_argument(
        "--reviewer-labels",
        type=Path,
        default=None,
        help="Optional reviewer-confirmed labels CSV (PATID_A, PATID_B, reviewer_label) from "
        "empi-service's scripts/export_reviewer_labels.py. Wins over --gold-labels for any "
        "pair present in both.",
    )
    p.add_argument("--split-seed", type=int, default=RANDOM_STATE)
    p.add_argument("--gate-threshold", type=float, default=0.30)
    p.add_argument("--deploy-gate-margin", type=float, default=0.02)
    p.add_argument("--model-dir", type=Path, default=Path("models/nonmatch_gate"))
    p.add_argument("--promote", action="store_true")
    p.add_argument("--force-promote", action="store_true")
    return p.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    args = parse_args()

    mlflow.set_experiment("empi-nonmatch-gate")
    with mlflow.start_run(run_name="nonmatch-gate-train") as run_ctx:
        mlflow.log_params(
            {
                **LGBM_PARAMS,
                "split_seed": args.split_seed,
                "gate_threshold": args.gate_threshold,
            }
        )
        result = run(args)
        gate_at_threshold = result.meta["test_metrics"]["gate_at_threshold"]
        mlflow.log_metrics(
            {
                "roc_auc": result.roc_auc,
                "pr_auc": result.pr_auc,
                **{
                    f"gate_{k}": v
                    for k, v in gate_at_threshold.items()
                    if isinstance(v, int | float)
                },
            }
        )
        mlflow.log_artifact(str(result.model_path), artifact_path="model")
        mlflow.log_artifact(
            str(registry_utils.meta_path_for(result.model_path)), artifact_path="model"
        )

        if mlflow.get_tracking_uri().startswith("azureml"):
            mlflow.register_model(f"runs:/{run_ctx.info.run_id}/model", "empi-nonmatch-gate")
        else:
            logger.info(
                "Not running under Azure ML tracking -- skipping model registry registration."
            )


if __name__ == "__main__":
    main()
