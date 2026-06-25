"""run_real_enhanced_3.py — score the real cohort with FSEnhanced3 (Phase E3-4).

*** RUN THIS ON THE VM ONLY. NEVER in the sandbox / off the VM. ***

This script never logs or prints individual PATID values, names, SSNs, DOBs, or
any other identifier — only aggregate counts, non-PHI score quantiles, and
test-split evaluation metrics.

enhanced_3 trains m supervised from the **real-cohort silver labels**
(`data/silver_labels/silver_labels_v1_2026_06_21.csv`, VM-only). The labels are
split into a TRAIN portion (positives → m via estimate_m_from_pairwise_labels)
and a held-out TEST portion (both classes → precision / recall / confusion
matrix). lambda is seeded from the deterministic rules; u is random-sampled on
the cohort.

Because the silver labels are Stage-3 deterministic confirmations, Stage 3 has
already removed them from `non_matches`. So for the production scoring pool you
will usually want `--score-full-candidate-pool` (the full pre-rules blocking
output). The held-out TEST split is always scored directly through the trained
model (a second linker built from the trained settings), so test metrics are
produced regardless of which scoring pool you choose.

Inputs (auto-resolved via `models.common.versioning.latest_versioned`; override
with CLI flags):
    data/processed/MDM_Population_cleaned_v*_*.parquet           (records, real PHI)
    data/silver_labels/silver_labels_v1_2026_06_21.csv          (PATID_A, PATID_B, silver_label[True/False])
    EITHER data/non_matches/non_matches_v*_*.parquet            (default scoring pool)
    OR     data/blocking/candidate_pairs_v*_*.parquet           (--score-full-candidate-pool)

Outputs:
    models/outputs/fs_splink_enhanced_3__<data-version>[_full_pool].parquet
        5-column cross-model eval schema (validation notebook §12 reads this).
    data/matches_model_v2/fs_splink_enhanced_3_matches_model__<data-version>[_full_pool].parquet
        ProbabilisticMatches contract (validated).
    models/artifacts/fs_splink_enhanced_3/diagnostics__<data-version>[_full_pool].json
        Non-PHI: tier counts, score quantiles, trained m/u settings, and the
        held-out test-split metrics (precision / recall / F1 / confusion).

USAGE:
    # Production scoring of the full candidate pool (recommended for eval).
    python -m models.experiments.fs_splink_enhanced_3.run_real_enhanced_3 \\
        --score-full-candidate-pool

    # Pin explicit paths + version tag.
    python -m models.experiments.fs_splink_enhanced_3.run_real_enhanced_3 \\
        --cleaned-index data/processed/MDM_Population_cleaned_v3_2026_06_11.parquet \\
        --candidate-pairs data/blocking/candidate_pairs_v4_2026_06_11.parquet \\
        --silver-labels data/silver_labels/silver_labels_v1_2026_06_21.csv \\
        --score-full-candidate-pool --data-version v4_2026_06_11
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from models.common.fs_base import _register_candidate_pairs  # noqa: E402
from models.common.versioning import (  # noqa: E402
    latest_versioned,
    version_tag_from_filename,
)
from models.experiments.fs_splink_enhanced_3.fs_enhanced_3 import (  # noqa: E402
    MODEL_NAME,
    FSEnhanced3,
)
from src.contracts import ProbabilisticMatches, validate as validate_contract  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths + naming conventions ────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "models" / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts" / MODEL_NAME
MATCHES_MODEL_V2_DIR = PROJECT_ROOT / "data" / "matches_model_v2"

CLEANED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_GLOBS = (
    "MDM_Population_cleaned_v*_*.parquet",  # standalone CLI
    "cleaned_*Z.parquet",                    # pipeline orchestrator
)

NON_MATCHES_DIR = PROJECT_ROOT / "data" / "non_matches"
NON_MATCHES_GLOBS = (
    "non_matches_v*_*.parquet",
    "non_matches_*Z.parquet",
)

CANDIDATE_PAIRS_DIRS = (
    PROJECT_ROOT / "data" / "blocking",
    PROJECT_ROOT / "src" / "features" / "outputs" / "blocking",
)
CANDIDATE_PAIRS_GLOBS = (
    "candidate_pairs_v*_*.parquet",
    "candidate_pairs_*Z.parquet",
)

DEFAULT_SILVER_LABELS = (
    PROJECT_ROOT / "data" / "silver_labels" / "silver_labels_v1_2026_06_21.csv"
)

# Production default (multi-core VM). FSEnhanced3's _estimate_u_with_guard
# auto-raises a clear RuntimeError if single-CPU; we catch and retry with 1e4.
U_MAX_PAIRS = 1e6

_RUN_ID_RE = __import__("re").compile(r"(\d{8}T\d{6}Z)")


def _derive_data_version(path: Path) -> str:
    tag = version_tag_from_filename(path)
    if tag != "unversioned":
        return tag
    m = _RUN_ID_RE.search(path.name)
    if m:
        return m.group(1)
    return path.stem


def _resolve_with_fallback(dirs, globs) -> Path:
    if isinstance(dirs, Path):
        dirs = (dirs,)
    searched: list[str] = []
    for d in dirs:
        searched.append(str(d))
        if not d.is_dir():
            continue
        try:
            return latest_versioned(d, globs[0])
        except FileNotFoundError:
            pass
    for d in dirs:
        if not d.is_dir():
            continue
        for pattern in globs[1:]:
            candidates = sorted(d.glob(pattern))
            if candidates:
                return candidates[-1]
    raise FileNotFoundError(
        f"No files matching {globs} in any of {searched}. Confirm the upstream "
        "stage has run, or pass the path explicitly."
    )


# ── Silver-label handling ─────────────────────────────────────────────────────
_REQUIRED_LABEL_COLS = ("PATID_A", "PATID_B")

# The silver-labels file encodes the class as booleans (True/False); accept those
# plus the common 0/1 and string spellings and normalize to int {0, 1}.
_BINARY_MAP = {
    True: 1, False: 0, 1: 1, 0: 0,
    "True": 1, "False": 0, "TRUE": 1, "FALSE": 0, "true": 1, "false": 0,
    "1": 1, "0": 0,
}


def _coerce_binary_label(series: pd.Series) -> pd.Series:
    """Normalize a label column (bool True/False, 0/1, or their string forms) to int."""
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
            f"have {sorted(df.columns)}. Expected PATID_A, PATID_B, {label_col!r} "
            "(label values True/False)."
        )
    df[label_col] = _coerce_binary_label(df[label_col])
    return df


def _canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with PATID_A/PATID_B reordered so PATID_A < PATID_B (matches the
    canonicalization FSModel.predict applies to scored output)."""
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


# ── Test-split scoring + metrics ──────────────────────────────────────────────
def _score_pairs_with_trained_model(
    model: FSEnhanced3,
    df_model: pd.DataFrame,
    trained_settings: dict,
    pairs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score an arbitrary set of pairs through the already-trained model.

    Builds a fresh Splink linker from the TRAINED settings (m/u/lambda already
    estimated) over the same records frame and the given candidate pairs — no
    retraining. `pairs_df` carries no n_blocks, so classify() applies no
    n_blocks bump: test metrics reflect the raw model score.
    """
    from splink import DuckDBAPI, Linker

    db_api = DuckDBAPI()
    _register_candidate_pairs(db_api, pairs_df, model.candidate_pairs_table_name)
    linker = Linker(df_model, trained_settings, db_api=db_api)
    predictions = model.predict(linker, pairs_df)
    return model.classify(predictions)


def _binary_metrics(y_true: pd.Series, y_pred_pos: pd.Series) -> dict:
    tp = int(((y_true == 1) & y_pred_pos).sum())
    fp = int(((y_true == 0) & y_pred_pos).sum())
    fn = int(((y_true == 1) & ~y_pred_pos).sum())
    tn = int(((y_true == 0) & ~y_pred_pos).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _evaluate_test_split(
    model: FSEnhanced3,
    df_model: pd.DataFrame,
    trained_settings: dict,
    test_labels: pd.DataFrame,
    label_col: str,
) -> dict:
    """Score the held-out test pairs and compute confusion + precision/recall."""
    canon = _canonicalize_pairs(test_labels)[["PATID_A", "PATID_B", label_col]]
    pairs = canon[["PATID_A", "PATID_B"]].drop_duplicates().reset_index(drop=True)
    logger.info("Scoring %d held-out test pairs for metrics...", len(pairs))
    classified = _score_pairs_with_trained_model(model, df_model, trained_settings, pairs)

    merged = classified.merge(canon, on=["PATID_A", "PATID_B"], how="inner")
    n_unscored = len(canon) - len(merged)
    if n_unscored:
        logger.warning(
            "%d/%d test pairs did not resolve in the records frame and were "
            "dropped from metrics.", n_unscored, len(canon),
        )

    y_true = merged[label_col]
    tier = merged["classification_tier"]
    # Tier × label confusion (rows = label, cols = tier).
    confusion = (
        merged.groupby([label_col, "classification_tier"]).size().unstack(fill_value=0)
    )
    logger.info("Test confusion (rows=label, cols=tier):\n%s", confusion)

    metrics = {
        "n_test_pairs": int(len(merged)),
        "n_test_positives": int((y_true == 1).sum()),
        "n_test_negatives": int((y_true == 0).sum()),
        "confusion_tier_by_label": {
            int(lbl): {t: int(confusion.loc[lbl].get(t, 0)) for t in confusion.columns}
            for lbl in confusion.index
        },
        # Strict: only auto_merge counts as a predicted positive.
        "metrics_auto_merge": _binary_metrics(y_true, tier == "auto_merge"),
        # Lenient: auto_merge OR human_review counts as a predicted positive.
        "metrics_merge_or_review": _binary_metrics(
            y_true, tier.isin(["auto_merge", "human_review"])
        ),
    }
    logger.info("Test metrics (auto_merge as positive): %s", metrics["metrics_auto_merge"])
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cleaned-index", type=Path, default=None,
                   help=f"Cleaned records parquet. Default: latest in {CLEANED_DIR}.")
    p.add_argument("--non-matches", type=Path, default=None,
                   help=f"Non-matches parquet. Default: latest in {NON_MATCHES_DIR}.")
    p.add_argument("--candidate-pairs", type=Path, default=None,
                   help="Full candidate-pairs parquet (with --score-full-candidate-pool).")
    p.add_argument("--score-full-candidate-pool", action="store_true",
                   help="Score the FULL pre-rules candidate pool (recommended). "
                        "Output filenames get a `_full_pool` suffix.")
    p.add_argument("--silver-labels", type=Path, default=DEFAULT_SILVER_LABELS,
                   help=f"Silver-labels CSV (PATID_A, PATID_B, label). Default: {DEFAULT_SILVER_LABELS}.")
    p.add_argument("--label-col", default="silver_label",
                   help="Label column in --silver-labels (default: 'silver_label'; "
                        "values True/False, coerced to {1,0}).")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction of silver labels held out for testing (default: 0.2).")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Random seed for the stratified train/test split (default: 42).")
    p.add_argument("--data-version", default=None,
                   help="Tag used in output filenames. Default: parsed from input.")
    p.add_argument("--u-max-pairs", type=float, default=U_MAX_PAIRS,
                   help="Splink u-sampling max_pairs. Default 1e6.")
    return p.parse_args()


def _resolve_scoring_pool(args: argparse.Namespace) -> Path:
    if args.score_full_candidate_pool:
        path = (
            args.candidate_pairs
            if args.candidate_pairs is not None
            else _resolve_with_fallback(CANDIDATE_PAIRS_DIRS, CANDIDATE_PAIRS_GLOBS)
        )
        if not path.exists():
            raise FileNotFoundError(f"Candidate-pairs parquet not found: {path}")
        logger.info("Scoring pool (full candidate pool) -> %s", path)
        return path
    path = (
        args.non_matches
        if args.non_matches is not None
        else _resolve_with_fallback(NON_MATCHES_DIR, NON_MATCHES_GLOBS)
    )
    if not path.exists():
        raise FileNotFoundError(f"Non-matches parquet not found: {path}")
    logger.info("Scoring pool (post-Stage-3 non_matches) -> %s", path)
    return path


def _train_and_score(
    df_clean: pd.DataFrame,
    df_model: pd.DataFrame,
    scoring_pool: pd.DataFrame,
    train_labels: pd.DataFrame,
    label_col: str,
    u_max_pairs: float,
) -> tuple[FSEnhanced3, pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """Train on the train split + score the production pool.

    Returns (model, classified, prob_matches, eval_schema, linker).
    """
    model = FSEnhanced3(
        labels_df=train_labels,
        label_col=label_col,
        include_address=True,
        u_max_pairs=u_max_pairs,
    )
    linker = model.build_linker(df_model, scoring_pool)
    model.train(linker, df_clean)
    predictions = model.predict(linker, scoring_pool)
    classified = model.classify(predictions)
    eval_schema = model.to_evaluation_schema(classified)
    prob_matches = model.to_probabilistic_matches(classified)
    return model, classified, prob_matches, eval_schema, linker


def main() -> None:
    args = parse_args()

    # ── Resolve inputs ────────────────────────────────────────────────────────
    if args.cleaned_index is None:
        args.cleaned_index = _resolve_with_fallback(CLEANED_DIR, CLEANED_GLOBS)
        logger.info("Auto-resolved --cleaned-index -> %s", args.cleaned_index)
    elif not args.cleaned_index.exists():
        raise FileNotFoundError(f"Cleaned index not found: {args.cleaned_index}")

    scoring_pool_path = _resolve_scoring_pool(args)
    if args.data_version is None:
        args.data_version = _derive_data_version(scoring_pool_path)
        logger.info("Auto-resolved --data-version -> %s", args.data_version)

    if not args.silver_labels.exists():
        raise FileNotFoundError(
            f"Silver labels not found: {args.silver_labels}. This file is VM-only "
            "(gitignored PHI). Pass --silver-labels or place it at the default path."
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading cleaned records from %s", args.cleaned_index)
    df_clean = pd.read_parquet(args.cleaned_index)
    logger.info("Cleaned records: %d rows, %d columns", len(df_clean), df_clean.shape[1])

    logger.info("Loading scoring pool from %s", scoring_pool_path)
    scoring_pool = pd.read_parquet(scoring_pool_path)
    logger.info("Scoring pool: %d pairs", len(scoring_pool))

    logger.info("Loading silver labels from %s", args.silver_labels)
    silver = _load_silver_labels(args.silver_labels, args.label_col)
    train_labels, test_labels = _stratified_split(
        silver, args.label_col, args.test_size, args.split_seed,
    )
    logger.info(
        "Silver labels: %d total (pos=%d) → train %d (pos=%d) / test %d (pos=%d)",
        len(silver), int((silver[args.label_col] == 1).sum()),
        len(train_labels), int((train_labels[args.label_col] == 1).sum()),
        len(test_labels), int((test_labels[args.label_col] == 1).sum()),
    )

    # ── Train + score (with single-CPU retry) ─────────────────────────────────
    # prepare_model_input once; reused for production scoring + test scoring.
    model_for_prep = FSEnhanced3(labels_df=train_labels, label_col=args.label_col)
    df_model = model_for_prep.prepare_model_input(df_clean)

    logger.info(
        "Training FSEnhanced3 (u_max_pairs=%.0e, include_address=True)...",
        args.u_max_pairs,
    )
    try:
        model, classified, prob_matches, eval_schema, linker = _train_and_score(
            df_clean, df_model, scoring_pool, train_labels,
            label_col=args.label_col, u_max_pairs=args.u_max_pairs,
        )
    except RuntimeError as exc:
        logger.warning("Retrying with u_max_pairs=1e4 due to: %s", exc)
        model, classified, prob_matches, eval_schema, linker = _train_and_score(
            df_clean, df_model, scoring_pool, train_labels,
            label_col=args.label_col, u_max_pairs=1e4,
        )

    # ── Held-out test-split evaluation ────────────────────────────────────────
    trained_settings = linker.misc.save_model_to_json()
    test_metrics = _evaluate_test_split(
        model, df_model, trained_settings, test_labels, args.label_col,
    )

    # ── Validate + write outputs ──────────────────────────────────────────────
    prob_matches = validate_contract(prob_matches, ProbabilisticMatches)
    suffix = "_full_pool" if args.score_full_candidate_pool else ""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = OUTPUTS_DIR / f"{MODEL_NAME}__{args.data_version}{suffix}.parquet"
    eval_schema.to_parquet(eval_path, index=False)
    logger.info("Wrote %d rows -> %s (eval_schema)", len(eval_schema), eval_path)

    MATCHES_MODEL_V2_DIR.mkdir(parents=True, exist_ok=True)
    pm_path = MATCHES_MODEL_V2_DIR / (
        f"{MODEL_NAME}_matches_model__{args.data_version}{suffix}.parquet"
    )
    prob_matches.to_parquet(pm_path, index=False)
    logger.info("Wrote %d rows -> %s (ProbabilisticMatches)", len(prob_matches), pm_path)

    # ── Aggregate-only summary (no PHI) ───────────────────────────────────────
    counts = eval_schema["predicted_tier"].value_counts().to_dict()
    logger.info("Tier breakdown: %s", counts)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = ARTIFACTS_DIR / f"diagnostics__{args.data_version}{suffix}.json"
    diagnostics = {
        "model_name": MODEL_NAME,
        "data_version": args.data_version,
        "scoring_pool_path": str(scoring_pool_path),
        "scoring_pool_rows": len(scoring_pool),
        "scored_full_candidate_pool": args.score_full_candidate_pool,
        "u_max_pairs": args.u_max_pairs,
        "silver_labels_path": str(args.silver_labels),
        "split": {
            "test_size": args.test_size,
            "split_seed": args.split_seed,
            "n_train": int(len(train_labels)),
            "n_train_positives": int((train_labels[args.label_col] == 1).sum()),
            "n_test": int(len(test_labels)),
        },
        "tier_counts": {k: int(v) for k, v in counts.items()},
        "score_quantiles": {
            "min": float(eval_schema["score"].min()),
            "p25": float(eval_schema["score"].quantile(0.25)),
            "median": float(eval_schema["score"].median()),
            "p75": float(eval_schema["score"].quantile(0.75)),
            "max": float(eval_schema["score"].max()),
        },
        "test_metrics": test_metrics,
    }
    try:
        diagnostics["trained_settings"] = trained_settings
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not embed trained settings: %s", exc)
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("Wrote diagnostics -> %s", diag_path)


if __name__ == "__main__":
    main()
