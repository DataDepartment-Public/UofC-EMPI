"""run_real_enhanced_2.py — score the real cohort with FSEnhanced2 (Phase E2-5).

*** RUN THIS ON THE VM ONLY. NEVER in the sandbox / off the VM. ***

This script never logs or prints individual PATID values, names, SSNs, DOBs,
or any other identifier — only aggregate counts and non-PHI score quantiles.

Inputs (auto-resolved via `models.common.versioning.latest_versioned`; override
with CLI flags):
    data/processed/MDM_Population_cleaned_v*_*.parquet      (records, real PHI)
    data/synthetic/synthetic_train_v3.csv                    (supervised m labels)
    EITHER:
      data/non_matches/non_matches_v*_*.parquet              (default scoring pool)
    OR:
      src/features/outputs/blocking/candidate_pairs_v*_*.parquet
                                                             (when --score-full-candidate-pool)

Outputs:
    models/outputs/fs_splink_enhanced_2__<data-version>[_full_pool].parquet
        5-column cross-model eval schema. The validation notebook §11 (E2-6)
        reads this for the 3-way head-to-head.
    data/matches_model_v2/fs_splink_enhanced_2_matches_model__<data-version>[_full_pool].parquet
        ProbabilisticMatches contract (validated). Union-ready for Stage 5.

The `--score-full-candidate-pool` flag exists for silver-labels evaluation
(see `data/silver_labels/`): silver labels reference pairs that Stage 3
already filtered out of non_matches, so they are absent from the default
scoring pool. Running with the flag scores the FULL pre-rules candidate
pool, making silver-labeled pairs available to the threshold sweep.

USAGE:
    # Production scoring run (post-Stage-3 non_matches, default).
    python -m models.experiments.fs_splink_enhanced_2.run_real_enhanced_2

    # Full-pool run for silver-labels validation.
    python -m models.experiments.fs_splink_enhanced_2.run_real_enhanced_2 \\
        --score-full-candidate-pool

    # Pin explicit paths and version tag.
    python -m models.experiments.fs_splink_enhanced_2.run_real_enhanced_2 \\
        --cleaned-index data/processed/MDM_Population_cleaned_v3_2026_06_11.parquet \\
        --non-matches data/non_matches/non_matches_v3_2026_06_11.parquet \\
        --labels data/synthetic/synthetic_train_v3.csv \\
        --data-version v3_2026_06_11
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

from models.common.versioning import latest_versioned, version_tag_from_filename  # noqa: E402
from models.experiments.fs_splink_enhanced_2.fs_enhanced_2 import (  # noqa: E402
    FSEnhanced2, MODEL_NAME,
)
from src.contracts import ProbabilisticMatches, validate as validate_contract  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths + naming conventions ────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "models" / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts" / MODEL_NAME
MATCHES_MODEL_V2_DIR = PROJECT_ROOT / "data" / "matches_model_v2"

CLEANED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_GLOB = "MDM_Population_cleaned_v*_*.parquet"

NON_MATCHES_DIR = PROJECT_ROOT / "data" / "non_matches"
NON_MATCHES_GLOB = "non_matches_v*_*.parquet"

CANDIDATE_PAIRS_DIR = PROJECT_ROOT / "src" / "features" / "outputs" / "blocking"
CANDIDATE_PAIRS_GLOB = "candidate_pairs_v*_*.parquet"

DEFAULT_LABELS = PROJECT_ROOT / "data" / "synthetic" / "synthetic_train_v3.csv"

# Production default (multi-core VM). FSEnhanced2's underlying
# _estimate_u_with_guard auto-raises a clear RuntimeError if this turns out to
# be single-CPU; we catch and retry with 1e4.
U_MAX_PAIRS = 1e6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cleaned-index", type=Path, default=None,
        help=f"Cleaned records parquet. Default: latest in {CLEANED_DIR}.",
    )
    p.add_argument(
        "--non-matches", type=Path, default=None,
        help=f"Non-matches parquet (Stage-3 output). Default: latest in {NON_MATCHES_DIR}.",
    )
    p.add_argument(
        "--candidate-pairs", type=Path, default=None,
        help=(
            f"Candidate-pairs parquet (full pre-rules pool). Used only with "
            f"--score-full-candidate-pool. Default: latest in {CANDIDATE_PAIRS_DIR}."
        ),
    )
    p.add_argument(
        "--score-full-candidate-pool", action="store_true",
        help=(
            "Score the FULL candidate pool from blocking, not the post-Stage-3 "
            "non_matches subset. Use for silver-labels validation: silver labels "
            "reference pairs Stage 3 already removed, so they are absent from "
            "non_matches by definition. Output filenames get a `_full_pool` suffix."
        ),
    )
    p.add_argument(
        "--labels", type=Path, default=DEFAULT_LABELS,
        help=f"Synthetic labeled-pairs CSV for supervised m-training. Default: {DEFAULT_LABELS}.",
    )
    p.add_argument(
        "--label-col", default="label",
        help="Binary label column in --labels (default: 'label'; values in {0, 1}).",
    )
    p.add_argument(
        "--data-version", default=None,
        help="Tag used in output filenames (e.g. v4_2026_06_11). Default: parsed from input.",
    )
    p.add_argument(
        "--u-max-pairs", type=float, default=U_MAX_PAIRS,
        help="Splink u-sampling max_pairs. Default 1e6.",
    )
    return p.parse_args()


def _resolve_scoring_pool(args: argparse.Namespace) -> Path:
    """Pick non_matches OR candidate_pairs based on --score-full-candidate-pool."""
    if args.score_full_candidate_pool:
        if args.candidate_pairs is None:
            path = latest_versioned(CANDIDATE_PAIRS_DIR, CANDIDATE_PAIRS_GLOB)
            logger.info("Auto-resolved --candidate-pairs -> %s", path)
        else:
            path = args.candidate_pairs
        if not path.exists():
            raise FileNotFoundError(f"Candidate-pairs parquet not found: {path}")
        return path
    if args.non_matches is None:
        path = latest_versioned(NON_MATCHES_DIR, NON_MATCHES_GLOB)
        logger.info("Auto-resolved --non-matches -> %s", path)
    else:
        path = args.non_matches
    if not path.exists():
        raise FileNotFoundError(f"Non-matches parquet not found: {path}")
    return path


def main() -> None:
    args = parse_args()

    # ── Resolve inputs ────────────────────────────────────────────────────────
    if args.cleaned_index is None:
        args.cleaned_index = latest_versioned(CLEANED_DIR, CLEANED_GLOB)
        logger.info("Auto-resolved --cleaned-index -> %s", args.cleaned_index)
    elif not args.cleaned_index.exists():
        raise FileNotFoundError(f"Cleaned index not found: {args.cleaned_index}")

    scoring_pool_path = _resolve_scoring_pool(args)
    if args.data_version is None:
        args.data_version = version_tag_from_filename(scoring_pool_path)
        logger.info("Auto-resolved --data-version -> %s", args.data_version)

    if not args.labels.exists():
        raise FileNotFoundError(
            f"Labels CSV not found: {args.labels}. Pass --labels or place the "
            "supervised-training pairs at the default location."
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading cleaned records from %s", args.cleaned_index)
    df_clean = pd.read_parquet(args.cleaned_index)
    logger.info("Cleaned records: %d rows, %d columns", len(df_clean), df_clean.shape[1])

    logger.info("Loading scoring pool from %s", scoring_pool_path)
    scoring_pool = pd.read_parquet(scoring_pool_path)
    logger.info("Scoring pool: %d pairs", len(scoring_pool))

    logger.info("Loading supervised labels from %s", args.labels)
    labels_df = pd.read_csv(args.labels)
    labels_df[args.label_col] = labels_df[args.label_col].astype(int)
    n_pos = int((labels_df[args.label_col] == 1).sum())
    logger.info(
        "Labels: %d rows total, %d positives for m-training",
        len(labels_df), n_pos,
    )

    # ── Train + score ─────────────────────────────────────────────────────────
    logger.info(
        "Training FSEnhanced2 (u_max_pairs=%.0e, include_address=True)...",
        args.u_max_pairs,
    )
    try:
        scored, prob_matches, eval_schema = _train_and_score(
            df_clean, scoring_pool, labels_df,
            label_col=args.label_col, u_max_pairs=args.u_max_pairs,
        )
    except RuntimeError as exc:
        # Single-CPU u-sampling salting issue. Should not occur on the VM,
        # but handle it gracefully if it does.
        logger.warning("Retrying with u_max_pairs=1e4 due to: %s", exc)
        scored, prob_matches, eval_schema = _train_and_score(
            df_clean, scoring_pool, labels_df,
            label_col=args.label_col, u_max_pairs=1e4,
        )

    # ── Validate + write ──────────────────────────────────────────────────────
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
    logger.info(
        "Wrote %d rows -> %s (ProbabilisticMatches)", len(prob_matches), pm_path,
    )

    # ── Aggregate-only summary (no PHI) ───────────────────────────────────────
    counts = eval_schema["predicted_tier"].value_counts().to_dict()
    logger.info("Tier breakdown: %s", counts)
    logger.info(
        "Score distribution: min=%.4f  p25=%.4f  median=%.4f  p75=%.4f  max=%.4f",
        eval_schema["score"].min(),
        eval_schema["score"].quantile(0.25),
        eval_schema["score"].median(),
        eval_schema["score"].quantile(0.75),
        eval_schema["score"].max(),
    )

    # ── Minimal diagnostics bundle (non-PHI: counts only) ─────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = ARTIFACTS_DIR / f"diagnostics__{args.data_version}{suffix}.json"
    diagnostics = {
        "model_name": MODEL_NAME,
        "data_version": args.data_version,
        "scoring_pool_path": str(scoring_pool_path),
        "scoring_pool_rows": len(scoring_pool),
        "scored_full_candidate_pool": args.score_full_candidate_pool,
        "u_max_pairs": args.u_max_pairs,
        "tier_counts": {k: int(v) for k, v in counts.items()},
        "score_quantiles": {
            "min": float(eval_schema["score"].min()),
            "p25": float(eval_schema["score"].quantile(0.25)),
            "median": float(eval_schema["score"].median()),
            "p75": float(eval_schema["score"].quantile(0.75)),
            "max": float(eval_schema["score"].max()),
        },
        "n_pos_labels_used": n_pos,
    }
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info("Wrote diagnostics -> %s", diag_path)


def _train_and_score(
    df_clean: pd.DataFrame,
    scoring_pool: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_col: str,
    u_max_pairs: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full FSEnhanced2 pipeline. Returns (classified, prob_matches, eval_schema)."""
    model = FSEnhanced2(
        labels_df=labels_df,
        label_col=label_col,
        include_address=True,
        u_max_pairs=u_max_pairs,
    )
    df_model = model.prepare_model_input(df_clean)
    linker = model.build_linker(df_model, scoring_pool)
    model.train(linker, df_clean)
    predictions = model.predict(linker, scoring_pool)
    classified = model.classify(predictions)
    eval_schema = model.to_evaluation_schema(classified)
    prob_matches = model.to_probabilistic_matches(classified)
    return classified, prob_matches, eval_schema


if __name__ == "__main__":
    main()
