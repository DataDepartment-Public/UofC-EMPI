"""
run_real_enhanced.py — run the FS Splink enhanced model against the real
AllianceChicago cleaned patient index + candidate pairs.

*** RUN THIS ON THE VM ONLY. NEVER in the sandbox / off the VM. ***

This script never logs or prints individual PATID values, names, SSNs, DOBs,
or any other identifier — only aggregate counts and the non-PHI diagnostics
bundle (trained m/u parameters), consistent with the HIPAA note in CLAUDE.md.

Inputs (auto-resolved to the highest-versioned parquet on disk via
``models.common.versioning.latest_versioned``; override with CLI flags):
    data/processed/MDM_Population_cleaned_v*_*.parquet
    src/features/outputs/blocking/candidate_pairs_v*_*.parquet

Outputs:
    models/outputs/fs_splink_enhanced__<data-version>.parquet
        Standardized 5-column evaluation frame (PATID_A, PATID_B, model_name,
        score, predicted_tier) for the team's comparison call.
    models/artifacts/fs_splink_enhanced/diagnostics__<data-version>.json
        Non-PHI diagnostics: trained m/u parameters, per-EM-session estimates,
        probability_two_random_records_match.

The ``<data-version>`` tag is auto-derived from the resolved candidate-pairs
filename (e.g. ``v4_2026_06_11``) so re-runs across data refreshes do not
overwrite each other. Override with ``--data-version`` if you need a custom tag.

Usage (from project root, with the project venv activated):
    python models/experiments/fs_splink_enhanced/run_real_enhanced.py

    # Or override paths / version tag explicitly:
    python models/experiments/fs_splink_enhanced/run_real_enhanced.py \\
        --cleaned-index data/processed/MDM_Population_cleaned_v3_2026_06_11.parquet \\
        --candidate-pairs src/features/outputs/blocking/candidate_pairs_v4_2026_06_11.parquet \\
        --data-version v4_2026_06_11
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Project root = three levels up from this file
# (models/experiments/fs_splink_enhanced/run_real_enhanced.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.experiments.fs_splink_enhanced.fs_enhanced import (
    FSEnhanced,
    MODEL_NAME,
)
from models.common.versioning import latest_versioned, version_tag_from_filename

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "models" / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts" / MODEL_NAME

# Auto-resolved at runtime (see main()) from these directories + glob patterns.
CLEANED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_GLOB = "MDM_Population_cleaned_v*_*.parquet"
PAIRS_DIR = PROJECT_ROOT / "src" / "features" / "outputs" / "blocking"
PAIRS_GLOB = "candidate_pairs_v*_*.parquet"

# E5 defaults (matched to FSEnhanced class-level ClassificationConfig).
_DEFAULT_AUTO_MERGE_THRESHOLD = 0.95
_DEFAULT_REVIEW_FLOOR = 0.40

# Production default (multi-core VM).
U_MAX_PAIRS = 1e6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cleaned-index", type=Path, default=None,
        help=(
            "Path to the cleaned patient index parquet. Defaults to the "
            f"highest-versioned file matching {CLEANED_GLOB} in {CLEANED_DIR}."
        ),
    )
    p.add_argument(
        "--candidate-pairs", type=Path, default=None,
        help=(
            "Path to the candidate-pairs parquet from blocking.py. Defaults to "
            f"the highest-versioned file matching {PAIRS_GLOB} in {PAIRS_DIR}."
        ),
    )
    p.add_argument(
        "--data-version", default=None,
        help=(
            "Tag used in output filenames, e.g. v4_2026_06_11. Defaults to "
            "the version tag parsed from the resolved candidate-pairs filename."
        ),
    )
    p.add_argument(
        "--u-max-pairs", type=float, default=U_MAX_PAIRS,
        help="Splink u-sampling max_pairs (default 1e6).",
    )
    p.add_argument(
        "--auto-merge-threshold", type=float, default=_DEFAULT_AUTO_MERGE_THRESHOLD,
        help=f"Auto-merge threshold (default {_DEFAULT_AUTO_MERGE_THRESHOLD}).",
    )
    p.add_argument(
        "--review-floor", type=float, default=_DEFAULT_REVIEW_FLOOR,
        help=f"Review floor (default {_DEFAULT_REVIEW_FLOOR}).",
    )
    p.add_argument(
        "--include-address", action="store_true", default=True,
        help="Include Address + Household_discount comparisons (default True).",
    )
    p.add_argument(
        "--no-address", dest="include_address", action="store_false",
        help="Exclude Address + Household_discount comparisons.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cleaned_index is None:
        args.cleaned_index = latest_versioned(CLEANED_DIR, CLEANED_GLOB)
        logger.info("Auto-resolved --cleaned-index -> %s", args.cleaned_index)
    elif not args.cleaned_index.exists():
        raise FileNotFoundError(
            f"Cleaned index not found: {args.cleaned_index}\n"
            "Pass --cleaned-index pointing at the current cleaned parquet."
        )

    if args.candidate_pairs is None:
        args.candidate_pairs = latest_versioned(PAIRS_DIR, PAIRS_GLOB)
        logger.info("Auto-resolved --candidate-pairs -> %s", args.candidate_pairs)
    elif not args.candidate_pairs.exists():
        raise FileNotFoundError(
            f"Candidate pairs not found: {args.candidate_pairs}\n"
            "Pass --candidate-pairs pointing at the current blocking output."
        )

    if args.data_version is None:
        args.data_version = version_tag_from_filename(args.candidate_pairs)
        logger.info("Auto-resolved --data-version -> %s", args.data_version)

    logger.info("Loading cleaned patient index from %s", args.cleaned_index)
    df_clean = pd.read_parquet(args.cleaned_index)
    logger.info("Loaded %d records", len(df_clean))

    candidate_pairs = pd.read_parquet(args.candidate_pairs)
    logger.info("Loaded %d candidate pairs from %s", len(candidate_pairs), args.candidate_pairs)

    logger.info(
        "Running FSEnhanced (include_address=%s, u_max_pairs=%.0e, "
        "auto_merge>=%.2f, review_floor>=%.2f)...",
        args.include_address, args.u_max_pairs,
        args.auto_merge_threshold, args.review_floor,
    )

    model = FSEnhanced(
        include_address=args.include_address,
        u_max_pairs=args.u_max_pairs,
        auto_merge_threshold=args.auto_merge_threshold,
        review_floor=args.review_floor,
    )

    try:
        classified = model.run(candidate_pairs, df_clean, full_output=True)
    except RuntimeError as exc:
        logger.warning("Retrying with u_max_pairs=1e4 due to: %s", exc)
        model = FSEnhanced(
            include_address=args.include_address,
            u_max_pairs=1e4,
            auto_merge_threshold=args.auto_merge_threshold,
            review_floor=args.review_floor,
        )
        classified = model.run(candidate_pairs, df_clean, full_output=True)

    # --- Dual-output projection from the rich classified frame ---------------
    # 1) Legacy 5-col eval-schema parquet — head-to-head input for notebook §10.
    # 2) ProbabilisticMatches parquet — union-ready Stage 4 contract, validated
    #    against src.contracts.ProbabilisticMatches.
    result = model.to_evaluation_schema(classified)
    prob_matches = model.to_probabilistic_matches(classified)
    from src.contracts import ProbabilisticMatches, validate as validate_contract
    prob_matches = validate_contract(prob_matches, ProbabilisticMatches)

    # --- Write both artifacts ------------------------------------------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"{MODEL_NAME}__{args.data_version}.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %d scored pairs -> %s (eval_schema)", len(result), out_path)

    from src.config import settings as _empi_settings
    _empi_settings.matches_model_dir.mkdir(parents=True, exist_ok=True)
    pm_path = _empi_settings.matches_model_dir / (
        f"{MODEL_NAME}_matches_model__{args.data_version}.parquet"
    )
    prob_matches.to_parquet(pm_path, index=False)
    logger.info(
        "Wrote %d scored pairs -> %s (ProbabilisticMatches)",
        len(prob_matches), pm_path,
    )

    # --- Aggregate-only summary (no PHI / identifiers printed) ---------------
    counts = result["predicted_tier"].value_counts().to_dict()
    logger.info("Tier breakdown: %s", counts)
    logger.info(
        "Score distribution: min=%.4f  p25=%.4f  median=%.4f  p75=%.4f  max=%.4f",
        result["score"].min(),
        result["score"].quantile(0.25),
        result["score"].median(),
        result["score"].quantile(0.75),
        result["score"].max(),
    )


if __name__ == "__main__":
    main()
