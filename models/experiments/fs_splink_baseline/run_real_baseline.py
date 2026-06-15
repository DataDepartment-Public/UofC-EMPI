"""
run_real_baseline.py — run the FS Splink baseline against the real
AllianceChicago cleaned patient index + candidate pairs.

*** RUN THIS ON THE VM ONLY. NEVER in the sandbox / off the VM. ***

This script never logs or prints individual PATID values, names, SSNs, DOBs,
or any other identifier — only aggregate counts and the non-PHI diagnostics
bundle (trained m/u parameters), consistent with fellegi_sunter_baseline.py's
HIPAA note.

Inputs (defaults match the implementation plan, Section 3.0, updated to the
2026-06-11 data refresh):
    data/processed/MDM_Population_cleaned_v3_2026_06_11.parquet
    src/features/outputs/blocking/candidate_pairs_v4_2026_06_11.parquet

Outputs:
    models/outputs/fs_splink_baseline__v4_2026_06_11.parquet
        Standardized 5-column evaluation frame (PATID_A, PATID_B, model_name,
        score, predicted_tier) for the team's comparison call.
    models/artifacts/fs_splink_baseline/diagnostics__v4_2026_06_11.json
        Non-PHI diagnostics: trained m/u parameters, per-EM-session estimates,
        probability_two_random_records_match.

Usage (from project root, with the project venv activated):
    python models/experiments/fs_splink_baseline/run_real_baseline.py

    # Or override paths / version tag explicitly:
    python models/experiments/fs_splink_baseline/run_real_baseline.py \\
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
# (models/experiments/fs_splink_baseline/run_real_baseline.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.experiments.fs_splink_baseline import fellegi_sunter_baseline as fs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "models" / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts" / fs.MODEL_NAME

# Defaults reflect the 2026-06-11 data refresh. Override via CLI flags if your
# team produces a newer cleaned index / candidate-pairs file.
DEFAULT_CLEANED_INDEX = (
    PROJECT_ROOT / "data" / "processed" / "MDM_Population_cleaned_v3_2026_06_11.parquet"
)
DEFAULT_CANDIDATE_PAIRS = (
    PROJECT_ROOT / "src" / "features" / "outputs" / "blocking"
    / "candidate_pairs_v4_2026_06_11.parquet"
)
DEFAULT_DATA_VERSION = "v4_2026_06_11"

# Production default (multi-core VM). The module's _estimate_u_with_guard
# auto-falls-back to 1e4 with a clear error if this host turns out to be
# single-CPU (NEW ISSUE A) — but on the VM this should not trigger.
U_MAX_PAIRS = 1e6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cleaned-index", type=Path, default=DEFAULT_CLEANED_INDEX,
        help="Path to the cleaned patient index parquet (Section 3.0).",
    )
    p.add_argument(
        "--candidate-pairs", type=Path, default=DEFAULT_CANDIDATE_PAIRS,
        help="Path to the candidate-pairs parquet from blocking.py (Section 3.0).",
    )
    p.add_argument(
        "--data-version", default=DEFAULT_DATA_VERSION,
        help="Tag used in output filenames, e.g. v4_2026_06_11.",
    )
    p.add_argument(
        "--u-max-pairs", type=float, default=U_MAX_PAIRS,
        help="Splink u-sampling max_pairs (default 1e6).",
    )
    p.add_argument(
        "--auto-merge-threshold", type=float, default=fs.DEFAULT_AUTO_MERGE_THRESHOLD,
        help=f"Default {fs.DEFAULT_AUTO_MERGE_THRESHOLD} (plan Section 8 starting point).",
    )
    p.add_argument(
        "--review-floor", type=float, default=fs.DEFAULT_REVIEW_FLOOR,
        help=f"Default {fs.DEFAULT_REVIEW_FLOOR} (plan Section 8 starting point).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.cleaned_index.exists():
        raise FileNotFoundError(
            f"Cleaned index not found: {args.cleaned_index}\n"
            "Pass --cleaned-index pointing at the current cleaned parquet."
        )
    if not args.candidate_pairs.exists():
        raise FileNotFoundError(
            f"Candidate pairs not found: {args.candidate_pairs}\n"
            "Pass --candidate-pairs pointing at the current blocking output."
        )

    logger.info("Loading cleaned patient index from %s", args.cleaned_index)
    df_clean = pd.read_parquet(args.cleaned_index)
    logger.info("Loaded %d records", len(df_clean))

    logger.info(
        "Running FS baseline (u_max_pairs=%.0e, auto_merge>=%.2f, review_floor>=%.2f)...",
        args.u_max_pairs, args.auto_merge_threshold, args.review_floor,
    )
    try:
        result, diagnostics = fs.run_fs_baseline(
            args.candidate_pairs,
            df_clean,
            auto_merge_threshold=args.auto_merge_threshold,
            review_floor=args.review_floor,
            u_max_pairs=args.u_max_pairs,
            return_diagnostics=True,
        )
    except RuntimeError as exc:
        # Single-CPU u-sampling salting issue (NEW ISSUE A). Should not occur
        # on the VM, but handle it gracefully if it does.
        logger.warning("Retrying with u_max_pairs=1e4 due to: %s", exc)
        result, diagnostics = fs.run_fs_baseline(
            args.candidate_pairs,
            df_clean,
            auto_merge_threshold=args.auto_merge_threshold,
            review_floor=args.review_floor,
            u_max_pairs=1e4,
            return_diagnostics=True,
        )

    # --- Write the standardized evaluation output ---------------------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"{fs.MODEL_NAME}__{args.data_version}.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %d scored pairs -> %s", len(result), out_path)

    # --- Write Phase A diagnostics (non-PHI: m/u parameters only) -----------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = ARTIFACTS_DIR / f"diagnostics__{args.data_version}.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    logger.info("Wrote diagnostics -> %s", diag_path)

    # --- Aggregate-only summary (no PHI / identifiers printed) --------------
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