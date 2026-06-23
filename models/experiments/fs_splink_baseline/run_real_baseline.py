"""
run_real_baseline.py — run the FS Splink baseline against the real
AllianceChicago cleaned patient index + candidate pairs.

*** RUN THIS ON THE VM ONLY. NEVER in the sandbox / off the VM. ***

This script never logs or prints individual PATID values, names, SSNs, DOBs,
or any other identifier — only aggregate counts and the non-PHI diagnostics
bundle (trained m/u parameters), consistent with HIPAA discipline.

Inputs (auto-resolved to the highest-versioned parquet on disk via
``models.common.versioning.latest_versioned``; override with CLI flags):
    data/processed/MDM_Population_cleaned_v*_*.parquet
    src/features/outputs/blocking/candidate_pairs_v*_*.parquet

Outputs:
    models/outputs/fs_splink_baseline__<data-version>.parquet
        Standardized 5-column evaluation frame (PATID_A, PATID_B, model_name,
        score, predicted_tier) for the team's comparison call.
    models/artifacts/fs_splink_baseline/diagnostics__<data-version>.json
        Non-PHI diagnostics: trained m/u parameters, per-EM-session estimates,
        probability_two_random_records_match.

The ``<data-version>`` tag is auto-derived from the resolved candidate-pairs
filename (e.g. ``v4_2026_06_11``) so re-runs across data refreshes do not
overwrite each other. Override with ``--data-version`` if you need a custom tag.

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
from typing import Any

# Project root = three levels up from this file
# (models/experiments/fs_splink_baseline/run_real_baseline.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.experiments.fs_splink_baseline.fs_baseline import FSBaseline, MODEL_NAME
from models.common.versioning import latest_versioned, version_tag_from_filename

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "models" / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts" / MODEL_NAME

# Auto-resolved at runtime (see resolve_inputs()) from these directories +
# glob patterns. Override via CLI flags to pin a specific file.
CLEANED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_GLOB = "MDM_Population_cleaned_v*_*.parquet"
PAIRS_DIR = PROJECT_ROOT / "src" / "features" / "outputs" / "blocking"
PAIRS_GLOB = "candidate_pairs_v*_*.parquet"

# Production default (multi-core VM). The module's _estimate_u_with_guard
# auto-falls-back to 1e4 with a clear error if this host turns out to be
# single-CPU — but on the VM this should not trigger.
U_MAX_PAIRS = 1e6

# Baseline thresholds (plan Section 8 starting points).
DEFAULT_AUTO_MERGE_THRESHOLD = 0.90
DEFAULT_REVIEW_FLOOR = 0.50


def _get_model_diagnostics(linker: Any) -> dict:
    """Return a non-PHI diagnostics bundle from the trained linker."""
    from splink import Linker  # local import
    diagnostics: dict = {}
    try:
        diagnostics["trained_settings"] = linker.misc.save_model_to_json()
    except AttributeError as exc:
        logger.warning("Splink save_model_to_json missing: %s", exc)
        diagnostics["trained_settings_error"] = str(exc)
    try:
        diagnostics["parameter_records"] = linker._settings_obj._parameters_as_detailed_records
    except AttributeError as exc:
        logger.warning("Splink _parameters_as_detailed_records missing: %s", exc)
        diagnostics["parameter_records_error"] = str(exc)
    try:
        diagnostics["per_session_estimates"] = linker._settings_obj._parameter_estimates_as_records
    except AttributeError as exc:
        logger.warning("Splink _parameter_estimates_as_records missing: %s", exc)
    diagnostics["probability_two_random_records_match"] = getattr(
        linker._settings_obj, "_probability_two_random_records_match", None
    )
    return diagnostics


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
        "--auto-merge-threshold", type=float, default=DEFAULT_AUTO_MERGE_THRESHOLD,
        help=f"Default {DEFAULT_AUTO_MERGE_THRESHOLD} (plan Section 8 starting point).",
    )
    p.add_argument(
        "--review-floor", type=float, default=DEFAULT_REVIEW_FLOOR,
        help=f"Default {DEFAULT_REVIEW_FLOOR} (plan Section 8 starting point).",
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

    logger.info("Loading candidate pairs from %s", args.candidate_pairs)
    candidate_pairs = pd.read_parquet(args.candidate_pairs)
    logger.info("Loaded %d candidate pairs", len(candidate_pairs))

    logger.info(
        "Running FS baseline (u_max_pairs=%.0e, auto_merge>=%.2f, review_floor>=%.2f)...",
        args.u_max_pairs, args.auto_merge_threshold, args.review_floor,
    )

    def _run_with_diagnostics(u_max_pairs: float):
        """Train the model and return (result_df, diagnostics)."""
        model = FSBaseline(
            u_max_pairs=u_max_pairs,
            auto_merge_threshold=args.auto_merge_threshold,
            review_floor=args.review_floor,
        )
        df_model = model.prepare_model_input(df_clean)
        linker = model.build_linker(df_model, candidate_pairs)
        model.train(linker, df_clean)
        predictions = model.predict(linker, candidate_pairs)
        classified = model.classify(predictions)
        result = model.to_evaluation_schema(classified)
        diagnostics = _get_model_diagnostics(linker)
        return result, diagnostics

    try:
        result, diagnostics = _run_with_diagnostics(args.u_max_pairs)
    except RuntimeError as exc:
        # Single-CPU u-sampling salting issue. Should not occur on the VM,
        # but handle it gracefully if it does.
        logger.warning("Retrying with u_max_pairs=1e4 due to: %s", exc)
        result, diagnostics = _run_with_diagnostics(1e4)

    # --- Write the standardized evaluation output ---------------------------
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"{MODEL_NAME}__{args.data_version}.parquet"
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
