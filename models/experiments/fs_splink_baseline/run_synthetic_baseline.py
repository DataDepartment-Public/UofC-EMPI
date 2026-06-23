"""
run_synthetic_baseline.py — generate a standardized evaluation output for the
FS Splink baseline, scored against the synthetic dataset.

Writes:
    models/outputs/fs_splink_baseline__synthetic.parquet

The output is the standardized 5-column evaluation frame (PATID_A, PATID_B,
model_name, score, predicted_tier) — the same contract every model under
models/experiments/ will write to models/outputs/, so they can later be
pooled and compared by evaluate.py (not yet built).

This script is for the *synthetic* dataset only. For real-data runs against
the AllianceChicago VM's cleaned index + candidate pairs, use run_real_baseline.py.

Usage (from project root, with empi_env activated):
    python -m models.experiments.fs_splink_baseline.run_synthetic_baseline
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Project root = three levels up from this file
# (models/experiments/fs_splink_baseline/run_synthetic_baseline.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.experiments.fs_splink_baseline.fs_baseline import FSBaseline, MODEL_NAME
from models.common import synthetic_data as sd
from src.preprocessing.blocking import run_batch_blocking, _compute_derived_columns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "models" / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts" / MODEL_NAME

DATA_VERSION = "synthetic"

# Production VM default is 1e6; this is only reduced if running on a
# single-CPU machine.
U_MAX_PAIRS = 1e6


def _get_model_diagnostics(linker: Any) -> dict:
    """Return a non-PHI diagnostics bundle from the trained linker."""
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


def main() -> None:
    logger.info("Building synthetic dataset and candidate pairs...")
    df_clean = sd.make_synthetic_patients()
    df_derived = _compute_derived_columns(df_clean.copy())
    candidate_pairs = run_batch_blocking(df_derived)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training and scoring (%d candidate pairs, u_max_pairs=%.0e)...",
        len(candidate_pairs), U_MAX_PAIRS,
    )

    def _run_with_diagnostics(u_max_pairs: float):
        model = FSBaseline(u_max_pairs=u_max_pairs)
        df_model = model.prepare_model_input(df_clean)
        linker = model.build_linker(df_model, candidate_pairs)
        model.train(linker, df_clean)
        predictions = model.predict(linker, candidate_pairs)
        classified = model.classify(predictions)
        result = model.to_evaluation_schema(classified)
        diagnostics = _get_model_diagnostics(linker)
        return result, diagnostics

    try:
        result, diagnostics = _run_with_diagnostics(U_MAX_PAIRS)
    except RuntimeError as exc:
        # Single-CPU u-sampling salting issue. Retry with the smaller
        # u_max_pairs used in the test suite.
        logger.warning("Retrying with u_max_pairs=1e4 due to: %s", exc)
        result, diagnostics = _run_with_diagnostics(1e4)

    # --- Write the standardized evaluation output ---------------------------
    out_path = OUTPUTS_DIR / f"{MODEL_NAME}__{DATA_VERSION}.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %d scored pairs -> %s", len(result), out_path)
    print(result.sort_values("score", ascending=False).to_string(index=False))

    # --- Write Phase A diagnostics for this model's own calibration ---------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = ARTIFACTS_DIR / f"diagnostics__{DATA_VERSION}.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    logger.info("Wrote diagnostics -> %s", diag_path)

    # --- Tier summary ---------------------------------------------------------
    counts = result["predicted_tier"].value_counts().to_dict()
    logger.info("Tier breakdown: %s", counts)


if __name__ == "__main__":
    main()
