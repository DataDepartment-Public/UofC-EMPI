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
the AllianceChicago VM's cleaned index + candidate pairs, a separate script
(or CLI flag) pointing at the real parquet paths from the implementation plan
(Section 3.0) will be added once that's ready to run on the VM.

Usage (from project root, with empi_env activated):
    python models/experiments/fs_splink_baseline/run_synthetic_baseline.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Project root = three levels up from this file
# (models/experiments/fs_splink_baseline/run_synthetic_baseline.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.experiments.fs_splink_baseline import fellegi_sunter_baseline as fs
from models.common import synthetic_data as sd
from src.features.blocking import run_batch_blocking, _compute_derived_columns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "models" / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts" / fs.MODEL_NAME

DATA_VERSION = "synthetic"

# Production VM default is 1e6; this is only reduced if running on a
# single-CPU machine (see NEW ISSUE A in fellegi_sunter_baseline.py).
U_MAX_PAIRS = 1e6


def main() -> None:
    logger.info("Building synthetic dataset and candidate pairs...")
    df_clean = sd.make_synthetic_patients()
    df_derived = _compute_derived_columns(df_clean.copy())
    candidate_pairs = run_batch_blocking(df_derived)

    tmp_pairs_path = OUTPUTS_DIR / "_tmp_synthetic_candidate_pairs.parquet"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_pairs.to_parquet(tmp_pairs_path)

    logger.info(
        "Training and scoring (%d candidate pairs, u_max_pairs=%.0e)...",
        len(candidate_pairs), U_MAX_PAIRS,
    )
    try:
        result, diagnostics = fs.run_fs_baseline(
            tmp_pairs_path,
            df_clean,
            u_max_pairs=U_MAX_PAIRS,
            return_diagnostics=True,
        )
    except RuntimeError as exc:
        # Single-CPU u-sampling salting issue (NEW ISSUE A). Retry with the
        # smaller u_max_pairs used in the test suite.
        logger.warning("Retrying with u_max_pairs=1e4 due to: %s", exc)
        result, diagnostics = fs.run_fs_baseline(
            tmp_pairs_path,
            df_clean,
            u_max_pairs=1e4,
            return_diagnostics=True,
        )
    finally:
        tmp_pairs_path.unlink(missing_ok=True)

    # --- Write the standardized evaluation output ---------------------------
    out_path = OUTPUTS_DIR / f"{fs.MODEL_NAME}__{DATA_VERSION}.parquet"
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