"""
run_synthetic_enhanced.py — generate a standardized evaluation output for the
FS Splink enhanced model, scored against the synthetic dataset.

Writes:
    models/outputs/fs_splink_enhanced__synthetic.parquet

The output is the standardized 5-column evaluation frame (PATID_A, PATID_B,
model_name, score, predicted_tier) — the same contract every model under
models/experiments/ writes to models/outputs/, so they can later be pooled and
compared.

This script is for the *synthetic* dataset only. For real-data runs against the
AllianceChicago VM's cleaned index + candidate pairs use run_real_enhanced.py.

Usage (from project root, with empi_env activated):
    python models/experiments/fs_splink_enhanced/run_synthetic_enhanced.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Project root = three levels up from this file
# (models/experiments/fs_splink_enhanced/run_synthetic_enhanced.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.experiments.fs_splink_enhanced.fs_enhanced import FSEnhanced, MODEL_NAME
from models.common import synthetic_data as sd
from src.preprocessing.blocking import run_batch_blocking, _compute_derived_columns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "models" / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts" / MODEL_NAME

DATA_VERSION = "synthetic"

# Production VM default; reduced automatically if single-CPU (see FSModel).
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

    # Synthetic fixture lacks address columns — use include_address=False.
    model = FSEnhanced(include_address=False, u_max_pairs=U_MAX_PAIRS)

    try:
        result = model.run(candidate_pairs, df_clean)
    except RuntimeError as exc:
        logger.warning("Retrying with u_max_pairs=1e4 due to: %s", exc)
        model = FSEnhanced(include_address=False, u_max_pairs=1e4)
        result = model.run(candidate_pairs, df_clean)
    finally:
        tmp_pairs_path.unlink(missing_ok=True)

    # --- Write the standardized evaluation output ----------------------------
    out_path = OUTPUTS_DIR / f"{MODEL_NAME}__{DATA_VERSION}.parquet"
    result.to_parquet(out_path, index=False)
    logger.info("Wrote %d scored pairs -> %s", len(result), out_path)
    print(result.sort_values("score", ascending=False).to_string(index=False))

    # --- Tier summary ---------------------------------------------------------
    counts = result["predicted_tier"].value_counts().to_dict()
    logger.info("Tier breakdown: %s", counts)


if __name__ == "__main__":
    main()
