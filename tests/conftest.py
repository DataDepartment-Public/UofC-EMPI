"""
Shared pytest configuration, path setup, and fixtures used across all
test layers (unit, integration, regression).

PATH SETUP:
    Adds the project root (EMPI_latest/) to sys.path so that imports
    like `from src.preprocessing.blocking import ...` resolve correctly
    regardless of where pytest is invoked from.
"""
import logging

import pandas as pd
import pytest

import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# This file lives at tests/conftest.py. The project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Quiet Splink's logging during tests ─────────────────────────────────────
logging.disable(logging.CRITICAL)


# ── FS Splink baseline: shared constants and fixtures ───────────────────────
#
# These fixtures train a real Splink model on the synthetic dataset and are
# shared across tests/unit, tests/integration, and tests/regression so that
# the (slow) training step happens at most once per test session.
#
# Adjust the two import paths below if fellegi_sunter_baseline.py /
# synthetic_data.py live elsewhere in your tree.
from models.experiments.fs_splink_baseline import fellegi_sunter_baseline as fs
from models.common import synthetic_data as sd
from src.preprocessing.blocking import run_batch_blocking, _compute_derived_columns

# Splink salts u-sampling by CPU count; on a single-CPU host that collapses to
# 1 and raises an error (see NEW ISSUE A in fellegi_sunter_baseline.py).
# Use a small u_max_pairs in tests; the production VM default is 1e6.
FS_U_MAX_PAIRS_SANDBOX = 1e4


@pytest.fixture(scope="session")
def fs_df_clean() -> pd.DataFrame:
    """The synthetic cleaned patient index (no derived columns persisted)."""
    return sd.make_synthetic_patients()


@pytest.fixture(scope="session")
def fs_candidate_pairs(fs_df_clean) -> pd.DataFrame:
    """Candidate pairs produced by the real blocking pipeline on synthetic data."""
    df_derived = _compute_derived_columns(fs_df_clean.copy())
    return run_batch_blocking(df_derived)


@pytest.fixture(scope="session")
def fs_candidate_pairs_path(fs_candidate_pairs, tmp_path_factory) -> str:
    """Persist candidate pairs to parquet so run_fs_baseline can load them."""
    p = tmp_path_factory.mktemp("fs_blocking") / "candidate_pairs.parquet"
    fs_candidate_pairs.to_parquet(p)
    return str(p)


@pytest.fixture(scope="session")
def fs_classified(fs_df_clean, fs_candidate_pairs_path) -> pd.DataFrame:
    """
    Full-output classified frame from one training run, shared across the
    integration and regression tests (training is the slow step, so we do it
    once per session).
    """
    return fs.run_fs_baseline(
        fs_candidate_pairs_path, fs_df_clean,
        u_max_pairs=FS_U_MAX_PAIRS_SANDBOX, full_output=True,
    )