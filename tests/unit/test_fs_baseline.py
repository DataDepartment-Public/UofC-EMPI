"""
tests/unit/test_fs_baseline.py

Fast unit tests for the FS Splink baseline (FSBaseline FSModel subclass).
None of these train a Splink model — they exercise pure data-prep, settings
construction, and post-processing logic in isolation.

For the full-pipeline integration tests (training + scoring synthetic data)
see tests/integration/test_fs_baseline.py. For trained-parameter sanity /
m-u snapshot checks see tests/regression/test_fs_baseline.py.

Shared fixtures (fs_df_clean, etc.) come from tests/conftest.py, but most
tests here only need the lightweight module imports plus `fs_df_clean` for
prepare_model_input.
"""

import pandas as pd
import pytest

from models.experiments.fs_splink_baseline.fs_baseline import (
    FSBaseline,
    MODEL_NAME,
    REQUIRED_MODEL_COLUMNS,
    _validate_required_columns,
)
from models.experiments.fs_splink_baseline.comparisons import (
    BASELINE_REGISTRY,
    COL_PHONES_ARRAY,
    COL_FIRST_NM,
    COL_LAST_NM,
)
from models.common.fs_base import ClassificationConfig, EMTraining
from src.preprocessing.blocking import _COL_PHONES_SET


# ===========================================================================
# Registry
# ===========================================================================
def test_baseline_registry_names():
    """Registry must expose 7 comparisons in the expected order."""
    expected = ["FirstNM", "LastNM", "BirthDT", "SSN", "Email", "Phones", "ZIP"]
    assert BASELINE_REGISTRY.names() == expected


# ===========================================================================
# FSBaseline class-level attributes
# ===========================================================================
def test_model_name():
    assert FSBaseline().model_name == "fs_splink_baseline"
    assert FSBaseline().model_name == MODEL_NAME


def test_classification_config_thresholds():
    cfg = FSBaseline().classification_config
    assert isinstance(cfg, ClassificationConfig)
    assert cfg.auto_merge_threshold == 0.90
    assert cfg.review_floor == 0.50


def test_training_is_em_training():
    t = FSBaseline().training
    assert isinstance(t, EMTraining)


def test_em_blocking_rules_length():
    """EMTraining must have 3 EM blocking rules (SSN, Email, Soundex)."""
    t = FSBaseline().training
    assert len(t.em_blocking_rules) == 3


def test_prior_rules_length():
    """EMTraining must have 3 prior rules (SSN, Email, DM-Last+DOB)."""
    t = FSBaseline().training
    assert t.prior_rules is not None
    assert len(t.prior_rules) == 3


def test_u_max_pairs_passthrough():
    """Constructor kwarg u_max_pairs must reach the EMTraining instance."""
    model = FSBaseline(u_max_pairs=1e4)
    assert model.training.u_max_pairs == 1e4


# ===========================================================================
# prepare_model_input
# ===========================================================================
def test_prepare_model_input_phones_array(fs_df_clean):
    """_phones_parsed set -> Phones_array sorted list; empty set -> []."""
    model_df = FSBaseline().prepare_model_input(fs_df_clean)
    assert COL_PHONES_ARRAY in model_df.columns

    # Every value is a list (never a set, never NaN scalar).
    assert model_df[COL_PHONES_ARRAY].apply(lambda v: isinstance(v, list)).all()

    # A record whose parsed phone set is empty must map to [] (not [''], NaN).
    empty_mask = model_df[_COL_PHONES_SET].apply(lambda s: not s)
    if empty_mask.any():
        empties = model_df.loc[empty_mask, COL_PHONES_ARRAY]
        assert all(v == [] for v in empties)

    # A record with phones must yield a sorted, de-duplicated list of strings.
    non_empty = model_df.loc[
        model_df[COL_PHONES_ARRAY].apply(lambda v: len(v) > 0), COL_PHONES_ARRAY
    ]
    assert len(non_empty) > 0, "fixture should contain at least one phoned record"
    for v in non_empty:
        assert v == sorted(v)
        assert all(isinstance(x, str) for x in v)


def test_prepare_model_input_required_columns_raises():
    """Missing the derivation inputs must raise a clear ValueError."""
    bad = pd.DataFrame({"not_a_real_column": [1, 2, 3]})
    with pytest.raises((ValueError, KeyError)):
        FSBaseline().prepare_model_input(bad)


def test_prepare_model_input_adds_shim_columns(fs_df_clean):
    """Inert PATID_A / PATID_B placeholders must be present for rule binding."""
    model_df = FSBaseline().prepare_model_input(fs_df_clean)
    for shim in ("PATID_A", "PATID_B"):
        assert shim in model_df.columns


# ===========================================================================
# _validate_required_columns
# ===========================================================================
def test_validate_required_columns_raises_on_missing():
    """Module-level helper must raise ValueError when required cols are absent."""
    bad = pd.DataFrame({"not_a_real_column": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_required_columns(bad)


def test_validate_required_columns_passes_when_present(fs_df_clean):
    """Must not raise when the frame has all required model columns."""
    from src.preprocessing.blocking import _compute_derived_columns
    df = _compute_derived_columns(fs_df_clean.copy())
    # Should not raise.
    _validate_required_columns(df)


# ===========================================================================
# build_settings
# ===========================================================================
def test_build_settings_structure():
    """dedupe_only, 7 comparisons, TF on exactly FN + LN + Email."""
    settings_dict = FSBaseline().build_settings()
    assert isinstance(settings_dict, dict)
    assert settings_dict["link_type"] == "dedupe_only"
    assert settings_dict["unique_id_column_name"] == "PATID"
    # FirstNM, LastNM, DOB, SSN, Email, Phones, ZIP (R2 Address comparison
    # reverted on 2026-06-15 pending ground-truth calibration).
    assert len(settings_dict["comparisons"]) == 7


def test_build_settings_retains_audit_columns():
    """Phase A keeps both retain flags True for full auditability."""
    d = FSBaseline().build_settings()
    assert d["retain_intermediate_calculation_columns"] is True
    assert d["retain_matching_columns"] is True


def test_build_settings_tf_fields():
    """TF adjustments must be ON for FirstNM, LastNM, Email; OFF elsewhere."""
    d = FSBaseline().build_settings()
    tf_fields = set()
    for comp in d["comparisons"]:
        col = comp.get("output_column_name", "")
        has_tf = any(
            lvl.get("tf_adjustment_column") for lvl in comp.get("comparison_levels", [])
        )
        if has_tf:
            tf_fields.add(col)

    assert COL_FIRST_NM in tf_fields
    assert COL_LAST_NM in tf_fields
    assert "Email" in tf_fields
    # Graded SSN / ZIP / DOB / phones must NOT carry TF.
    assert "SSN" not in tf_fields
    assert "ZIP" not in tf_fields
