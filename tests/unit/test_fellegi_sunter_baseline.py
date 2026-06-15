"""
tests/unit/test_fellegi_sunter_baseline.py

Fast unit tests for the Fellegi-Sunter Splink baseline (implementation plan
Section 10). None of these train a Splink model — they exercise pure
data-prep, settings construction, and post-processing logic in isolation.

For the full-pipeline integration tests (training + scoring synthetic data)
see tests/integration/test_fellegi_sunter_baseline.py. For trained-parameter
sanity / m-u snapshot checks see tests/regression/test_fellegi_sunter_baseline.py.

Shared fixtures (fs_df_clean, etc.) come from tests/conftest.py, but most
tests here only need the lightweight `fs` and `sd` module imports plus
`fs_df_clean` for prepare_model_input.
"""

import pandas as pd
import pytest

from models.experiments.fs_splink_baseline import fellegi_sunter_baseline as fs
from src.features.blocking import COL_FIRST_NM, COL_LAST_NM, COL_EMAIL, _COL_PHONES_SET


# ===========================================================================
# prepare_model_input
# ===========================================================================
def test_prepare_model_input_phones_array(fs_df_clean):
    """_phones_parsed set -> Phones_array sorted list; empty set -> []."""
    model = fs.prepare_model_input(fs_df_clean)
    assert fs.COL_PHONES_ARRAY in model.columns

    # Every value is a list (never a set, never NaN scalar).
    assert model[fs.COL_PHONES_ARRAY].apply(lambda v: isinstance(v, list)).all()

    # A record whose parsed phone set is empty must map to [] (not [''], NaN).
    # _COL_PHONES_SET is the derived parsed-set column (actual Python sets),
    # present on the model frame after _compute_derived_columns runs.
    empty_mask = model[_COL_PHONES_SET].apply(lambda s: not s)
    if empty_mask.any():
        empties = model.loc[empty_mask, fs.COL_PHONES_ARRAY]
        assert all(v == [] for v in empties)

    # A record with phones must yield a sorted, de-duplicated list of strings.
    non_empty = model.loc[
        model[fs.COL_PHONES_ARRAY].apply(lambda v: len(v) > 0), fs.COL_PHONES_ARRAY
    ]
    assert len(non_empty) > 0, "fixture should contain at least one phoned record"
    for v in non_empty:
        assert v == sorted(v)
        assert all(isinstance(x, str) for x in v)


def test_prepare_model_input_required_columns_raises():
    """Missing the derivation inputs must raise a clear ValueError."""
    # A frame that _compute_derived_columns cannot turn into the required cols.
    bad = pd.DataFrame({"not_a_real_column": [1, 2, 3]})
    with pytest.raises((ValueError, KeyError)):
        fs.prepare_model_input(bad)


def test_prepare_model_input_adds_shim_columns(fs_df_clean):
    """Inert PATID_A / PATID_B placeholders must be present for rule binding."""
    model = fs.prepare_model_input(fs_df_clean)
    for shim in fs._PAIR_SHIM_COLUMNS:
        assert shim in model.columns


# ===========================================================================
# build_settings
# ===========================================================================
def test_build_settings_structure():
    """dedupe_only, 8 comparisons, TF on exactly FN + LN + Email."""
    settings = fs.build_settings()
    d = settings.get_settings("duckdb").as_dict()

    assert d["link_type"] == "dedupe_only"
    assert d["unique_id_column_name"] == fs.COL_PATID
    # Post-R2: FirstNM, LastNM, DOB, SSN, Email, Phones, ZIP, Address.
    assert len(d["comparisons"]) == 8

    # Term-frequency adjustments must be ON for FirstNM, LastNM, Email and
    # OFF everywhere else. We detect TF by inspecting comparison levels.
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
    # Post-R1 Email is a CustomComparison with output_column_name="Email".
    assert "Email" in tf_fields
    # Graded SSN / ZIP / DOB / phones / Address must NOT carry TF.
    assert "SSN" not in tf_fields
    assert "ZIP" not in tf_fields
    assert "Address" not in tf_fields


def test_build_settings_retains_audit_columns():
    """Phase A keeps both retain flags True for full auditability."""
    settings = fs.build_settings()
    d = settings.get_settings("duckdb").as_dict()
    assert d["retain_intermediate_calculation_columns"] is True
    assert d["retain_matching_columns"] is True


# ===========================================================================
# classify_pairs (boundary values)
# ===========================================================================
@pytest.mark.parametrize(
    "score, expected",
    [
        (0.00, "no_match"),
        (0.49, "no_match"),
        (0.50, "human_review"),   # review floor is inclusive
        (0.89, "human_review"),
        (0.90, "auto_merge"),     # auto-merge threshold is inclusive
        (1.00, "auto_merge"),
    ],
)
def test_classify_pairs_thresholds(score, expected):
    df = pd.DataFrame({"match_probability": [score]})
    out = fs.classify_pairs(df)
    assert out["classification_tier"].iloc[0] == expected


def test_classify_pairs_rejects_bad_thresholds():
    df = pd.DataFrame({"match_probability": [0.5]})
    with pytest.raises(ValueError):
        fs.classify_pairs(df, auto_merge_threshold=0.3, review_floor=0.6)


# ===========================================================================
# to_evaluation_schema
# ===========================================================================
def test_to_evaluation_schema_contract():
    """Exact five columns, in order, with correct value mapping."""
    rich = pd.DataFrame(
        {
            "PATID_A": ["P0001", "P0003"],
            "PATID_B": ["P0002", "P0004"],
            "match_probability": [0.97, 0.20],
            "classification_tier": ["auto_merge", "no_match"],
            "match_weight": [5.0, -2.0],          # extra cols must be dropped
            "gamma_SSN": [3, 0],
        }
    )
    out = fs.to_evaluation_schema(rich)

    assert list(out.columns) == fs.EVAL_SCHEMA_COLUMNS
    assert (out["model_name"] == fs.MODEL_NAME).all()
    assert out["score"].tolist() == [0.97, 0.20]
    assert out["predicted_tier"].tolist() == ["auto_merge", "no_match"]
    assert out["PATID_A"].tolist() == ["P0001", "P0003"]


def test_to_evaluation_schema_missing_columns_raises():
    with pytest.raises(ValueError):
        fs.to_evaluation_schema(pd.DataFrame({"PATID_A": ["x"]}))