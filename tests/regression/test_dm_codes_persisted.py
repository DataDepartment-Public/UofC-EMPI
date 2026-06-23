"""Phase E2-2 regression tests: DM codes persisted by cleaning.

Pins three invariants:

1. `transform_dataframe()` emits `_dm_LastNM` and `_dm_FirstNM` columns whose
   values are byte-for-byte identical to what blocking's local `_dm_primary`
   used to compute. (Authoritative implementation now lives in transformations;
   blocking imports it.)

2. `_compute_derived_columns()` in blocking reuses the persisted DM columns
   when they already exist on the input frame — no redundant compute, and the
   final values are unchanged either way.

3. End-to-end blocking output (`run_batch_blocking`) is byte-identical between
   a cleaned frame that carries the persisted DM columns and one that does
   not (the fallback path).
"""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from src.preprocessing.transformations import _dm_primary, transform_dataframe
from src.preprocessing.blocking import (
    _COL_DM_FN,
    _COL_DM_LN,
    _compute_derived_columns,
    run_batch_blocking,
)


# ── Shared fixture: a small raw cohort that exercises edge cases ─────────────
def _raw_cohort() -> pd.DataFrame:
    return pd.DataFrame({
        "PATID":         ["P01", "P02", "P03", "P04", "P05"],
        "FirstNM":       ["Catherine", "KATHRYN", None, "Bob", "RoBeRt"],
        "LastNM":        ["Smith", "Smyth", "Jones", "O'Brien", "OBrien"],
        "MiddleNM":      [None, None, None, None, None],
        "SuffixNM":      [None, None, None, None, None],
        "BirthDT":       ["1980-01-15", "1980-01-15", "1990-06-22",
                          "1975-03-04", "1975-03-04"],
        "SSN":           [None, None, None, None, None],
        "AddressLine1":  [None, None, None, None, None],
        "AddressLine2":  [None, None, None, None, None],
        "CityNM":        [None, None, None, None, None],
        "ZipCD":         [None, None, None, None, None],
        "StateCD":       [None, None, None, None, None],
        "CountryNM":     [None, None, None, None, None],
        "PrimaryPhoneNBR": [None, None, None, None, None],
        "Phone01NBR":    [None, None, None, None, None],
        "Phone02NBR":    [None, None, None, None, None],
        "Phone03NBR":    [None, None, None, None, None],
        "Email":         [None, None, None, None, None],
        "SexAtBirthDSC": ["FEMALE", "FEMALE", "MALE", "MALE", "MALE"],
    })


# ── 1. transform_dataframe emits the DM columns ──────────────────────────────
def test_transform_dataframe_emits_dm_lastnm_and_firstnm():
    cleaned = transform_dataframe(_raw_cohort())
    assert _COL_DM_LN in cleaned.columns
    assert _COL_DM_FN in cleaned.columns
    # Catherine and KATHRYN both clean to "CATHERINE"/"KATHRYN" — the test is
    # really about computation, not phonetic-equality theory.
    assert cleaned[_COL_DM_LN].notna().sum() >= 4
    assert cleaned[_COL_DM_FN].notna().sum() >= 4


def test_dm_lastnm_values_match_direct_dm_primary_call():
    cleaned = transform_dataframe(_raw_cohort())
    expected = cleaned["LastNM_clean"].apply(
        lambda x: _dm_primary(x) if pd.notna(x) else None
    )
    pdt.assert_series_equal(
        cleaned[_COL_DM_LN].rename(None),
        expected.rename(None),
        check_names=False,
    )


def test_dm_firstnm_values_match_direct_dm_primary_call():
    cleaned = transform_dataframe(_raw_cohort())
    expected = cleaned["FirstNM_clean"].apply(
        lambda x: _dm_primary(x) if pd.notna(x) else None
    )
    pdt.assert_series_equal(
        cleaned[_COL_DM_FN].rename(None),
        expected.rename(None),
        check_names=False,
    )


def test_dm_handles_null_and_empty_names_as_none():
    cleaned = transform_dataframe(_raw_cohort())
    # P03 has FirstNM=None in raw -> FirstNM_clean is null -> DM is None.
    p03 = cleaned[cleaned["PATID"] == "P03"].iloc[0]
    assert p03[_COL_DM_FN] is None or pd.isna(p03[_COL_DM_FN])


# ── 2. blocking._compute_derived_columns reuses persisted columns ────────────
def test_compute_derived_columns_reuses_persisted_dm():
    cleaned = transform_dataframe(_raw_cohort())
    dm_ln_persisted = cleaned[_COL_DM_LN].copy()
    dm_fn_persisted = cleaned[_COL_DM_FN].copy()

    derived = _compute_derived_columns(cleaned)
    pdt.assert_series_equal(
        derived[_COL_DM_LN].reset_index(drop=True),
        dm_ln_persisted.reset_index(drop=True),
        check_names=False,
    )
    pdt.assert_series_equal(
        derived[_COL_DM_FN].reset_index(drop=True),
        dm_fn_persisted.reset_index(drop=True),
        check_names=False,
    )


def test_compute_derived_columns_recomputes_when_persisted_columns_absent():
    """Fallback path: an older cleaned parquet without DM columns still works."""
    cleaned = transform_dataframe(_raw_cohort())
    cleaned_no_dm = cleaned.drop(columns=[_COL_DM_LN, _COL_DM_FN])

    derived = _compute_derived_columns(cleaned_no_dm)
    # Computed values must equal the persisted values.
    expected_ln = cleaned[_COL_DM_LN]
    expected_fn = cleaned[_COL_DM_FN]
    pdt.assert_series_equal(
        derived[_COL_DM_LN].reset_index(drop=True),
        expected_ln.reset_index(drop=True),
        check_names=False,
    )
    pdt.assert_series_equal(
        derived[_COL_DM_FN].reset_index(drop=True),
        expected_fn.reset_index(drop=True),
        check_names=False,
    )


# ── 3. End-to-end blocking output unchanged (the regression gate) ────────────
def test_batch_blocking_output_unchanged_with_or_without_persisted_dm():
    """If both code paths produce the same DM values, batch blocking must
    produce identical candidate pairs regardless of whether the cleaned frame
    carries the persisted columns."""
    cleaned = transform_dataframe(_raw_cohort())
    cleaned_no_dm = cleaned.drop(columns=[_COL_DM_LN, _COL_DM_FN])

    pairs_with    = run_batch_blocking(cleaned).sort_values(["PATID_A", "PATID_B"]).reset_index(drop=True)
    pairs_without = run_batch_blocking(cleaned_no_dm).sort_values(["PATID_A", "PATID_B"]).reset_index(drop=True)
    pdt.assert_frame_equal(pairs_with, pairs_without)
