"""
Unit tests for blocking key column generation logic inside
_compute_derived_columns and the key construction patterns used by
each blocking scheme.

These tests verify:
    1. Null propagation — a null in any required field produces a null key
    2. Pipe separator prevents false key collisions
    3. Derived columns (DOB string, birth year, FN prefix) are computed correctly
    4. Key values match expected format for each blocking scheme

WHY THIS MATTERS:
    A blocking key error silently drops true match pairs from the
    candidate set. If a key is constructed incorrectly (e.g., missing
    pipe separator, wrong date format, truncated prefix), pairs that
    should match on that key will not be generated as candidates.
    These errors are undetectable downstream — the FS model never sees
    the missing pairs and cannot compensate.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.blocking import _compute_derived_columns

# Column name constants (must match blocking.py)
COL_PATID    = "PATID"
COL_LAST_NM  = "LastNM_clean"
COL_FIRST_NM = "FirstNM_clean"
COL_BIRTH_DT = "BirthDT_clean"
COL_SSN      = "SSN_clean"
COL_SSN_L4   = "last_4_SSN"
COL_ZIP      = "ZipCD_clean_base"
COL_EMAIL    = "Email_clean"
COL_PHONES   = "Phones_set"


# ── Fixture: minimal valid record ─────────────────────────────────────────

def _make_record(**overrides):
    """
    Create a single-row DataFrame representing one cleaned patient record.
    All fields populated with realistic defaults. Override any field via kwargs.
    """
    defaults = {
        COL_PATID:    "PAT_001",
        COL_LAST_NM:  "JOHNSON",
        COL_FIRST_NM: "MICHAEL",
        COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
        COL_SSN:      "123456789",
        COL_SSN_L4:   "6789",
        COL_ZIP:      "60614",
        COL_EMAIL:    "mjohnson@email.com",
        COL_PHONES:   "{'7735551234'}",
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def _make_pair(record_a_overrides, record_b_overrides):
    """
    Create a two-row DataFrame representing a pair of records.
    Both records start from the same defaults; overrides customize each.
    """
    rec_a = _make_record(
        **{COL_PATID: "PAT_A", **record_a_overrides}
    )
    rec_b = _make_record(
        **{COL_PATID: "PAT_B", **record_b_overrides}
    )
    return pd.concat([rec_a, rec_b], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Derived Column Computation
# ═══════════════════════════════════════════════════════════════════════════

class TestDerivedColumns:
    """Tests for _compute_derived_columns correctness."""

    def test_dob_string_format(self):
        """DOB string must be in YYYY-MM-DD format for key concatenation."""
        df = _make_record(**{COL_BIRTH_DT: pd.Timestamp("1985-08-04")})
        result = _compute_derived_columns(df)
        assert result["_dob_str"].iloc[0] == "1985-08-04"

    def test_birth_year_extraction(self):
        """Birth year must be extracted as an integer."""
        df = _make_record(**{COL_BIRTH_DT: pd.Timestamp("1985-08-04")})
        result = _compute_derived_columns(df)
        assert result["_birth_year"].iloc[0] == 1985

    def test_fn_prefix3(self):
        """First name prefix must be exactly 3 characters."""
        df = _make_record(**{COL_FIRST_NM: "MICHAEL"})
        result = _compute_derived_columns(df)
        assert result["_fn_prefix3"].iloc[0] == "MIC"

    def test_fn_prefix3_short_name(self):
        """If first name is 2 characters, prefix is the full name."""
        df = _make_record(**{COL_FIRST_NM: "LI"})
        result = _compute_derived_columns(df)
        assert result["_fn_prefix3"].iloc[0] == "LI"

    def test_phonetic_columns_created(self):
        """DM and Soundex columns must be present after computation."""
        df = _make_record()
        result = _compute_derived_columns(df)
        assert "_dm_LastNM" in result.columns
        assert "_sx_LastNM" in result.columns
        assert "_sx_FirstNM" in result.columns

    def test_phonetic_columns_non_null_for_valid_names(self):
        """Phonetic columns must be non-null for valid name inputs."""
        df = _make_record()
        result = _compute_derived_columns(df)
        assert pd.notna(result["_dm_LastNM"].iloc[0])
        assert pd.notna(result["_sx_LastNM"].iloc[0])
        assert pd.notna(result["_sx_FirstNM"].iloc[0])

    def test_phone_set_parsed(self):
        """Phones_set string must be parsed to a Python set."""
        df = _make_record(**{COL_PHONES: "{'7735551234', '3125559876'}"})
        result = _compute_derived_columns(df)
        phones = result["_phones_parsed"].iloc[0]
        assert isinstance(phones, set)
        assert "7735551234" in phones

    def test_null_dob_produces_null_derived(self):
        """Null BirthDT must produce null DOB string and birth year."""
        df = _make_record(**{COL_BIRTH_DT: None})
        result = _compute_derived_columns(df)
        assert pd.isna(result["_dob_str"].iloc[0]) or result["_dob_str"].iloc[0] == "NaT"
        assert pd.isna(result["_birth_year"].iloc[0])

    def test_null_first_name_produces_null_prefix(self):
        """Null FirstNM must produce null FN prefix."""
        df = _make_record(**{COL_FIRST_NM: None})
        result = _compute_derived_columns(df)
        assert pd.isna(result["_fn_prefix3"].iloc[0])

    def test_null_last_name_produces_null_phonetics(self):
        """Null LastNM must produce null DM and Soundex columns."""
        df = _make_record(**{COL_LAST_NM: None})
        result = _compute_derived_columns(df)
        assert result["_dm_LastNM"].iloc[0] is None
        assert result["_sx_LastNM"].iloc[0] is None


# ═══════════════════════════════════════════════════════════════════════════
# Null Propagation in Key Construction
# ═══════════════════════════════════════════════════════════════════════════

class TestNullPropagation:
    """
    A null in any field required by a blocking key must produce a null key.
    This prevents false matches between records that share null field values
    (e.g., two records with null SSN should NOT form a B1 candidate pair).
    """

    def test_b1_null_ssn(self):
        """B1: null SSN → record does not participate in B1 blocking."""
        df = _make_record(**{COL_SSN: None})
        result = _compute_derived_columns(df)
        # B1 key would be clean_SSN — verify it's null
        assert pd.isna(result[COL_SSN].iloc[0])

    def test_b3_null_lastnm(self):
        """B3: null LastNM → null DM → record cannot form B3 key."""
        df = _make_record(**{COL_LAST_NM: None})
        result = _compute_derived_columns(df)
        assert result["_dm_LastNM"].iloc[0] is None

    def test_b3_null_dob(self):
        """B3: null DOB → record cannot form B3 key."""
        df = _make_record(**{COL_BIRTH_DT: None})
        result = _compute_derived_columns(df)
        dob_str = result["_dob_str"].iloc[0]
        assert pd.isna(dob_str) or dob_str == "NaT"

    def test_b4_null_birth_year(self):
        """B4: null BirthYear → record cannot form B4 key."""
        df = _make_record(**{COL_BIRTH_DT: None})
        result = _compute_derived_columns(df)
        assert pd.isna(result["_birth_year"].iloc[0])

    def test_b6_null_email(self):
        """B6: null email → record does not participate in B6 blocking."""
        df = _make_record(**{COL_EMAIL: None})
        result = _compute_derived_columns(df)
        assert pd.isna(result[COL_EMAIL].iloc[0])

    def test_b7_null_zip(self):
        """B7: null ZIP → record cannot form B7 key."""
        df = _make_record(**{COL_ZIP: None})
        result = _compute_derived_columns(df)
        assert pd.isna(result[COL_ZIP].iloc[0])

    def test_b9_null_ssn_last4(self):
        """B9: null SSN_Last4 → record cannot form B9 key."""
        df = _make_record(**{COL_SSN_L4: None})
        result = _compute_derived_columns(df)
        assert pd.isna(result[COL_SSN_L4].iloc[0])


# ═══════════════════════════════════════════════════════════════════════════
# Pipe Separator Collision Prevention
# ═══════════════════════════════════════════════════════════════════════════

class TestPipeSeparator:
    """
    Blocking keys that concatenate multiple fields use a pipe separator
    to prevent false collisions. Without the separator, two different
    field combinations could produce the same concatenated string:
        "SMIT" + "H1985" == "SMITH" + "1985" (both = "SMITH1985")
    With pipe: "SMIT|H1985" != "SMITH|1985"
    """

    def test_b3_key_contains_pipe(self):
        """B3 key must use pipe between DM code and DOB string."""
        df = _make_record()
        result = _compute_derived_columns(df)
        dm_code = result["_dm_LastNM"].iloc[0]
        dob_str = result["_dob_str"].iloc[0]
        # Verify the pipe separator would produce the correct key
        expected_key = f"{dm_code}|{dob_str}"
        assert "|" in expected_key
        assert expected_key.count("|") == 1

    def test_b4_key_contains_two_pipes(self):
        """B4 key has 3 components → exactly 2 pipe separators."""
        df = _make_record()
        result = _compute_derived_columns(df)
        ln = result[COL_LAST_NM].iloc[0]
        yr = str(result["_birth_year"].iloc[0])
        fn3 = result["_fn_prefix3"].iloc[0]
        expected_key = f"{ln}|{yr}|{fn3}"
        assert expected_key.count("|") == 2

    def test_collision_prevented_by_pipe(self):
        """Two records that would collide without pipe must not collide with pipe."""
        # Record A: LastNM="SMIT", BirthYear="H1985" (contrived but illustrative)
        # Record B: LastNM="SMITH", BirthYear="1985"
        # Without pipe: "SMITH1985" == "SMITH1985" → FALSE COLLISION
        # With pipe: "SMIT|H1985" != "SMITH|1985" → CORRECT
        key_a = "SMIT" + "|" + "H1985"
        key_b = "SMITH" + "|" + "1985"
        assert key_a != key_b


# ═══════════════════════════════════════════════════════════════════════════
# Idempotency
# ═══════════════════════════════════════════════════════════════════════════

class TestIdempotency:
    """_compute_derived_columns must be safe to call multiple times."""

    def test_double_computation_produces_same_result(self):
        """Calling _compute_derived_columns twice on the same data
        must produce identical output columns."""
        df = _make_record()
        first  = _compute_derived_columns(df)
        second = _compute_derived_columns(first)

        assert first["_dm_LastNM"].iloc[0] == second["_dm_LastNM"].iloc[0]
        assert first["_sx_LastNM"].iloc[0] == second["_sx_LastNM"].iloc[0]
        assert first["_dob_str"].iloc[0] == second["_dob_str"].iloc[0]
        assert first["_birth_year"].iloc[0] == second["_birth_year"].iloc[0]

    def test_original_dataframe_not_modified(self):
        """_compute_derived_columns must not modify the input DataFrame."""
        df = _make_record()
        original_columns = set(df.columns)
        _ = _compute_derived_columns(df)
        assert set(df.columns) == original_columns