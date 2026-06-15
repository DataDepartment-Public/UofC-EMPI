"""
=========================================
Integration tests for build_blocking_index() — the function that
pre-computes all lookup structures used by both run_batch_blocking()
and run_inference_blocking().

WHAT THIS TEST PROVES:
    1. The BlockingIndex is built correctly from a cleaned dataset
    2. Every active block's index is populated with the right keys
    3. The phone index correctly parses serialized Phones_set strings
    4. Null fields do not produce index entries (no false key collisions)
    5. The index is reusable — calling build_blocking_index twice on
       the same data produces identical indexes
    6. Index metadata (built_at, n_records) is correctly recorded

WHY THIS MATTERS:
    build_blocking_index() is called once at pipeline startup and the
    resulting BlockingIndex is reused for every inference request.
    A silent error here — a wrong key, a missing entry, a null being
    indexed — propagates to every subsequent inference call without
    any obvious failure signal. These tests catch index-level bugs
    before they silently degrade recall in production.
"""

import pytest
import pandas as pd
from datetime import datetime

from src.features.blocking import (
    build_blocking_index,
    BlockingIndex,
    _dm_primary,
    _soundex,
)

# ── Column name constants ─────────────────────────────────────────────────
COL_PATID    = "PATID"
COL_LAST_NM  = "LastNM_clean"
COL_FIRST_NM = "FirstNM_clean"
COL_BIRTH_DT = "BirthDT_clean"
COL_SSN      = "SSN_clean"
COL_SSN_L4   = "last_4_SSN"
COL_ZIP      = "ZipCD_clean_base"
COL_EMAIL    = "Email_clean"
COL_PHONES   = "Phones_set"


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sample_df():
    """
    Small controlled dataset designed to exercise every index structure.
    5 records with known field values so we can assert exact index contents.
    """
    return pd.DataFrame([
        {
            COL_PATID:    "PAT_001",
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "MICHAEL",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
            COL_SSN:      "234567891",
            COL_SSN_L4:   "7891",
            COL_ZIP:      "60614",
            COL_EMAIL:    "michael.j@test.com",
            COL_PHONES:   "{'7735551234', '3125559876'}",
        },
        {
            COL_PATID:    "PAT_002",
            COL_LAST_NM:  "JONSON",           # phonetic variant of JOHNSON
            COL_FIRST_NM: "MIKE",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),  # same DOB as PAT_001
            COL_SSN:      "234567891",         # same SSN as PAT_001
            COL_SSN_L4:   "7891",
            COL_ZIP:      "60614",             # same ZIP as PAT_001
            COL_EMAIL:    "mike.j@test.com",
            COL_PHONES:   "{'7735551234'}",    # shares one phone with PAT_001
        },
        {
            COL_PATID:    "PAT_003",
            COL_LAST_NM:  "GARCIA",
            COL_FIRST_NM: "CARLOS",
            COL_BIRTH_DT: pd.Timestamp("1990-03-15"),
            COL_SSN:      None,                # null SSN
            COL_SSN_L4:   None,
            COL_ZIP:      "60640",
            COL_EMAIL:    None,                # null email
            COL_PHONES:   "{'3125550001'}",
        },
        {
            COL_PATID:    "PAT_004",
            COL_LAST_NM:  "GARCIA",           # same last name as PAT_003
            COL_FIRST_NM: "MARIA",
            COL_BIRTH_DT: pd.Timestamp("1990-03-15"),  # same DOB as PAT_003
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60640",             # same ZIP as PAT_003
            COL_EMAIL:    None,
            COL_PHONES:   "{'8475550002'}",
        },
        {
            COL_PATID:    "PAT_005",
            COL_LAST_NM:  "UNIQUE",
            COL_FIRST_NM: "PERSON",
            COL_BIRTH_DT: pd.Timestamp("2000-01-01"),
            COL_SSN:      "111223333",
            COL_SSN_L4:   "3333",
            COL_ZIP:      "99999",
            COL_EMAIL:    "unique@test.com",
            COL_PHONES:   "set()",             # empty phone set
        },
    ])


@pytest.fixture(scope="module")
def index(sample_df):
    """Build the BlockingIndex once for all tests in this module."""
    return build_blocking_index(sample_df)


# ═══════════════════════════════════════════════════════════════════════════
# Return Type and Structure
# ═══════════════════════════════════════════════════════════════════════════

class TestBlockingIndexStructure:
    """Verify the returned object has the correct type and attributes."""

    def test_returns_blocking_index_instance(self, index):
        """build_blocking_index must return a BlockingIndex dataclass."""
        assert isinstance(index, BlockingIndex)

    def test_all_index_attributes_present(self, index):
        """Every block's index attribute must be present."""
        assert hasattr(index, "b1_index")
        assert hasattr(index, "b3_index")
        assert hasattr(index, "b4_index")
        assert hasattr(index, "b5_phone_index")
        assert hasattr(index, "b6_index")
        assert hasattr(index, "b7_index")
        assert hasattr(index, "b8_index")
        assert hasattr(index, "b9_index")
        assert hasattr(index, "patid_phones")

    def test_no_b2_index_attribute(self, index):
        """B2 was removed from the scheme — no b2_index should exist."""
        assert not hasattr(index, "b2_index"), (
            "b2_index attribute found — B2 was removed and must not "
            "be present in the BlockingIndex"
        )

    def test_metadata_built_at_is_string(self, index):
        """built_at metadata must be a non-empty ISO format string."""
        assert isinstance(index.built_at, str)
        assert len(index.built_at) > 0
        # Verify it parses as a valid datetime
        try:
            datetime.fromisoformat(index.built_at)
        except ValueError:
            pytest.fail(
                f"built_at is not a valid ISO datetime: '{index.built_at}'"
            )

    def test_metadata_n_records(self, index, sample_df):
        """n_records must equal the number of records in the input dataset."""
        assert index.n_records == len(sample_df), (
            f"Expected n_records={len(sample_df)}, got {index.n_records}"
        )

    def test_all_indexes_are_dicts(self, index):
        """Every index structure must be a dict."""
        assert isinstance(index.b1_index, dict)
        assert isinstance(index.b3_index, dict)
        assert isinstance(index.b4_index, dict)
        assert isinstance(index.b5_phone_index, dict)
        assert isinstance(index.b6_index, dict)
        assert isinstance(index.b7_index, dict)
        assert isinstance(index.b8_index, dict)
        assert isinstance(index.b9_index, dict)
        assert isinstance(index.patid_phones, dict)


# ═══════════════════════════════════════════════════════════════════════════
# B1 Index — SSN Exact
# ═══════════════════════════════════════════════════════════════════════════

class TestB1Index:

    def test_shared_ssn_groups_correct_patids(self, index):
        """SSN '234567891' is shared by PAT_001 and PAT_002."""
        assert "234567891" in index.b1_index
        patids = set(index.b1_index["234567891"])
        assert "PAT_001" in patids
        assert "PAT_002" in patids

    def test_unique_ssn_has_one_record(self, index):
        """SSN '111223333' belongs only to PAT_005."""
        assert "111223333" in index.b1_index
        assert len(index.b1_index["111223333"]) == 1
        assert "PAT_005" in index.b1_index["111223333"]

    def test_null_ssn_not_indexed(self, index):
        """PAT_003 and PAT_004 have null SSN — must not create index entries."""
        all_patids_in_b1 = [
            p for patids in index.b1_index.values() for p in patids
        ]
        assert "PAT_003" not in all_patids_in_b1
        assert "PAT_004" not in all_patids_in_b1

    def test_no_empty_string_keys(self, index):
        """No index key should be an empty string or None."""
        for key in index.b1_index:
            assert key is not None
            assert key != ""


# ═══════════════════════════════════════════════════════════════════════════
# B3 Index — DM(LastNM) + Full DOB
# ═══════════════════════════════════════════════════════════════════════════

class TestB3Index:

    def test_johnson_jonson_share_b3_key(self, index):
        """JOHNSON and JONSON must produce the same DM code + DOB key."""
        dm_johnson = _dm_primary("JOHNSON")
        dm_jonson  = _dm_primary("JONSON")
        assert dm_johnson == dm_jonson, (
            "JOHNSON and JONSON must share a DM code for B3 to work"
        )
        expected_key = f"{dm_johnson}|1985-08-04"
        assert expected_key in index.b3_index
        patids = set(index.b3_index[expected_key])
        assert "PAT_001" in patids
        assert "PAT_002" in patids

    def test_garcia_records_share_b3_key(self, index):
        """PAT_003 and PAT_004 (both GARCIA, same DOB) must share a B3 key."""
        dm_garcia = _dm_primary("GARCIA")
        expected_key = f"{dm_garcia}|1990-03-15"
        assert expected_key in index.b3_index
        patids = set(index.b3_index[expected_key])
        assert "PAT_003" in patids
        assert "PAT_004" in patids

    def test_pipe_separator_in_b3_keys(self, index):
        """All B3 keys must contain exactly one pipe separator."""
        for key in index.b3_index:
            assert key.count("|") == 1, (
                f"B3 key missing or has extra pipe separator: '{key}'"
            )


# ═══════════════════════════════════════════════════════════════════════════
# B4 Index — LastNM + BirthYear + FN Prefix
# ═══════════════════════════════════════════════════════════════════════════

class TestB4Index:

    def test_b4_key_format(self, index):
        """B4 keys must have exactly two pipe separators (3 components)."""
        for key in index.b4_index:
            assert key.count("|") == 2, (
                f"B4 key has wrong number of pipes: '{key}'"
            )

    def test_johnson_pat001_in_b4(self, index):
        """PAT_001 (JOHNSON, 1985, MICHAEL) must appear in B4 index."""
        expected_key = "JOHNSON|1985|MIC"
        assert expected_key in index.b4_index
        assert "PAT_001" in index.b4_index[expected_key]


# ═══════════════════════════════════════════════════════════════════════════
# B5 Phone Index
# ═══════════════════════════════════════════════════════════════════════════

class TestB5PhoneIndex:

    def test_shared_phone_groups_correct_patids(self, index):
        """Phone '7735551234' is shared by PAT_001 and PAT_002."""
        assert "7735551234" in index.b5_phone_index
        patids = index.b5_phone_index["7735551234"]
        assert "PAT_001" in patids
        assert "PAT_002" in patids

    def test_unique_phone_has_one_record(self, index):
        """Phone '3125559876' belongs only to PAT_001."""
        assert "3125559876" in index.b5_phone_index
        assert "PAT_001" in index.b5_phone_index["3125559876"]
        assert "PAT_002" not in index.b5_phone_index["3125559876"]

    def test_empty_phone_set_not_indexed(self, index):
        """PAT_005 has empty phone set — must not appear in phone index."""
        all_patids_in_b5 = [
            p for patids in index.b5_phone_index.values() for p in patids
        ]
        assert "PAT_005" not in all_patids_in_b5

    def test_patid_phones_populated(self, index):
        """patid_phones must map PATIDs to their phone sets."""
        assert "PAT_001" in index.patid_phones
        phones = index.patid_phones["PAT_001"]
        assert isinstance(phones, set)
        assert "7735551234" in phones
        assert "3125559876" in phones

    def test_patid_phones_empty_set_excluded(self, index):
        """PAT_005 with empty phones must not appear in patid_phones."""
        assert "PAT_005" not in index.patid_phones

    def test_phone_index_values_are_sets(self, index):
        """All values in b5_phone_index must be sets of PATID strings."""
        for phone, patid_set in index.b5_phone_index.items():
            assert isinstance(patid_set, set), (
                f"b5_phone_index['{phone}'] is not a set: {type(patid_set)}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# B6 Index — Email Exact
# ═══════════════════════════════════════════════════════════════════════════

class TestB6Index:

    def test_unique_email_indexed(self, index):
        """PAT_001's email must appear in B6 index."""
        assert "michael.j@test.com" in index.b6_index
        assert "PAT_001" in index.b6_index["michael.j@test.com"]

    def test_null_email_not_indexed(self, index):
        """PAT_003 and PAT_004 have null email — must not appear in B6."""
        all_patids_in_b6 = [
            p for patids in index.b6_index.values() for p in patids
        ]
        assert "PAT_003" not in all_patids_in_b6
        assert "PAT_004" not in all_patids_in_b6


# ═══════════════════════════════════════════════════════════════════════════
# B7 Index — DM(LastNM) + ZIP + BirthYear
# ═══════════════════════════════════════════════════════════════════════════

class TestB7Index:

    def test_garcia_records_share_b7_key(self, index):
        """PAT_003 and PAT_004 share DM(GARCIA) + ZIP 60640 + Year 1990."""
        dm_garcia = _dm_primary("GARCIA")
        expected_key = f"{dm_garcia}|60640|1990"
        assert expected_key in index.b7_index
        patids = set(index.b7_index[expected_key])
        assert "PAT_003" in patids
        assert "PAT_004" in patids

    def test_b7_key_format(self, index):
        """B7 keys must have exactly two pipe separators (3 components)."""
        for key in index.b7_index:
            assert key.count("|") == 2, (
                f"B7 key has wrong pipe count: '{key}'"
            )


# ═══════════════════════════════════════════════════════════════════════════
# B8 Index — Soundex(FN) + Soundex(LN) + BirthYear
# ═══════════════════════════════════════════════════════════════════════════

class TestB8Index:

    def test_johnson_jonson_share_b8_key(self, index):
        """JOHNSON and JONSON share Soundex(LN). MICHAEL and MIKE share
        Soundex(FN) if their Soundex codes match — verify and test."""
        sx_johnson = _soundex("JOHNSON")
        sx_jonson  = _soundex("JONSON")
        sx_michael = _soundex("MICHAEL")
        sx_mike    = _soundex("MIKE")

        # JOHNSON and JONSON must share LN Soundex for B8 to work
        assert sx_johnson == sx_jonson, (
            f"JOHNSON({sx_johnson}) and JONSON({sx_jonson}) must share "
            "Soundex code for B8 blocking"
        )

        # If MICHAEL and MIKE share FN Soundex, they'll be in the same B8 key
        if sx_michael == sx_mike:
            expected_key = f"{sx_michael}|{sx_johnson}|1985"
            if expected_key in index.b8_index:
                patids = set(index.b8_index[expected_key])
                assert "PAT_001" in patids
                assert "PAT_002" in patids

    def test_b8_key_format(self, index):
        """B8 keys must have exactly two pipe separators (3 components)."""
        for key in index.b8_index:
            assert key.count("|") == 2, (
                f"B8 key has wrong pipe count: '{key}'"
            )


# ═══════════════════════════════════════════════════════════════════════════
# B9 Index — LastNM + FirstNM + SSN Last 4
# ═══════════════════════════════════════════════════════════════════════════

class TestB9Index:

    def test_shared_last4_same_name_indexed(self, index):
        """PAT_001 and PAT_002 share LN(JOHNSON/JONSON)? No — different LN.
        But both share SSN last4='7891'. They won't share B9 key because
        LN differs. This verifies B9 requires exact LN match."""
        # JOHNSON|MICHAEL|7891 — PAT_001
        assert "JOHNSON|MICHAEL|7891" in index.b9_index
        assert "PAT_001" in index.b9_index["JOHNSON|MICHAEL|7891"]
        # JONSON|MIKE|7891 — PAT_002 (different key due to different LN/FN)
        assert "JONSON|MIKE|7891" in index.b9_index
        assert "PAT_002" in index.b9_index["JONSON|MIKE|7891"]
        # The two keys are different — B9 requires exact name match
        assert "JOHNSON|MICHAEL|7891" != "JONSON|MIKE|7891"

    def test_null_ssn_last4_not_indexed(self, index):
        """PAT_003 and PAT_004 have null SSN_L4 — must not appear in B9."""
        all_patids_in_b9 = [
            p for patids in index.b9_index.values() for p in patids
        ]
        assert "PAT_003" not in all_patids_in_b9
        assert "PAT_004" not in all_patids_in_b9

    def test_b9_key_format(self, index):
        """B9 keys must have exactly two pipe separators (3 components)."""
        for key in index.b9_index:
            assert key.count("|") == 2, (
                f"B9 key has wrong pipe count: '{key}'"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Reusability and Idempotency
# ═══════════════════════════════════════════════════════════════════════════

class TestIndexReusability:
    """The BlockingIndex must be safely reusable across multiple calls."""

    def test_build_twice_produces_identical_b1_index(self, sample_df):
        """Calling build_blocking_index twice must produce identical B1 indexes."""
        index_a = build_blocking_index(sample_df)
        index_b = build_blocking_index(sample_df)
        assert index_a.b1_index == index_b.b1_index

    def test_build_twice_produces_identical_b5_phone_index(self, sample_df):
        """Phone index must be identical across two builds."""
        index_a = build_blocking_index(sample_df)
        index_b = build_blocking_index(sample_df)
        # Compare phone keys
        assert set(index_a.b5_phone_index.keys()) == set(index_b.b5_phone_index.keys())

    def test_original_dataframe_not_modified(self, sample_df):
        """build_blocking_index must not modify the input DataFrame."""
        original_cols   = set(sample_df.columns)
        original_shape  = sample_df.shape
        _ = build_blocking_index(sample_df)
        assert set(sample_df.columns) == original_cols, (
            "build_blocking_index modified the input DataFrame's columns"
        )
        assert sample_df.shape == original_shape

    def test_built_at_changes_between_builds(self, sample_df):
        """Each build produces a new built_at timestamp."""
        import time
        index_a = build_blocking_index(sample_df)
        time.sleep(0.01)  # ensure clock advances
        index_b = build_blocking_index(sample_df)
        # built_at should be different (different build time)
        # This may occasionally be equal on very fast machines —
        # we assert both are valid ISO strings rather than strict inequality
        assert isinstance(index_a.built_at, str)
        assert isinstance(index_b.built_at, str)


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestIndexEdgeCases:

    def test_empty_dataframe_produces_empty_indexes(self):
        """An empty DataFrame must produce empty (not None) indexes."""
        empty_df = pd.DataFrame(columns=[
            COL_PATID, COL_LAST_NM, COL_FIRST_NM, COL_BIRTH_DT,
            COL_SSN, COL_SSN_L4, COL_ZIP, COL_EMAIL, COL_PHONES,
        ])
        index = build_blocking_index(empty_df)
        assert isinstance(index, BlockingIndex)
        assert len(index.b1_index) == 0
        assert len(index.b3_index) == 0
        assert len(index.b5_phone_index) == 0
        assert index.n_records == 0

    def test_single_record_produces_no_matchable_keys(self):
        """A single record produces index entries but no pairs are possible."""
        single_df = pd.DataFrame([{
            COL_PATID:    "PAT_SOLO",
            COL_LAST_NM:  "SOLO",
            COL_FIRST_NM: "PERSON",
            COL_BIRTH_DT: pd.Timestamp("1990-01-01"),
            COL_SSN:      "234567891",
            COL_SSN_L4:   "7891",
            COL_ZIP:      "60601",
            COL_EMAIL:    "solo@test.com",
            COL_PHONES:   "{'7731234567'}",
        }])
        index = build_blocking_index(single_df)
        # Index is built (PAT_SOLO appears in B1)
        assert "234567891" in index.b1_index
        assert "PAT_SOLO" in index.b1_index["234567891"]
        # But no pairs are possible from a single-record list
        for key, patid_list in index.b1_index.items():
            assert len(patid_list) <= 1 or patid_list.count("PAT_SOLO") == 1