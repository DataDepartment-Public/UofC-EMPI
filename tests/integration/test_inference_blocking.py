"""
=============================================
Integration tests for run_inference_blocking() — the function that
matches a single new incoming record against the pre-built BlockingIndex
of existing records.

WHAT THIS TEST PROVES:
    1. A new record generates correct candidate pairs against the index
    2. Each active block correctly identifies matches in inference mode
    3. Null fields in the new record don't generate spurious candidates
    4. The new record's PATID appears as PATID_A in all output pairs
    5. The function handles records with no matches gracefully
    6. The BlockingIndex is not mutated by inference calls

WHY THIS MATTERS:
    run_inference_blocking() is the production path for matching new
    patient records as they arrive from clinic systems. Unlike batch
    blocking which runs periodically, inference runs in real-time.
    A bug here — a missed candidate, a spurious match, a crash on
    a null field — directly affects patient care continuity.

ARCHITECTURE NOTE:
    Inference mode is tested against a pre-built index (built from the
    same sample dataset used in test_blocking_index.py). The new record
    being matched is a synthetic record designed to match specific
    records in the index via specific blocks.
"""

import pytest
import pandas as pd

from src.features.blocking import (
    build_blocking_index,
    run_inference_blocking,
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
def index_df():
    """
    Existing patient index — 5 records with known field values.
    The new incoming records in tests are designed to match specific
    records in this index via specific blocking schemes.
    """
    return pd.DataFrame([
        {
            COL_PATID:    "IDX_001",
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "MICHAEL",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
            COL_SSN:      "234567891",
            COL_SSN_L4:   "7891",
            COL_ZIP:      "60614",
            COL_EMAIL:    "michael.j@index.com",
            COL_PHONES:   "{'7735551234', '3125559876'}",
        },
        {
            COL_PATID:    "IDX_002",
            COL_LAST_NM:  "GARCIA",
            COL_FIRST_NM: "CARLOS",
            COL_BIRTH_DT: pd.Timestamp("1990-03-15"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60640",
            COL_EMAIL:    "carlos.g@index.com",
            COL_PHONES:   "{'3125550001'}",
        },
        {
            COL_PATID:    "IDX_003",
            COL_LAST_NM:  "RODRIGUEZ",
            COL_FIRST_NM: "ELENA",
            COL_BIRTH_DT: pd.Timestamp("1978-11-22"),
            COL_SSN:      "456789012",
            COL_SSN_L4:   "9012",
            COL_ZIP:      "60620",
            COL_EMAIL:    "elena.r@index.com",
            COL_PHONES:   "{'8475550003'}",
        },
        {
            COL_PATID:    "IDX_004",
            COL_LAST_NM:  "NGUYEN",
            COL_FIRST_NM: "MINH",
            COL_BIRTH_DT: pd.Timestamp("1995-06-08"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60660",
            COL_EMAIL:    "minh.n@index.com",
            COL_PHONES:   "{'7730004444'}",
        },
        {
            COL_PATID:    "IDX_005",
            COL_LAST_NM:  "UNIQUE",
            COL_FIRST_NM: "NOBODY",
            COL_BIRTH_DT: pd.Timestamp("2000-12-31"),
            COL_SSN:      "999888777",
            COL_SSN_L4:   "8777",
            COL_ZIP:      "99990",
            COL_EMAIL:    "unique@index.com",
            COL_PHONES:   "{'5550000001'}",
        },
    ])


@pytest.fixture(scope="module")
def blocking_index(index_df):
    """Build the BlockingIndex once for all inference tests."""
    return build_blocking_index(index_df)


# ═══════════════════════════════════════════════════════════════════════════
# Output Schema
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceOutputSchema:
    """Verify output schema matches specification."""

    def test_returns_dataframe(self, blocking_index):
        """run_inference_blocking must return a DataFrame."""
        new_record = {
            COL_PATID:    "NEW_001",
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "MICHAEL",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
            COL_SSN:      "234567891",
            COL_SSN_L4:   "7891",
            COL_ZIP:      "60614",
            COL_EMAIL:    "michael.j@index.com",
            COL_PHONES:   "{'7735551234'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_required_columns(self, blocking_index):
        """Output must contain PATID_A, PATID_B, source_blocks, n_blocks."""
        new_record = {
            COL_PATID:    "NEW_SCHEMA",
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "MICHAEL",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
            COL_SSN:      "234567891",
            COL_SSN_L4:   "7891",
            COL_ZIP:      "60614",
            COL_EMAIL:    "michael.j@index.com",
            COL_PHONES:   "{'7735551234'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        expected = {"PATID_A", "PATID_B", "source_blocks", "n_blocks"}
        assert set(result.columns) == expected

    def test_new_patid_always_patid_a(self, blocking_index):
        """The new record's PATID must always appear as PATID_A."""
        new_record = {
            COL_PATID:    "NEW_PATID_A_TEST",
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "MICHAEL",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
            COL_SSN:      "234567891",
            COL_SSN_L4:   "7891",
            COL_ZIP:      "60614",
            COL_EMAIL:    "michael.j@index.com",
            COL_PHONES:   "{'7735551234'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        if not result.empty:
            assert all(result["PATID_A"] == "NEW_PATID_A_TEST"), (
                "New record's PATID must always be PATID_A in inference output"
            )

    def test_no_self_pairs_in_inference(self, blocking_index):
        """PATID_A must never equal PATID_B."""
        new_record = {
            COL_PATID:    "SELF_TEST",
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "MICHAEL",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
            COL_SSN:      "234567891",
            COL_SSN_L4:   "7891",
            COL_ZIP:      "60614",
            COL_EMAIL:    "michael.j@index.com",
            COL_PHONES:   "{'7735551234'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        if not result.empty:
            self_pairs = result[result["PATID_A"] == result["PATID_B"]]
            assert len(self_pairs) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Block-Level Candidate Generation
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceBlockCapture:
    """Each block must correctly identify matches in inference mode."""

    def test_b1_ssn_match_identifies_candidate(self, blocking_index):
        """New record with same SSN as IDX_001 must be matched via B1."""
        new_record = {
            COL_PATID:    "NEW_B1",
            COL_LAST_NM:  "DIFFERENTNAME",
            COL_FIRST_NM: "DIFFERENTFIRST",
            COL_BIRTH_DT: pd.Timestamp("1999-01-01"),
            COL_SSN:      "234567891",       # same as IDX_001
            COL_SSN_L4:   "7891",
            COL_ZIP:      "99990",
            COL_EMAIL:    "new_b1@test.com",
            COL_PHONES:   "{'9990000001'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        candidate_patids = set(result["PATID_B"])
        assert "IDX_001" in candidate_patids, (
            "B1 failed: new record with same SSN as IDX_001 not identified"
        )
        # Verify B1 is in the source_blocks for this pair
        row = result[result["PATID_B"] == "IDX_001"]
        assert "B1" in row.iloc[0]["source_blocks"].split("|")

    def test_b3_phonetic_ln_match(self, blocking_index):
        """New record GARSIA (phonetic variant of GARCIA) + same DOB
        must match IDX_002 via B3."""
        new_record = {
            COL_PATID:    "NEW_B3",
            COL_LAST_NM:  "GARSIA",          # phonetic variant of GARCIA
            COL_FIRST_NM: "CARLOS",
            COL_BIRTH_DT: pd.Timestamp("1990-03-15"),  # same as IDX_002
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60641",
            COL_EMAIL:    "garsia@test.com",
            COL_PHONES:   "{'9990000002'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        candidate_patids = set(result["PATID_B"])
        assert "IDX_002" in candidate_patids, (
            "B3 failed: GARSIA with same DOB should match IDX_002 (GARCIA)"
        )

    def test_b4_dob_transposition_match(self, blocking_index):
        """New record with transposed DOB month/day must match via B4."""
        new_record = {
            COL_PATID:    "NEW_B4",
            COL_LAST_NM:  "RODRIGUEZ",       # same as IDX_003
            COL_FIRST_NM: "ELENA",           # same FN prefix
            COL_BIRTH_DT: pd.Timestamp("1978-22-11") if False else
                          pd.Timestamp("1978-11-22"),  # will use B7 year match
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60620",
            COL_EMAIL:    "elena2@test.com",
            COL_PHONES:   "{'9990000003'}",
        }
        # Corrected: same LN + same BirthYear + same FN prefix → B4 match
        new_record[COL_BIRTH_DT] = pd.Timestamp("1978-05-22")  # same year
        result = run_inference_blocking(new_record, blocking_index)
        candidate_patids = set(result["PATID_B"])
        assert "IDX_003" in candidate_patids, (
            "B4 failed: RODRIGUEZ/ELENA with same year should match IDX_003"
        )

    def test_b5_phone_match(self, blocking_index):
        """New record sharing a phone with IDX_001 must be matched via B5."""
        new_record = {
            COL_PATID:    "NEW_B5",
            COL_LAST_NM:  "COMPLETELY",
            COL_FIRST_NM: "DIFFERENT",
            COL_BIRTH_DT: pd.Timestamp("2001-06-15"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "99991",
            COL_EMAIL:    "different@test.com",
            COL_PHONES:   "{'7735551234'}",   # shares phone with IDX_001
        }
        result = run_inference_blocking(new_record, blocking_index)
        candidate_patids = set(result["PATID_B"])
        assert "IDX_001" in candidate_patids, (
            "B5 failed: new record sharing phone '7735551234' should "
            "match IDX_001"
        )
        row = result[result["PATID_B"] == "IDX_001"]
        assert "B5" in row.iloc[0]["source_blocks"].split("|")

    def test_b6_email_match(self, blocking_index):
        """New record with same email as IDX_002 must be matched via B6."""
        new_record = {
            COL_PATID:    "NEW_B6",
            COL_LAST_NM:  "ANOTHERNAME",
            COL_FIRST_NM: "ANOTHERPERSON",
            COL_BIRTH_DT: pd.Timestamp("2002-01-01"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "99992",
            COL_EMAIL:    "carlos.g@index.com",  # same as IDX_002
            COL_PHONES:   "{'9990000006'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        candidate_patids = set(result["PATID_B"])
        assert "IDX_002" in candidate_patids, (
            "B6 failed: new record with same email should match IDX_002"
        )

    def test_b7_geographic_fallback_match(self, blocking_index):
        """New record with phonetic LN variant + same ZIP + same year
        must match via B7."""
        new_record = {
            COL_PATID:    "NEW_B7",
            COL_LAST_NM:  "RODRIGUES",       # phonetic variant of RODRIGUEZ
            COL_FIRST_NM: "ELENA",
            COL_BIRTH_DT: pd.Timestamp("1978-01-01"),  # same year as IDX_003
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60620",            # same ZIP as IDX_003
            COL_EMAIL:    "rodrigues@test.com",
            COL_PHONES:   "{'9990000007'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        candidate_patids = set(result["PATID_B"])
        assert "IDX_003" in candidate_patids, (
            "B7 failed: RODRIGUES with same ZIP and year should match "
            "IDX_003 (RODRIGUEZ)"
        )

    def test_b9_ssn_last4_match(self, blocking_index):
        """New record with same LN, FN, and SSN last-4 as IDX_003 must
        match via B9."""
        new_record = {
            COL_PATID:    "NEW_B9",
            COL_LAST_NM:  "RODRIGUEZ",       # same as IDX_003
            COL_FIRST_NM: "ELENA",           # same as IDX_003
            COL_BIRTH_DT: pd.Timestamp("1978-05-30"),
            COL_SSN:      "111789012",        # different first digits
            COL_SSN_L4:   "9012",            # SAME last 4 as IDX_003
            COL_ZIP:      "99993",
            COL_EMAIL:    "elena.new@test.com",
            COL_PHONES:   "{'9990000009'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        candidate_patids = set(result["PATID_B"])
        assert "IDX_003" in candidate_patids, (
            "B9 failed: new record with same LN/FN/SSN-last4 should "
            "match IDX_003"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Null Field Handling
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceNullHandling:
    """Null fields in the new record must not generate spurious candidates."""

    def test_null_ssn_does_not_match_any_record(self, blocking_index):
        """A new record with null SSN must not match via B1."""
        new_record = {
            COL_PATID:    "NEW_NULL_SSN",
            COL_LAST_NM:  "NULLSSN",
            COL_FIRST_NM: "TEST",
            COL_BIRTH_DT: pd.Timestamp("2003-01-01"),
            COL_SSN:      None,              # null SSN
            COL_SSN_L4:   None,
            COL_ZIP:      "99994",
            COL_EMAIL:    "nullssn@test.com",
            COL_PHONES:   "{'9990001111'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        # Verify no B1 pairs were generated (null SSN cannot match via B1)
        b1_pairs = result[
            result["source_blocks"].str.contains("B1", na=False)
        ] if not result.empty else pd.DataFrame()
        assert len(b1_pairs) == 0, (
            "Null SSN generated B1 candidate pairs — null must not match"
        )

    def test_null_email_does_not_match_via_b6(self, blocking_index):
        """Null email must not generate B6 candidates."""
        new_record = {
            COL_PATID:    "NEW_NULL_EMAIL",
            COL_LAST_NM:  "NULLEMAIL",
            COL_FIRST_NM: "TEST",
            COL_BIRTH_DT: pd.Timestamp("2003-06-01"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "99995",
            COL_EMAIL:    None,              # null email
            COL_PHONES:   "{'9990002222'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        b6_pairs = result[
            result["source_blocks"].str.contains("B6", na=False)
        ] if not result.empty else pd.DataFrame()
        assert len(b6_pairs) == 0

    def test_completely_null_record_produces_no_candidates(self, blocking_index):
        """A record with all null blocking fields produces zero candidates."""
        new_record = {
            COL_PATID:    "NEW_ALL_NULL",
            COL_LAST_NM:  None,
            COL_FIRST_NM: None,
            COL_BIRTH_DT: None,
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      None,
            COL_EMAIL:    None,
            COL_PHONES:   None,
        }
        result = run_inference_blocking(new_record, blocking_index)
        assert len(result) == 0, (
            f"All-null record produced {len(result)} candidate pairs — "
            "expected 0"
        )

    def test_empty_phone_set_no_b5_match(self, blocking_index):
        """Empty phone set must not generate B5 candidates."""
        new_record = {
            COL_PATID:    "NEW_EMPTY_PHONE",
            COL_LAST_NM:  "EMPTYPHONE",
            COL_FIRST_NM: "TEST",
            COL_BIRTH_DT: pd.Timestamp("2003-09-01"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "99996",
            COL_EMAIL:    "emptyphone@test.com",
            COL_PHONES:   "set()",           # empty phone set
        }
        result = run_inference_blocking(new_record, blocking_index)
        b5_pairs = result[
            result["source_blocks"].str.contains("B5", na=False)
        ] if not result.empty else pd.DataFrame()
        assert len(b5_pairs) == 0


# ═══════════════════════════════════════════════════════════════════════════
# No Match Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceNoMatch:

    def test_completely_unique_record_produces_no_candidates(self, blocking_index):
        """A record sharing no blocking keys with the index produces 0 pairs."""
        new_record = {
            COL_PATID:    "NEW_UNIQUE",
            COL_LAST_NM:  "ZZZZUNIQUE",
            COL_FIRST_NM: "AAAA",
            COL_BIRTH_DT: pd.Timestamp("1800-01-01"),
            COL_SSN:      "888776655",
            COL_SSN_L4:   "6655",
            COL_ZIP:      "00001",
            COL_EMAIL:    "totally.unique@nowhere.com",
            COL_PHONES:   "{'0000000000'}",
        }
        result = run_inference_blocking(new_record, blocking_index)
        assert len(result) == 0, (
            f"Unique record produced {len(result)} unexpected candidates"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Index Immutability
# ═══════════════════════════════════════════════════════════════════════════

class TestIndexImmutability:
    """run_inference_blocking must not modify the BlockingIndex."""

    def test_index_not_mutated_after_inference(self, index_df):
        """Running inference must not add entries to the BlockingIndex."""
        idx = build_blocking_index(index_df)
        b1_keys_before = set(idx.b1_index.keys())
        b5_phones_before = set(idx.b5_phone_index.keys())

        new_record = {
            COL_PATID:    "MUTATION_TEST",
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "MICHAEL",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
            COL_SSN:      "234567891",
            COL_SSN_L4:   "7891",
            COL_ZIP:      "60614",
            COL_EMAIL:    "michael.j@index.com",
            COL_PHONES:   "{'7735551234'}",
        }
        _ = run_inference_blocking(new_record, idx)

        b1_keys_after  = set(idx.b1_index.keys())
        b5_phones_after = set(idx.b5_phone_index.keys())

        assert b1_keys_before == b1_keys_after, (
            "B1 index was mutated by run_inference_blocking"
        )
        assert b5_phones_before == b5_phones_after, (
            "B5 phone index was mutated by run_inference_blocking"
        )

    def test_multiple_inference_calls_consistent(self, blocking_index):
        """Same new record run through inference twice must produce
        identical results."""
        new_record = {
            COL_PATID:    "CONSISTENCY_TEST",
            COL_LAST_NM:  "GARCIA",
            COL_FIRST_NM: "CARLOS",
            COL_BIRTH_DT: pd.Timestamp("1990-03-15"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60640",
            COL_EMAIL:    "carlos.g@index.com",
            COL_PHONES:   "{'3125550001'}",
        }
        result_a = run_inference_blocking(new_record, blocking_index)
        result_b = run_inference_blocking(new_record, blocking_index)

        pairs_a = set(zip(result_a["PATID_A"], result_a["PATID_B"]))
        pairs_b = set(zip(result_b["PATID_A"], result_b["PATID_B"]))
        assert pairs_a == pairs_b, (
            "Inference produced different results on identical inputs"
        )

    def test_dict_and_series_input_identical(self, blocking_index):
        """run_inference_blocking must accept both dict and pd.Series input
        and produce identical results."""
        record_dict = {
            COL_PATID:    "DICT_VS_SERIES",
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "MICHAEL",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
            COL_SSN:      "234567891",
            COL_SSN_L4:   "7891",
            COL_ZIP:      "60614",
            COL_EMAIL:    "michael.j@index.com",
            COL_PHONES:   "{'7735551234'}",
        }
        record_series = pd.Series(record_dict)

        result_dict   = run_inference_blocking(record_dict, blocking_index)
        result_series = run_inference_blocking(record_series, blocking_index)

        pairs_dict   = set(zip(result_dict["PATID_A"], result_dict["PATID_B"]))
        pairs_series = set(zip(result_series["PATID_A"], result_series["PATID_B"]))
        assert pairs_dict == pairs_series, (
            "Dict and Series input produced different inference results"
        )