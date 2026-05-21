"""
Unit tests for candidate pair canonicalization, deduplication, and
output schema validation.

These tests verify:
    1. Pair ordering: PATID_A is always alphabetically before PATID_B
    2. No self-pairs: PATID_A != PATID_B for every pair
    3. No duplicate pairs: each (PATID_A, PATID_B) appears exactly once
    4. Source blocks metadata: pipe-delimited, sorted, no duplicates
    5. Output schema: correct column names, types, and constraints

WHY THIS MATTERS:
    Splink expects canonical pair ordering. If (A, B) and (B, A) both
    appear as separate pairs, the FS model scores them independently,
    producing duplicate match probabilities and inflating the auto-match
    count. If self-pairs exist, the model wastes compute scoring a record
    against itself (which always produces a perfect match score).
"""

import pytest
import numpy as np
import pandas as pd

from src.features.blocking import (
    run_batch_blocking,
    _compute_derived_columns,
    _pairs_from_index,
    _build_key_index,
    _pair_blocks_to_dataframe,
)

# Column name constants
COL_PATID    = "PATID"
COL_LAST_NM  = "LastNM_clean"
COL_FIRST_NM = "FirstNM_clean"
COL_BIRTH_DT = "BirthDT_clean"
COL_SSN      = "SSN_clean"
COL_SSN_L4   = "last_4_SSN"
COL_ZIP      = "ZipCD_clean_base"
COL_EMAIL    = "Email_clean"
COL_PHONES   = "Phones_set"


# ── Fixture: small synthetic dataset for pair tests ──────────────────────

def _make_small_dataset():
    """
    Create a small dataset with known blocking relationships.
    3 records where PAT_001 and PAT_002 share a last name + DOB,
    and PAT_003 is distinct from both.
    """
    return pd.DataFrame([
        {
            COL_PATID:    "PAT_002",   # intentionally not alphabetical
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "MICHAEL",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
            COL_SSN:      "123456789",
            COL_SSN_L4:   "6789",
            COL_ZIP:      "60614",
            COL_EMAIL:    "mjohnson@email.com",
            COL_PHONES:   "{'7735551234'}",
        },
        {
            COL_PATID:    "PAT_001",   # comes first alphabetically
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "MIKE",
            COL_BIRTH_DT: pd.Timestamp("1985-08-04"),
            COL_SSN:      "123456780",   # different SSN
            COL_SSN_L4:   "6780",
            COL_ZIP:      "60614",
            COL_EMAIL:    "mike.j@email.com",
            COL_PHONES:   "{'7735551234'}",   # same phone as PAT_002
        },
        {
            COL_PATID:    "PAT_003",
            COL_LAST_NM:  "GARCIA",
            COL_FIRST_NM: "MARIA",
            COL_BIRTH_DT: pd.Timestamp("1990-03-15"),
            COL_SSN:      "987654321",
            COL_SSN_L4:   "4321",
            COL_ZIP:      "60640",
            COL_EMAIL:    "mgarcia@email.com",
            COL_PHONES:   "{'3125559876'}",
        },
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Pair Ordering (Canonicalization)
# ═══════════════════════════════════════════════════════════════════════════

class TestPairOrdering:
    """PATID_A must be alphabetically before PATID_B in every pair."""

    def test_canonical_ordering_in_batch_output(self):
        """All pairs from run_batch_blocking must have PATID_A < PATID_B."""
        df = _make_small_dataset()
        pairs = run_batch_blocking(df)

        for _, row in pairs.iterrows():
            assert row["PATID_A"] < row["PATID_B"], (
                f"Pair not canonical: PATID_A='{row['PATID_A']}' "
                f"is not < PATID_B='{row['PATID_B']}'"
            )

    def test_pair_order_independent_of_input_order(self):
        """The same canonical pair must be produced regardless of which
        record appears first in the input DataFrame."""
        df = _make_small_dataset()
        pairs_original = run_batch_blocking(df)

        # Reverse the input row order
        df_reversed = df.iloc[::-1].reset_index(drop=True)
        pairs_reversed = run_batch_blocking(df_reversed)

        # Same pairs in both outputs
        set_original = set(
            zip(pairs_original["PATID_A"], pairs_original["PATID_B"])
        )
        set_reversed = set(
            zip(pairs_reversed["PATID_A"], pairs_reversed["PATID_B"])
        )
        assert set_original == set_reversed, (
            "Input row order affected pair generation"
        )

    def test_internal_pairs_from_index_canonical(self):
        """_pairs_from_index must produce canonical (sorted) pairs."""
        pair_blocks = {}
        test_index = {"KEY_1": ["PAT_C", "PAT_A", "PAT_B"]}
        _pairs_from_index(test_index, "TEST", pair_blocks)

        for pair in pair_blocks:
            assert pair[0] < pair[1], (
                f"Internal pair not canonical: {pair}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# No Self-Pairs
# ═══════════════════════════════════════════════════════════════════════════

class TestNoSelfPairs:
    """No record should ever be paired with itself."""

    def test_no_self_pairs_in_batch_output(self):
        """PATID_A must never equal PATID_B."""
        df = _make_small_dataset()
        pairs = run_batch_blocking(df)

        self_pairs = pairs[pairs["PATID_A"] == pairs["PATID_B"]]
        assert len(self_pairs) == 0, (
            f"Found {len(self_pairs)} self-pairs in output"
        )

    def test_singleton_block_produces_no_pairs(self):
        """A block with exactly one record must produce zero pairs."""
        pair_blocks = {}
        test_index = {"SINGLETON_KEY": ["PAT_ONLY"]}
        _pairs_from_index(test_index, "TEST", pair_blocks)
        assert len(pair_blocks) == 0


# ═══════════════════════════════════════════════════════════════════════════
# No Duplicate Pairs
# ═══════════════════════════════════════════════════════════════════════════

class TestNoDuplicatePairs:
    """Each canonical pair must appear exactly once in the output."""

    def test_no_duplicate_pairs_in_batch_output(self):
        """Deduplication must ensure each pair appears once."""
        df = _make_small_dataset()
        pairs = run_batch_blocking(df)

        pair_tuples = list(
            zip(pairs["PATID_A"], pairs["PATID_B"])
        )
        assert len(pair_tuples) == len(set(pair_tuples)), (
            "Duplicate pairs found in batch output"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Source Blocks Metadata
# ═══════════════════════════════════════════════════════════════════════════

class TestSourceBlocksMetadata:
    """source_blocks column must be correctly formatted."""

    def test_source_blocks_is_pipe_delimited(self):
        """Source blocks must use pipe as delimiter."""
        df = _make_small_dataset()
        pairs = run_batch_blocking(df)

        for _, row in pairs.iterrows():
            blocks_str = row["source_blocks"]
            assert isinstance(blocks_str, str)
            # Must contain only valid block IDs and pipes
            parts = blocks_str.split("|")
            valid_blocks = {"B1", "B3", "B4", "B5", "B6", "B7", "B8", "B9"}
            for part in parts:
                assert part in valid_blocks, (
                    f"Invalid block ID '{part}' in source_blocks: "
                    f"'{blocks_str}'"
                )

    def test_source_blocks_sorted(self):
        """Block IDs within source_blocks must be sorted."""
        df = _make_small_dataset()
        pairs = run_batch_blocking(df)

        for _, row in pairs.iterrows():
            parts = row["source_blocks"].split("|")
            assert parts == sorted(parts), (
                f"source_blocks not sorted: '{row['source_blocks']}'"
            )

    def test_source_blocks_no_duplicates(self):
        """No block ID should appear twice in source_blocks."""
        df = _make_small_dataset()
        pairs = run_batch_blocking(df)

        for _, row in pairs.iterrows():
            parts = row["source_blocks"].split("|")
            assert len(parts) == len(set(parts)), (
                f"Duplicate block in source_blocks: '{row['source_blocks']}'"
            )

    def test_n_blocks_matches_source_blocks_count(self):
        """n_blocks must equal the number of pipe-delimited block IDs."""
        df = _make_small_dataset()
        pairs = run_batch_blocking(df)

        for _, row in pairs.iterrows():
            n_parts = len(row["source_blocks"].split("|"))
            assert row["n_blocks"] == n_parts, (
                f"n_blocks={row['n_blocks']} but source_blocks has "
                f"{n_parts} entries: '{row['source_blocks']}'"
            )

    def test_b2_never_appears_in_source_blocks(self):
        """B2 was removed from the scheme. It must never appear in output."""
        df = _make_small_dataset()
        pairs = run_batch_blocking(df)

        for _, row in pairs.iterrows():
            parts = row["source_blocks"].split("|")
            assert "B2" not in parts, (
                f"Removed block B2 found in source_blocks: "
                f"'{row['source_blocks']}'"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Output Schema Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestOutputSchema:
    """The output DataFrame must have the exact schema expected by splink."""

    def test_required_columns_present(self):
        """Output must contain exactly the 4 specified columns."""
        df = _make_small_dataset()
        pairs = run_batch_blocking(df)

        expected_cols = {"PATID_A", "PATID_B", "source_blocks", "n_blocks"}
        assert set(pairs.columns) == expected_cols, (
            f"Expected columns {expected_cols}, got {set(pairs.columns)}"
        )

    def test_column_types(self):
        """Column dtypes must be correct for downstream consumption."""
        df = _make_small_dataset()
        pairs = run_batch_blocking(df)

        if not pairs.empty:
            # Accept both legacy object dtype and modern pandas StringDtype
            string_dtypes = {object, pd.StringDtype()}
            assert pairs["PATID_A"].dtype in string_dtypes or \
               pd.api.types.is_string_dtype(pairs["PATID_A"]), \
               f"PATID_A dtype unexpected: {pairs['PATID_A'].dtype}"
            assert pairs["PATID_B"].dtype in string_dtypes or \
               pd.api.types.is_string_dtype(pairs["PATID_B"]), \
               f"PATID_B dtype unexpected: {pairs['PATID_B'].dtype}"
            assert pairs["source_blocks"].dtype in string_dtypes or \
               pd.api.types.is_string_dtype(pairs["source_blocks"]), \
               f"source_blocks dtype unexpected: {pairs['source_blocks'].dtype}"
            assert pairs["n_blocks"].dtype in (
                np.int64, np.int32, int
            ), f"n_blocks dtype is {pairs['n_blocks'].dtype}"

    def test_empty_dataset_produces_empty_output(self):
        """An empty input DataFrame must produce an empty output
        with the correct schema (no crash)."""
        empty_df = pd.DataFrame(columns=[
            COL_PATID, COL_LAST_NM, COL_FIRST_NM, COL_BIRTH_DT,
            COL_SSN, COL_SSN_L4, COL_ZIP, COL_EMAIL, COL_PHONES,
        ])
        pairs = run_batch_blocking(empty_df)

        assert len(pairs) == 0
        expected_cols = {"PATID_A", "PATID_B", "source_blocks", "n_blocks"}
        assert set(pairs.columns) == expected_cols

    def test_single_record_produces_no_pairs(self):
        """A dataset with only one record must produce zero pairs."""
        df = pd.DataFrame([{
            COL_PATID:    "PAT_ONLY",
            COL_LAST_NM:  "SMITH",
            COL_FIRST_NM: "JOHN",
            COL_BIRTH_DT: pd.Timestamp("1990-01-01"),
            COL_SSN:      "111223333",
            COL_SSN_L4:   "3333",
            COL_ZIP:      "60601",
            COL_EMAIL:    "jsmith@email.com",
            COL_PHONES:   "{'3125551111'}",
        }])
        pairs = run_batch_blocking(df)
        assert len(pairs) == 0