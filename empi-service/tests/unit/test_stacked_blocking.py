"""Unit tests for the stacked blocker orchestration
(src/preprocessing/stacked_blocking.py)."""

import pandas as pd

from src.contracts import CandidatePairs, validate
from src.preprocessing.stacked_blocking import (
    run_stacked_blocking,
    union_candidate_pairs,
)

_DOB = pd.Timestamp("1980-05-05")


def _pairs(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"PATID_A": a, "PATID_B": b, "source_blocks": s, "n_blocks": len(s.split("|"))}
         for a, b, s in rows]
    )


def _clean(records: list[dict]) -> pd.DataFrame:
    base = {
        "FirstNM_clean": None, "LastNM_clean": None, "BirthDT_clean": None,
        "SSN_clean": None, "Email_clean": None, "AddressLine1_clean": None,
        "SexAtBirthDSC_clean": None, "Phones_set": None,
        "ZipCD_clean_base": None, "last_4_SSN": None, "valid_record": True,
    }
    return pd.DataFrame([{**base, **r} for r in records])


class TestUnion:
    def test_merges_source_blocks_and_counts(self):
        a = _pairs([("A", "B", "B3")])
        b = _pairs([("A", "B", "QGRAM")])
        out = union_candidate_pairs(a, b)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["source_blocks"] == "B3|QGRAM"
        assert row["n_blocks"] == 2

    def test_distinct_pairs_preserved(self):
        a = _pairs([("A", "B", "B3")])
        b = _pairs([("C", "D", "QGRAM")])
        out = union_candidate_pairs(a, b)
        assert set(zip(out["PATID_A"], out["PATID_B"])) == {("A", "B"), ("C", "D")}

    def test_duplicate_block_token_not_double_counted(self):
        a = _pairs([("A", "B", "B3")])
        b = _pairs([("A", "B", "B3")])
        out = union_candidate_pairs(a, b)
        assert out.iloc[0]["source_blocks"] == "B3"
        assert out.iloc[0]["n_blocks"] == 1

    def test_empty_inputs(self):
        empty = pd.DataFrame(columns=["PATID_A", "PATID_B", "source_blocks", "n_blocks"])
        assert union_candidate_pairs(empty, empty).empty

    def test_output_satisfies_candidate_pairs_contract(self):
        out = union_candidate_pairs(_pairs([("A", "B", "B3")]), _pairs([("A", "B", "QGRAM")]))
        validate(out, CandidatePairs, allow_empty=False)


class TestRunStackedBlocking:
    def test_typo_pair_surfaces_and_contract_holds(self):
        # SMITH/SMYTH share an exact DOB; the 8-block scheme splits them but the
        # q-gram leg recovers them, and the pair survives the stacked pipeline.
        df = _clean([
            {"PATID": "A", "FirstNM_clean": "JOHN", "LastNM_clean": "SMITH", "BirthDT_clean": _DOB},
            {"PATID": "B", "FirstNM_clean": "JOHN", "LastNM_clean": "SMYTH", "BirthDT_clean": _DOB},
        ])
        out = run_stacked_blocking(df)
        validate(out, CandidatePairs)
        pairset = set(zip(out["PATID_A"], out["PATID_B"]))
        assert ("A", "B") in pairset
        # The pair carries the q-gram provenance.
        row = out[(out["PATID_A"] == "A") & (out["PATID_B"] == "B")].iloc[0]
        assert "QGRAM" in row["source_blocks"]
