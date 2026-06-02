"""
Unit tests for the deterministic matching-rule engine in
src/models/deterministic_rules.py.

Coverage:
    - Each of the six rules fires on a constructed agreeing pair.
    - The winning rule is the highest-confidence rule that fired; rules_fired
      lists every rule that fired.
    - Null / missing fields never count as agreement; missing attribute columns
      degrade gracefully instead of crashing.
    - Phone agreement uses set intersection of Phones_set (incl. the np.ndarray
      form Parquet produces).
    - The is_suspicious flag follows the guide's definition exactly.
    - Passthrough columns survive; empty / malformed inputs are handled.
    - assign_clusters links transitive matches; get_match_stats reports
      coverage, quality, and cluster stats.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.deterministic_rules import (
    RULES,
    apply_rules,
    assign_clusters,
    get_match_stats,
    get_non_matches,
    _parse_phone_set,
)

COL_PATID = "PATID"

_DEFAULTS = {
    "FirstNM_clean": None,
    "LastNM_clean": None,
    "BirthDT_clean": None,
    "SSN_clean": None,
    "Email_clean": None,
    "AddressLine1_clean": None,
    "SexAtBirthDSC_clean": None,
    "Phones_set": "set()",
}


def _clean_df(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_DEFAULTS, **r} for r in records])


def _pairs(*pairs: tuple[str, str], blocks: str = "B1") -> pd.DataFrame:
    return pd.DataFrame(
        [{"PATID_A": a, "PATID_B": b, "source_blocks": blocks, "n_blocks": 1}
         for a, b in pairs]
    )


def _one(out: pd.DataFrame) -> pd.Series:
    assert len(out) == 1
    return out.iloc[0]


# ── Each rule fires ────────────────────────────────────────────────────────────
class TestEachRuleFires:
    def test_exact_ssn(self):
        df = _clean_df([
            {COL_PATID: "A", "SSN_clean": "123456789", "LastNM_clean": "SMITH"},
            {COL_PATID: "B", "SSN_clean": "123456789", "LastNM_clean": "JONES"},
        ])
        row = _one(apply_rules(_pairs(("A", "B")), df))
        assert row["match_rule"] == "EXACT_SSN"
        assert row["confidence"] == 1.0

    def test_email_exact(self):
        # Different names, shared email → EMAIL_EXACT (and nothing else).
        df = _clean_df([
            {COL_PATID: "A", "Email_clean": "shared@x.com", "LastNM_clean": "A"},
            {COL_PATID: "B", "Email_clean": "shared@x.com", "LastNM_clean": "B"},
        ])
        row = _one(apply_rules(_pairs(("A", "B")), df))
        assert row["match_rule"] == "EMAIL_EXACT"
        assert row["confidence"] == 0.995

    def test_name_dob_email(self):
        # Name+DOB+email agree but SSN absent → EMAIL_EXACT wins, but
        # NAME_DOB_EMAIL must also be among the fired rules.
        df = _clean_df([
            {COL_PATID: "A", "FirstNM_clean": "JANE", "LastNM_clean": "DOE",
             "BirthDT_clean": pd.Timestamp("1990-01-01"),
             "Email_clean": "j@x.com"},
            {COL_PATID: "B", "FirstNM_clean": "JANE", "LastNM_clean": "DOE",
             "BirthDT_clean": pd.Timestamp("1990-01-01"),
             "Email_clean": "j@x.com"},
        ])
        row = _one(apply_rules(_pairs(("A", "B")), df))
        assert "NAME_DOB_EMAIL" in row["rules_fired"].split("|")
        assert "EMAIL_EXACT" in row["rules_fired"].split("|")

    def test_name_dob_phone(self):
        df = _clean_df([
            {COL_PATID: "A", "FirstNM_clean": "JOHN", "LastNM_clean": "LEE",
             "BirthDT_clean": pd.Timestamp("1980-05-05"),
             "Phones_set": "{'3125551234', '7735559999'}"},
            {COL_PATID: "B", "FirstNM_clean": "JOHN", "LastNM_clean": "LEE",
             "BirthDT_clean": pd.Timestamp("1980-05-05"),
             "Phones_set": "{'3125551234'}"},
        ])
        assert _one(apply_rules(_pairs(("A", "B")), df))["match_rule"] \
            == "NAME_DOB_PHONE"

    def test_name_dob_sex(self):
        df = _clean_df([
            {COL_PATID: "A", "FirstNM_clean": "JANE", "LastNM_clean": "DOE",
             "BirthDT_clean": pd.Timestamp("1990-01-01"),
             "SexAtBirthDSC_clean": "FEMALE"},
            {COL_PATID: "B", "FirstNM_clean": "JANE", "LastNM_clean": "DOE",
             "BirthDT_clean": pd.Timestamp("1990-01-01"),
             "SexAtBirthDSC_clean": "FEMALE"},
        ])
        row = _one(apply_rules(_pairs(("A", "B")), df))
        assert row["match_rule"] == "NAME_DOB_SEX"
        assert bool(row["is_suspicious"]) is False

    def test_name_dob_address(self):
        df = _clean_df([
            {COL_PATID: "A", "FirstNM_clean": "JOE", "LastNM_clean": "ROE",
             "BirthDT_clean": pd.Timestamp("1975-03-03"),
             "AddressLine1_clean": "123 MAIN ST"},
            {COL_PATID: "B", "FirstNM_clean": "JOE", "LastNM_clean": "ROE",
             "BirthDT_clean": pd.Timestamp("1975-03-03"),
             "AddressLine1_clean": "123 MAIN ST"},
        ])
        assert _one(apply_rules(_pairs(("A", "B")), df))["match_rule"] \
            == "NAME_DOB_ADDRESS"


# ── Rule precedence ────────────────────────────────────────────────────────────
class TestPrecedence:
    def test_highest_confidence_wins(self):
        df = _clean_df([
            {COL_PATID: "A", "SSN_clean": "123456789", "FirstNM_clean": "JANE",
             "LastNM_clean": "DOE", "BirthDT_clean": pd.Timestamp("1990-01-01"),
             "SexAtBirthDSC_clean": "FEMALE"},
            {COL_PATID: "B", "SSN_clean": "123456789", "FirstNM_clean": "JANE",
             "LastNM_clean": "DOE", "BirthDT_clean": pd.Timestamp("1990-01-01"),
             "SexAtBirthDSC_clean": "FEMALE"},
        ])
        row = _one(apply_rules(_pairs(("A", "B")), df))
        assert row["match_rule"] == "EXACT_SSN"
        assert set(row["rules_fired"].split("|")) == {"EXACT_SSN", "NAME_DOB_SEX"}

    def test_rules_are_confidence_sorted(self):
        confidences = [r.confidence for r in RULES]
        assert confidences == sorted(confidences, reverse=True)


# ── Agreement edge cases ───────────────────────────────────────────────────────
class TestAgreementEdges:
    def test_both_null_ssn_does_not_match(self):
        df = _clean_df([{COL_PATID: "A"}, {COL_PATID: "B"}])
        assert apply_rules(_pairs(("A", "B")), df).empty

    def test_one_null_field_does_not_agree(self):
        df = _clean_df([
            {COL_PATID: "A", "SSN_clean": "123456789"},
            {COL_PATID: "B", "SSN_clean": None},
        ])
        assert apply_rules(_pairs(("A", "B")), df).empty

    def test_partial_name_dob_does_not_fire(self):
        # Name+DOB agree but no 4th field → no NAME_DOB_* rule should fire.
        df = _clean_df([
            {COL_PATID: "A", "FirstNM_clean": "JANE", "LastNM_clean": "DOE",
             "BirthDT_clean": pd.Timestamp("1990-01-01")},
            {COL_PATID: "B", "FirstNM_clean": "JANE", "LastNM_clean": "DOE",
             "BirthDT_clean": pd.Timestamp("1990-01-01")},
        ])
        assert apply_rules(_pairs(("A", "B")), df).empty

    def test_missing_attribute_column_degrades_gracefully(self):
        # df_clean lacks SexAtBirthDSC_clean entirely — NAME_DOB_SEX cannot fire,
        # but the call must not crash and other rules still work.
        df = pd.DataFrame([
            {COL_PATID: "A", "SSN_clean": "123456789"},
            {COL_PATID: "B", "SSN_clean": "123456789"},
        ])
        row = _one(apply_rules(_pairs(("A", "B")), df))
        assert row["match_rule"] == "EXACT_SSN"


# ── Phone parsing ──────────────────────────────────────────────────────────────
class TestPhoneParsing:
    def test_parse_serialized_set(self):
        assert _parse_phone_set("{'123', '456'}") == frozenset({"123", "456"})

    def test_parse_numpy_array(self):
        # Parquet returns Phones_set as an np.ndarray — must not be dropped.
        arr = np.array(["111", "222"], dtype=object)
        assert _parse_phone_set(arr) == frozenset({"111", "222"})

    def test_parse_empty_forms(self):
        for empty in ("set()", "{}", "[]", "", "nan", None, float("nan")):
            assert _parse_phone_set(empty) == frozenset()

    def test_no_shared_phone_does_not_match(self):
        df = _clean_df([
            {COL_PATID: "A", "FirstNM_clean": "JOHN", "LastNM_clean": "LEE",
             "BirthDT_clean": pd.Timestamp("1980-05-05"),
             "Phones_set": "{'111'}"},
            {COL_PATID: "B", "FirstNM_clean": "JOHN", "LastNM_clean": "LEE",
             "BirthDT_clean": pd.Timestamp("1980-05-05"),
             "Phones_set": "{'222'}"},
        ])
        assert apply_rules(_pairs(("A", "B")), df).empty


# ── Suspicious flag ────────────────────────────────────────────────────────────
class TestSuspiciousFlag:
    def _ssn_pair(self, **b_over) -> pd.Series:
        a = {COL_PATID: "A", "SSN_clean": "123456789", "FirstNM_clean": "JANE",
             "LastNM_clean": "DOE", "BirthDT_clean": pd.Timestamp("1990-01-01")}
        b = {**a, COL_PATID: "B", **b_over}
        return _one(apply_rules(_pairs(("A", "B")), _clean_df([a, b])))

    def test_not_suspicious_when_all_agree(self):
        assert bool(self._ssn_pair()["is_suspicious"]) is False

    def test_suspicious_on_dob_disagreement(self):
        assert bool(self._ssn_pair(
            BirthDT_clean=pd.Timestamp("1991-02-02"))["is_suspicious"]) is True

    def test_suspicious_on_lastname_disagreement(self):
        assert bool(self._ssn_pair(LastNM_clean="SMITH")["is_suspicious"]) is True

    def test_suspicious_when_both_ssn_present_differ(self):
        # Match via NAME_DOB_SEX, but both SSNs present and differing.
        a = {COL_PATID: "A", "FirstNM_clean": "JANE", "LastNM_clean": "DOE",
             "BirthDT_clean": pd.Timestamp("1990-01-01"),
             "SexAtBirthDSC_clean": "FEMALE", "SSN_clean": "111111111"}
        b = {**a, COL_PATID: "B", "SSN_clean": "222222222"}
        row = _one(apply_rules(_pairs(("A", "B")), _clean_df([a, b])))
        assert row["match_rule"] == "NAME_DOB_SEX"
        assert bool(row["is_suspicious"]) is True


# ── Output shape / passthrough ─────────────────────────────────────────────────
class TestOutputShape:
    def test_passthrough_columns_preserved(self):
        df = _clean_df([
            {COL_PATID: "A", "SSN_clean": "123456789"},
            {COL_PATID: "B", "SSN_clean": "123456789"},
        ])
        out = apply_rules(_pairs(("A", "B"), blocks="B1|B9"), df)
        row = _one(out)
        assert row["source_blocks"] == "B1|B9"
        assert row["n_blocks"] == 1

    def test_unmatched_pairs_dropped(self):
        df = _clean_df([
            {COL_PATID: "A", "SSN_clean": "123456789"},
            {COL_PATID: "B", "SSN_clean": "123456789"},
            {COL_PATID: "C", "SSN_clean": "999999999"},
            {COL_PATID: "D"},
        ])
        out = apply_rules(_pairs(("A", "B"), ("C", "D")), df)
        assert list(zip(out["PATID_A"], out["PATID_B"])) == [("A", "B")]

    def test_empty_candidate_pairs_returns_schema(self):
        df = _clean_df([{COL_PATID: "A"}])
        out = apply_rules(pd.DataFrame(columns=["PATID_A", "PATID_B"]), df)
        assert out.empty
        assert {"PATID_A", "PATID_B", "match_rule", "confidence",
                "rules_fired", "is_suspicious"} <= set(out.columns)

    def test_missing_required_columns_raises(self):
        df = _clean_df([{COL_PATID: "A"}])
        with pytest.raises(ValueError, match="PATID_B"):
            apply_rules(pd.DataFrame({"PATID_A": ["A"]}), df)

    def test_df_clean_without_patid_raises(self):
        df = pd.DataFrame([{"SSN_clean": "123456789"}])
        with pytest.raises(ValueError, match="PATID"):
            apply_rules(_pairs(("A", "B")), df)


# ── Non-matches (downstream input) ───────────────────────────────────────────────
class TestNonMatches:
    def test_complement_of_matches(self):
        df = _clean_df([
            {COL_PATID: "A", "SSN_clean": "123456789"},
            {COL_PATID: "B", "SSN_clean": "123456789"},
            {COL_PATID: "C", "SSN_clean": "999999999"},
            {COL_PATID: "D"},
        ])
        pairs = _pairs(("A", "B"), ("C", "D"))
        matches = apply_rules(pairs, df)
        non = get_non_matches(pairs, matches)
        assert list(zip(non["PATID_A"], non["PATID_B"])) == [("C", "D")]

    def test_partition_is_exhaustive_and_disjoint(self):
        df = _clean_df([
            {COL_PATID: "A", "SSN_clean": "111111111"},
            {COL_PATID: "B", "SSN_clean": "111111111"},
            {COL_PATID: "C", "SSN_clean": "222222222"},
            {COL_PATID: "D"},
        ])
        pairs = _pairs(("A", "B"), ("C", "D"))
        matches = apply_rules(pairs, df)
        non = get_non_matches(pairs, matches)
        assert len(matches) + len(non) == len(pairs)
        matched = set(zip(matches["PATID_A"], matches["PATID_B"]))
        unmatched = set(zip(non["PATID_A"], non["PATID_B"]))
        assert matched.isdisjoint(unmatched)
        assert matched | unmatched == set(zip(pairs["PATID_A"], pairs["PATID_B"]))

    def test_blocking_schema_preserved(self):
        df = _clean_df([{COL_PATID: "C"}, {COL_PATID: "D"}])
        pairs = _pairs(("C", "D"), blocks="B3|B7")
        non = get_non_matches(pairs, apply_rules(pairs, df))
        row = non.iloc[0]
        assert list(non.columns) == ["PATID_A", "PATID_B", "source_blocks", "n_blocks"]
        assert row["source_blocks"] == "B3|B7"

    def test_no_matches_returns_all_pairs(self):
        df = _clean_df([{COL_PATID: "C"}, {COL_PATID: "D"}])
        pairs = _pairs(("C", "D"))
        non = get_non_matches(pairs, apply_rules(pairs, df))
        assert len(non) == 1

    def test_empty_candidate_pairs(self):
        empty = pd.DataFrame(columns=["PATID_A", "PATID_B"])
        assert get_non_matches(empty, empty).empty

    def test_missing_required_columns_raises(self):
        with pytest.raises(ValueError, match="PATID_B"):
            get_non_matches(pd.DataFrame({"PATID_A": ["A"]}), pd.DataFrame())


# ── Clustering ─────────────────────────────────────────────────────────────────
class TestClustering:
    def test_transitive_link_one_cluster(self):
        clusters = assign_clusters(
            pd.DataFrame({"PATID_A": ["A", "B"], "PATID_B": ["B", "C"]})
        )
        assert clusters["A"] == clusters["B"] == clusters["C"]

    def test_separate_components_distinct_ids(self):
        clusters = assign_clusters(
            pd.DataFrame({"PATID_A": ["A", "C"], "PATID_B": ["B", "D"]})
        )
        assert clusters["A"] == clusters["B"]
        assert clusters["C"] == clusters["D"]
        assert clusters["A"] != clusters["C"]

    def test_cluster_ids_are_contiguous(self):
        clusters = assign_clusters(
            pd.DataFrame({"PATID_A": ["A", "C", "E"],
                          "PATID_B": ["B", "D", "F"]})
        )
        assert sorted(set(clusters.values())) == [0, 1, 2]


# ── Stats ──────────────────────────────────────────────────────────────────────
class TestStats:
    def test_empty_matches_returns_empty_dict(self):
        assert get_match_stats(pd.DataFrame(
            columns=["match_rule", "confidence", "is_suspicious",
                     "PATID_A", "PATID_B"])) == {}

    def test_stats_fields(self):
        df = _clean_df([
            {COL_PATID: "A", "SSN_clean": "123456789", "LastNM_clean": "SMITH"},
            {COL_PATID: "B", "SSN_clean": "123456789", "LastNM_clean": "SMITH"},
        ])
        stats = get_match_stats(apply_rules(_pairs(("A", "B")), df), n_records=10)
        assert stats["total_matches"] == 1
        assert stats["patients_matched"] == 2
        assert stats["coverage_rate"] == 20.0
        assert stats["max_cluster_size"] == 2
        assert stats["avg_confidence"] == 1.0

    def test_coverage_none_without_n_records(self):
        df = _clean_df([
            {COL_PATID: "A", "SSN_clean": "123456789"},
            {COL_PATID: "B", "SSN_clean": "123456789"},
        ])
        stats = get_match_stats(apply_rules(_pairs(("A", "B")), df))
        assert stats["coverage_rate"] is None
