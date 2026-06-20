"""Unit tests for src/evaluation/blocking_eval.py.

Covers the pure pair-set / wide-candidate logic plus one end-to-end recall
computation (rules confirm an SSN+DOB pair that production blocking also emits,
so recall is 100%).
"""

import pandas as pd

from src.evaluation.blocking_eval import (
    BlockingEvalReport,
    _canonical,
    _pairset,
    evaluate_blocking,
    wide_candidate_pairs,
)

COL = {
    "PATID": "PATID",
    "last": "LastNM_clean",
    "first": "FirstNM_clean",
    "dob": "BirthDT_clean",
    "ssn": "SSN_clean",
    "ssn4": "last_4_SSN",
    "zip": "ZipCD_clean_base",
    "email": "Email_clean",
    "phones": "Phones_set",
    "sex": "SexAtBirthDSC_clean",
}


def _rec(patid, last, first, dob, ssn, zipc, email, phone):
    return {
        COL["PATID"]: patid,
        COL["last"]: last,
        COL["first"]: first,
        COL["dob"]: pd.Timestamp(dob),
        COL["ssn"]: ssn,
        COL["ssn4"]: ssn[-4:] if ssn else None,
        COL["zip"]: zipc,
        COL["email"]: email,
        COL["phones"]: f"{{'{phone}'}}",
        COL["sex"]: "FEMALE",
    }


# ── Pure helpers ────────────────────────────────────────────────────────────────
class TestPairHelpers:
    def test_canonical_is_sorted(self):
        assert _canonical("B", "A") == ("A", "B")
        assert _canonical("A", "B") == ("A", "B")

    def test_pairset_canonicalizes(self):
        df = pd.DataFrame({"PATID_A": ["B", "C"], "PATID_B": ["A", "D"]})
        assert _pairset(df) == {("A", "B"), ("C", "D")}

    def test_pairset_empty(self):
        assert _pairset(pd.DataFrame(columns=["PATID_A", "PATID_B"])) == set()


# ── Wide candidate generation ───────────────────────────────────────────────────
class TestWideCandidates:
    def _df(self):
        return pd.DataFrame([
            _rec("P01", "SMITH", "JANE", "1990-01-01", "111111111", "60601",
                 "a@x.com", "3120000001"),
            _rec("P02", "SMITH", "JOHN", "1990-01-01", "222222222", "60602",
                 "b@x.com", "3120000002"),
            _rec("P03", "JONES", "MARY", "1985-05-05", "333333331", "60603",
                 "c@x.com", "3120000003"),
        ])

    def test_sample_generates_all_pairs(self):
        wide, records = wide_candidate_pairs(self._df(), method="sample",
                                             sample_size=10, seed=1)
        assert len(records) == 3
        assert len(wide) == 3  # C(3, 2)
        assert _pairset(wide) == {("P01", "P02"), ("P01", "P03"), ("P02", "P03")}

    def test_loose_groups_by_dob_and_soundex(self):
        # P01 & P02 share DOB (and Soundex(SMITH)); P03 shares neither.
        wide, records = wide_candidate_pairs(self._df(), method="loose")
        assert ("P01", "P02") in _pairset(wide)
        assert ("P01", "P03") not in _pairset(wide)

    def test_invalid_method_raises(self):
        try:
            wide_candidate_pairs(self._df(), method="bogus")
        except ValueError as e:
            assert "method" in str(e)
        else:
            raise AssertionError("expected ValueError")


# ── End-to-end recall ───────────────────────────────────────────────────────────
class TestEvaluateBlocking:
    def test_confirmed_and_blocked_pair_gives_full_recall(self):
        # P01 & P02 share SSN + DOB -> SSN_DOB confirms; same SSN -> blocking B1
        # emits the pair -> caught -> recall 100%.
        df = pd.DataFrame([
            _rec("P01", "SMITH", "JANE", "1990-01-01", "123456789", "60601",
                 "a@x.com", "3120000001"),
            _rec("P02", "JONES", "JOHN", "1990-01-01", "123456789", "60602",
                 "b@x.com", "3120000002"),
            _rec("P03", "ADAMS", "MARY", "1975-03-03", "999999991", "60603",
                 "c@x.com", "3120000003"),
            _rec("P04", "BAKER", "PAUL", "1960-07-07", "999999992", "60604",
                 "d@x.com", "3120000004"),
        ])
        report = evaluate_blocking(df, method="sample", sample_size=10, seed=1)
        assert isinstance(report, BlockingEvalReport)
        assert report.headline["rule_confirmed (R)"] == 1
        assert report.recall == 100.0
        assert report.missed_examples.empty
        assert "B1" in report.hits_by_block
        # to_text renders without error
        assert "BLOCKING RECALL" in report.to_text()
