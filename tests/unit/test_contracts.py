"""Unit tests for the stage-boundary data contracts in src/contracts.py.

Coverage:
    - Each schema accepts a well-formed frame and rejects the violations the
      contract is meant to catch (bad SSN/zip/sex, out-of-range confidence,
      non-canonical pair ordering, n_blocks < 1).
    - Phones_set accepts list / ndarray / set / null forms.
    - validate() skips empty frames; assert_patid_coverage() catches lineage
      mismatches.

Requires: pandera, pyarrow (for the engines), numpy, pandas.
"""

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from src.contracts import (
    CandidatePairs,
    CleanedRecords,
    Matches,
    NonMatches,
    assert_patid_coverage,
    validate,
)

_SCHEMA_ERR = (SchemaError, SchemaErrors)


def _cleaned(**override) -> pd.DataFrame:
    base = {
        "PATID": ["A", "B"],
        "FirstNM_clean": ["JOHN", "JANE"],
        "LastNM_clean": ["SMITH", "DOE"],
        "BirthDT_clean": pd.to_datetime(["1990-01-01", "1985-05-05"]),
        "SSN_clean": ["123456789", np.nan],
        "last_4_SSN": ["6789", "4321"],
        "Email_clean": ["a@example.com", np.nan],
        "ZipCD_clean_base": ["60614", "02134"],
        "AddressLine1_clean": ["1 MAIN ST", np.nan],
        "SexAtBirthDSC_clean": ["MALE", "FEMALE"],
        "Phones_set": [["7732001234"], []],
        "valid_record": [True, True],
    }
    base.update(override)
    return pd.DataFrame(base)


def _pairs(**override) -> pd.DataFrame:
    base = {
        "PATID_A": ["A"],
        "PATID_B": ["B"],
        "source_blocks": ["B1"],
        "n_blocks": [1],
    }
    base.update(override)
    return pd.DataFrame(base)


def _matches(**override) -> pd.DataFrame:
    base = {
        "PATID_A": ["A"],
        "PATID_B": ["B"],
        "match_rule": ["EXACT_SSN"],
        "confidence": [1.0],
        "rules_fired": ["EXACT_SSN"],
        "is_suspicious": [False],
        "high_fanout_ssn": [False],
        "cluster_id": [0],
        "source_blocks": ["B1"],
        "n_blocks": [1],
    }
    base.update(override)
    return pd.DataFrame(base)


class TestCleanedRecords:
    def test_valid_frame_passes(self):
        validate(_cleaned(), CleanedRecords, allow_empty=False)

    @pytest.mark.parametrize(
        "phones",
        [[["7732001234"], ["3122001234"]],
         [np.array(["7732001234"]), np.array([])],
         [{"7732001234"}, set()],
         [np.nan, np.nan]],
    )
    def test_phones_set_accepts_collection_and_null_forms(self, phones):
        validate(_cleaned(Phones_set=phones), CleanedRecords, allow_empty=False)

    def test_rejects_malformed_ssn(self):
        with pytest.raises(_SCHEMA_ERR):
            validate(_cleaned(SSN_clean=["12345", "123456789"]), CleanedRecords)

    def test_rejects_unknown_sex_value(self):
        with pytest.raises(_SCHEMA_ERR):
            validate(_cleaned(SexAtBirthDSC_clean=["MALE", "X"]), CleanedRecords)

    def test_rejects_malformed_zip(self):
        with pytest.raises(_SCHEMA_ERR):
            validate(_cleaned(ZipCD_clean_base=["6061", "02134"]), CleanedRecords)


class TestCandidatePairs:
    def test_valid_frame_passes(self):
        validate(_pairs(), CandidatePairs, allow_empty=False)
        validate(_pairs(), NonMatches, allow_empty=False)  # inherits the schema

    def test_rejects_non_canonical_order(self):
        with pytest.raises(_SCHEMA_ERR):
            validate(_pairs(PATID_A=["B"], PATID_B=["A"]), CandidatePairs)

    def test_rejects_zero_blocks(self):
        with pytest.raises(_SCHEMA_ERR):
            validate(_pairs(n_blocks=[0]), CandidatePairs)

    def test_rejects_unexpected_column(self):
        with pytest.raises(_SCHEMA_ERR):
            validate(_pairs(surprise=["x"]), CandidatePairs)


class TestMatches:
    def test_valid_frame_passes(self):
        validate(_matches(), Matches, allow_empty=False)

    def test_rejects_out_of_range_confidence(self):
        with pytest.raises(_SCHEMA_ERR):
            validate(_matches(confidence=[0.5]), Matches)

    def test_rejects_unknown_rule(self):
        with pytest.raises(_SCHEMA_ERR):
            validate(_matches(match_rule=["NOT_A_RULE"]), Matches)

    def test_empty_frame_is_skipped(self):
        out = validate(pd.DataFrame(), Matches)  # allow_empty default True
        assert out.empty


class TestPatidCoverage:
    def test_passes_when_covered(self):
        assert_patid_coverage(_pairs(), _cleaned())  # A,B ⊆ {A,B}

    def test_raises_on_lineage_mismatch(self):
        clean_missing_b = _cleaned(
            PATID=["A", "C"], LastNM_clean=["SMITH", "ROE"],
            FirstNM_clean=["JOHN", "CARL"],
        )
        with pytest.raises(ValueError, match="Lineage mismatch"):
            assert_patid_coverage(_pairs(), clean_missing_b)
