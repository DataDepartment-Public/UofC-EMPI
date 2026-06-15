"""Unit tests for the per-field cleaners and orchestrator in
src/data/transformations.py — previously the largest untested module.

Coverage is representative, not exhaustive: the highest-risk standardization
logic (leading-zero preservation, structural SSN/phone/zip rules, junk-value
nullification, invalid-record flagging) plus the cross-field derivations and
`valid_record` propagation that the downstream contract depends on.

Requires: pandas, numpy, unidecode (per requirements.txt).
"""

import pandas as pd
import pytest

from src.data.transformations import (
    clean_birth_date,
    clean_email,
    clean_first_name,
    clean_phone,
    clean_ssn,
    clean_zip,
    derive_phones_set,
    transform_dataframe,
)


class TestCleanSSN:
    def test_strips_punctuation_and_keeps_last4(self):
        # 536-90-1234 is structurally valid and not sequential/junk.
        assert clean_ssn("536-90-1234") == ("536901234", "1234")

    def test_left_pads_eight_digits(self):
        clean, last4 = clean_ssn("53690123")
        assert clean == "053690123"  # leading zero preserved
        assert last4 == "0123"

    def test_nullifies_all_same_digits(self):
        # last_4 is only extracted from a fully-validated SSN, so an invalid
        # value yields (nan, nan).
        clean, last4 = clean_ssn("000000000")
        assert pd.isna(clean) and pd.isna(last4)

    def test_nullifies_invalid_area_900(self):
        clean, _ = clean_ssn("900112222")
        assert pd.isna(clean)

    def test_nullifies_known_junk_value(self):
        clean, last4 = clean_ssn("111223333")
        assert pd.isna(clean) and pd.isna(last4)

    def test_nullifies_dominant_digit_placeholder(self):
        # 333333330 / 003333333 pass structural + stdnum checks but are clerical
        # placeholders (one digit fills >=7 of 9 positions). Must be rejected,
        # and an invalid SSN yields no last_4.
        for junk in ("333333330", "003333333"):
            clean, l4 = clean_ssn(junk)
            assert pd.isna(clean) and pd.isna(l4)

    def test_keeps_valid_low_repeat_ssn(self):
        # A genuine SSN with some repeated digits must survive.
        clean, _ = clean_ssn("536901234")
        assert clean == "536901234"

    def test_nullifies_sequential_runs(self):
        # Full ascending/descending digit runs are placeholders, not just the
        # two values that used to be hardcoded.
        for junk in ("012345678", "123456789", "234567890", "987654321"):
            assert pd.isna(clean_ssn(junk)[0])


class TestCleanZip:
    def test_base_only(self):
        assert clean_zip("60614") == ("60614", None) or clean_zip("60614")[0] == "60614"

    def test_splits_base_and_ext(self):
        assert clean_zip("60614-1234") == ("60614", "1234")

    def test_left_pads_four_digits(self):
        base, ext = clean_zip("2134")
        assert base == "02134"  # leading zero preserved
        assert pd.isna(ext)

    def test_nullifies_placeholder(self):
        base, ext = clean_zip("00000")
        assert pd.isna(base) and pd.isna(ext)


class TestCleanPhone:
    def test_ten_digit_passthrough(self):
        assert clean_phone("(773) 200-1234") == "7732001234"

    def test_strips_leading_country_code(self):
        assert clean_phone("1-773-200-1234") == "7732001234"

    @pytest.mark.parametrize("bad", ["773-200-12", "0732001234", "2112001234", "1111111111"])
    def test_nullifies_invalid(self, bad):
        assert pd.isna(clean_phone(bad))


class TestCleanEmail:
    def test_lowercases_valid(self):
        assert clean_email("John.Doe@Gmail.COM") == "john.doe@gmail.com"

    @pytest.mark.parametrize("bad", ["noemail@x.com", "ab@x.com", "test@test.com", "notanemail"])
    def test_nullifies_junk(self, bad):
        assert pd.isna(clean_email(bad))


class TestCleanFirstName:
    def test_unicode_and_uppercase(self):
        assert clean_first_name("josé") == ("JOSE", False)

    def test_strips_title(self):
        assert clean_first_name("Mr Smith") == ("SMITH", False)

    def test_text_null_to_nan(self):
        value, invalid = clean_first_name("UNKNOWN")
        assert pd.isna(value) and invalid is False

    def test_flags_invalid_placeholder(self):
        value, invalid = clean_first_name("BABYBOY")
        assert value == "BABYBOY" and invalid is True


class TestCleanBirthDate:
    def test_parses_valid(self):
        assert clean_birth_date("1990-05-20") == pd.Timestamp("1990-05-20")

    @pytest.mark.parametrize("bad", ["2099-01-01", "1899-12-31", "garbage"])
    def test_nullifies_out_of_range(self, bad):
        assert pd.isna(clean_birth_date(bad))


class TestDerivePhonesSet:
    def test_dedup_sorted_non_null(self):
        assert derive_phones_set("7732001234", None, "3122001234", "7732001234") == [
            "3122001234",
            "7732001234",
        ]


class TestTransformDataframe:
    def _raw(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "PATID": ["1", "2", "3"],
                "FirstNM": ["John", "BABYBOY", "UNKNOWN"],
                "LastNM": ["Smith", "Jones", "UNKNOWN"],
                "BirthDT": ["1990-05-20", "1985-01-01", "1992-03-03"],
                "SSN": ["536-90-1234", "536901234", "000000000"],
                "PrimaryPhoneNBR": ["7732001234", None, None],
                "SexAtBirthDSC": ["MALE", "FEMALE", "OTHER"],
            }
        )

    def test_keeps_raw_and_clean_columns(self):
        out = transform_dataframe(self._raw())
        assert out.loc[0, "FirstNM_raw"] == "John"
        assert out.loc[0, "FirstNM_clean"] == "JOHN"

    def test_ssn_clean_and_last4(self):
        out = transform_dataframe(self._raw())
        assert out.loc[0, "SSN_clean"] == "536901234"
        assert out.loc[0, "last_4_SSN"] == "1234"

    def test_phones_set_is_list(self):
        out = transform_dataframe(self._raw())
        phones = out.loc[0, "Phones_set"]
        assert isinstance(phones, list) and "7732001234" in phones

    def test_valid_record_propagation(self):
        out = transform_dataframe(self._raw())
        # row 1: BABYBOY first name → invalid; row 2: both names null → invalid.
        assert out["valid_record"].tolist() == [True, False, False]
