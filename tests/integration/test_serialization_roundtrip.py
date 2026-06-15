"""Integration test for the cleaned-dataset Parquet handoff (stage 1 → 2/3).

This is the contract test for the serialization gap the project hit before: the
multi-valued `Phones_set` column is a Python list in memory, has no native Arrow
type, and must survive `write_cleaned → read_parquet` in a form the downstream
phone parsers accept. It asserts:

    1. write_cleaned + read_parquet preserves every column and the phone values;
    2. the round-tripped frame still satisfies the CleanedRecords contract;
    3. deterministic_rules._parse_phone_set recovers the original phone set from
       whatever form Parquet produced (list/ndarray).

Requires: pyarrow, pandera, pandas, numpy.
"""

import numpy as np
import pandas as pd

from src.contracts import CleanedRecords, validate
from src.data.clean import write_cleaned
from src.models.deterministic_rules import _parse_phone_set


def _cleaned() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PATID": ["A", "B", "C"],
            "FirstNM_clean": ["JOHN", "JANE", np.nan],
            "LastNM_clean": ["SMITH", "DOE", "ROE"],
            "BirthDT_clean": pd.to_datetime(["1990-01-01", "1985-05-05", "1972-12-12"]),
            "SSN_clean": ["123456789", np.nan, "987654301"],
            "last_4_SSN": ["6789", "4321", "4301"],
            "Email_clean": ["a@example.com", np.nan, "c@example.com"],
            "ZipCD_clean_base": ["60614", "02134", "94105"],
            "AddressLine1_clean": ["1 MAIN ST", np.nan, "9 OAK AVE"],
            "SexAtBirthDSC_clean": ["MALE", "FEMALE", "OTHER"],
            "Phones_set": [["7732001234", "3122005678"], [], ["4155009999"]],
            "valid_record": [True, True, True],
        }
    )


def test_cleaned_parquet_roundtrip_preserves_phones(tmp_path):
    original = _cleaned()
    out_path = tmp_path / "cleaned_v1.parquet"

    returned = write_cleaned(original, out_path)
    assert returned == out_path and out_path.exists()
    # write_cleaned must not mutate the caller's frame.
    assert isinstance(original.loc[0, "Phones_set"], list)

    loaded = pd.read_parquet(out_path)

    assert list(loaded.columns) == list(original.columns)
    assert loaded["PATID"].tolist() == ["A", "B", "C"]
    # Leading zeros / id-like strings survive.
    assert loaded.loc[0, "ZipCD_clean_base"] == "60614"
    assert loaded.loc[0, "last_4_SSN"] == "6789"

    # The round-tripped frame still honors the contract.
    validate(loaded, CleanedRecords, allow_empty=False)

    # The phone set is recoverable in whatever form Parquet produced.
    assert _parse_phone_set(loaded.loc[0, "Phones_set"]) == {"7732001234", "3122005678"}
    assert _parse_phone_set(loaded.loc[1, "Phones_set"]) == set()
    assert _parse_phone_set(loaded.loc[2, "Phones_set"]) == {"4155009999"}
