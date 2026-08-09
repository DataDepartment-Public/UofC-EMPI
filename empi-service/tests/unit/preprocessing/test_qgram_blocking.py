"""Unit tests for the q-gram blocking pass (src/preprocessing/qgram_blocking.py)."""

import pandas as pd

from src.preprocessing.qgram_blocking import QGRAM_SOURCE_LABEL, run_qgram_blocking

_DOB = pd.Timestamp("1980-05-05")


def _df(records: list[dict]) -> pd.DataFrame:
    base = {
        "FirstNM_clean": None,
        "LastNM_clean": None,
        "BirthDT_clean": None,
        "valid_record": True,
    }
    return pd.DataFrame([{**base, **r} for r in records])


def test_recovers_lastname_typo_within_dob_block():
    # SMITH vs SMYTH share an exact DOB but a typo that breaks Soundex AND Double
    # Metaphone — the residual the 8-block scheme misses. q-gram catches it.
    df = _df([
        {"PATID": "A", "FirstNM_clean": "JOHN", "LastNM_clean": "SMITH", "BirthDT_clean": _DOB},
        {"PATID": "B", "FirstNM_clean": "JOHN", "LastNM_clean": "SMYTH", "BirthDT_clean": _DOB},
    ])
    out = run_qgram_blocking(df, threshold=0.30, min_df=1)
    assert list(zip(out["PATID_A"], out["PATID_B"])) == [("A", "B")]
    row = out.iloc[0]
    assert row["source_blocks"] == QGRAM_SOURCE_LABEL
    assert row["n_blocks"] == 1


def test_canonical_ordering_enforced():
    # Input order B,A → output must be canonical A < B.
    df = _df([
        {"PATID": "B", "FirstNM_clean": "JOHN", "LastNM_clean": "SMITH", "BirthDT_clean": _DOB},
        {"PATID": "A", "FirstNM_clean": "JOHN", "LastNM_clean": "SMYTH", "BirthDT_clean": _DOB},
    ])
    out = run_qgram_blocking(df, threshold=0.30, min_df=1)
    assert bool((out["PATID_A"] < out["PATID_B"]).all())


def test_different_dob_blocks_never_paired():
    # Same names but different birth dates → different coarse blocks → no pair,
    # however similar the names are.
    df = _df([
        {"PATID": "A", "FirstNM_clean": "JOHN", "LastNM_clean": "SMITH",
         "BirthDT_clean": pd.Timestamp("1980-05-05")},
        {"PATID": "B", "FirstNM_clean": "JOHN", "LastNM_clean": "SMITH",
         "BirthDT_clean": pd.Timestamp("1991-02-02")},
    ])
    assert run_qgram_blocking(df, threshold=0.30, min_df=1).empty


def test_dissimilar_names_below_threshold_not_paired():
    df = _df([
        {"PATID": "A", "FirstNM_clean": "JOHN", "LastNM_clean": "SMITH", "BirthDT_clean": _DOB},
        {"PATID": "B", "FirstNM_clean": "MARIA", "LastNM_clean": "GONZALEZ", "BirthDT_clean": _DOB},
    ])
    assert run_qgram_blocking(df, threshold=0.30, min_df=1).empty


def test_birthyear_block_kind_is_broader():
    # Different exact DOB but same birth year: fulldob splits them, birthyear groups
    # them, so the typo pair only surfaces under birthyear.
    df = _df([
        {"PATID": "A", "FirstNM_clean": "JOHN", "LastNM_clean": "SMITH",
         "BirthDT_clean": pd.Timestamp("1980-05-05")},
        {"PATID": "B", "FirstNM_clean": "JOHN", "LastNM_clean": "SMYTH",
         "BirthDT_clean": pd.Timestamp("1980-09-09")},
    ])
    assert run_qgram_blocking(df, block_kind="fulldob", threshold=0.30, min_df=1).empty
    out = run_qgram_blocking(df, block_kind="birthyear", threshold=0.30, min_df=1)
    assert list(zip(out["PATID_A"], out["PATID_B"])) == [("A", "B")]


def test_invalid_records_excluded():
    df = _df([
        {"PATID": "A", "FirstNM_clean": "JOHN", "LastNM_clean": "SMITH",
         "BirthDT_clean": _DOB, "valid_record": False},
        {"PATID": "B", "FirstNM_clean": "JOHN", "LastNM_clean": "SMYTH",
         "BirthDT_clean": _DOB, "valid_record": True},
    ])
    assert run_qgram_blocking(df, threshold=0.30, min_df=1).empty


def test_null_dob_excluded():
    df = _df([
        {"PATID": "A", "FirstNM_clean": "JOHN", "LastNM_clean": "SMITH", "BirthDT_clean": None},
        {"PATID": "B", "FirstNM_clean": "JOHN", "LastNM_clean": "SMYTH", "BirthDT_clean": None},
    ])
    assert run_qgram_blocking(df, threshold=0.30, min_df=1).empty


def test_empty_and_singleton_inputs():
    assert run_qgram_blocking(_df([]), min_df=1).empty
    assert run_qgram_blocking(
        _df([{"PATID": "A", "FirstNM_clean": "JOHN", "LastNM_clean": "SMITH",
              "BirthDT_clean": _DOB}]), min_df=1
    ).empty


def test_threshold_raises_recall_tradeoff():
    # A high threshold drops a borderline pair that a low threshold keeps.
    df = _df([
        {"PATID": "A", "FirstNM_clean": "JON", "LastNM_clean": "SMITH", "BirthDT_clean": _DOB},
        {"PATID": "B", "FirstNM_clean": "JOHN", "LastNM_clean": "SMYTHE", "BirthDT_clean": _DOB},
    ])
    low = run_qgram_blocking(df, threshold=0.10, min_df=1)
    high = run_qgram_blocking(df, threshold=0.99, min_df=1)
    assert len(low) >= len(high)
    assert high.empty
