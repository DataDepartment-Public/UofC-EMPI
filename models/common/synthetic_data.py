"""
synthetic_data.py — Synthetic patient fixtures for the Fellegi-Sunter baseline.

PURPOSE
-------
The real patient index (`MDM_Population_cleaned_v2_*.parquet`) is PHI and lives
only on the AllianceChicago VM. This module produces a small, **non-PHI**,
fully-synthetic stand-in that mirrors the real *schema* (the columns
`_compute_derived_columns()` reads, plus the non-derived columns the Fellegi-
Sunter settings need) so that `fellegi_sunter_baseline.py` can be built and
validated in a sandbox before being ported to the VM.

DESIGN NOTES
------------
* Records are **hard-coded, not random**. The known-truth duplicate pairs are
  therefore explicit and the fixture is deterministic — required for the
  regression / m-u snapshot test (plan Section 10.3).
* Every engineered duplicate targets a specific error scenario from plan
  Section 10.2 AND is constructed to survive at least one real block in
  `blocking.py`, so it actually reaches the scorer rather than being filtered
  out upstream.
* `Phones_set` is stored as a **string-serialized set** (e.g. "{'3125551111'}")
  to match the cleaning pipeline's serialization, which `blocking._parse_phone_set`
  expects. Empty phone sets are stored as the literal string "set()".
* No real names, SSNs, emails, or phone numbers are used. SSNs are obviously
  synthetic; phone numbers use the 555 reserved exchange.

PUBLIC API
----------
    make_synthetic_patients() -> pd.DataFrame
    ground_truth_pairs()      -> set[tuple[str, str]]   # canonical (A<B) true matches
    ground_truth_nonmatches() -> set[tuple[str, str]]   # canonical (A<B) known non-matches
"""

from __future__ import annotations

import pandas as pd

# Column names mirror blocking.py's COL_* constants (single source of truth).
from src.preprocessing.blocking import (
    COL_PATID,
    COL_FIRST_NM,
    COL_LAST_NM,
    COL_BIRTH_DT,
    COL_SSN,
    COL_SSN_LAST4,
    COL_ZIP,
    COL_EMAIL,
    COL_PHONES,
)


def _phones(*nums: str) -> str:
    """Serialize phone numbers the way the cleaning pipeline does: str(set)."""
    s = {str(n) for n in nums if n}
    return str(s) if s else "set()"


# ---------------------------------------------------------------------------
# Record table.
#
# Each row is one patient *record*. Two records with the same `_truth_id`
# represent the same real person (a true duplicate). `_truth_id` and
# `_scenario` are bookkeeping columns for tests — they are dropped before the
# frame is handed to blocking / Splink (the real cleaned parquet has no such
# columns).
# ---------------------------------------------------------------------------
_RECORDS = [
    # ===================================================================
    # PAIR 1 — clean exact duplicate (sanity high-scorer). Blocks: B1/B3/B4/B6/B9
    # ===================================================================
    dict(_truth_id="T01", _scenario="exact_duplicate",
         PATID="P0001", FirstNM_clean="CATHERINE", LastNM_clean="OBRIEN",
         BirthDT_clean="1985-03-15", SSN_clean="900000001", last_4_SSN="0001",
         ZipCD_clean_base="60614", Email_clean="cobrien@example.com",
         Phones_set=_phones("3125550101")),
    dict(_truth_id="T01", _scenario="exact_duplicate",
         PATID="P0002", FirstNM_clean="CATHERINE", LastNM_clean="OBRIEN",
         BirthDT_clean="1985-03-15", SSN_clean="900000001", last_4_SSN="0001",
         ZipCD_clean_base="60614", Email_clean="cobrien@example.com",
         Phones_set=_phones("3125550101")),

    # ===================================================================
    # PAIR 2 — last-name typo, SSN matches. Blocks: B1 (SSN exact)
    # ===================================================================
    dict(_truth_id="T02", _scenario="name_typo",
         PATID="P0003", FirstNM_clean="JONATHAN", LastNM_clean="SMITH",
         BirthDT_clean="1990-07-12", SSN_clean="900000002", last_4_SSN="0002",
         ZipCD_clean_base="60625", Email_clean="jsmith@example.com",
         Phones_set=_phones("7735550202")),
    dict(_truth_id="T02", _scenario="name_typo",
         PATID="P0004", FirstNM_clean="JONATHAN", LastNM_clean="SMYTH",  # typo
         BirthDT_clean="1990-07-12", SSN_clean="900000002", last_4_SSN="0002",
         ZipCD_clean_base="60625", Email_clean="jsmith@example.com",
         Phones_set=_phones("7735550202")),

    # ===================================================================
    # PAIR 3 — DOB digit transposition (12 -> 21, 9 days), SSN matches. Blocks: B1
    # ===================================================================
    dict(_truth_id="T03", _scenario="dob_transposition",
         PATID="P0005", FirstNM_clean="MARIA", LastNM_clean="GARCIA",
         BirthDT_clean="1978-11-12", SSN_clean="900000003", last_4_SSN="0003",
         ZipCD_clean_base="60647", Email_clean="mgarcia@example.com",
         Phones_set=_phones("3125550303")),
    dict(_truth_id="T03", _scenario="dob_transposition",
         PATID="P0006", FirstNM_clean="MARIA", LastNM_clean="GARCIA",
         BirthDT_clean="1978-11-21", SSN_clean="900000003", last_4_SSN="0003",  # day transposed
         ZipCD_clean_base="60647", Email_clean="mgarcia@example.com",
         Phones_set=_phones("3125550303")),

    # ===================================================================
    # PAIR 4 — SSN missing but phone matching. Blocks: B5 (phone), B3/B4 (name+DOB)
    # ===================================================================
    dict(_truth_id="T04", _scenario="ssn_missing_phone_match",
         PATID="P0007", FirstNM_clean="DAVID", LastNM_clean="NGUYEN",
         BirthDT_clean="1965-02-28", SSN_clean="900000004", last_4_SSN="0004",
         ZipCD_clean_base="60618", Email_clean="dnguyen@example.com",
         Phones_set=_phones("7735550404")),
    dict(_truth_id="T04", _scenario="ssn_missing_phone_match",
         PATID="P0008", FirstNM_clean="DAVID", LastNM_clean="NGUYEN",
         BirthDT_clean="1965-02-28", SSN_clean=None, last_4_SSN=None,  # SSN missing
         ZipCD_clean_base="60618", Email_clean=None,                   # email also missing
         Phones_set=_phones("7735550404")),                            # same phone

    # ===================================================================
    # PAIR 5 — last-4 SSN match only (full SSN missing on B). Blocks: B9, B3/B4
    # ===================================================================
    dict(_truth_id="T05", _scenario="ssn_last4_only",
         PATID="P0009", FirstNM_clean="SARAH", LastNM_clean="JOHNSON",
         BirthDT_clean="1992-09-05", SSN_clean="900000005", last_4_SSN="0005",
         ZipCD_clean_base="60616", Email_clean="sjohnson@example.com",
         Phones_set=_phones("3125550505")),
    dict(_truth_id="T05", _scenario="ssn_last4_only",
         PATID="P0010", FirstNM_clean="SARAH", LastNM_clean="JOHNSON",
         BirthDT_clean="1992-09-05", SSN_clean=None, last_4_SSN="0005",  # only last4 known
         ZipCD_clean_base="60616", Email_clean="sjohnson@example.com",
         Phones_set=_phones("3125550606")),                              # different phone

    # ===================================================================
    # PAIR 6 — email username match, different domain. Blocks: B3 (DM-LN+DOB), B4
    # ===================================================================
    dict(_truth_id="T06", _scenario="email_diff_domain",
         PATID="P0011", FirstNM_clean="MICHAEL", LastNM_clean="BROWN",
         BirthDT_clean="1988-12-20", SSN_clean="900000006", last_4_SSN="0006",
         ZipCD_clean_base="60622", Email_clean="mbrown@gmail.example",
         Phones_set=_phones("7735550707")),
    dict(_truth_id="T06", _scenario="email_diff_domain",
         PATID="P0012", FirstNM_clean="MICHAEL", LastNM_clean="BROWN",
         BirthDT_clean="1988-12-20", SSN_clean=None, last_4_SSN=None,
         ZipCD_clean_base="60622", Email_clean="mbrown@yahoo.example",  # same user, diff domain
         Phones_set=_phones("7735550808")),

    # ===================================================================
    # PAIR 7 — ZIP 3-digit prefix only + thin record (many nulls on B).
    #          Blocks: B3 (DM-LN + full DOB), B4
    # ===================================================================
    dict(_truth_id="T07", _scenario="zip_prefix_thin",
         PATID="P0013", FirstNM_clean="EMILY", LastNM_clean="DAVIS",
         BirthDT_clean="1995-04-10", SSN_clean="900000007", last_4_SSN="0007",
         ZipCD_clean_base="60614", Email_clean="edavis@example.com",
         Phones_set=_phones("3125550909")),
    dict(_truth_id="T07", _scenario="zip_prefix_thin",
         PATID="P0014", FirstNM_clean="EMILY", LastNM_clean="DAVIS",
         BirthDT_clean="1995-04-10", SSN_clean=None, last_4_SSN=None,
         ZipCD_clean_base="60611", Email_clean=None,        # same '606' prefix, diff full ZIP
         Phones_set=_phones()),                              # empty phone set

    # ===================================================================
    # PAIR 8 — phonetic last-name variant (Soundex/DM equal). Blocks: B8, maybe B3
    # ===================================================================
    dict(_truth_id="T08", _scenario="phonetic_lastname",
         PATID="P0015", FirstNM_clean="PATRICK", LastNM_clean="SHAUGHNESSY",
         BirthDT_clean="1972-06-30", SSN_clean="900000008", last_4_SSN="0008",
         ZipCD_clean_base="60634", Email_clean="pshaughnessy@example.com",
         Phones_set=_phones("7735551212")),
    dict(_truth_id="T08", _scenario="phonetic_lastname",
         PATID="P0016", FirstNM_clean="PATRICK", LastNM_clean="SHAUNESSY",  # phonetic variant
         BirthDT_clean="1972-06-30", SSN_clean=None, last_4_SSN=None,
         ZipCD_clean_base="60634", Email_clean=None,
         Phones_set=_phones()),

    # ===================================================================
    # KNOWN NON-MATCH A — two different people, same LastNM + birth year +
    # first-3 of first name (block via B4) but everything identifying differs.
    # FS must score this BELOW 0.5.
    # ===================================================================
    dict(_truth_id="N01a", _scenario="nonmatch_b4_collision",
         PATID="P0017", FirstNM_clean="JENNIFER", LastNM_clean="WILSON",
         BirthDT_clean="1983-05-09", SSN_clean="900000101", last_4_SSN="0101",
         ZipCD_clean_base="60640", Email_clean="jwilson@example.com",
         Phones_set=_phones("7735551001")),
    dict(_truth_id="N01b", _scenario="nonmatch_b4_collision",
         PATID="P0018", FirstNM_clean="JENNA", LastNM_clean="WILSON",  # same 'JEN' prefix
         BirthDT_clean="1983-11-22", SSN_clean="900000102", last_4_SSN="0102",
         ZipCD_clean_base="60630", Email_clean="jenna.wilson@example.com",
         Phones_set=_phones("3125551002")),

    # ===================================================================
    # KNOWN NON-MATCH B — two unrelated people sharing one phone number
    # (a clinic line keyed onto both). Block via B5. FS must score BELOW 0.5.
    # ===================================================================
    dict(_truth_id="N02a", _scenario="nonmatch_shared_phone",
         PATID="P0019", FirstNM_clean="KEVIN", LastNM_clean="MARTIN",
         BirthDT_clean="1991-08-14", SSN_clean="900000103", last_4_SSN="0103",
         ZipCD_clean_base="60615", Email_clean="kmartin@example.com",
         Phones_set=_phones("3125550000")),
    dict(_truth_id="N02b", _scenario="nonmatch_shared_phone",
         PATID="P0020", FirstNM_clean="LISA", LastNM_clean="ANDERSON",
         BirthDT_clean="1976-03-22", SSN_clean="900000104", last_4_SSN="0104",
         ZipCD_clean_base="60619", Email_clean="landerson@example.com",
         Phones_set=_phones("3125550000")),                     # same clinic phone

    # ===================================================================
    # SINGLETONS — records that should pair with no one (realism for u-estimation).
    # ===================================================================
    dict(_truth_id="S01", _scenario="singleton",
         PATID="P0021", FirstNM_clean="GEORGE", LastNM_clean="WASHINGTON",
         BirthDT_clean="1955-07-04", SSN_clean="900000201", last_4_SSN="0201",
         ZipCD_clean_base="60652", Email_clean="gwashington@example.com",
         Phones_set=_phones("7735552001")),
    dict(_truth_id="S02", _scenario="singleton",
         PATID="P0022", FirstNM_clean="AISHA", LastNM_clean="KHAN",
         BirthDT_clean="2001-01-30", SSN_clean="900000202", last_4_SSN="0202",
         ZipCD_clean_base="60607", Email_clean="akhan@example.com",
         Phones_set=_phones("3125552002")),
    dict(_truth_id="S03", _scenario="singleton",
         PATID="P0023", FirstNM_clean="WEI", LastNM_clean="CHEN",
         BirthDT_clean="1969-10-18", SSN_clean="900000203", last_4_SSN="0203",
         ZipCD_clean_base="60660", Email_clean="wchen@example.com",
         Phones_set=_phones("7735552003")),
    dict(_truth_id="S04", _scenario="singleton",
         PATID="P0024", FirstNM_clean="FATIMA", LastNM_clean="ALI",
         BirthDT_clean="1998-06-22", SSN_clean="900000204", last_4_SSN="0204",
         ZipCD_clean_base="60629", Email_clean="fali@example.com",
         Phones_set=_phones("3125552004")),
]

# Bookkeeping columns that exist only in the fixture, never in real data.
_FIXTURE_ONLY_COLS = ["_truth_id", "_scenario"]


def make_synthetic_patients(include_truth_columns: bool = False) -> pd.DataFrame:
    """
    Build the synthetic cleaned-patient DataFrame.

    Parameters
    ----------
    include_truth_columns : bool
        If True, retain the `_truth_id` / `_scenario` bookkeeping columns
        (useful for tests). If False (default), drop them so the frame matches
        the real cleaned-parquet schema exactly.

    Returns
    -------
    pd.DataFrame
        One row per patient record, columns matching the real cleaned schema
        (the inputs to `_compute_derived_columns()` plus the non-derived
        identifier columns).
    """
    df = pd.DataFrame(_RECORDS)
    # Enforce string dtype on identifier columns (object), preserving None as NULL.
    str_cols = [
        COL_PATID, COL_FIRST_NM, COL_LAST_NM, COL_BIRTH_DT,
        COL_SSN, COL_SSN_LAST4, COL_ZIP, COL_EMAIL, COL_PHONES,
    ]
    for c in str_cols:
        df[c] = df[c].astype("object")
    if not include_truth_columns:
        df = df.drop(columns=_FIXTURE_ONLY_COLS)
    return df


def _canon(a: str, b: str) -> tuple[str, str]:
    """Return the pair in canonical (alphabetically-lower-first) order."""
    return (a, b) if a < b else (b, a)


def ground_truth_pairs() -> set[tuple[str, str]]:
    """Canonical (PATID_A, PATID_B) pairs that are genuinely the same person."""
    df = make_synthetic_patients(include_truth_columns=True)
    pairs: set[tuple[str, str]] = set()
    for _truth_id, grp in df[df["_truth_id"].str.startswith("T")].groupby("_truth_id"):
        patids = sorted(grp[COL_PATID])
        for i in range(len(patids)):
            for j in range(i + 1, len(patids)):
                pairs.add(_canon(patids[i], patids[j]))
    return pairs


def ground_truth_nonmatches() -> set[tuple[str, str]]:
    """
    Canonical pairs that block together but are genuinely different people
    (the false positives the scorer must push below the match threshold).
    """
    return {
        _canon("P0017", "P0018"),  # B4 last-name+birthyear collision
        _canon("P0019", "P0020"),  # shared clinic phone (B5)
    }


if __name__ == "__main__":
    # Non-PHI summary only (safe to print): shapes, dtypes, null rates, scenarios.
    patients = make_synthetic_patients(include_truth_columns=True)
    print("=== synthetic patient frame ===")
    print("shape:", patients.shape)
    print("\nscenario counts:")
    print(patients["_scenario"].value_counts())
    print("\nground-truth true-match pairs:", len(ground_truth_pairs()))
    print("ground-truth non-match pairs:", len(ground_truth_nonmatches()))