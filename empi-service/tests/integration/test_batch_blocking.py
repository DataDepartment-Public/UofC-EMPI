"""
=========================================
Integration test for run_batch_blocking() using a 300-record synthetic
dataset with 15 known match pairs — one pair per block for each of the
8 active blocks, plus 7 additional pairs that test multi-block capture
and boundary conditions.

WHAT THIS TEST PROVES:
    The blocking pipeline correctly generates candidate pairs for real
    match scenarios across all 8 active blocking schemes. Each known
    pair is designed to be captured by exactly one specific block (its
    target block) and verified to appear in the output. Additionally,
    pairs designed to be NON-candidates are verified to be absent.

WHY SYNTHETIC DATA:
    Synthetic data is zero-PHI, deterministic, reproducible on any
    machine, and allows exact control over which pairs each block
    should and should not capture. Real data cannot provide these
    guarantees for a test suite.

DATASET STRUCTURE:
    - 270 background records: realistic but unique — no shared blocking
      keys, so they produce no candidate pairs among themselves
    - 30 records forming 15 known match pairs, injected at specific
      positions, each pair sharing exactly the blocking key of their
      target block

KNOWN PAIR DESIGN:
    B1  — SSN exact:                    pair shares clean_SSN
    B3  — DM(LN) + full DOB:            pair shares phonetic LN + exact DOB
    B3b — DM(LN) + full DOB (variant):  phonetic variant catches what B2 missed
    B4  — LN + BirthYear + FN prefix:   DOB transposition pair
    B5  — Phone set intersection:        pair shares one phone number
    B5b — Phone set (secondary column):  phone in different position
    B6  — Email exact:                   pair shares clean email
    B7  — DM(LN) + ZIP + BirthYear:     geographic fallback pair
    B7b — DM(LN) + ZIP + BirthYear:     DOB-missing variant
    B8  — Soundex(FN)+Soundex(LN)+Year: both names phonetically variant
    B8b — Soundex catch-all:             coarser encoding catches what B3 misses
    B9  — LN + FN + SSN last 4:         SSN typo in first 5 digits
    B9b — LN + FN + SSN last 4:         different first name, same last 4
    MULTI — pair captured by B3 + B5:   multi-block capture verified
    NONE — pair sharing no blocking key: must NOT appear in output
"""

import pytest
import pandas as pd

from src.preprocessing.blocking import run_batch_blocking

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
# Synthetic Dataset Builder
# ═══════════════════════════════════════════════════════════════════════════

# Diverse last name pool — chosen to produce distinct Soundex codes
# so background records do not collide on B8 (Soundex FN + Soundex LN + Year)
_BG_LAST_NAMES = [
    "ABERNATHY", "BJORNSTAD", "CZAJKOWSKI", "DELVECCHIO", "EVANGELISTA",
    "FJELSTAD", "GHEORGHIU", "HASHIMOTO", "IOANNIDIS", "JAKUBOWSKI",
    "KLEINHEMPEL", "LAFRAMBOISE", "MWANGI", "NZINGA", "ODUYA",
    "PRZYBYSZEWSKI", "QUIROGA", "RASMUSSEN", "SZCZEPANSKI", "TJIONG",
    "UZODINMA", "VANDENBERGHE", "WOJCIECHOWSKI", "XANTHOPOULOS", "YABLONSKY",
    "ZELAZOWSKI", "ACHTERBERG", "BARANAUSKAS", "CRISTODOULOU", "DWORACZYK",
]

_BG_FIRST_NAMES = [
    "AALIYA", "BOGUMIL", "CHIBUEZE", "DAGMARA", "EFEMENA",
    "FIORENTINA", "GRZEGORZ", "HYUNJIN", "INGEBORG", "JABARI",
    "KWABENA", "LILAVATI", "METODIJA", "NNAMDI", "OLUWASEUN",
    "PRZEMYSLAW", "QUIRINA", "RADOSLAVA", "SAOIRSE", "TATENDA",
    "UCHENNA", "VLADISLAVA", "WANJIKU", "XIULAN", "YEVHENIYA",
    "ZYGMUNT", "AGNIESZKA", "BALACHANDRAN", "CHIAMAKA", "DESISLAVA",
    ]

def _make_background_record(idx: int) -> dict:
    """
    Generate a unique background record guaranteed to share no blocking
    keys with any other background record or known pair.

    Uses a curated name pool with high phonetic diversity specifically
    chosen to avoid Soundex collisions on B8. Each record has a fully
    unique SSN, email, phone, DOB, and ZIP.
    """
    # Select names from the pool using modulo — the 30-name pools combined
    # with unique birth years ensure no two background records share a
    # B8 key (Soundex(FN) | Soundex(LN) | BirthYear)
    last_nm  = _BG_LAST_NAMES[idx % len(_BG_LAST_NAMES)]
    first_nm = _BG_FIRST_NAMES[idx % len(_BG_FIRST_NAMES)]

    # Spread birth years across a 270-year range to guarantee uniqueness
    # Each idx gets a unique year in range 1750-2019
    birth_year = 1750 + idx
    # Keep month/day realistic
    month = (idx % 12) + 1
    day   = (idx % 28) + 1

    return {
        COL_PATID:    f"BG_{idx:04d}",
        COL_LAST_NM:  last_nm,
        COL_FIRST_NM: first_nm,
        COL_BIRTH_DT: pd.Timestamp(f"{birth_year}-{month:02d}-{day:02d}"),
        COL_SSN:      f"{100000000 + idx:09d}",
        COL_SSN_L4:   f"{100000000 + idx:09d}"[-4:],
        COL_ZIP:      f"{10000 + idx:05d}",
        COL_EMAIL:    f"bg{idx:04d}@uniquetest.com",
        COL_PHONES:   f"{{'555{idx:07d}'}}",
    }


def _build_synthetic_dataset() -> tuple[pd.DataFrame, list[dict]]:
    """
    Build the 300-record synthetic dataset and return it alongside
    the known pairs manifest for test assertions.

    Returns
    -------
    df : pd.DataFrame
        The full 300-record dataset (270 background + 30 pair records)
    known_pairs : list of dict
        Each dict has keys:
            patid_a       (str) canonical first PATID
            patid_b       (str) canonical second PATID
            target_block  (str) block expected to capture this pair
            should_exist  (bool) True = must be a candidate, False = must NOT
    """
    records = []
    known_pairs = []

    # ── 270 background records ────────────────────────────────────────────
    for i in range(1, 271):
        records.append(_make_background_record(i))

    # ── Known pair 1: B1 — SSN Exact ─────────────────────────────────────
    # Both records share SSN "234567891". All other fields differ.
    records.append({
        COL_PATID:    "KP_B1_A",
        COL_LAST_NM:  "WALKERB1A",
        COL_FIRST_NM: "THOMAS",
        COL_BIRTH_DT: pd.Timestamp("1975-03-12"),
        COL_SSN:      "234567891",
        COL_SSN_L4:   "7891",
        COL_ZIP:      "60601",
        COL_EMAIL:    "twalker_a@test.com",
        COL_PHONES:   "{'7731110001'}",
    })
    records.append({
        COL_PATID:    "KP_B1_B",
        COL_LAST_NM:  "WALKERB1B",       # different last name
        COL_FIRST_NM: "THOMAS",
        COL_BIRTH_DT: pd.Timestamp("1980-07-22"),  # different DOB
        COL_SSN:      "234567891",        # SAME SSN → B1 match
        COL_SSN_L4:   "7891",
        COL_ZIP:      "60602",
        COL_EMAIL:    "twalker_b@test.com",
        COL_PHONES:   "{'7731110002'}",
    })
    known_pairs.append({
        "patid_a": "KP_B1_A", "patid_b": "KP_B1_B",
        "target_block": "B1", "should_exist": True,
    })

    # ── Known pair 2: B3 — DM(LastNM) + Full DOB ─────────────────────────
    # GARCIA and GARSIA share Double Metaphone primary code.
    # Both have exact same DOB.
    records.append({
        COL_PATID:    "KP_B3_A",
        COL_LAST_NM:  "GARCIA",
        COL_FIRST_NM: "CARLOS",
        COL_BIRTH_DT: pd.Timestamp("1988-05-20"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60610",
        COL_EMAIL:    "carlos.garcia@test.com",
        COL_PHONES:   "{'7732220001'}",
    })
    records.append({
        COL_PATID:    "KP_B3_B",
        COL_LAST_NM:  "GARSIA",           # phonetic variant → same DM code
        COL_FIRST_NM: "CARLOS",
        COL_BIRTH_DT: pd.Timestamp("1988-05-20"),  # SAME DOB → B3 match
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60611",
        COL_EMAIL:    "carlos.garsia@test.com",
        COL_PHONES:   "{'7732220002'}",
    })
    known_pairs.append({
        "patid_a": "KP_B3_A", "patid_b": "KP_B3_B",
        "target_block": "B3", "should_exist": True,
    })

    # ── Known pair 3: B3b — Phonetic variant B3 (JOHNSON / JONSON) ───────
    records.append({
        COL_PATID:    "KP_B3B_A",
        COL_LAST_NM:  "JOHNSON",
        COL_FIRST_NM: "SARAH",
        COL_BIRTH_DT: pd.Timestamp("1992-11-03"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60620",
        COL_EMAIL:    "sarah.johnson@test.com",
        COL_PHONES:   "{'7733330001'}",
    })
    records.append({
        COL_PATID:    "KP_B3B_B",
        COL_LAST_NM:  "JONSON",           # phonetic variant
        COL_FIRST_NM: "SARAH",
        COL_BIRTH_DT: pd.Timestamp("1992-11-03"),  # same DOB
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60621",
        COL_EMAIL:    "sarah.jonson@test.com",
        COL_PHONES:   "{'7733330002'}",
    })
    known_pairs.append({
        "patid_a": "KP_B3B_A", "patid_b": "KP_B3B_B",
        "target_block": "B3", "should_exist": True,
    })

    # ── Known pair 4: B4 — LN + BirthYear + FN Prefix (DOB transposition) ─
    # Same last name and birth year, FN prefix matches.
    # DOBs differ by month/day swap — B3 would miss this, B4 catches it.
    records.append({
        COL_PATID:    "KP_B4_A",
        COL_LAST_NM:  "RODRIGUEZ",
        COL_FIRST_NM: "CARLOS",
        COL_BIRTH_DT: pd.Timestamp("1979-08-04"),  # Aug 4
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60630",
        COL_EMAIL:    "carlos.rod_a@test.com",
        COL_PHONES:   "{'7734440001'}",
    })
    records.append({
        COL_PATID:    "KP_B4_B",
        COL_LAST_NM:  "RODRIGUEZ",        # same last name
        COL_FIRST_NM: "CARLOS",           # same FN → prefix "CAR" matches
        COL_BIRTH_DT: pd.Timestamp("1979-04-08"),  # Apr 8 — transposed
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60631",
        COL_EMAIL:    "carlos.rod_b@test.com",
        COL_PHONES:   "{'7734440002'}",
    })
    known_pairs.append({
        "patid_a": "KP_B4_A", "patid_b": "KP_B4_B",
        "target_block": "B4", "should_exist": True,
    })

    # ── Known pair 5: B5 — Phone Set Intersection ─────────────────────────
    # Records share one phone number. All name/DOB fields differ.
    records.append({
        COL_PATID:    "KP_B5_A",
        COL_LAST_NM:  "PATELB5A",
        COL_FIRST_NM: "PRIYA",
        COL_BIRTH_DT: pd.Timestamp("1986-09-14"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60640",
        COL_EMAIL:    "priya_a@test.com",
        COL_PHONES:   "{'3125550100', '7735550200'}",  # two phones
    })
    records.append({
        COL_PATID:    "KP_B5_B",
        COL_LAST_NM:  "PATELB5B",         # different last name
        COL_FIRST_NM: "RAHUL",            # different first name
        COL_BIRTH_DT: pd.Timestamp("1990-02-28"),  # different DOB
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60641",
        COL_EMAIL:    "rahul_b@test.com",
        COL_PHONES:   "{'3125550100'}",   # shares 3125550100 → B5 match
    })
    known_pairs.append({
        "patid_a": "KP_B5_A", "patid_b": "KP_B5_B",
        "target_block": "B5", "should_exist": True,
    })

    # ── Known pair 6: B5b — Phone in secondary position ───────────────────
    records.append({
        COL_PATID:    "KP_B5B_A",
        COL_LAST_NM:  "NGUYENB5B",
        COL_FIRST_NM: "MINH",
        COL_BIRTH_DT: pd.Timestamp("1983-06-17"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60650",
        COL_EMAIL:    "minh_a@test.com",
        COL_PHONES:   "{'7735550301', '8475550302'}",
    })
    records.append({
        COL_PATID:    "KP_B5B_B",
        COL_LAST_NM:  "NGUYENB5C",
        COL_FIRST_NM: "LINH",
        COL_BIRTH_DT: pd.Timestamp("1987-12-01"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60651",
        COL_EMAIL:    "linh_b@test.com",
        COL_PHONES:   "{'8475550302'}",   # shares 8475550302 → B5 match
    })
    known_pairs.append({
        "patid_a": "KP_B5B_A", "patid_b": "KP_B5B_B",
        "target_block": "B5", "should_exist": True,
    })

    # ── Known pair 7: B6 — Email Exact ────────────────────────────────────
    records.append({
        COL_PATID:    "KP_B6_A",
        COL_LAST_NM:  "SMITHB6A",
        COL_FIRST_NM: "JAMES",
        COL_BIRTH_DT: pd.Timestamp("1971-04-25"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60660",
        COL_EMAIL:    "shared.email.b6@test.com",  # shared email
        COL_PHONES:   "{'7735550401'}",
    })
    records.append({
        COL_PATID:    "KP_B6_B",
        COL_LAST_NM:  "SMITHB6B",
        COL_FIRST_NM: "JIMMY",
        COL_BIRTH_DT: pd.Timestamp("1971-04-25"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60661",
        COL_EMAIL:    "shared.email.b6@test.com",  # SAME email → B6 match
        COL_PHONES:   "{'7735550402'}",
    })
    known_pairs.append({
        "patid_a": "KP_B6_A", "patid_b": "KP_B6_B",
        "target_block": "B6", "should_exist": True,
    })

    # ── Known pair 8: B7 — DM(LN) + ZIP + BirthYear ──────────────────────
    # DOB differs between records (one has it, one has default Jan 1st).
    # Same phonetic LN, same ZIP, same birth year → B7 catches it.
    records.append({
        COL_PATID:    "KP_B7_A",
        COL_LAST_NM:  "MARTINEZ",
        COL_FIRST_NM: "ELENA",
        COL_BIRTH_DT: pd.Timestamp("1984-09-12"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60670",
        COL_EMAIL:    "elena.m_a@test.com",
        COL_PHONES:   "{'7735550501'}",
    })
    records.append({
        COL_PATID:    "KP_B7_B",
        COL_LAST_NM:  "MARTINES",         # phonetic variant → same DM
        COL_FIRST_NM: "ELENA",
        COL_BIRTH_DT: pd.Timestamp("1984-01-01"),  # default DOB
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60670",            # SAME ZIP → B7 match on year+zip+DM
        COL_EMAIL:    "elena.m_b@test.com",
        COL_PHONES:   "{'7735550502'}",
    })
    known_pairs.append({
        "patid_a": "KP_B7_A", "patid_b": "KP_B7_B",
        "target_block": "B7", "should_exist": True,
    })

    # ── Known pair 9: B7b — Geographic fallback (no DOB at all) ──────────
    records.append({
        COL_PATID:    "KP_B7B_A",
        COL_LAST_NM:  "GUTIERREZ",
        COL_FIRST_NM: "PEDRO",
        COL_BIRTH_DT: pd.Timestamp("1976-07-18"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60680",
        COL_EMAIL:    "pedro.g_a@test.com",
        COL_PHONES:   "{'7735550601'}",
    })
    records.append({
        COL_PATID:    "KP_B7B_B",
        COL_LAST_NM:  "GUTIERRES",        # phonetic variant
        COL_FIRST_NM: "PEDRO",
        COL_BIRTH_DT: pd.Timestamp("1976-03-22"),  # different month/day
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60680",            # same ZIP + same year → B7
        COL_EMAIL:    "pedro.g_b@test.com",
        COL_PHONES:   "{'7735550602'}",
    })
    known_pairs.append({
        "patid_a": "KP_B7B_A", "patid_b": "KP_B7B_B",
        "target_block": "B7", "should_exist": True,
    })

    # ── Known pair 10: B8 — Soundex(FN)+Soundex(LN)+BirthYear ────────────
    # Both names have phonetic variants across first AND last name.
    # MIKHAIL/MICHAEL + KOVALENKO/KOVALENCO — both share Soundex + birth year.
    records.append({
        COL_PATID:    "KP_B8_A",
        COL_LAST_NM:  "KOVALENKO",
        COL_FIRST_NM: "MIKHAIL",
        COL_BIRTH_DT: pd.Timestamp("1969-02-14"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60690",
        COL_EMAIL:    "mikhail.k_a@test.com",
        COL_PHONES:   "{'7735550701'}",
    })
    records.append({
        COL_PATID:    "KP_B8_B",
        COL_LAST_NM:  "KOVALENCO",        # phonetic variant
        COL_FIRST_NM: "MICHAEL",          # phonetic variant
        COL_BIRTH_DT: pd.Timestamp("1969-08-22"),  # same year, diff month/day
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60691",
        COL_EMAIL:    "michael.k_b@test.com",
        COL_PHONES:   "{'7735550702'}",
    })
    known_pairs.append({
        "patid_a": "KP_B8_A", "patid_b": "KP_B8_B",
        "target_block": "B8", "should_exist": True,
    })

    # ── Known pair 11: B8b — Soundex catch-all ────────────────────────────
    records.append({
        COL_PATID:    "KP_B8B_A",
        COL_LAST_NM:  "SCHMIDT",
        COL_FIRST_NM: "HANS",
        COL_BIRTH_DT: pd.Timestamp("1955-11-30"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60700",
        COL_EMAIL:    "hans.s_a@test.com",
        COL_PHONES:   "{'7735550801'}",
    })
    records.append({
        COL_PATID:    "KP_B8B_B",
        COL_LAST_NM:  "SHMIDT",           # phonetic variant
        COL_FIRST_NM: "HANS",
        COL_BIRTH_DT: pd.Timestamp("1955-06-15"),  # same year, diff month/day
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60701",
        COL_EMAIL:    "hans.s_b@test.com",
        COL_PHONES:   "{'7735550802'}",
    })
    known_pairs.append({
        "patid_a": "KP_B8B_A", "patid_b": "KP_B8B_B",
        "target_block": "B8", "should_exist": True,
    })

    # ── Known pair 12: B9 — LN + FN + SSN Last 4 ─────────────────────────
    # SSN differs in the first 5 digits (typo) but last 4 are identical.
    # Same exact last name and first name → B9 catches it.
    records.append({
        COL_PATID:    "KP_B9_A",
        COL_LAST_NM:  "THOMPSONB9",
        COL_FIRST_NM: "DAVID",
        COL_BIRTH_DT: pd.Timestamp("1981-03-08"),
        COL_SSN:      "345678901",
        COL_SSN_L4:   "8901",
        COL_ZIP:      "60710",
        COL_EMAIL:    "david.t_a@test.com",
        COL_PHONES:   "{'7735550901'}",
    })
    records.append({
        COL_PATID:    "KP_B9_B",
        COL_LAST_NM:  "THOMPSONB9",       # same last name
        COL_FIRST_NM: "DAVID",            # same first name
        COL_BIRTH_DT: pd.Timestamp("1981-09-14"),  # different DOB
        COL_SSN:      "345271901",        # typo in first 5 digits
        COL_SSN_L4:   "1901",             # different last 4 — B9 won't match
        COL_ZIP:      "60711",
        COL_EMAIL:    "david.t_b@test.com",
        COL_PHONES:   "{'7735550902'}",
    })
    # NOTE: The above B9 pair actually tests that B9 does NOT match when
    # last-4 differs. The correct B9 pair needs matching last-4.
    # Replacing with a proper B9 test pair:
    records[-1] = {
        COL_PATID:    "KP_B9_B",
        COL_LAST_NM:  "THOMPSONB9",       # same last name
        COL_FIRST_NM: "DAVID",            # same first name
        COL_BIRTH_DT: pd.Timestamp("1981-09-14"),  # different DOB
        COL_SSN:      "999678901",        # typo in first 3 digits
        COL_SSN_L4:   "8901",             # SAME last 4 → B9 match
        COL_ZIP:      "60711",
        COL_EMAIL:    "david.t_b@test.com",
        COL_PHONES:   "{'7735550902'}",
    }
    known_pairs.append({
        "patid_a": "KP_B9_A", "patid_b": "KP_B9_B",
        "target_block": "B9", "should_exist": True,
    })

    # ── Known pair 13: B9b — Different first name, same last + last 4 ────
    records.append({
        COL_PATID:    "KP_B9B_A",
        COL_LAST_NM:  "WILSONB9B",
        COL_FIRST_NM: "ROBERT",
        COL_BIRTH_DT: pd.Timestamp("1973-01-22"),
        COL_SSN:      "456789012",
        COL_SSN_L4:   "9012",
        COL_ZIP:      "60720",
        COL_EMAIL:    "robert.w_a@test.com",
        COL_PHONES:   "{'7735551001'}",
    })
    records.append({
        COL_PATID:    "KP_B9B_B",
        COL_LAST_NM:  "WILSONB9B",        # same last name
        COL_FIRST_NM: "ROBERT",           # same first name
        COL_BIRTH_DT: pd.Timestamp("1973-07-04"),
        COL_SSN:      "123789012",        # typo in first 3 digits
        COL_SSN_L4:   "9012",             # SAME last 4 → B9 match
        COL_ZIP:      "60721",
        COL_EMAIL:    "robert.w_b@test.com",
        COL_PHONES:   "{'7735551002'}",
    })
    known_pairs.append({
        "patid_a": "KP_B9B_A", "patid_b": "KP_B9B_B",
        "target_block": "B9", "should_exist": True,
    })

    # ── Known pair 14: MULTI — Captured by B3 AND B5 ─────────────────────
    # Records share both a phonetic last name + exact DOB (B3) AND a phone
    # number (B5). The pair must appear with source_blocks containing both.
    records.append({
        COL_PATID:    "KP_MULTI_A",
        COL_LAST_NM:  "GONZALEZ",
        COL_FIRST_NM: "LUCIA",
        COL_BIRTH_DT: pd.Timestamp("1994-07-07"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60730",
        COL_EMAIL:    "lucia.g_a@test.com",
        COL_PHONES:   "{'3125551100', '7735551101'}",
    })
    records.append({
        COL_PATID:    "KP_MULTI_B",
        COL_LAST_NM:  "GONZALES",         # phonetic variant → B3 match
        COL_FIRST_NM: "LUCIA",
        COL_BIRTH_DT: pd.Timestamp("1994-07-07"),  # same DOB → B3 match
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "60731",
        COL_EMAIL:    "lucia.g_b@test.com",
        COL_PHONES:   "{'3125551100'}",   # shared phone → B5 match
    })
    known_pairs.append({
        "patid_a": "KP_MULTI_A", "patid_b": "KP_MULTI_B",
        "target_block": "B3",  # primary — B5 also captures, verified separately
        "should_exist": True,
    })

    # ── Known pair 15: NONE — Must NOT appear as a candidate ──────────────
    # Records share no blocking key on any of the 8 active schemes.
    # Different last name, different DOB, different phone, no SSN, no email match.
    records.append({
        COL_PATID:    "KP_NONE_A",
        COL_LAST_NM:  "AAAA_UNIQUE",
        COL_FIRST_NM: "FIRSTONLY",
        COL_BIRTH_DT: pd.Timestamp("2001-01-15"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "99998",
        COL_EMAIL:    "unique_a_none@test.com",
        COL_PHONES:   "{'9990001111'}",
    })
    records.append({
        COL_PATID:    "KP_NONE_B",
        COL_LAST_NM:  "ZZZZ_UNIQUE",
        COL_FIRST_NM: "LASTONLY",
        COL_BIRTH_DT: pd.Timestamp("2002-06-20"),
        COL_SSN:      None,
        COL_SSN_L4:   None,
        COL_ZIP:      "99997",
        COL_EMAIL:    "unique_b_none@test.com",
        COL_PHONES:   "{'9990002222'}",
    })
    known_pairs.append({
        "patid_a": "KP_NONE_A", "patid_b": "KP_NONE_B",
        "target_block": "NONE",
        "should_exist": False,  # must NOT be a candidate
    })

    df = pd.DataFrame(records)
    return df, known_pairs


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def synthetic_data():
    """Build the synthetic dataset once per module — not per test."""
    return _build_synthetic_dataset()


@pytest.fixture(scope="module")
def candidate_pairs(synthetic_data):
    """Run batch blocking once per module and cache the result."""
    df, _ = synthetic_data
    return run_batch_blocking(df)


@pytest.fixture(scope="module")
def pair_set(candidate_pairs):
    """Set of canonical (PATID_A, PATID_B) tuples for fast lookup."""
    return set(
        zip(candidate_pairs["PATID_A"], candidate_pairs["PATID_B"])
    )


def _canonical(a: str, b: str) -> tuple:
    """Return canonical (lower, higher) pair regardless of input order."""
    return (a, b) if a < b else (b, a)


# ═══════════════════════════════════════════════════════════════════════════
# Core: Known Pair Capture Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestKnownPairCapture:
    """
    Every known match pair must be captured by at least one blocking scheme.
    Every known non-match pair must NOT appear in the candidate set.
    """

    def test_b1_ssn_exact_captures_pair(self, pair_set):
        """B1: records sharing exact SSN must be candidates."""
        pair = _canonical("KP_B1_A", "KP_B1_B")
        assert pair in pair_set, (
            "B1 (SSN Exact) failed to generate candidate pair "
            "for records sharing clean_SSN='234567891'"
        )

    def test_b3_phonetic_ln_exact_dob_garcia_garsia(self, pair_set):
        """B3: GARCIA/GARSIA with same DOB must be candidates."""
        pair = _canonical("KP_B3_A", "KP_B3_B")
        assert pair in pair_set, (
            "B3 (DM-LN + DOB) failed to generate candidate pair "
            "for GARCIA/GARSIA with matching DOB"
        )

    def test_b3_phonetic_ln_johnson_jonson(self, pair_set):
        """B3: JOHNSON/JONSON with same DOB must be candidates."""
        pair = _canonical("KP_B3B_A", "KP_B3B_B")
        assert pair in pair_set, (
            "B3 failed to generate candidate pair for JOHNSON/JONSON"
        )

    def test_b4_dob_transposition_captures_pair(self, pair_set):
        """B4: records with transposed DOB month/day must be candidates."""
        pair = _canonical("KP_B4_A", "KP_B4_B")
        assert pair in pair_set, (
            "B4 (LN + BirthYear + FN-Prefix) failed to generate candidate "
            "pair for RODRIGUEZ with DOBs 1979-08-04 and 1979-04-08"
        )

    def test_b5_shared_phone_primary(self, pair_set):
        """B5: records sharing a phone number must be candidates."""
        pair = _canonical("KP_B5_A", "KP_B5_B")
        assert pair in pair_set, (
            "B5 (Phone Set) failed to generate candidate pair "
            "for records sharing phone '3125550100'"
        )

    def test_b5_shared_phone_secondary(self, pair_set):
        """B5: shared phone in secondary position must be candidates."""
        pair = _canonical("KP_B5B_A", "KP_B5B_B")
        assert pair in pair_set, (
            "B5 failed to generate candidate pair "
            "for records sharing phone '8475550302'"
        )

    def test_b6_email_exact_captures_pair(self, pair_set):
        """B6: records sharing exact email must be candidates."""
        pair = _canonical("KP_B6_A", "KP_B6_B")
        assert pair in pair_set, (
            "B6 (Email Exact) failed to generate candidate pair "
            "for records sharing 'shared.email.b6@test.com'"
        )

    def test_b7_geographic_fallback_captures_pair(self, pair_set):
        """B7: phonetic LN + same ZIP + same year must be candidates."""
        pair = _canonical("KP_B7_A", "KP_B7_B")
        assert pair in pair_set, (
            "B7 (DM-LN + ZIP + Year) failed to generate candidate pair "
            "for MARTINEZ/MARTINES with matching ZIP and year"
        )

    def test_b7_dob_variant_captures_pair(self, pair_set):
        """B7: GUTIERREZ/GUTIERRES with same ZIP and year must be candidates."""
        pair = _canonical("KP_B7B_A", "KP_B7B_B")
        assert pair in pair_set, (
            "B7 failed for GUTIERREZ/GUTIERRES geographic fallback pair"
        )

    def test_b8_both_names_phonetic_variant(self, pair_set):
        """B8: both FN and LN phonetic variants with same year — candidates."""
        pair = _canonical("KP_B8_A", "KP_B8_B")
        assert pair in pair_set, (
            "B8 (Soundex FN+LN+Year) failed to generate candidate pair "
            "for MIKHAIL/MICHAEL + KOVALENKO/KOVALENCO with same year"
        )

    def test_b8_catchall_schmidt_shmidt(self, pair_set):
        """B8: SCHMIDT/SHMIDT with same year must be candidates."""
        pair = _canonical("KP_B8B_A", "KP_B8B_B")
        assert pair in pair_set, (
            "B8 failed for SCHMIDT/SHMIDT catch-all pair"
        )

    def test_b9_ssn_last4_captures_pair(self, pair_set):
        """B9: same LN + FN + SSN-last4 with typo in first 5 — candidates."""
        pair = _canonical("KP_B9_A", "KP_B9_B")
        assert pair in pair_set, (
            "B9 (LN + FN + SSN-L4) failed to generate candidate pair "
            "for THOMPSONB9/DAVID with matching last 4 SSN digits"
        )

    def test_b9_second_pair(self, pair_set):
        """B9: second pair with same LN + FN + last-4 pattern."""
        pair = _canonical("KP_B9B_A", "KP_B9B_B")
        assert pair in pair_set, (
            "B9 failed for WILSONB9B/ROBERT second pair"
        )

    def test_multi_block_pair_captured(self, pair_set):
        """MULTI: pair captured by both B3 and B5 must appear as candidate."""
        pair = _canonical("KP_MULTI_A", "KP_MULTI_B")
        assert pair in pair_set, (
            "Multi-block pair (B3+B5) was not generated as a candidate"
        )

    def test_non_candidate_pair_absent(self, pair_set):
        """NONE: records sharing no blocking key must NOT be candidates."""
        pair = _canonical("KP_NONE_A", "KP_NONE_B")
        assert pair not in pair_set, (
            "Non-candidate pair appeared in candidate set — "
            "a blocking scheme is generating spurious pairs"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Block Source Metadata Test
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiBlockMetadata:
    """Verify source_blocks metadata is correct for known pairs."""

    def test_multi_block_pair_has_both_b3_and_b5(self, candidate_pairs):
        """The MULTI pair must show both B3 and B5 in source_blocks."""
        pair = _canonical("KP_MULTI_A", "KP_MULTI_B")
        row = candidate_pairs[
            (candidate_pairs["PATID_A"] == pair[0]) &
            (candidate_pairs["PATID_B"] == pair[1])
        ]
        assert len(row) == 1, "Multi-block pair not found in output"
        blocks = set(row.iloc[0]["source_blocks"].split("|"))
        assert "B3" in blocks, "B3 not in source_blocks for multi-block pair"
        assert "B5" in blocks, "B5 not in source_blocks for multi-block pair"
        assert row.iloc[0]["n_blocks"] >= 2, (
            "n_blocks should be >= 2 for multi-block pair"
        )

    def test_b1_pair_source_blocks_contains_b1(self, candidate_pairs):
        """B1 pair source_blocks must include B1."""
        pair = _canonical("KP_B1_A", "KP_B1_B")
        row = candidate_pairs[
            (candidate_pairs["PATID_A"] == pair[0]) &
            (candidate_pairs["PATID_B"] == pair[1])
        ]
        assert len(row) == 1
        blocks = set(row.iloc[0]["source_blocks"].split("|"))
        assert "B1" in blocks


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Scale Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetScale:
    """Verify output scale is within expected ranges for this dataset."""

    def test_total_pairs_within_expected_range(self, candidate_pairs):
        """
        With 270 unique background records (no shared keys) and 15 known
        pairs, total output should be at minimum 14 pairs (NONE pair excluded)
        and should not explode to an unreasonable number.
        """
        n_pairs = len(candidate_pairs)
        assert n_pairs >= 14, (
            f"Expected at least 14 candidate pairs, got {n_pairs}"
        )
        # Upper bound: with 300 records, max reasonable pairs << 300*299/2=44850
        # Background records share no keys, so pairs come only from known pairs
        # and any accidental key collisions. Set a generous upper bound.
        assert n_pairs < 5000, (
            f"Candidate pair count {n_pairs} seems unreasonably large "
            f"for a 300-record dataset with isolated background records"
        )

    def test_no_background_record_pairs(self, candidate_pairs):
        """
        Background records (BG_XXXX) are designed to share no blocking keys.
        No BG-to-BG pairs should appear in the output.
        """
        bg_pairs = candidate_pairs[
            candidate_pairs["PATID_A"].str.startswith("BG_") &
            candidate_pairs["PATID_B"].str.startswith("BG_")
        ]
        assert len(bg_pairs) == 0, (
            f"{len(bg_pairs)} background-to-background pairs found. "
            "Background records should share no blocking keys."
        )

    def test_all_blocks_contributed_pairs(self, candidate_pairs):
        """
        Every active block should have generated at least one pair
        in this dataset since we designed a known pair for each block.
        """
        all_blocks_used = set()
        for blocks_str in candidate_pairs["source_blocks"]:
            for blk in blocks_str.split("|"):
                all_blocks_used.add(blk)

        active_blocks = {"B1", "B3", "B4", "B5", "B6", "B7", "B8", "B9"}
        missing = active_blocks - all_blocks_used
        assert len(missing) == 0, (
            f"These blocks generated zero pairs: {missing}. "
            f"Each block should have captured its designed known pair."
        )

    def test_b2_not_in_any_source_blocks(self, candidate_pairs):
        """Removed block B2 must not appear in any pair's source_blocks."""
        for blocks_str in candidate_pairs["source_blocks"]:
            assert "B2" not in blocks_str.split("|"), (
                f"Removed block B2 found in source_blocks: '{blocks_str}'"
            )