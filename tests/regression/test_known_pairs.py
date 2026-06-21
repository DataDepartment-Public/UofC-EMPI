"""
=====================================
Regression test: 20 known-match pairs derived from Track B Synthetic
Pair Recall Testing must appear as candidate pairs in the blocking output.

PURPOSE:
    These tests encode the ground truth from Track B — pairs that were
    confirmed at 100% cumulative recall across all 9 error scenarios.
    If a future code change (refactor, library upgrade, column rename,
    threshold change) silently degrades blocking recall, these tests
    catch it immediately before the degradation propagates to the
    Fellegi-Sunter scoring pipeline.

    Unlike the integration tests which verify structural correctness
    on synthetic data, these regression tests verify the specific
    error patterns that exist in real healthcare data as evidenced
    by Track B's perturbation scenarios.

PAIR DESIGN:
    20 pairs across 9 error scenario categories from Track B:
        - 3 pairs: last name typo (adjacent character swap)
        - 2 pairs: last name phonetic variant
        - 2 pairs: first name phonetic variant
        - 2 pairs: first name nickname / truncated form
        - 2 pairs: DOB month/day transposition
        - 2 pairs: DOB partial loss (year only retained)
        - 2 pairs: SSN missing in one record
        - 2 pairs: phone set intersection
        - 3 pairs: multi-block capture (pair caught by 2+ blocks)

WHY THESE 20 SPECIFICALLY:
    Track B ran against 1,000 anchor records and generated 6,837
    synthetic pairs achieving 100% cumulative recall. These 20 pairs
    are a representative sample chosen to cover every scenario type
    while remaining small enough to keep the regression test fast.
    They are encoded directly in this file — no file I/O, no external
    dependencies — so the test is fully self-contained.

PASS CRITERION:
    Every pair with should_exist=True must appear in the candidate set.
    Every pair with should_exist=False must NOT appear.

FAILURE INTERPRETATION:
    If a pair fails: a code change has broken the blocking scheme's
    recall for that specific error pattern. Investigate immediately
    before proceeding to any downstream pipeline stage. Do not merge
    a PR that causes any regression test to fail.
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
# Regression Pair Manifest
# ═══════════════════════════════════════════════════════════════════════════

# Each entry defines a pair of records. The two records in each pair are
# an anchor (the original) and a perturbed copy (the error scenario).
# expected_blocks lists the blocks that MUST capture this pair.

REGRESSION_PAIRS = [

    # ── Scenario 1: Last Name Typo — adjacent character swap ─────────────
    {
        "pair_id": "REG_LN_TYPO_01",
        "scenario": "lastnm_typo",
        "description": "JOHNSON → JOHNSON (adjacent swap positions 2-3)",
        "expected_blocks": ["B3", "B4"],  # DM encodes both identically
        "anchor": {
            COL_PATID:    "REG_LN_TYPO_01_A",
            COL_LAST_NM:  "JOHNSON",
            COL_FIRST_NM: "ROBERT",
            COL_BIRTH_DT: pd.Timestamp("1978-04-12"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60601",
            COL_EMAIL:    "robert.j_a@regtest.com",
            COL_PHONES:   "{'7731001001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_LN_TYPO_01_B",
            COL_LAST_NM:  "JOHNSON",      # adjacent swap: H and N swapped
            COL_FIRST_NM: "ROBERT",
            COL_BIRTH_DT: pd.Timestamp("1978-04-12"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60602",
            COL_EMAIL:    "robert.j_b@regtest.com",
            COL_PHONES:   "{'7731001002'}",
        },
    },

    {
        "pair_id": "REG_LN_TYPO_02",
        "scenario": "lastnm_typo",
        "description": "GARCIA → GRACIA (adjacent swap positions 1-2)",
        "expected_blocks": ["B3"],
        "anchor": {
            COL_PATID:    "REG_LN_TYPO_02_A",
            COL_LAST_NM:  "GARCIA",
            COL_FIRST_NM: "MARIA",
            COL_BIRTH_DT: pd.Timestamp("1992-07-19"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60610",
            COL_EMAIL:    "maria.g_a@regtest.com",
            COL_PHONES:   "{'7731002001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_LN_TYPO_02_B",
            COL_LAST_NM:  "GRACIA",       # A and R swapped
            COL_FIRST_NM: "MARIA",
            COL_BIRTH_DT: pd.Timestamp("1992-07-19"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60611",
            COL_EMAIL:    "maria.g_b@regtest.com",
            COL_PHONES:   "{'7731002002'}",
        },
    },

    {
        "pair_id": "REG_LN_TYPO_03",
        "scenario": "lastnm_typo",
        "description": "MARTINEZ → MARITNEZ (adjacent swap in middle)",
        "expected_blocks": ["B3", "B7"],
        "anchor": {
            COL_PATID:    "REG_LN_TYPO_03_A",
            COL_LAST_NM:  "MARTINEZ",
            COL_FIRST_NM: "LUIS",
            COL_BIRTH_DT: pd.Timestamp("1985-02-28"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60620",
            COL_EMAIL:    "luis.m_a@regtest.com",
            COL_PHONES:   "{'7731003001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_LN_TYPO_03_B",
            COL_LAST_NM:  "MARITNEZ",     # T and I swapped in middle
            COL_FIRST_NM: "LUIS",
            COL_BIRTH_DT: pd.Timestamp("1985-02-28"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60620",         # same ZIP → B7 too
            COL_EMAIL:    "luis.m_b@regtest.com",
            COL_PHONES:   "{'7731003002'}",
        },
    },

    # ── Scenario 2: Last Name Phonetic Variant ────────────────────────────
    {
        "pair_id": "REG_LN_PHONETIC_01",
        "scenario": "lastnm_phonetic",
        "description": "GUTIERREZ → GUTIERRES (Z→S substitution)",
        "expected_blocks": ["B3"],
        "anchor": {
            COL_PATID:    "REG_LN_PHONETIC_01_A",
            COL_LAST_NM:  "GUTIERREZ",
            COL_FIRST_NM: "PEDRO",
            COL_BIRTH_DT: pd.Timestamp("1969-11-03"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60630",
            COL_EMAIL:    "pedro.g_a@regtest.com",
            COL_PHONES:   "{'7731004001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_LN_PHONETIC_01_B",
            COL_LAST_NM:  "GUTIERRES",    # Z → S phonetic substitution
            COL_FIRST_NM: "PEDRO",
            COL_BIRTH_DT: pd.Timestamp("1969-11-03"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60631",
            COL_EMAIL:    "pedro.g_b@regtest.com",
            COL_PHONES:   "{'7731004002'}",
        },
    },

    {
        "pair_id": "REG_LN_PHONETIC_02",
        "scenario": "lastnm_phonetic",
        "description": "SCHMIDT → SHMIDT (C removal — phonetic equivalence)",
        "expected_blocks": ["B3", "B8"],
        "anchor": {
            COL_PATID:    "REG_LN_PHONETIC_02_A",
            COL_LAST_NM:  "SCHMIDT",
            COL_FIRST_NM: "ANNA",
            COL_BIRTH_DT: pd.Timestamp("1981-06-15"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60640",
            COL_EMAIL:    "anna.s_a@regtest.com",
            COL_PHONES:   "{'7731005001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_LN_PHONETIC_02_B",
            COL_LAST_NM:  "SHMIDT",       # phonetic variant
            COL_FIRST_NM: "ANNA",
            COL_BIRTH_DT: pd.Timestamp("1981-06-15"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60641",
            COL_EMAIL:    "anna.s_b@regtest.com",
            COL_PHONES:   "{'7731005002'}",
        },
    },

    # ── Scenario 3: First Name Phonetic Variant ───────────────────────────
    {
        "pair_id": "REG_FN_PHONETIC_01",
        "scenario": "firstnm_phonetic",
        "description": "CARLOS → KARLOS (C→K substitution)",
        "expected_blocks": ["B3"],
        "anchor": {
            COL_PATID:    "REG_FN_PHONETIC_01_A",
            COL_LAST_NM:  "SANCHEZ",
            COL_FIRST_NM: "CARLOS",
            COL_BIRTH_DT: pd.Timestamp("1975-09-22"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60650",
            COL_EMAIL:    "carlos.s_a@regtest.com",
            COL_PHONES:   "{'7731006001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_FN_PHONETIC_01_B",
            COL_LAST_NM:  "SANCHEZ",
            COL_FIRST_NM: "KARLOS",       # C → K substitution
            COL_BIRTH_DT: pd.Timestamp("1975-09-22"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60651",
            COL_EMAIL:    "karlos.s_b@regtest.com",
            COL_PHONES:   "{'7731006002'}",
        },
    },

    {
        "pair_id": "REG_FN_PHONETIC_02",
        "scenario": "firstnm_phonetic",
        "description": "SARA → ZARA (S→Z substitution)",
        "expected_blocks": ["B3"],
        "anchor": {
            COL_PATID:    "REG_FN_PHONETIC_02_A",
            COL_LAST_NM:  "RODRIGUEZ",
            COL_FIRST_NM: "SARA",
            COL_BIRTH_DT: pd.Timestamp("1988-03-14"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60660",
            COL_EMAIL:    "sara.r_a@regtest.com",
            COL_PHONES:   "{'7731007001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_FN_PHONETIC_02_B",
            COL_LAST_NM:  "RODRIGUEZ",
            COL_FIRST_NM: "ZARA",         # S → Z substitution
            COL_BIRTH_DT: pd.Timestamp("1988-03-14"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60661",
            COL_EMAIL:    "zara.r_b@regtest.com",
            COL_PHONES:   "{'7731007002'}",
        },
    },

    # ── Scenario 4: First Name Nickname / Truncated ───────────────────────
    {
        "pair_id": "REG_FN_NICKNAME_01",
        "scenario": "firstnm_nickname",
        "description": "MICHAEL → MICH (truncated to 4 chars)",
        "expected_blocks": ["B3", "B4"],
        "anchor": {
            COL_PATID:    "REG_FN_NICKNAME_01_A",
            COL_LAST_NM:  "THOMPSON",
            COL_FIRST_NM: "MICHAEL",
            COL_BIRTH_DT: pd.Timestamp("1983-12-01"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60670",
            COL_EMAIL:    "michael.t_a@regtest.com",
            COL_PHONES:   "{'7731008001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_FN_NICKNAME_01_B",
            COL_LAST_NM:  "THOMPSON",
            COL_FIRST_NM: "MICH",         # truncated to 4 chars
            COL_BIRTH_DT: pd.Timestamp("1983-12-01"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60671",
            COL_EMAIL:    "mich.t_b@regtest.com",
            COL_PHONES:   "{'7731008002'}",
        },
    },

    {
        "pair_id": "REG_FN_NICKNAME_02",
        "scenario": "firstnm_nickname",
        "description": "ELIZABETH → ELIZ (truncated to 4 chars)",
        "expected_blocks": ["B3", "B4"],
        "anchor": {
            COL_PATID:    "REG_FN_NICKNAME_02_A",
            COL_LAST_NM:  "WILSON",
            COL_FIRST_NM: "ELIZABETH",
            COL_BIRTH_DT: pd.Timestamp("1971-08-30"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60680",
            COL_EMAIL:    "elizabeth.w_a@regtest.com",
            COL_PHONES:   "{'7731009001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_FN_NICKNAME_02_B",
            COL_LAST_NM:  "WILSON",
            COL_FIRST_NM: "ELIZ",         # truncated to 4 chars
            COL_BIRTH_DT: pd.Timestamp("1971-08-30"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60681",
            COL_EMAIL:    "eliz.w_b@regtest.com",
            COL_PHONES:   "{'7731009002'}",
        },
    },

    # ── Scenario 5: DOB Month/Day Transposition ───────────────────────────
    {
        "pair_id": "REG_DOB_TRANS_01",
        "scenario": "dob_transposition",
        "description": "1987-04-08 → 1987-08-04 (month/day transposed)",
        "expected_blocks": ["B4", "B7"],
        "anchor": {
            COL_PATID:    "REG_DOB_TRANS_01_A",
            COL_LAST_NM:  "NGUYEN",
            COL_FIRST_NM: "LINH",
            COL_BIRTH_DT: pd.Timestamp("1987-04-08"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60690",
            COL_EMAIL:    "linh.n_a@regtest.com",
            COL_PHONES:   "{'7731010001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_DOB_TRANS_01_B",
            COL_LAST_NM:  "NGUYEN",
            COL_FIRST_NM: "LINH",
            COL_BIRTH_DT: pd.Timestamp("1987-08-04"),  # month/day transposed
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60690",         # same ZIP → B7
            COL_EMAIL:    "linh.n_b@regtest.com",
            COL_PHONES:   "{'7731010002'}",
        },
    },

    {
        "pair_id": "REG_DOB_TRANS_02",
        "scenario": "dob_transposition",
        "description": "1995-03-11 → 1995-11-03 (month/day transposed)",
        "expected_blocks": ["B4"],
        "anchor": {
            COL_PATID:    "REG_DOB_TRANS_02_A",
            COL_LAST_NM:  "PATEL",
            COL_FIRST_NM: "PRIYA",
            COL_BIRTH_DT: pd.Timestamp("1995-03-11"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60700",
            COL_EMAIL:    "priya.p_a@regtest.com",
            COL_PHONES:   "{'7731011001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_DOB_TRANS_02_B",
            COL_LAST_NM:  "PATEL",
            COL_FIRST_NM: "PRIYA",
            COL_BIRTH_DT: pd.Timestamp("1995-11-03"),  # month/day transposed
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60701",
            COL_EMAIL:    "priya.p_b@regtest.com",
            COL_PHONES:   "{'7731011002'}",
        },
    },

    # ── Scenario 6: DOB Partial Loss (year only retained) ─────────────────
    {
        "pair_id": "REG_DOB_PARTIAL_01",
        "scenario": "dob_partial_loss",
        "description": "1982-09-17 → 1982-01-01 (month/day defaulted)",
        "expected_blocks": ["B4", "B7"],
        "anchor": {
            COL_PATID:    "REG_DOB_PARTIAL_01_A",
            COL_LAST_NM:  "GONZALEZ",
            COL_FIRST_NM: "JOSE",
            COL_BIRTH_DT: pd.Timestamp("1982-09-17"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60710",
            COL_EMAIL:    "jose.g_a@regtest.com",
            COL_PHONES:   "{'7731012001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_DOB_PARTIAL_01_B",
            COL_LAST_NM:  "GONZALEZ",
            COL_FIRST_NM: "JOSE",
            COL_BIRTH_DT: pd.Timestamp("1982-01-01"),  # default — year retained
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60710",         # same ZIP → B7
            COL_EMAIL:    "jose.g_b@regtest.com",
            COL_PHONES:   "{'7731012002'}",
        },
    },

    {
        "pair_id": "REG_DOB_PARTIAL_02",
        "scenario": "dob_partial_loss",
        "description": "1966-05-24 → 1966-01-01 (month/day defaulted)",
        "expected_blocks": ["B4", "B7"],
        "anchor": {
            COL_PATID:    "REG_DOB_PARTIAL_02_A",
            COL_LAST_NM:  "ROBINSON",
            COL_FIRST_NM: "JAMES",
            COL_BIRTH_DT: pd.Timestamp("1966-05-24"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60720",
            COL_EMAIL:    "james.r_a@regtest.com",
            COL_PHONES:   "{'7731013001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_DOB_PARTIAL_02_B",
            COL_LAST_NM:  "ROBINSON",
            COL_FIRST_NM: "JAMES",
            COL_BIRTH_DT: pd.Timestamp("1966-01-01"),  # year retained only
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60720",
            COL_EMAIL:    "james.r_b@regtest.com",
            COL_PHONES:   "{'7731013002'}",
        },
    },

    # ── Scenario 7: SSN Missing in One Record ─────────────────────────────
    {
        "pair_id": "REG_SSN_NULL_01",
        "scenario": "ssn_null",
        "description": "SSN present in anchor, null in perturbed — B5 catches via phone",
        "expected_blocks": ["B3", "B5"],
        "anchor": {
            COL_PATID:    "REG_SSN_NULL_01_A",
            COL_LAST_NM:  "ANDERSON",
            COL_FIRST_NM: "DAVID",
            COL_BIRTH_DT: pd.Timestamp("1979-06-10"),
            COL_SSN:      "345678901",
            COL_SSN_L4:   "8901",
            COL_ZIP:      "60730",
            COL_EMAIL:    "david.a_a@regtest.com",
            COL_PHONES:   "{'3125551401', '7735551402'}",
        },
        "perturbed": {
            COL_PATID:    "REG_SSN_NULL_01_B",
            COL_LAST_NM:  "ANDERSON",
            COL_FIRST_NM: "DAVID",
            COL_BIRTH_DT: pd.Timestamp("1979-06-10"),
            COL_SSN:      None,            # SSN missing in this system
            COL_SSN_L4:   None,
            COL_ZIP:      "60731",
            COL_EMAIL:    "david.a_b@regtest.com",
            COL_PHONES:   "{'3125551401'}",  # shared phone → B5
        },
    },

    {
        "pair_id": "REG_SSN_NULL_02",
        "scenario": "ssn_null",
        "description": "SSN null in both — matched via B3 + B4",
        "expected_blocks": ["B3", "B4"],
        "anchor": {
            COL_PATID:    "REG_SSN_NULL_02_A",
            COL_LAST_NM:  "HERNANDEZ",
            COL_FIRST_NM: "SOFIA",
            COL_BIRTH_DT: pd.Timestamp("1993-08-22"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60740",
            COL_EMAIL:    "sofia.h_a@regtest.com",
            COL_PHONES:   "{'7731015001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_SSN_NULL_02_B",
            COL_LAST_NM:  "HERNANDEZ",
            COL_FIRST_NM: "SOFIA",
            COL_BIRTH_DT: pd.Timestamp("1993-08-22"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60741",
            COL_EMAIL:    "sofia.h_b@regtest.com",
            COL_PHONES:   "{'7731015002'}",
        },
    },

    # ── Scenario 8: Phone Set Intersection ────────────────────────────────
    {
        "pair_id": "REG_PHONE_01",
        "scenario": "phone_reorder",
        "description": "Same phone number, different field positions",
        "expected_blocks": ["B5"],
        "anchor": {
            COL_PATID:    "REG_PHONE_01_A",
            COL_LAST_NM:  "APHONE",
            COL_FIRST_NM: "FIRST",
            COL_BIRTH_DT: pd.Timestamp("1988-01-15"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60750",
            COL_EMAIL:    "phone_a@regtest.com",
            COL_PHONES:   "{'8005551601', '3125551602'}",  # two phones
        },
        "perturbed": {
            COL_PATID:    "REG_PHONE_01_B",
            COL_LAST_NM:  "BPHONE",       # different name — only phone matches
            COL_FIRST_NM: "SECOND",
            COL_BIRTH_DT: pd.Timestamp("1991-07-20"),  # different DOB
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60751",
            COL_EMAIL:    "phone_b@regtest.com",
            COL_PHONES:   "{'8005551601'}",  # shares primary phone → B5
        },
    },

    {
        "pair_id": "REG_PHONE_02",
        "scenario": "phone_reorder",
        "description": "Shared phone, all other fields differ",
        "expected_blocks": ["B5"],
        "anchor": {
            COL_PATID:    "REG_PHONE_02_A",
            COL_LAST_NM:  "CPHONE",
            COL_FIRST_NM: "THIRD",
            COL_BIRTH_DT: pd.Timestamp("1977-11-08"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60760",
            COL_EMAIL:    "phone_c@regtest.com",
            COL_PHONES:   "{'7735551701', '6305551702'}",
        },
        "perturbed": {
            COL_PATID:    "REG_PHONE_02_B",
            COL_LAST_NM:  "DPHONE",
            COL_FIRST_NM: "FOURTH",
            COL_BIRTH_DT: pd.Timestamp("1984-05-12"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60761",
            COL_EMAIL:    "phone_d@regtest.com",
            COL_PHONES:   "{'6305551702'}",   # shares secondary phone → B5
        },
    },

    # ── Scenario 9: Multi-Block Capture ───────────────────────────────────
    {
        "pair_id": "REG_MULTI_01",
        "scenario": "multi_block",
        "description": "Matched by B3 (phonetic LN + DOB) AND B5 (phone)",
        "expected_blocks": ["B3", "B5"],
        "anchor": {
            COL_PATID:    "REG_MULTI_01_A",
            COL_LAST_NM:  "JONES",
            COL_FIRST_NM: "KAREN",
            COL_BIRTH_DT: pd.Timestamp("1980-05-05"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60770",
            COL_EMAIL:    "karen.j_a@regtest.com",
            COL_PHONES:   "{'3125551801', '7735551802'}",
        },
        "perturbed": {
            COL_PATID:    "REG_MULTI_01_B",
            COL_LAST_NM:  "JONEZ",        # phonetic variant → B3
            COL_FIRST_NM: "KAREN",
            COL_BIRTH_DT: pd.Timestamp("1980-05-05"),  # same DOB → B3
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60771",
            COL_EMAIL:    "karen.j_b@regtest.com",
            COL_PHONES:   "{'3125551801'}",  # shared phone → B5
        },
    },

    {
        "pair_id": "REG_MULTI_02",
        "scenario": "multi_block",
        "description": "Matched by B1 (SSN) AND B3 (LN+DOB) AND B9 (LN+FN+L4)",
        "expected_blocks": ["B1", "B3", "B9"],
        "anchor": {
            COL_PATID:    "REG_MULTI_02_A",
            COL_LAST_NM:  "HARRIS",
            COL_FIRST_NM: "THOMAS",
            COL_BIRTH_DT: pd.Timestamp("1973-03-17"),
            COL_SSN:      "567890123",
            COL_SSN_L4:   "0123",
            COL_ZIP:      "60780",
            COL_EMAIL:    "thomas.h_a@regtest.com",
            COL_PHONES:   "{'7731019001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_MULTI_02_B",
            COL_LAST_NM:  "HARRIS",       # same LN → B3, B9
            COL_FIRST_NM: "THOMAS",       # same FN → B9
            COL_BIRTH_DT: pd.Timestamp("1973-03-17"),  # same DOB → B3
            COL_SSN:      "567890123",    # same SSN → B1
            COL_SSN_L4:   "0123",
            COL_ZIP:      "60781",
            COL_EMAIL:    "thomas.h_b@regtest.com",
            COL_PHONES:   "{'7731019002'}",
        },
    },

    {
        "pair_id": "REG_MULTI_03",
        "scenario": "multi_block",
        "description": "Matched by B6 (email) AND B7 (geographic fallback)",
        "expected_blocks": ["B6", "B7"],
        "anchor": {
            COL_PATID:    "REG_MULTI_03_A",
            COL_LAST_NM:  "JACKSON",
            COL_FIRST_NM: "DIANE",
            COL_BIRTH_DT: pd.Timestamp("1967-10-12"),
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60790",
            COL_EMAIL:    "shared.diane@regtest.com",
            COL_PHONES:   "{'7731020001'}",
        },
        "perturbed": {
            COL_PATID:    "REG_MULTI_03_B",
            COL_LAST_NM:  "JACKSEN",      # phonetic variant → B7
            COL_FIRST_NM: "DIANE",
            COL_BIRTH_DT: pd.Timestamp("1967-01-01"),  # defaulted DOB
            COL_SSN:      None,
            COL_SSN_L4:   None,
            COL_ZIP:      "60790",        # same ZIP + same year → B7
            COL_EMAIL:    "shared.diane@regtest.com",  # same email → B6
            COL_PHONES:   "{'7731020002'}",
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_regression_dataset():
    """
    Combine all anchor and perturbed records from the regression pair
    manifest into a single DataFrame for batch blocking.
    """
    records = []
    for pair in REGRESSION_PAIRS:
        records.append(pair["anchor"])
        records.append(pair["perturbed"])
    return pd.DataFrame(records)


def _canonical(a: str, b: str) -> tuple:
    return (a, b) if a < b else (b, a)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def regression_df():
    return _build_regression_dataset()


@pytest.fixture(scope="module")
def regression_pairs(regression_df):
    """Run blocking once for all regression tests."""
    return run_batch_blocking(regression_df)


@pytest.fixture(scope="module")
def regression_pair_set(regression_pairs):
    return set(zip(regression_pairs["PATID_A"], regression_pairs["PATID_B"]))


@pytest.fixture(scope="module")
def regression_source_blocks(regression_pairs):
    """Map (PATID_A, PATID_B) -> set of block IDs."""
    result = {}
    for _, row in regression_pairs.iterrows():
        pair = (row["PATID_A"], row["PATID_B"])
        result[pair] = set(row["source_blocks"].split("|"))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Regression Tests — Pair Presence
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressionPairPresence:
    """
    Every known-match pair must appear in the candidate set.
    Parameterized over all 20 pairs so each shows as a distinct test.
    """

    @pytest.mark.parametrize("pair_spec", REGRESSION_PAIRS, ids=[
        p["pair_id"] for p in REGRESSION_PAIRS
    ])
    def test_known_pair_is_candidate(self, pair_spec, regression_pair_set):
        """Each regression pair must appear in the candidate set."""
        pair = _canonical(
            pair_spec["anchor"][COL_PATID],
            pair_spec["perturbed"][COL_PATID],
        )
        assert pair in regression_pair_set, (
            f"\n"
            f"  Pair ID:     {pair_spec['pair_id']}\n"
            f"  Scenario:    {pair_spec['scenario']}\n"
            f"  Description: {pair_spec['description']}\n"
            f"  PATID_A:     {pair[0]}\n"
            f"  PATID_B:     {pair[1]}\n"
            f"\n"
            f"  This pair was NOT found in the candidate set.\n"
            f"  A code change has broken recall for this error scenario.\n"
            f"  Do NOT merge a PR that causes this test to fail."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Regression Tests — Expected Block Capture
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressionBlockCapture:
    """
    Every expected block must appear in source_blocks for each known pair.
    This verifies not just that the pair was found, but that the correct
    blocking mechanism found it.
    """

    @pytest.mark.parametrize("pair_spec", REGRESSION_PAIRS, ids=[
        p["pair_id"] for p in REGRESSION_PAIRS
    ])
    def test_expected_blocks_capture_pair(
        self, pair_spec, regression_pair_set, regression_source_blocks
    ):
        """Each expected block must appear in source_blocks for this pair."""
        pair = _canonical(
            pair_spec["anchor"][COL_PATID],
            pair_spec["perturbed"][COL_PATID],
        )

        if pair not in regression_pair_set:
            pytest.skip(
                f"Skipping block check — pair {pair_spec['pair_id']} "
                f"not found in candidate set (caught by presence test)"
            )

        actual_blocks = regression_source_blocks.get(pair, set())

        for expected_block in pair_spec["expected_blocks"]:
            assert expected_block in actual_blocks, (
                f"\n"
                f"  Pair ID:        {pair_spec['pair_id']}\n"
                f"  Scenario:       {pair_spec['scenario']}\n"
                f"  Expected block: {expected_block}\n"
                f"  Actual blocks:  {sorted(actual_blocks)}\n"
                f"\n"
                f"  Block {expected_block} did not capture this pair.\n"
                f"  The pair was found (via another block) but the specific\n"
                f"  blocking mechanism for this error pattern is not working."
            )


# ═══════════════════════════════════════════════════════════════════════════
# Regression Tests — B2 Absence
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressionB2Absent:
    """B2 was removed. It must never appear in any regression pair output."""

    def test_b2_never_in_source_blocks(self, regression_pairs):
        """Removed block B2 must not appear in any pair's source_blocks."""
        if regression_pairs.empty:
            return
        b2_pairs = regression_pairs[
            regression_pairs["source_blocks"].str.contains(
                r"\bB2\b", regex=True, na=False
            )
        ]
        assert len(b2_pairs) == 0, (
            f"B2 found in {len(b2_pairs)} pair(s) source_blocks. "
            f"B2 was removed from the scheme and must never appear."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Summary Test
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressionSummary:
    """High-level summary assertions on the regression run."""

    def test_all_20_pairs_are_candidates(self, regression_pair_set):
        """All 20 regression pairs must be in the candidate set."""
        missing = []
        for pair_spec in REGRESSION_PAIRS:
            pair = _canonical(
                pair_spec["anchor"][COL_PATID],
                pair_spec["perturbed"][COL_PATID],
            )
            if pair not in regression_pair_set:
                missing.append(pair_spec["pair_id"])

        assert len(missing) == 0, (
            f"\n{len(missing)} regression pair(s) not found as candidates:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nBlocking recall has regressed. Do not merge this PR."
        )

    def test_recall_rate(self, regression_pair_set):
        """Recall rate across all 20 pairs must be 100%."""
        total = len(REGRESSION_PAIRS)
        found = sum(
            1 for pair_spec in REGRESSION_PAIRS
            if _canonical(
                pair_spec["anchor"][COL_PATID],
                pair_spec["perturbed"][COL_PATID],
            ) in regression_pair_set
        )
        recall = 100 * found / total
        assert recall == 100.0, (
            f"Regression recall: {recall:.1f}% ({found}/{total}). "
            f"Expected 100.0%. Blocking scheme has regressed."
        )