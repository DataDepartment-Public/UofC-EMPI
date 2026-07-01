"""tests/unit/test_fs_enhanced_4_gate.py

Unit tests for apply_corroboration_gate (corroboration_gate.py).
All frames are tiny synthetic fixtures — no PHI, no Splink training.

Column name constants are imported from src.contracts to guard against
rename drift (the gate itself imports them the same way).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.contracts import ADDRESS1, EMAIL, PATID, PHONES, SSN
from models.experiments.fs_splink_enhanced_4.corroboration_gate import (
    CORROBORATION_COL,
    DEMOTED_REASON,
    apply_corroboration_gate,
)

# ── Shared tiny df_clean builders ────────────────────────────────────────────


def _mk_clean(*rows) -> pd.DataFrame:
    """Build a minimal df_clean from keyword-dicts. Each dict must include PATID."""
    defaults = {
        PATID: None,
        SSN: None,
        EMAIL: None,
        ADDRESS1: None,
        PHONES: None,
    }
    records = [{**defaults, **row} for row in rows]
    return pd.DataFrame(records)


def _mk_classified(**kwargs) -> pd.DataFrame:
    """Build a single-row df_classified. Caller must supply PATID_A, PATID_B, tier."""
    return pd.DataFrame(
        {
            "PATID_A": [kwargs["PATID_A"]],
            "PATID_B": [kwargs["PATID_B"]],
            "classification_tier": [kwargs["tier"]],
        }
    )


# ── Person-unique corroboration: SSN exact ───────────────────────────────────

def test_ssn_exact_agreement_keeps_auto_merge():
    """SSN exact match is person-unique → auto_merge stays auto_merge."""
    df_clean = _mk_clean(
        {PATID: "P1", SSN: "123456789"},
        {PATID: "P2", SSN: "123456789"},
    )
    df_classified = _mk_classified(PATID_A="P1", PATID_B="P2", tier="auto_merge")
    out = apply_corroboration_gate(df_classified, df_clean)
    assert out["classification_tier"].iloc[0] == "auto_merge"
    assert pd.isna(out[CORROBORATION_COL].iloc[0])


# ── Person-unique corroboration: Email exact ──────────────────────────────────

def test_email_exact_agreement_keeps_auto_merge():
    """Email exact match is person-unique → auto_merge stays auto_merge."""
    df_clean = _mk_clean(
        {PATID: "P1", EMAIL: "alice@example.com"},
        {PATID: "P2", EMAIL: "alice@example.com"},
    )
    df_classified = _mk_classified(PATID_A="P1", PATID_B="P2", tier="auto_merge")
    out = apply_corroboration_gate(df_classified, df_clean)
    assert out["classification_tier"].iloc[0] == "auto_merge"
    assert pd.isna(out[CORROBORATION_COL].iloc[0])


# ── Household corroboration: both phone AND address ───────────────────────────

def test_two_household_signals_keep_auto_merge():
    """Shared phone AND exact address = two household signals → stays auto_merge."""
    df_clean = _mk_clean(
        {PATID: "P1", PHONES: "{'5551234'}", ADDRESS1: "100 Maple Ave"},
        {PATID: "P2", PHONES: "{'5551234'}", ADDRESS1: "100 Maple Ave"},
    )
    df_classified = _mk_classified(PATID_A="P1", PATID_B="P2", tier="auto_merge")
    out = apply_corroboration_gate(df_classified, df_clean)
    assert out["classification_tier"].iloc[0] == "auto_merge"
    assert pd.isna(out[CORROBORATION_COL].iloc[0])


# ── Single household: phone only (no address match) → DEMOTED ────────────────

def test_single_household_phone_only_demotes():
    """Shared phone but different addresses = household_count == 1 → DEMOTED."""
    df_clean = _mk_clean(
        {PATID: "P1", PHONES: "{'5551234'}", ADDRESS1: "100 Maple Ave"},
        {PATID: "P2", PHONES: "{'5551234'}", ADDRESS1: "200 Oak Dr"},
    )
    df_classified = _mk_classified(PATID_A="P1", PATID_B="P2", tier="auto_merge")
    out = apply_corroboration_gate(df_classified, df_clean)
    assert out["classification_tier"].iloc[0] == "human_review"
    assert out[CORROBORATION_COL].iloc[0] == DEMOTED_REASON


# ── Single household: address only (no shared phone) → DEMOTED ───────────────

def test_single_household_address_only_demotes():
    """Exact address but no phone set = household_count == 1 → DEMOTED."""
    df_clean = _mk_clean(
        {PATID: "P1", PHONES: None, ADDRESS1: "100 Maple Ave"},
        {PATID: "P2", PHONES: None, ADDRESS1: "100 Maple Ave"},
    )
    df_classified = _mk_classified(PATID_A="P1", PATID_B="P2", tier="auto_merge")
    out = apply_corroboration_gate(df_classified, df_clean)
    assert out["classification_tier"].iloc[0] == "human_review"
    assert out[CORROBORATION_COL].iloc[0] == DEMOTED_REASON


# ── No corroboration: DOB+name-only style pair → DEMOTED ─────────────────────

def test_no_corroboration_demotes():
    """Conflicting SSN/Email, no phone, no address → DEMOTED (name/DOB alone not enough)."""
    df_clean = _mk_clean(
        {PATID: "P1", SSN: "111111111", EMAIL: "a@x.com", PHONES: None, ADDRESS1: None},
        {PATID: "P2", SSN: "222222222", EMAIL: "b@x.com", PHONES: None, ADDRESS1: None},
    )
    df_classified = _mk_classified(PATID_A="P1", PATID_B="P2", tier="auto_merge")
    out = apply_corroboration_gate(df_classified, df_clean)
    assert out["classification_tier"].iloc[0] == "human_review"
    assert out[CORROBORATION_COL].iloc[0] == DEMOTED_REASON


# ── human_review pair: gate never touches non-auto_merge ─────────────────────

def test_human_review_pair_is_untouched():
    """A pair already in human_review must remain human_review with corroboration None."""
    df_clean = _mk_clean(
        {PATID: "P1", SSN: None, EMAIL: None, PHONES: None, ADDRESS1: None},
        {PATID: "P2", SSN: None, EMAIL: None, PHONES: None, ADDRESS1: None},
    )
    df_classified = _mk_classified(PATID_A="P1", PATID_B="P2", tier="human_review")
    out = apply_corroboration_gate(df_classified, df_clean)
    assert out["classification_tier"].iloc[0] == "human_review"
    assert pd.isna(out[CORROBORATION_COL].iloc[0])


# ── Inputs not mutated ────────────────────────────────────────────────────────

def test_inputs_not_mutated():
    """apply_corroboration_gate must return a copy; original frames unchanged."""
    df_clean = _mk_clean(
        {PATID: "P1", SSN: None},
        {PATID: "P2", SSN: None},
    )
    df_classified = _mk_classified(PATID_A="P1", PATID_B="P2", tier="auto_merge")

    original_tier = df_classified["classification_tier"].copy()
    _ = apply_corroboration_gate(df_classified, df_clean)

    # CORROBORATION_COL must NOT have been added to the original df_classified.
    assert CORROBORATION_COL not in df_classified.columns
    # classification_tier in the original must be unchanged.
    pd.testing.assert_series_equal(
        df_classified["classification_tier"], original_tier
    )


# ── Graceful no-op when all corroborating columns missing ────────────────────

def test_graceful_noop_when_all_corroborating_columns_absent():
    """df_clean with only PATID (no SSN/Email/Phone/Address) → no demotions, all None."""
    df_clean = pd.DataFrame({PATID: ["P1", "P2"]})
    df_classified = pd.DataFrame(
        {
            "PATID_A": ["P1"],
            "PATID_B": ["P2"],
            "classification_tier": ["auto_merge"],
        }
    )
    out = apply_corroboration_gate(df_classified, df_clean)
    # Tier must be unchanged (no wrongful demotion).
    assert out["classification_tier"].iloc[0] == "auto_merge"
    # CORROBORATION_COL present but all None.
    assert CORROBORATION_COL in out.columns
    assert out[CORROBORATION_COL].isna().all()


# ── Empty df_classified ───────────────────────────────────────────────────────

def test_empty_df_classified_returns_empty_with_corroboration_col():
    """Empty input must return an empty frame that already has CORROBORATION_COL."""
    df_clean = _mk_clean({PATID: "P1"})
    empty = pd.DataFrame(columns=["PATID_A", "PATID_B", "classification_tier"])
    out = apply_corroboration_gate(empty, df_clean)
    assert len(out) == 0
    assert CORROBORATION_COL in out.columns


# ── Multi-row + non-default index: alignment / vectorization guard ───────────

def test_multi_row_non_default_index_aligns_outcomes():
    """A mixed batch on a NON-default index: each row must land in the correct
    tier/reason. Guards against positional (.values) misalignment that a
    single-row test can never surface, and covers no_match-untouched too."""
    df_clean = _mk_clean(
        {PATID: "P1", SSN: "123456789"},                                  # keep (SSN)
        {PATID: "P2", SSN: "123456789"},
        {PATID: "P3", PHONES: "{'5551234'}", ADDRESS1: "100 Maple Ave"},  # demote (phone only)
        {PATID: "P4", PHONES: "{'5551234'}", ADDRESS1: "200 Oak Dr"},
        {PATID: "P5", PHONES: "{'5559999'}", ADDRESS1: "300 Pine St"},    # keep (phone+addr)
        {PATID: "P6", PHONES: "{'5559999'}", ADDRESS1: "300 Pine St"},
        {PATID: "P7"},                                                    # human_review untouched
        {PATID: "P8"},
        {PATID: "P9"},                                                    # no_match untouched
        {PATID: "P10"},
    )
    df_classified = pd.DataFrame(
        {
            "PATID_A": ["P1", "P3", "P5", "P7", "P9"],
            "PATID_B": ["P2", "P4", "P6", "P8", "P10"],
            "classification_tier": [
                "auto_merge", "auto_merge", "auto_merge", "human_review", "no_match",
            ],
        },
        index=[10, 20, 30, 40, 50],  # non-default index
    )
    out = apply_corroboration_gate(df_classified, df_clean)

    assert out.loc[10, "classification_tier"] == "auto_merge"
    assert pd.isna(out.loc[10, CORROBORATION_COL])
    assert out.loc[20, "classification_tier"] == "human_review"
    assert out.loc[20, CORROBORATION_COL] == DEMOTED_REASON
    assert out.loc[30, "classification_tier"] == "auto_merge"
    assert pd.isna(out.loc[30, CORROBORATION_COL])
    assert out.loc[40, "classification_tier"] == "human_review"
    assert pd.isna(out.loc[40, CORROBORATION_COL])
    assert out.loc[50, "classification_tier"] == "no_match"
    assert pd.isna(out.loc[50, CORROBORATION_COL])


# ── ValueError when df_clean lacks PATID ─────────────────────────────────────

def test_raises_valueerror_when_df_clean_lacks_patid():
    """df_clean missing PATID must raise ValueError immediately."""
    df_clean_no_patid = pd.DataFrame({SSN: ["123456789"]})
    df_classified = _mk_classified(PATID_A="P1", PATID_B="P2", tier="auto_merge")
    with pytest.raises(ValueError, match="PATID"):
        apply_corroboration_gate(df_classified, df_clean_no_patid)
