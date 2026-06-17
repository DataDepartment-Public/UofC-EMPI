"""
tests/unit/test_fs_enhanced_e4_levels.py

Structural tests for the E4 comparison-level additions:
  - FirstNM and LastNM each get an explicit "JW < 0.5" mismatch level
    inserted between JW>=0.85 and the trailing ElseLevel.
  - SSN gets an explicit "Both populated, full 9-digit mismatch" level
    inserted between "Last 4 digits match" and the trailing ElseLevel.
  - A new Household_discount composite comparison is added (gated on
    include_address — production has the address columns, synthetic does
    not).

All four new levels have locked m/u priors per manual_priors. We test that
the priors are present on the right levels here; the integration test that
the priors survive an EM pass is in tests/regression/.
"""

from __future__ import annotations

import pytest

from models.experiments.fs_splink_enhanced.fellegi_sunter_enhanced import (
    build_settings,
)
from models.experiments.fs_splink_enhanced.manual_priors import (
    FIRSTNM_JW_LT_05_MU,
    LASTNM_JW_LT_05_MU,
    SSN_FULL_MISMATCH_MU,
    HOUSEHOLD_DISCOUNT_MU,
)


# ============================================================================
# FirstNM / LastNM — explicit JW < 0.5 mismatch level
# ============================================================================
@pytest.mark.parametrize(
    "ocn,priors",
    [
        ("FirstNM_clean", FIRSTNM_JW_LT_05_MU),
        ("LastNM_clean", LASTNM_JW_LT_05_MU),
    ],
)
def test_name_comparison_has_jw_lt_05_level_with_locked_priors(ocn, priors):
    settings = build_settings()
    comp = next(c for c in settings["comparisons"] if c["output_column_name"] == ocn)
    labels = [lv.get("label_for_charts") for lv in comp["comparison_levels"]]
    target_label = next(iter(priors))  # exactly one entry per prior dict
    assert target_label in labels, (
        f"comparison {ocn} missing JW<0.5 level (have: {labels})"
    )
    # Position: should be at index -2 (right before ElseLevel at -1).
    assert comp["comparison_levels"][-2]["label_for_charts"] == target_label
    # Locked priors present on that level.
    lv = comp["comparison_levels"][-2]
    m_expected, u_expected = priors[target_label]
    assert lv["m_probability"] == pytest.approx(m_expected)
    assert lv["u_probability"] == pytest.approx(u_expected)
    assert lv["fix_m_probability"] is True
    assert lv["fix_u_probability"] is True


# ============================================================================
# SSN — explicit 5-9 conflict level
# ============================================================================
def test_ssn_comparison_has_explicit_mismatch_level_with_locked_priors():
    settings = build_settings()
    ssn = next(c for c in settings["comparisons"] if c["output_column_name"] == "SSN")
    labels = [lv.get("label_for_charts") for lv in ssn["comparison_levels"]]
    target_label = "Both populated, full 9-digit mismatch"
    assert target_label in labels, f"SSN missing 5-9 conflict level (have: {labels})"
    # Position right before Else.
    assert ssn["comparison_levels"][-2]["label_for_charts"] == target_label

    lv = ssn["comparison_levels"][-2]
    m_expected, u_expected = SSN_FULL_MISMATCH_MU[target_label]
    assert lv["m_probability"] == pytest.approx(m_expected)
    assert lv["u_probability"] == pytest.approx(u_expected)
    assert lv["fix_m_probability"] is True
    assert lv["fix_u_probability"] is True


# ============================================================================
# Household_discount — composite comparison
# ============================================================================
def test_household_discount_comparison_present_when_address_included():
    settings = build_settings(include_address=True)
    ocns = [c["output_column_name"] for c in settings["comparisons"]]
    assert "Household_discount" in ocns


def test_household_discount_comparison_absent_when_address_skipped():
    """Household_discount is gated together with Address — both need the
    address columns, so when include_address=False both are skipped."""
    settings = build_settings(include_address=False)
    ocns = [c["output_column_name"] for c in settings["comparisons"]]
    assert "Household_discount" not in ocns
    assert "Address" not in ocns


def test_household_discount_has_locked_prior():
    settings = build_settings()
    hd = next(
        c for c in settings["comparisons"]
        if c["output_column_name"] == "Household_discount"
    )
    # 3 levels: null, fire, else
    assert len(hd["comparison_levels"]) == 3
    target_label = "Household indicator without identity match"
    fire_level = next(
        lv for lv in hd["comparison_levels"]
        if lv.get("label_for_charts") == target_label
    )
    m_expected, u_expected = HOUSEHOLD_DISCOUNT_MU[target_label]
    assert fire_level["m_probability"] == pytest.approx(m_expected)
    assert fire_level["u_probability"] == pytest.approx(u_expected)
    assert fire_level["fix_m_probability"] is True
    assert fire_level["fix_u_probability"] is True


# ============================================================================
# Top-level comparison count after all E4 additions
# ============================================================================
def test_total_comparison_count_with_all_additions():
    """Baseline 7 + Address (E3) + Household_discount (E4) = 9.
    Levels added to existing comparisons (FirstNM, LastNM, SSN JW<0.5 /
    explicit-mismatch) do not add new comparisons, only new levels within."""
    settings = build_settings(include_address=True)
    assert len(settings["comparisons"]) == 9


def test_total_comparison_count_without_address():
    """Without address columns, both Address and Household_discount drop."""
    settings = build_settings(include_address=False)
    assert len(settings["comparisons"]) == 7
