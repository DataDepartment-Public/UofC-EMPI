"""
tests/unit/test_fs_enhanced_e4_levels.py

Structural tests for the E4 comparison-level additions via FSEnhanced and its
``ENHANCED_REGISTRY`` (or the ``build_registry`` factory):

  - FirstNM and LastNM each get an explicit "JW < 0.5" mismatch level
    inserted between JW>=0.85 and the trailing ElseLevel.
  - SSN gets an explicit "Both populated, full 9-digit mismatch" level
    inserted between "Last 4 digits match" and the trailing ElseLevel.
  - A new Household_discount composite comparison is added (gated on
    include_address — production has the address columns, synthetic does not).
  - Address comparison present when include_address=True.

All four new levels have locked m/u priors per manual_priors. We test that
the priors are present on the right levels here; the integration test that
the priors survive an EM pass is in tests/regression/.
"""

from __future__ import annotations

import pytest

from models.experiments.fs_splink_enhanced.comparisons import (
    ENHANCED_REGISTRY,
    build_registry,
)
from models.experiments.fs_splink_enhanced.manual_priors import (
    FIRSTNM_JW_LT_05_MU,
    LASTNM_JW_LT_05_MU,
    SSN_FULL_MISMATCH_MU,
    HOUSEHOLD_DISCOUNT_MU,
)


# ============================================================================
# Registry contents — with / without address
# ============================================================================
def test_registry_with_address_has_9_comparisons():
    """Baseline 7 + Household_discount (E4) + Address (E3) = 9."""
    reg = build_registry(include_address=True)
    assert len(reg) == 9


def test_registry_without_address_has_7_comparisons():
    """Without address columns, both Address and Household_discount drop."""
    reg = build_registry(include_address=False)
    assert len(reg) == 7
    assert "Household_discount" not in reg
    assert "Address" not in reg


def test_enhanced_registry_alias_equals_full_registry():
    """ENHANCED_REGISTRY is the include_address=True shortcut."""
    assert ENHANCED_REGISTRY.names() == build_registry(include_address=True).names()


def test_registry_names_order_with_address():
    names = build_registry(include_address=True).names()
    assert names == [
        "FirstNM", "LastNM", "BirthDT", "SSN", "Email", "Phones", "ZIP",
        "Household_discount", "Address",
    ]


def test_registry_names_order_without_address():
    names = build_registry(include_address=False).names()
    assert names == [
        "FirstNM", "LastNM", "BirthDT", "SSN", "Email", "Phones", "ZIP",
    ]


# ============================================================================
# Verify E4 levels are present in the built dicts
# ============================================================================
# NOTE: locked m/u priors are applied by apply_manual_priors() inside
# FSEnhanced.build_settings(), NOT in the individual builder functions.
# Tests that check priors must use FSEnhanced().build_settings(), not
# the raw builder().

def _get_comparison_from_settings(include_address: bool, name: str) -> dict:
    """Get a comparison dict from FSEnhanced.build_settings() (priors applied)."""
    from models.experiments.fs_splink_enhanced.fs_enhanced import FSEnhanced
    settings = FSEnhanced(include_address=include_address).build_settings()
    output_col_names = {
        "FirstNM": "FirstNM_clean",
        "LastNM": "LastNM_clean",
        "BirthDT": "_dob_str",
        "SSN": "SSN",
        "Email": "Email",
        "Phones": "Phones_array",
        "ZIP": "ZIP",
        "Household_discount": "Household_discount",
        "Address": "Address",
    }
    ocn = output_col_names.get(name, name)
    return next(c for c in settings["comparisons"] if c["output_column_name"] == ocn)


@pytest.mark.parametrize(
    "comp_name,prior_dict",
    [
        ("FirstNM",  FIRSTNM_JW_LT_05_MU),
        ("LastNM",   LASTNM_JW_LT_05_MU),
    ],
)
def test_name_comparison_has_jw_lt_05_level_with_locked_priors(comp_name, prior_dict):
    comp = _get_comparison_from_settings(include_address=False, name=comp_name)
    labels = [lv.get("label_for_charts") for lv in comp["comparison_levels"]]
    target_label = next(iter(prior_dict))
    assert target_label in labels, (
        f"comparison {comp_name} missing JW<0.5 level (have: {labels})"
    )
    # Position: should be at index -2 (right before ElseLevel at -1).
    assert comp["comparison_levels"][-2]["label_for_charts"] == target_label
    # Locked priors present on that level.
    lv = comp["comparison_levels"][-2]
    m_expected, u_expected = prior_dict[target_label]
    assert lv["m_probability"] == pytest.approx(m_expected)
    assert lv["u_probability"] == pytest.approx(u_expected)
    assert lv["fix_m_probability"] is True
    assert lv["fix_u_probability"] is True


def test_ssn_comparison_has_explicit_mismatch_level_with_locked_priors():
    comp = _get_comparison_from_settings(include_address=False, name="SSN")
    labels = [lv.get("label_for_charts") for lv in comp["comparison_levels"]]
    target_label = "Both populated, full 9-digit mismatch"
    assert target_label in labels, f"SSN missing 5-9 conflict level (have: {labels})"
    # Position right before Else.
    assert comp["comparison_levels"][-2]["label_for_charts"] == target_label

    lv = comp["comparison_levels"][-2]
    m_expected, u_expected = SSN_FULL_MISMATCH_MU[target_label]
    assert lv["m_probability"] == pytest.approx(m_expected)
    assert lv["u_probability"] == pytest.approx(u_expected)
    assert lv["fix_m_probability"] is True
    assert lv["fix_u_probability"] is True


# ============================================================================
# Household_discount — composite comparison
# ============================================================================
def test_household_discount_comparison_present_when_address_included():
    reg = build_registry(include_address=True)
    assert "Household_discount" in reg


def test_household_discount_comparison_absent_when_address_skipped():
    """Household_discount is gated together with Address."""
    reg = build_registry(include_address=False)
    assert "Household_discount" not in reg
    assert "Address" not in reg


def test_household_discount_has_locked_prior():
    comp = _get_comparison_from_settings(include_address=True, name="Household_discount")
    # 3 levels: null, fire, else
    assert len(comp["comparison_levels"]) == 3
    target_label = "Household indicator without identity match"
    fire_level = next(
        lv for lv in comp["comparison_levels"]
        if lv.get("label_for_charts") == target_label
    )
    m_expected, u_expected = HOUSEHOLD_DISCOUNT_MU[target_label]
    assert fire_level["m_probability"] == pytest.approx(m_expected)
    assert fire_level["u_probability"] == pytest.approx(u_expected)
    assert fire_level["fix_m_probability"] is True
    assert fire_level["fix_u_probability"] is True


# ============================================================================
# FSEnhanced — registry wired correctly at instance level
# ============================================================================
def test_fs_enhanced_registry_with_address():
    from models.experiments.fs_splink_enhanced.fs_enhanced import FSEnhanced
    m = FSEnhanced(include_address=True)
    assert len(m.registry) == 9
    assert "Household_discount" in m.registry
    assert "Address" in m.registry


def test_fs_enhanced_registry_without_address():
    from models.experiments.fs_splink_enhanced.fs_enhanced import FSEnhanced
    m = FSEnhanced(include_address=False)
    assert len(m.registry) == 7
    assert "Household_discount" not in m.registry
    assert "Address" not in m.registry
