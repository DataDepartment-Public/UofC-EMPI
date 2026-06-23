"""
tests/unit/test_fs_enhanced_manual_priors.py

Unit tests for manual_priors.apply_manual_priors and the FSEnhanced.build_settings
Address-comparison wiring.  None of these train a Splink model — they exercise
dict mutation and settings-structure assertions only.

Sections:
  A. FSEnhanced.build_settings() structure assertions (replaces the old
     build_settings() functional-module tests)
  B. FSEnhanced().train() calls apply_manual_priors (mock-patched)
  C. apply_manual_priors standalone contract (unchanged — tests the helper
     module itself, not FSEnhanced)
  D. Module-level prior values — sanity ranges
"""

from __future__ import annotations

from unittest.mock import call, patch

import pandas as pd
import pytest

from models.experiments.fs_splink_enhanced.fs_enhanced import FSEnhanced
from models.experiments.fs_splink_enhanced.manual_priors import (
    ADDRESS_MU,
    PHONES_MU,
    apply_manual_priors,
)


# ============================================================================
# A. FSEnhanced.build_settings() — structure assertions
# ============================================================================
def test_build_settings_default_includes_address():
    m = FSEnhanced(include_address=True)
    settings = m.build_settings()
    ocns = [c["output_column_name"] for c in settings["comparisons"]]
    assert "Address" in ocns
    # Sanity: all baseline comparisons present.
    for required in ("FirstNM_clean", "LastNM_clean", "_dob_str", "SSN", "Email",
                     "Phones_array", "ZIP"):
        assert required in ocns, f"missing {required}"
    # 9 comparisons total (7 baseline + Household_discount + Address).
    assert len(settings["comparisons"]) == 9


def test_build_settings_include_address_false_drops_address():
    m = FSEnhanced(include_address=False)
    settings = m.build_settings()
    ocns = [c["output_column_name"] for c in settings["comparisons"]]
    assert "Address" not in ocns
    assert "Household_discount" not in ocns
    assert len(settings["comparisons"]) == 7


def test_build_settings_address_comparison_has_4_levels_with_locked_priors():
    m = FSEnhanced(include_address=True)
    settings = m.build_settings()
    address_comp = next(
        c for c in settings["comparisons"] if c["output_column_name"] == "Address"
    )
    assert len(address_comp["comparison_levels"]) == 4

    # Skip the null level; check the 3 evidential levels have locked m/u.
    locked_levels = [
        lv for lv in address_comp["comparison_levels"]
        if lv.get("label_for_charts") in ADDRESS_MU
    ]
    assert len(locked_levels) == 3
    for lv in locked_levels:
        m_expected, u_expected = ADDRESS_MU[lv["label_for_charts"]]
        assert lv["m_probability"] == pytest.approx(m_expected)
        assert lv["u_probability"] == pytest.approx(u_expected)
        assert lv["fix_m_probability"] is True
        assert lv["fix_u_probability"] is True


def test_build_settings_phones_comparison_has_locked_priors():
    model = FSEnhanced(include_address=False)
    settings = model.build_settings()
    phones_comp = next(
        c for c in settings["comparisons"] if c["output_column_name"] == "Phones_array"
    )
    locked_levels = [
        lv for lv in phones_comp["comparison_levels"]
        if lv.get("label_for_charts") in PHONES_MU
    ]
    # All 3 Phones evidential levels must be locked.
    assert len(locked_levels) == 3
    for lv in locked_levels:
        m_expected, u_expected = PHONES_MU[lv["label_for_charts"]]
        assert lv["m_probability"] == pytest.approx(m_expected)
        assert lv["u_probability"] == pytest.approx(u_expected)
        assert lv["fix_m_probability"] is True
        assert lv["fix_u_probability"] is True


# ============================================================================
# B. FSEnhanced.train() calls apply_manual_priors via build_settings()
# ============================================================================
def test_train_calls_apply_manual_priors_via_build_settings():
    """apply_manual_priors is invoked inside build_settings(), which is called
    by build_linker(), which is called by FSModel.run() before train().

    We verify the priors are locked on the final settings dict produced by
    build_settings() — this is the integration-level assertion that the priors
    flow through to the linker.
    """
    m = FSEnhanced(include_address=True)
    settings = m.build_settings()

    # Phones priors should be locked in the output dict.
    phones = next(
        c for c in settings["comparisons"] if c["output_column_name"] == "Phones_array"
    )
    locked = {
        lv["label_for_charts"]: (lv.get("fix_m_probability"), lv.get("fix_u_probability"))
        for lv in phones["comparison_levels"]
        if lv.get("label_for_charts") in PHONES_MU
    }
    assert all(v == (True, True) for v in locked.values()), (
        f"Expected all Phones levels to have fix_m/u=True; got: {locked}"
    )


def test_train_calls_apply_manual_priors_once(tmp_path):
    """Patch apply_manual_priors to confirm it is called exactly once per
    build_settings() call."""
    m = FSEnhanced(include_address=True)
    with patch(
        "models.experiments.fs_splink_enhanced.fs_enhanced.apply_manual_priors",
        wraps=apply_manual_priors,
    ) as mock_priors:
        _ = m.build_settings()

    assert mock_priors.call_count == 1, (
        f"apply_manual_priors expected 1 call, got {mock_priors.call_count}"
    )
    # The argument must be a dict (the settings dict).
    (settings_arg,), _ = mock_priors.call_args
    assert isinstance(settings_arg, dict)
    assert "comparisons" in settings_arg


# ============================================================================
# C. apply_manual_priors standalone contract (helper-module tests — unchanged)
# ============================================================================
def test_apply_manual_priors_is_no_op_when_target_comparison_absent():
    """If no Address/Phones comparison in the dict, apply_manual_priors must
    not raise — it just warns and returns the dict unchanged."""
    settings = {
        "comparisons": [
            {
                "output_column_name": "FirstNM_clean",
                "comparison_levels": [
                    {"label_for_charts": "Exact match", "m_probability": 0.5,
                     "u_probability": 0.001}
                ],
            }
        ]
    }
    result = apply_manual_priors(settings)
    assert result is settings  # returns the same object (in-place mutation)
    lv = result["comparisons"][0]["comparison_levels"][0]
    assert lv["m_probability"] == 0.5
    assert "fix_m_probability" not in lv


def test_apply_manual_priors_accepts_override_dicts():
    """Custom address_mu / phones_mu override the module defaults."""
    custom_address = {"Exact match on AddressLine1_clean": (0.99, 0.01)}
    m = FSEnhanced(include_address=True)
    settings = m.build_settings()
    apply_manual_priors(settings, address_mu=custom_address)

    address_comp = next(
        c for c in settings["comparisons"] if c["output_column_name"] == "Address"
    )
    exact_level = next(
        lv for lv in address_comp["comparison_levels"]
        if lv.get("label_for_charts") == "Exact match on AddressLine1_clean"
    )
    assert exact_level["m_probability"] == pytest.approx(0.99)
    assert exact_level["u_probability"] == pytest.approx(0.01)


# ============================================================================
# D. Module-level prior values — sanity ranges
# ============================================================================
@pytest.mark.parametrize("priors", [ADDRESS_MU, PHONES_MU])
def test_priors_are_valid_probabilities(priors):
    for label, (m, u) in priors.items():
        assert 0.0 < m < 1.0, f"{label}: m={m} out of (0,1)"
        assert 0.0 < u < 1.0, f"{label}: u={u} out of (0,1)"
