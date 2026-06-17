"""
tests/unit/test_fs_enhanced_manual_priors.py

Unit tests for manual_priors.apply_manual_priors and the build_settings
Address-comparison wiring. None of these train a Splink model — they
exercise dict mutation and settings-structure assertions only.

The integration test that asserts the priors survive an EM training pass
is in tests/regression/test_fs_enhanced_priors_survive_em.py (added if/when
a fast-enough fixture exists; the synthetic fixture currently lacks address
columns, which is why include_address=False is the sandbox-path default).
"""

from __future__ import annotations

import pandas as pd
import pytest

from models.experiments.fs_splink_enhanced.fellegi_sunter_enhanced import (
    build_settings,
)
from models.experiments.fs_splink_enhanced.manual_priors import (
    ADDRESS_MU,
    PHONES_MU,
    apply_manual_priors,
)


# ============================================================================
# build_settings — structure assertions
# ============================================================================
def test_build_settings_default_includes_address():
    settings = build_settings()  # include_address=True default
    ocns = [c["output_column_name"] for c in settings["comparisons"]]
    assert "Address" in ocns
    # Sanity: still has the baseline comparisons.
    for required in ("FirstNM_clean", "LastNM_clean", "_dob_str", "SSN", "Email",
                     "Phones_array", "ZIP"):
        assert required in ocns, f"missing {required}"
    # 8 comparisons total (was 7 in baseline; +Address)
    assert len(settings["comparisons"]) == 8


def test_build_settings_include_address_false_drops_it():
    settings = build_settings(include_address=False)
    ocns = [c["output_column_name"] for c in settings["comparisons"]]
    assert "Address" not in ocns
    assert len(settings["comparisons"]) == 7


def test_address_comparison_has_4_levels_with_locked_priors():
    settings = build_settings()
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


def test_phones_comparison_has_locked_priors():
    settings = build_settings()
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
# apply_manual_priors — dict mutation contract
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
    # The unrelated comparison is untouched.
    lv = result["comparisons"][0]["comparison_levels"][0]
    assert lv["m_probability"] == 0.5
    assert "fix_m_probability" not in lv


def test_apply_manual_priors_accepts_override_dicts():
    """Custom address_mu / phones_mu override the module defaults."""
    custom_address = {"Exact match on AddressLine1_clean": (0.99, 0.01)}
    settings = build_settings()  # default uses ADDRESS_MU
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
# Module-level prior values — sanity ranges
# ============================================================================
@pytest.mark.parametrize("priors", [ADDRESS_MU, PHONES_MU])
def test_priors_are_valid_probabilities(priors):
    for label, (m, u) in priors.items():
        assert 0.0 < m < 1.0, f"{label}: m={m} out of (0,1)"
        assert 0.0 < u < 1.0, f"{label}: u={u} out of (0,1)"
