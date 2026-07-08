"""Unit tests for the frozen production comparison registry."""

from __future__ import annotations

from src.models.fs_matcher.comparisons import (
    DEFAULT_PRIOR_RECALL,
    FS_MATCHER_REGISTRY,
    PRIOR_RULES,
    build_registry,
)

_EXPECTED_ORDER = ["FirstNM", "LastNM", "BirthDT", "SSN", "Email", "Phones", "Address"]


def test_registry_has_seven_comparisons_in_order():
    reg = build_registry(include_address=True)
    assert reg.names() == _EXPECTED_ORDER
    assert len(reg.build_all()) == 7


def test_default_registry_matches_build():
    assert FS_MATCHER_REGISTRY.names() == build_registry(True).names()


def test_each_comparison_has_a_null_level_and_is_two_level():
    """Frozen structure: each field is null + (exact-ish) + all-other."""
    for comp in build_registry().build_all():
        levels = comp["comparison_levels"]
        assert any(lv.get("is_null_level") for lv in levels), comp["output_column_name"]
        # null + substantive + all-other == 3 levels for the two-level design
        assert len(levels) == 3, (comp["output_column_name"], len(levels))


def test_include_address_false_drops_address():
    reg = build_registry(include_address=False)
    assert "Address" not in reg.names()
    assert len(reg) == 6


def test_prior_rules_are_five_and_ssn_dob_requires_ssn_and_dob():
    assert len(PRIOR_RULES) == 5
    assert 0.0 < DEFAULT_PRIOR_RECALL <= 1.0
    # First prior rule is SSN_DOB — must require BOTH SSN and DOB agreement.
    assert "SSN_clean" in PRIOR_RULES[0] and "_dob_str" in PRIOR_RULES[0]


def test_tf_adjusted_fields_are_first_last_email():
    """FirstNM / LastNM / Email carry term-frequency adjustments (their exact
    level gets a TF-scaled Bayes factor)."""
    reg = {c["output_column_name"]: c for c in build_registry().build_all()}
    for col in ("FirstNM_clean", "LastNM_clean", "Email_clean"):
        exact = next(lv for lv in reg[col]["comparison_levels"]
                     if "Exact" in lv.get("label_for_charts", ""))
        assert exact.get("tf_adjustment_column") == col
