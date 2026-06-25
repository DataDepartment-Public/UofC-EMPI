"""tests/unit/test_fs_enhanced_3_comparisons.py

Structural tests for the fs_splink_enhanced_3 comparison registry.

enhanced_3 is the deliberately-simple FS variant: seven comparisons, each a
two-level Exact / All-other distinction preceded by a standard Splink null level
(null / exact / else). Term-frequency adjustments are enabled on the
high-cardinality identity fields only (FirstNM, LastNM, Email).
"""

from __future__ import annotations

import pytest

from models.experiments.fs_splink_enhanced_3.comparisons import (
    DEFAULT_PRIOR_RECALL,
    ENHANCED_3_REGISTRY,
    PRIOR_RULES,
    build_registry,
)


# ── Registry contents ────────────────────────────────────────────────────────
def test_registry_with_address_has_7_comparisons():
    reg = build_registry(include_address=True)
    assert len(reg) == 7


def test_registry_without_address_has_6_comparisons():
    reg = build_registry(include_address=False)
    assert len(reg) == 6
    assert "Address" not in reg


def test_enhanced_3_registry_alias_equals_full_registry():
    assert ENHANCED_3_REGISTRY.names() == build_registry(include_address=True).names()


def test_registry_names_and_order():
    assert build_registry(include_address=True).names() == [
        "FirstNM", "LastNM", "BirthDT", "SSN", "Email", "Phones", "Address",
    ]


# ── Every comparison is exactly null / exact / all-other (3 levels) ───────────
@pytest.mark.parametrize(
    "comp_name",
    ["FirstNM", "LastNM", "BirthDT", "SSN", "Email", "Phones", "Address"],
)
def test_each_comparison_has_three_levels(comp_name):
    reg = build_registry(include_address=True)
    by_name = {s.name: s for s in reg}
    comp = by_name[comp_name].builder()
    levels = comp["comparison_levels"]
    assert len(levels) == 3, (
        f"{comp_name} should be null/exact/else (3 levels), got "
        f"{[lv.get('label_for_charts') for lv in levels]}"
    )
    # First level is the null (no-evidence) level; last is the catch-all Else.
    assert levels[0].get("is_null_level") is True
    last_sql = levels[-1].get("sql_condition", "").strip().upper()
    assert last_sql == "ELSE"


# ── Term-frequency adjustments: identity fields on, the rest off ─────────────
def _exact_level(comp: dict) -> dict:
    """The middle (substantive, non-null, non-else) level of a 3-level comp."""
    return comp["comparison_levels"][1]


@pytest.mark.parametrize("comp_name", ["FirstNM", "LastNM", "Email"])
def test_tf_enabled_on_identity_fields(comp_name):
    reg = build_registry(include_address=True)
    comp = {s.name: s for s in reg}[comp_name].builder()
    assert _exact_level(comp).get("tf_adjustment_column") is not None


@pytest.mark.parametrize("comp_name", ["BirthDT", "SSN", "Phones", "Address"])
def test_tf_disabled_on_non_identity_fields(comp_name):
    reg = build_registry(include_address=True)
    comp = {s.name: s for s in reg}[comp_name].builder()
    assert _exact_level(comp).get("tf_adjustment_column") is None


# ── Phones is the multi-valued "shared phone" comparison ──────────────────────
def test_phones_uses_array_intersection():
    comp = {s.name: s for s in build_registry()}["Phones"].builder()
    fire = comp["comparison_levels"][1]
    assert "list_intersect" in fire["sql_condition"]


# ── Prior rules (lambda seeding) ──────────────────────────────────────────────
def test_prior_rules_translate_deterministic_rules():
    # One rule per deterministic RULE (SSN+DOB, NAME+DOB+{EMAIL,PHONE,SEX,ADDRESS}).
    assert len(PRIOR_RULES) == 5
    # All use the l./r. linker-alias predicate form.
    assert all(rule.startswith("l.") and " r." in rule for rule in PRIOR_RULES)
    # The strongest rule pairs SSN with DOB (mirrors RULES[0] = SSN_DOB).
    assert any("SSN_clean = r.SSN_clean" in rule and "_dob_str" in rule for rule in PRIOR_RULES)


def test_default_prior_recall_in_range():
    assert 0.0 < DEFAULT_PRIOR_RECALL <= 1.0
