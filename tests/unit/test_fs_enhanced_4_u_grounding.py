"""tests/unit/test_fs_enhanced_4_u_grounding.py

Unit tests for FSEnhanced4.build_settings() u-grounding and projection helpers.
No Splink model training takes place: build_settings() is a metadata operation
(SettingsCreator → dict) that does NOT open a DuckDB connection or estimate m/u.

Tested behaviours:
  1. With a real df_clean, SSN and Email exact levels get grounded u = 1/n_distinct
     and fix_u_probability=True; _grounded_u dict is populated.
  2. Grounded exact levels do NOT gain fix_m_probability (m stays supervised).
  3. Fallback: _df_clean is None → exact levels fall back to hardcoded defaults.
  4. Fallback: column missing from _df_clean → that comparison falls back; the
     other is still grounded from actual data.
  5. Both exact levels were actually located (_grounded_u has both keys) — guards
     against the 'Exact match on <col>' label coupling silently breaking.
  6. to_evaluation_schema() and to_probabilistic_matches() drop the 'corroboration'
     column that FSEnhanced4.classify() adds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.contracts import EMAIL, SSN
from models.experiments.fs_splink_enhanced_4.corroboration_gate import CORROBORATION_COL
from models.experiments.fs_splink_enhanced_4.fs_enhanced_4 import FSEnhanced4

# ── Helpers ───────────────────────────────────────────────────────────────────

# Minimal silver-label frame accepted by SupervisedTraining.__init__.
_MINIMAL_LABELS = pd.DataFrame({"PATID_A": ["a"], "PATID_B": ["b"], "label": [1]})


def _fresh_model() -> FSEnhanced4:
    """Return a freshly constructed FSEnhanced4 with no df_clean attached."""
    return FSEnhanced4(labels_df=_MINIMAL_LABELS)


def _find_comparison(settings: dict, out_col_name: str) -> dict | None:
    """Return the comparison dict whose output_column_name matches out_col_name."""
    return next(
        (c for c in settings.get("comparisons", [])
         if c.get("output_column_name") == out_col_name),
        None,
    )


def _find_exact_level(comp: dict, col: str) -> dict | None:
    """Return the level dict with label_for_charts == 'Exact match on {col}'."""
    target = f"Exact match on {col}"
    return next(
        (lvl for lvl in comp.get("comparison_levels", [])
         if lvl.get("label_for_charts") == target),
        None,
    )


# ── 1 + 2. Grounded u with real df_clean ─────────────────────────────────────

def test_grounded_u_uses_one_over_n_distinct():
    """SSN: 3 distinct values → u = 1/3; Email: 2 distinct values → u = 1/2."""
    m = _fresh_model()
    # SSN_clean distinct: "1", "2", "3" (None dropped) → n=3 → u=1/3
    # Email_clean distinct: "x", "y" (None dropped, "x" deduplicated) → n=2 → u=1/2
    m._df_clean = pd.DataFrame(
        {SSN: ["1", "2", "3", None], EMAIL: ["x", "x", "y", None]}
    )
    settings = m.build_settings()

    ssn_comp = _find_comparison(settings, "SSN")
    assert ssn_comp is not None, "SSN comparison not found in settings"
    ssn_level = _find_exact_level(ssn_comp, SSN)
    assert ssn_level is not None, f"'Exact match on {SSN}' level not found in SSN comparison"

    email_comp = _find_comparison(settings, "Email")
    assert email_comp is not None, "Email comparison not found in settings"
    email_level = _find_exact_level(email_comp, EMAIL)
    assert email_level is not None, f"'Exact match on {EMAIL}' level not found in Email comparison"

    assert ssn_level["u_probability"] == pytest.approx(1 / 3)
    assert ssn_level.get("fix_u_probability") is True

    assert email_level["u_probability"] == pytest.approx(1 / 2)
    assert email_level.get("fix_u_probability") is True

    assert m._grounded_u == {"SSN": pytest.approx(1 / 3), "Email": pytest.approx(1 / 2)}


def test_grounded_exact_levels_do_not_have_fix_m_probability():
    """m stays supervised: grounded exact levels must NOT have fix_m_probability=True."""
    m = _fresh_model()
    m._df_clean = pd.DataFrame({SSN: ["1", "2"], EMAIL: ["x", "y"]})
    settings = m.build_settings()

    ssn_level = _find_exact_level(_find_comparison(settings, "SSN"), SSN)
    email_level = _find_exact_level(_find_comparison(settings, "Email"), EMAIL)

    assert ssn_level is not None
    assert email_level is not None
    # fix_m_probability must be absent or explicitly False — never True.
    assert ssn_level.get("fix_m_probability") is not True
    assert email_level.get("fix_m_probability") is not True


# ── 3. Fallback when _df_clean is None ───────────────────────────────────────

def test_fallback_u_when_df_clean_is_none():
    """With no df_clean attached, exact levels fall back to hardcoded defaults."""
    m = _fresh_model()
    assert m._df_clean is None
    settings = m.build_settings()

    ssn_level = _find_exact_level(_find_comparison(settings, "SSN"), SSN)
    email_level = _find_exact_level(_find_comparison(settings, "Email"), EMAIL)

    assert ssn_level is not None
    assert email_level is not None
    assert ssn_level["u_probability"] == pytest.approx(1e-6)
    assert email_level["u_probability"] == pytest.approx(5e-4)


# ── 4. Fallback when one column is missing from df_clean ─────────────────────

def test_fallback_when_email_column_missing():
    """SSN present → grounded from data; Email absent → falls back to 5e-4."""
    m = _fresh_model()
    m._df_clean = pd.DataFrame({SSN: ["1", "2"]})  # no Email_clean column
    settings = m.build_settings()

    ssn_level = _find_exact_level(_find_comparison(settings, "SSN"), SSN)
    email_level = _find_exact_level(_find_comparison(settings, "Email"), EMAIL)

    assert ssn_level is not None
    assert email_level is not None
    # SSN: 2 distinct values → u = 1/2
    assert ssn_level["u_probability"] == pytest.approx(1 / 2)
    # Email: column absent → fallback
    assert email_level["u_probability"] == pytest.approx(5e-4)


# ── 5. Both keys in _grounded_u ──────────────────────────────────────────────

def test_both_grounded_u_keys_present():
    """_grounded_u must contain both 'SSN' and 'Email' keys after build_settings().

    This is the coupling guard: if the comparison or level label is ever renamed,
    _ground_untrained_u logs a warning and skips — _grounded_u would be missing
    the key, and this test would catch it immediately.
    """
    m = _fresh_model()
    m._df_clean = pd.DataFrame({SSN: ["a", "b", "c"], EMAIL: ["x@y.com", "z@y.com", "w@y.com"]})
    m.build_settings()
    assert "SSN" in m._grounded_u, (
        "'SSN' key missing from _grounded_u — 'Exact match on SSN_clean' label "
        "may have changed in the comparison definition."
    )
    assert "Email" in m._grounded_u, (
        "'Email' key missing from _grounded_u — 'Exact match on Email_clean' label "
        "may have changed in the comparison definition."
    )


# ── 6. Projection: corroboration column dropped by to_evaluation_schema / to_probabilistic_matches ──

def test_to_evaluation_schema_drops_corroboration():
    """corroboration column added by classify() must be dropped in the eval output."""
    m = _fresh_model()
    frame = pd.DataFrame(
        {
            "PATID_A": ["P1"],
            "PATID_B": ["P2"],
            "match_probability": [0.97],
            "match_weight": [4.5],
            "classification_tier": ["auto_merge"],
            CORROBORATION_COL: [None],
        }
    )
    out = m.to_evaluation_schema(frame)
    assert CORROBORATION_COL not in out.columns
    assert list(out.columns) == list(m.eval_schema_columns)


def test_to_probabilistic_matches_drops_corroboration():
    """corroboration column must not appear in the ProbabilisticMatches output."""
    m = _fresh_model()
    frame = pd.DataFrame(
        {
            "PATID_A": ["P1"],
            "PATID_B": ["P2"],
            "match_probability": [0.97],
            "match_weight": [4.5],
            "classification_tier": ["auto_merge"],
            CORROBORATION_COL: [None],
        }
    )
    out = m.to_probabilistic_matches(frame)
    assert CORROBORATION_COL not in out.columns
    assert list(out.columns) == list(m.probabilistic_matches_columns)
