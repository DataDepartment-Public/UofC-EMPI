"""
tests/regression/test_fs_enhanced_vetoes.py

Regression tests for the four deterministic patient-safety vetoes in
`models/experiments/fs_splink_enhanced/deterministic_vetoes.py`.

These tests construct small in-memory df_predictions + df_clean frames
that hit each veto's trigger exactly. No Splink training is involved, so
the suite is fast and isolated from the probabilistic scoring layer.

The veto contract under test:
    apply_vetoes(df_predictions, df_clean) -> df_predictions with
        `veto_reason: str | None` added. Pairs whose identifiers hit a
        rule get the first-firing rule name; others get None.
    Precedence (high to low):
        ssn_conflict > dob_year_gap > gender_conflict > dob_and_ssn4_conflict
"""

from __future__ import annotations

import pandas as pd
import pytest

from models.experiments.fs_splink_enhanced.deterministic_vetoes import (
    apply_vetoes,
    VETO_REASON_COL,
    VETO_PRECEDENCE,
    DOB_YEAR_GAP_THRESHOLD,
)
from src.contracts import PATID, SSN, SSN_LAST4, BIRTH_DT, SEX


# ============================================================================
# Helpers
# ============================================================================
def _build_clean(rows: list[dict]) -> pd.DataFrame:
    """Construct a minimal df_clean carrying the columns the vetoes need."""
    required = [PATID, SSN, SSN_LAST4, BIRTH_DT, SEX]
    return pd.DataFrame(
        [{c: r.get(c) for c in required} for r in rows]
    )


def _build_predictions(pairs: list[tuple[str, str, float]]) -> pd.DataFrame:
    """A predictions frame with PATID_A/PATID_B and match_probability."""
    return pd.DataFrame(
        pairs, columns=["PATID_A", "PATID_B", "match_probability"]
    )


# ============================================================================
# Per-veto firing tests
# ============================================================================
def test_ssn_conflict_fires_when_both_populated_and_differ():
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: "123456789", SSN_LAST4: "6789",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "MALE"},
            {PATID: "B", SSN: "987654321", SSN_LAST4: "4321",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "MALE"},
        ]
    )
    df_predictions = _build_predictions([("A", "B", 0.98)])
    out = apply_vetoes(df_predictions, df_clean)
    assert out.loc[0, VETO_REASON_COL] == "ssn_conflict"


def test_ssn_conflict_does_not_fire_when_one_side_null():
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: "123456789", SSN_LAST4: "6789",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "MALE"},
            {PATID: "B", SSN: None, SSN_LAST4: None,
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "MALE"},
        ]
    )
    df_predictions = _build_predictions([("A", "B", 0.98)])
    out = apply_vetoes(df_predictions, df_clean)
    assert out.loc[0, VETO_REASON_COL] is None


def test_dob_year_gap_fires_at_threshold():
    # 5-year gap is the threshold (>= triggers).
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: None, SSN_LAST4: None,
             BIRTH_DT: pd.Timestamp("1980-06-15"), SEX: "MALE"},
            {PATID: "B", SSN: None, SSN_LAST4: None,
             BIRTH_DT: pd.Timestamp(f"{1980 + DOB_YEAR_GAP_THRESHOLD}-06-15"),
             SEX: "MALE"},
        ]
    )
    out = apply_vetoes(_build_predictions([("A", "B", 0.92)]), df_clean)
    assert out.loc[0, VETO_REASON_COL] == "dob_year_gap"


def test_dob_year_gap_does_not_fire_below_threshold():
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: None, SSN_LAST4: None,
             BIRTH_DT: pd.Timestamp("1980-06-15"), SEX: "MALE"},
            {PATID: "B", SSN: None, SSN_LAST4: None,
             BIRTH_DT: pd.Timestamp(f"{1980 + DOB_YEAR_GAP_THRESHOLD - 1}-06-15"),
             SEX: "MALE"},
        ]
    )
    out = apply_vetoes(_build_predictions([("A", "B", 0.92)]), df_clean)
    assert out.loc[0, VETO_REASON_COL] is None


def test_gender_conflict_strict_male_female_fires():
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: None, SSN_LAST4: None,
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "MALE"},
            {PATID: "B", SSN: None, SSN_LAST4: None,
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "FEMALE"},
        ]
    )
    out = apply_vetoes(_build_predictions([("A", "B", 0.93)]), df_clean)
    assert out.loc[0, VETO_REASON_COL] == "gender_conflict"


@pytest.mark.parametrize("sex_a,sex_b", [("MALE", "OTHER"), ("OTHER", "MALE"),
                                         ("FEMALE", "OTHER"), ("OTHER", "FEMALE")])
def test_gender_conflict_does_not_fire_with_other(sex_a, sex_b):
    """OTHER mismatching MALE or FEMALE must NOT veto — clinical equity
    requirement for transgender / nonbinary patients."""
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: None, SSN_LAST4: None,
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: sex_a},
            {PATID: "B", SSN: None, SSN_LAST4: None,
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: sex_b},
        ]
    )
    out = apply_vetoes(_build_predictions([("A", "B", 0.93)]), df_clean)
    assert out.loc[0, VETO_REASON_COL] is None


def test_dob_and_ssn4_conflict_fires():
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: None, SSN_LAST4: "6789",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "OTHER"},
            {PATID: "B", SSN: None, SSN_LAST4: "4321",
             BIRTH_DT: pd.Timestamp("1981-02-20"), SEX: "OTHER"},
        ]
    )
    out = apply_vetoes(_build_predictions([("A", "B", 0.91)]), df_clean)
    assert out.loc[0, VETO_REASON_COL] == "dob_and_ssn4_conflict"


def test_dob_and_ssn4_does_not_fire_when_only_one_differs():
    # Same DOB, different last-4-SSN — neither veto fires.
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: None, SSN_LAST4: "6789",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "OTHER"},
            {PATID: "B", SSN: None, SSN_LAST4: "4321",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "OTHER"},
        ]
    )
    out = apply_vetoes(_build_predictions([("A", "B", 0.91)]), df_clean)
    assert out.loc[0, VETO_REASON_COL] is None


# ============================================================================
# Precedence
# ============================================================================
def test_precedence_ssn_conflict_wins_over_dob_gap():
    """When multiple rules would fire, the first in VETO_PRECEDENCE wins."""
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: "111111111", SSN_LAST4: "1111",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "MALE"},
            {PATID: "B", SSN: "222222222", SSN_LAST4: "2222",
             BIRTH_DT: pd.Timestamp("2000-01-15"), SEX: "FEMALE"},
        ]
    )
    out = apply_vetoes(_build_predictions([("A", "B", 0.99)]), df_clean)
    assert out.loc[0, VETO_REASON_COL] == "ssn_conflict"
    assert VETO_PRECEDENCE[0] == "ssn_conflict"


# ============================================================================
# Clean pairs and frame-level invariants
# ============================================================================
def test_clean_pair_gets_no_veto():
    """A pair with full identifier agreement must carry veto_reason=None."""
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: "123456789", SSN_LAST4: "6789",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "FEMALE"},
            {PATID: "B", SSN: "123456789", SSN_LAST4: "6789",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "FEMALE"},
        ]
    )
    out = apply_vetoes(_build_predictions([("A", "B", 0.97)]), df_clean)
    assert out.loc[0, VETO_REASON_COL] is None


def test_apply_vetoes_does_not_mutate_inputs():
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: "111111111", SSN_LAST4: "1111",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "MALE"},
            {PATID: "B", SSN: "222222222", SSN_LAST4: "2222",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "MALE"},
        ]
    )
    df_predictions = _build_predictions([("A", "B", 0.99)])
    cols_in = list(df_predictions.columns)
    n_rows_in = len(df_predictions)
    _ = apply_vetoes(df_predictions, df_clean)
    assert list(df_predictions.columns) == cols_in
    assert len(df_predictions) == n_rows_in
    assert VETO_REASON_COL not in df_predictions.columns


def test_apply_vetoes_handles_empty_frame():
    df_clean = _build_clean(
        [
            {PATID: "A", SSN: "111111111", SSN_LAST4: "1111",
             BIRTH_DT: pd.Timestamp("1980-01-15"), SEX: "MALE"},
        ]
    )
    out = apply_vetoes(_build_predictions([]), df_clean)
    assert VETO_REASON_COL in out.columns
    assert len(out) == 0


def test_apply_vetoes_raises_only_on_missing_patid():
    """PATID is the only hard-required column on df_clean (the join key).
    Other column gaps degrade specific vetoes to no-ops with a warning."""
    df_predictions = _build_predictions([("A", "B", 0.5)])
    with pytest.raises(ValueError, match="missing the required join key"):
        apply_vetoes(df_predictions, pd.DataFrame({"not_patid": ["x"]}))


def test_apply_vetoes_skips_gender_conflict_when_sex_column_missing(caplog):
    """The synthetic fixture predates SexAtBirthDSC_clean; the gender_conflict
    veto must degrade to a no-op (with a logged warning) rather than crash."""
    df_clean = pd.DataFrame(
        {
            PATID: ["A", "B"],
            SSN: ["111111111", "222222222"],
            SSN_LAST4: ["1111", "2222"],
            BIRTH_DT: [pd.Timestamp("1980-01-15"), pd.Timestamp("1980-01-15")],
            # SEX column intentionally absent.
        }
    )
    # SSN conflict still fires (its column is present); gender_conflict skips.
    df_predictions = _build_predictions([("A", "B", 0.99)])
    with caplog.at_level("WARNING"):
        out = apply_vetoes(df_predictions, df_clean)
    assert out.loc[0, VETO_REASON_COL] == "ssn_conflict"
    assert any("skipping gender_conflict" in rec.message for rec in caplog.records)


# ============================================================================
# Wire-in: classify_pairs must force tier=no_match for any veto_reason
# ============================================================================
def test_classify_pairs_honors_veto_reason_over_high_score():
    """A pair with score=0.99 AND a veto_reason set must land in no_match."""
    from models.experiments.fs_splink_enhanced.fellegi_sunter_enhanced import (
        classify_pairs,
    )

    df = pd.DataFrame(
        {
            "match_probability": [0.99, 0.99, 0.20],
            VETO_REASON_COL: ["ssn_conflict", None, None],
        }
    )
    out = classify_pairs(df, auto_merge_threshold=0.90, review_floor=0.50)
    assert out.loc[0, "classification_tier"] == "no_match"
    # Sanity: the second high-score pair (no veto) still auto_merges, and
    # the low-score pair still no_matches via the normal threshold path.
    assert out.loc[1, "classification_tier"] == "auto_merge"
    assert out.loc[2, "classification_tier"] == "no_match"


def test_classify_pairs_works_without_veto_column():
    """Backwards-compat: if the predictions frame has no veto_reason column,
    classify_pairs must behave exactly like the baseline."""
    from models.experiments.fs_splink_enhanced.fellegi_sunter_enhanced import (
        classify_pairs,
    )

    df = pd.DataFrame({"match_probability": [0.99, 0.70, 0.20]})
    out = classify_pairs(df, auto_merge_threshold=0.90, review_floor=0.50)
    assert out["classification_tier"].tolist() == [
        "auto_merge", "human_review", "no_match",
    ]
