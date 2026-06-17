"""
tests/unit/test_fs_enhanced_e5_thresholds_bump.py

Unit tests for E5:
  - DEFAULT_AUTO_MERGE_THRESHOLD = 0.95 / DEFAULT_REVIEW_FLOOR = 0.40
  - n_blocks score bump applied in log-odds space (capped, threshold-gated)
  - to_probabilistic_matches projection columns + values
  - Boundary parametrize across the new tier boundaries
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.experiments.fs_splink_enhanced import fellegi_sunter_enhanced as fs


# ============================================================================
# Threshold defaults
# ============================================================================
def test_default_thresholds_are_e5_values():
    assert fs.DEFAULT_AUTO_MERGE_THRESHOLD == 0.95
    assert fs.DEFAULT_REVIEW_FLOOR == 0.40


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.00, "no_match"),
        (0.39, "no_match"),
        (0.40, "human_review"),   # inclusive review floor
        (0.94, "human_review"),
        (0.95, "auto_merge"),     # inclusive auto-merge threshold
        (1.00, "auto_merge"),
    ],
)
def test_classify_pairs_thresholds_e5(score, expected):
    df = pd.DataFrame({"match_probability": [score]})
    out = fs.classify_pairs(df)
    assert out["classification_tier"].iloc[0] == expected


# ============================================================================
# n_blocks bump
# ============================================================================
def test_n_blocks_bump_no_op_when_column_absent():
    """No n_blocks column -> classify_pairs behaves unchanged."""
    df = pd.DataFrame({"match_probability": [0.7]})
    out = fs.classify_pairs(df)
    assert out["match_probability"].iloc[0] == pytest.approx(0.7)


def test_n_blocks_bump_below_threshold_no_change():
    """n_blocks <= 2 (default threshold) -> no bump."""
    df = pd.DataFrame({"match_probability": [0.7, 0.7], "n_blocks": [1, 2]})
    out = fs.classify_pairs(df)
    assert out["match_probability"].iloc[0] == pytest.approx(0.7)
    assert out["match_probability"].iloc[1] == pytest.approx(0.7)


def test_n_blocks_bump_one_bit_at_threshold_plus_one():
    """n_blocks=3 -> +1 bit. p=0.5 (weight=0) -> weight=1 -> p=2/3."""
    df = pd.DataFrame({"match_probability": [0.5], "n_blocks": [3]})
    out = fs.classify_pairs(df)
    # +1 bit: weight 0 -> 1 -> p = 1/(1+2^-1) = 2/3
    assert out["match_probability"].iloc[0] == pytest.approx(2.0 / 3.0)


def test_n_blocks_bump_capped_at_max_bits():
    """n_blocks=20 should cap at +4 bits, not +18."""
    df = pd.DataFrame({"match_probability": [0.5], "n_blocks": [20]})
    out = fs.classify_pairs(df)
    # weight 0 -> 4 -> p = 1/(1+2^-4) = 16/17 ~ 0.9412
    assert out["match_probability"].iloc[0] == pytest.approx(16.0 / 17.0, abs=1e-6)


def test_n_blocks_bump_keeps_match_weight_in_sync():
    """If match_weight is present, it must reflect the bumped value."""
    df = pd.DataFrame({
        "match_probability": [0.5],
        "match_weight": [0.0],
        "n_blocks": [4],  # +2 bits
    })
    out = fs.classify_pairs(df)
    assert out["match_weight"].iloc[0] == pytest.approx(2.0)


def test_n_blocks_bump_can_move_pair_across_threshold():
    """A pair at p=0.85 with n_blocks=5 (+3 bits) should cross into auto_merge."""
    df = pd.DataFrame({"match_probability": [0.85], "n_blocks": [5]})
    out = fs.classify_pairs(df)
    # weight at p=0.85 is log2(0.85/0.15) ≈ 2.50; +3 bits = 5.50; p ~0.978
    assert out["match_probability"].iloc[0] > 0.95
    assert out["classification_tier"].iloc[0] == "auto_merge"


# ============================================================================
# to_probabilistic_matches projection
# ============================================================================
def test_to_probabilistic_matches_columns_and_values():
    rich = pd.DataFrame({
        "PATID_A": ["P0001", "P0003"],
        "PATID_B": ["P0002", "P0004"],
        "match_probability": [0.97, 0.20],
        "match_weight": [5.0, -2.0],
        "classification_tier": ["auto_merge", "no_match"],
        "veto_reason": [None, "ssn_conflict"],
        "source_blocks": ["B1|B3", "B5"],
        "n_blocks": [2, 1],
        "gamma_FirstNM_clean": [4, 0],  # should be dropped
    })
    out = fs.to_probabilistic_matches(rich)
    assert list(out.columns) == list(fs.PROBABILISTIC_MATCHES_COLUMNS)
    assert (out["match_source"] == "model").all()
    assert out["score"].tolist() == [0.97, 0.20]
    # pandas DataFrame construction may convert Python None -> NaN on object
    # columns. Either is valid (contract is nullable=True). Test on null-ness
    # for the empty slot and equality for the populated one.
    assert pd.isna(out["veto_reason"].iloc[0])
    assert out["veto_reason"].iloc[1] == "ssn_conflict"


def test_to_probabilistic_matches_handles_missing_optional_columns():
    """source_blocks/n_blocks/veto_reason can be absent (e.g., synthetic)."""
    rich = pd.DataFrame({
        "PATID_A": ["P0001"],
        "PATID_B": ["P0002"],
        "match_probability": [0.5],
        "match_weight": [0.0],
        "classification_tier": ["human_review"],
    })
    out = fs.to_probabilistic_matches(rich)
    assert out["veto_reason"].iloc[0] is None
    assert out["source_blocks"].iloc[0] is None
    assert pd.isna(out["n_blocks"].iloc[0])


def test_to_probabilistic_matches_raises_on_missing_required():
    """Missing match_weight -> clear ValueError."""
    rich = pd.DataFrame({
        "PATID_A": ["P0001"],
        "PATID_B": ["P0002"],
        "match_probability": [0.5],
        "classification_tier": ["human_review"],
    })
    with pytest.raises(ValueError, match="missing columns"):
        fs.to_probabilistic_matches(rich)


# ============================================================================
# Veto override still wins, even with n_blocks bump
# ============================================================================
def test_veto_wins_over_n_blocks_bump():
    """A pair with a veto_reason must land in no_match even if the n_blocks
    bump would otherwise lift its score above auto_merge."""
    from models.experiments.fs_splink_enhanced.deterministic_vetoes import (
        VETO_REASON_COL,
    )
    df = pd.DataFrame({
        "match_probability": [0.85],
        "n_blocks": [10],  # large bump that would push p > 0.95
        VETO_REASON_COL: ["ssn_conflict"],
    })
    out = fs.classify_pairs(df)
    assert out["classification_tier"].iloc[0] == "no_match"
