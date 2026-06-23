"""
tests/unit/test_fs_enhanced_e5_thresholds_bump.py

Unit tests for E5 via FSEnhanced:
  - ClassificationConfig defaults: auto_merge=0.95, review_floor=0.40,
    n_blocks_bump_threshold=2, n_blocks_bump_max_bits=4.0
  - n_blocks score bump applied in log-odds space (capped, threshold-gated)
  - FSEnhanced.classify() applies tiers + n_blocks bump correctly
  - to_probabilistic_matches projection columns + values (includes veto_reason)
  - Boundary parametrize across the new tier boundaries
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.experiments.fs_splink_enhanced.fs_enhanced import (
    FSEnhanced,
    PROBABILISTIC_MATCHES_COLUMNS,
)


# ============================================================================
# ClassificationConfig defaults
# ============================================================================
def test_fs_enhanced_classification_config_defaults():
    m = FSEnhanced()
    assert m.classification_config.auto_merge_threshold == 0.95
    assert m.classification_config.review_floor == 0.40
    assert m.classification_config.n_blocks_bump_threshold == 2
    assert m.classification_config.n_blocks_bump_max_bits == 4.0


def test_fs_enhanced_threshold_overrides():
    m = FSEnhanced(auto_merge_threshold=0.80, review_floor=0.30)
    assert m.classification_config.auto_merge_threshold == 0.80
    assert m.classification_config.review_floor == 0.30
    # n_blocks settings preserved from class defaults.
    assert m.classification_config.n_blocks_bump_threshold == 2
    assert m.classification_config.n_blocks_bump_max_bits == 4.0


# ============================================================================
# Tier boundaries (no n_blocks column — pure threshold test)
# ============================================================================
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
def test_classify_thresholds_e5(score, expected):
    m = FSEnhanced()
    df = pd.DataFrame({"match_probability": [score], "PATID_A": ["A"], "PATID_B": ["B"]})
    out = m.classify(df)
    assert out["classification_tier"].iloc[0] == expected


# ============================================================================
# n_blocks bump
# ============================================================================
def _make_pred(score, n_blocks=None, weight=None):
    d = {"match_probability": [score], "PATID_A": ["A"], "PATID_B": ["B"]}
    if n_blocks is not None:
        d["n_blocks"] = [n_blocks]
    if weight is not None:
        d["match_weight"] = [weight]
    return pd.DataFrame(d)


def test_n_blocks_bump_no_op_when_column_absent():
    m = FSEnhanced()
    df = _make_pred(0.7)
    out = m.classify(df)
    assert out["match_probability"].iloc[0] == pytest.approx(0.7)


def test_n_blocks_bump_below_threshold_no_change():
    """n_blocks <= 2 (default threshold) -> no bump."""
    m = FSEnhanced()
    for n in (1, 2):
        df = _make_pred(0.7, n_blocks=n)
        out = m.classify(df)
        assert out["match_probability"].iloc[0] == pytest.approx(0.7)


def test_n_blocks_bump_one_bit_at_threshold_plus_one():
    """n_blocks=3 -> +1 bit. p=0.5 (weight=0) -> weight=1 -> p=2/3."""
    m = FSEnhanced()
    df = _make_pred(0.5, n_blocks=3)
    out = m.classify(df)
    assert out["match_probability"].iloc[0] == pytest.approx(2.0 / 3.0)


def test_n_blocks_bump_capped_at_max_bits():
    """n_blocks=20 should cap at +4 bits, not +18."""
    m = FSEnhanced()
    df = _make_pred(0.5, n_blocks=20)
    out = m.classify(df)
    # weight 0 -> 4 -> p = 1/(1+2^-4) = 16/17 ~ 0.9412
    assert out["match_probability"].iloc[0] == pytest.approx(16.0 / 17.0, abs=1e-6)


def test_n_blocks_bump_keeps_match_weight_in_sync():
    """If match_weight is present, it must reflect the bumped value."""
    m = FSEnhanced()
    df = _make_pred(0.5, n_blocks=4, weight=0.0)
    out = m.classify(df)
    assert out["match_weight"].iloc[0] == pytest.approx(2.0)


def test_n_blocks_bump_can_move_pair_across_threshold():
    """A pair at p=0.85 with n_blocks=5 (+3 bits) should cross into auto_merge."""
    m = FSEnhanced()
    df = _make_pred(0.85, n_blocks=5)
    out = m.classify(df)
    # weight at p=0.85 is log2(0.85/0.15) ≈ 2.50; +3 bits = 5.50; p ~0.978
    assert out["match_probability"].iloc[0] > 0.95
    assert out["classification_tier"].iloc[0] == "auto_merge"


# ============================================================================
# to_probabilistic_matches projection (FSEnhanced override adds veto_reason)
# ============================================================================
def test_to_probabilistic_matches_columns_and_values():
    m = FSEnhanced()
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
    out = m.to_probabilistic_matches(rich)
    assert list(out.columns) == list(PROBABILISTIC_MATCHES_COLUMNS)
    assert (out["match_source"] == "model").all()
    assert out["score"].tolist() == [0.97, 0.20]
    assert pd.isna(out["veto_reason"].iloc[0])
    assert out["veto_reason"].iloc[1] == "ssn_conflict"


def test_to_probabilistic_matches_handles_missing_optional_columns():
    """source_blocks/n_blocks/veto_reason can be absent (e.g., synthetic)."""
    m = FSEnhanced()
    rich = pd.DataFrame({
        "PATID_A": ["P0001"],
        "PATID_B": ["P0002"],
        "match_probability": [0.5],
        "match_weight": [0.0],
        "classification_tier": ["human_review"],
    })
    out = m.to_probabilistic_matches(rich)
    assert out["veto_reason"].iloc[0] is None
    assert out["source_blocks"].iloc[0] is None
    assert pd.isna(out["n_blocks"].iloc[0])


def test_to_probabilistic_matches_raises_on_missing_required():
    """Missing match_weight -> clear ValueError."""
    m = FSEnhanced()
    rich = pd.DataFrame({
        "PATID_A": ["P0001"],
        "PATID_B": ["P0002"],
        "match_probability": [0.5],
        "classification_tier": ["human_review"],
    })
    with pytest.raises(ValueError, match="missing columns"):
        m.to_probabilistic_matches(rich)


# ============================================================================
# Veto override wins over n_blocks bump
# ============================================================================
def test_veto_wins_over_n_blocks_bump():
    """Veto must force no_match even when the n_blocks bump would otherwise
    push the pair into auto_merge.

    Setup:
    - PATID_A="A" has SSN "111111111"; PATID_B="B" has SSN "222222222".
      These differ → ssn_conflict veto fires.
    - The predictions row carries n_blocks=10, which gives +8 bits of bump
      (capped at max_bits=4.0 → effectively +4 bits), pushing p=0.85 well
      above the 0.95 auto_merge threshold without the veto.
    - After classify() the pair must land in no_match, not auto_merge.
    """
    from src.contracts import PATID, SSN, SSN_LAST4, BIRTH_DT, SEX

    m = FSEnhanced()
    # df_clean with a genuine SSN conflict between A and B.
    df_clean = pd.DataFrame({
        PATID: ["A", "B"],
        SSN: ["111111111", "222222222"],
        SSN_LAST4: ["1111", "2222"],
        BIRTH_DT: [pd.Timestamp("1980-01-01"), pd.Timestamp("1980-01-01")],
        SEX: ["MALE", "MALE"],
    })
    m._df_clean = df_clean

    # n_blocks=10 → +4-bit bump (capped) → p=0.85 crosses into auto_merge
    # territory without the veto.  With the veto, must be forced to no_match.
    df = pd.DataFrame({
        "match_probability": [0.85],
        "n_blocks": [10],
        "PATID_A": ["A"],
        "PATID_B": ["B"],
    })
    out = m.classify(df)

    assert out["classification_tier"].iloc[0] == "no_match", (
        "SSN-conflict veto must override the n_blocks bump and force no_match"
    )
    assert out["veto_reason"].iloc[0] == "ssn_conflict", (
        "veto_reason must identify the firing rule"
    )
