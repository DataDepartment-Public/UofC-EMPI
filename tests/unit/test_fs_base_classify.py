"""Unit tests for FSModel.classify + _apply_n_blocks_bump.

These don't need Splink — they exercise pure pandas/numpy paths.
"""

from __future__ import annotations

import pandas as pd
import pytest

from models.common.fs_base import (
    ClassificationConfig,
    ComparisonRegistry,
    ComparisonSpec,
    FSModel,
    SupervisedTraining,
)


# ── A minimal concrete subclass for testing the base methods ──────────────────
class _DummyFS(FSModel):
    model_name = "dummy_fs"
    registry = ComparisonRegistry([ComparisonSpec("X", lambda: {})])
    classification_config = ClassificationConfig()

    def __init__(self, cfg: ClassificationConfig | None = None):
        if cfg is not None:
            self.classification_config = cfg
        # training intentionally unused in classify/projection tests; provide a
        # trivially valid instance for completeness.
        self.training = SupervisedTraining(
            pd.DataFrame({"label": [0, 1]}), label_col="label",
        )

    def prepare_model_input(self, df_clean):
        return df_clean

    def build_settings(self):
        return {"comparisons": self.registry.build_all()}


# ─── ClassificationConfig invariants ──────────────────────────────────────────
def test_classification_config_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="review_floor"):
        ClassificationConfig(auto_merge_threshold=0.4, review_floor=0.95)


def test_classification_config_rejects_negative_max_bits():
    with pytest.raises(ValueError, match="max_bits"):
        ClassificationConfig(n_blocks_bump_max_bits=-1.0)


# ─── Boundary parametrize: thresholds are inclusive at the floor ──────────────
@pytest.mark.parametrize(
    "score, expected",
    [
        (0.00, "no_match"),
        (0.39, "no_match"),
        (0.40, "human_review"),  # inclusive review floor
        (0.94, "human_review"),
        (0.95, "auto_merge"),    # inclusive auto-merge threshold
        (1.00, "auto_merge"),
    ],
)
def test_classify_thresholds_inclusive(score, expected):
    model = _DummyFS()
    df = pd.DataFrame({"match_probability": [score]})
    out = model.classify(df)
    assert out["classification_tier"].iloc[0] == expected


# ─── n_blocks bump ────────────────────────────────────────────────────────────
def test_n_blocks_bump_no_op_when_column_absent():
    model = _DummyFS()
    df = pd.DataFrame({"match_probability": [0.7]})
    out = model.classify(df)
    assert out["match_probability"].iloc[0] == pytest.approx(0.7)


def test_n_blocks_bump_below_threshold_no_change():
    model = _DummyFS()
    df = pd.DataFrame({"match_probability": [0.7, 0.7], "n_blocks": [1, 2]})
    out = model.classify(df)
    assert out["match_probability"].iloc[0] == pytest.approx(0.7)
    assert out["match_probability"].iloc[1] == pytest.approx(0.7)


def test_n_blocks_bump_one_bit_at_threshold_plus_one():
    """n_blocks=3 (threshold=2, max_bits=4) -> +1 bit. p=0.5 -> p=2/3."""
    model = _DummyFS()
    df = pd.DataFrame({"match_probability": [0.5], "n_blocks": [3]})
    out = model.classify(df)
    assert out["match_probability"].iloc[0] == pytest.approx(2.0 / 3.0)


def test_n_blocks_bump_capped_at_max_bits():
    model = _DummyFS()
    df = pd.DataFrame({"match_probability": [0.5], "n_blocks": [20]})
    out = model.classify(df)
    # weight 0 -> 4 -> p = 16/17
    assert out["match_probability"].iloc[0] == pytest.approx(16.0 / 17.0, abs=1e-6)


def test_n_blocks_bump_keeps_match_weight_in_sync():
    model = _DummyFS()
    df = pd.DataFrame({
        "match_probability": [0.5],
        "match_weight": [0.0],
        "n_blocks": [4],  # +2 bits
    })
    out = model.classify(df)
    assert out["match_weight"].iloc[0] == pytest.approx(2.0)


def test_n_blocks_bump_can_move_pair_across_threshold():
    """p=0.85 + n_blocks=5 (+3 bits) crosses 0.95 auto_merge boundary."""
    model = _DummyFS()
    df = pd.DataFrame({"match_probability": [0.85], "n_blocks": [5]})
    out = model.classify(df)
    assert out["match_probability"].iloc[0] > 0.95
    assert out["classification_tier"].iloc[0] == "auto_merge"


# ─── Custom thresholds via ClassificationConfig override ──────────────────────
def test_custom_classification_config_is_respected():
    model = _DummyFS(ClassificationConfig(
        auto_merge_threshold=0.80, review_floor=0.30,
        n_blocks_bump_threshold=10,  # effectively disable bump in this test
        n_blocks_bump_max_bits=4.0,
    ))
    df = pd.DataFrame({"match_probability": [0.29, 0.30, 0.79, 0.80]})
    out = model.classify(df)
    assert out["classification_tier"].tolist() == [
        "no_match", "human_review", "human_review", "auto_merge",
    ]
