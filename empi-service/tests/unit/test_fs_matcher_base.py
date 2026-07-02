"""Unit tests for the lifted FS base (src/models/fs_matcher/base.py).

Ports the still-relevant assertions from the retired research
`test_fs_base_{classify,projections,registry}` suites onto the production base:
ClassificationConfig invariants, FSModel.classify tiers, the n_blocks log-odds
bump, the two output projections, and ComparisonRegistry immutability. These are
splink-free (base.py imports splink only lazily inside training/predict).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.fs_matcher.base import (
    ClassificationConfig,
    ComparisonRegistry,
    ComparisonSpec,
    FSModel,
)


class _ConcreteModel(FSModel):
    """Minimal concrete FSModel for classify/projection tests (never trained)."""

    model_name = "test_model"

    def __init__(self, cfg: ClassificationConfig | None = None):
        self.registry = ComparisonRegistry([ComparisonSpec("x", lambda: {"output_column_name": "x"})])
        self.classification_config = cfg or ClassificationConfig()
        self.training = None

    def prepare_model_input(self, df_clean):  # pragma: no cover - unused here
        return df_clean

    def build_settings(self):  # pragma: no cover - unused here
        return {}


# ─── ClassificationConfig invariants ──────────────────────────────────────────
def test_classification_config_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="review_floor"):
        ClassificationConfig(auto_merge_threshold=0.4, review_floor=0.95)


def test_classification_config_rejects_negative_max_bits():
    with pytest.raises(ValueError, match="max_bits"):
        ClassificationConfig(n_blocks_bump_max_bits=-1.0)


# ─── classify: tier thresholds (inclusive) ────────────────────────────────────
@pytest.mark.parametrize(
    "score, expected",
    [
        (0.39, "no_match"),
        (0.40, "human_review"),   # review_floor inclusive
        (0.94, "human_review"),
        (0.95, "auto_merge"),     # auto_merge_threshold inclusive
        (0.999, "auto_merge"),
    ],
)
def test_classify_thresholds_inclusive(score, expected):
    model = _ConcreteModel()
    out = model.classify(pd.DataFrame({"match_probability": [score]}))
    assert out["classification_tier"].iloc[0] == expected


# ─── n_blocks bump ────────────────────────────────────────────────────────────
def test_n_blocks_bump_no_op_when_column_absent():
    model = _ConcreteModel()
    out = model.classify(pd.DataFrame({"match_probability": [0.7]}))
    assert out["match_probability"].iloc[0] == pytest.approx(0.7)


def test_n_blocks_bump_below_threshold_no_change():
    model = _ConcreteModel()
    out = model.classify(pd.DataFrame({"match_probability": [0.7, 0.7], "n_blocks": [1, 2]}))
    assert out["match_probability"].tolist() == pytest.approx([0.7, 0.7])


def test_n_blocks_bump_one_bit_at_threshold_plus_one():
    """n_blocks=3 (threshold=2, max_bits=4) -> +1 bit; p=0.5 -> 2/3."""
    model = _ConcreteModel()
    out = model.classify(pd.DataFrame({"match_probability": [0.5], "n_blocks": [3]}))
    assert out["match_probability"].iloc[0] == pytest.approx(2 / 3, abs=1e-6)


def test_n_blocks_bump_capped_at_max_bits():
    """n_blocks=20 -> capped at +4 bits; p=0.5 -> 16/17."""
    model = _ConcreteModel()
    out = model.classify(pd.DataFrame({"match_probability": [0.5], "n_blocks": [20]}))
    assert out["match_probability"].iloc[0] == pytest.approx(16 / 17, abs=1e-6)


def test_n_blocks_bump_keeps_match_weight_in_sync():
    model = _ConcreteModel()
    df = pd.DataFrame({"match_probability": [0.5], "match_weight": [0.0], "n_blocks": [4]})
    out = model.classify(df)  # +2 bits
    assert out["match_weight"].iloc[0] == pytest.approx(2.0, abs=1e-6)


def test_n_blocks_bump_can_move_pair_across_threshold():
    model = _ConcreteModel()
    # p=0.85 (weight ~2.50 bits) + n_blocks=5 (+3 bits) -> crosses 0.95.
    out = model.classify(pd.DataFrame({"match_probability": [0.85], "n_blocks": [5]}))
    assert out["classification_tier"].iloc[0] == "auto_merge"


# ─── projections ──────────────────────────────────────────────────────────────
def _classified() -> pd.DataFrame:
    return pd.DataFrame({
        "PATID_A": ["a", "a"], "PATID_B": ["b", "c"],
        "match_probability": [0.98, 0.10], "match_weight": [6.0, -3.0],
        "classification_tier": ["auto_merge", "no_match"],
        "source_blocks": ["SSN", "EMAIL"], "n_blocks": [2, 1],
    })


def test_to_evaluation_schema_projection():
    out = _ConcreteModel().to_evaluation_schema(_classified())
    assert list(out.columns) == ["PATID_A", "PATID_B", "model_name", "score", "predicted_tier"]
    assert (out["model_name"] == "test_model").all()
    assert out["score"].tolist() == pytest.approx([0.98, 0.10])


def test_to_probabilistic_matches_projection():
    out = _ConcreteModel().to_probabilistic_matches(_classified())
    assert list(out.columns) == [
        "PATID_A", "PATID_B", "match_source", "score", "match_weight",
        "classification_tier", "source_blocks", "n_blocks",
    ]
    assert (out["match_source"] == "model").all()


def test_projection_raises_on_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        _ConcreteModel().to_probabilistic_matches(pd.DataFrame({"PATID_A": ["a"]}))


# ─── ComparisonRegistry immutability / ordering ───────────────────────────────
def _spec(name: str) -> ComparisonSpec:
    return ComparisonSpec(name=name, builder=lambda n=name: {"output_column_name": n})


def test_registry_preserves_order_and_rejects_duplicates():
    assert ComparisonRegistry([_spec("a"), _spec("b")]).names() == ["a", "b"]
    with pytest.raises(ValueError, match="duplicate"):
        ComparisonRegistry([_spec("a"), _spec("a")])


def test_registry_mutations_return_new_registry():
    original = ComparisonRegistry([_spec("a"), _spec("b")])
    assert original.with_added(_spec("c")).names() == ["a", "b", "c"]
    assert original.with_added(_spec("z"), position=0).names() == ["z", "a", "b"]
    assert original.with_removed("a").names() == ["b"]
    assert original.names() == ["a", "b"]  # original untouched


def test_registry_build_all_is_lazy():
    calls: list[str] = []
    reg = ComparisonRegistry([ComparisonSpec("a", lambda: (calls.append("a"), {"v": 1})[1])])
    assert calls == []
    reg.build_all()
    assert calls == ["a"]
