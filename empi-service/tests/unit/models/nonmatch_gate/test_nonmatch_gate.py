"""Unit tests for NonMatchGate (src/models/nonmatch_gate/gate.py).

The gate is the pipeline's confident-non-match filter: it splits the rules'
`non_matches` pool into plausible pairs (which reach the ML matcher) and
confident non-matches (discarded). These tests cover the threshold boundary,
the two-tier projection, and `apply()`'s passthrough-column preservation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts import (
    ClassificationResults,
    TIER_HUMAN_REVIEW,
    TIER_NO_MATCH,
    validate,
)
from src.models.nonmatch_gate.gate import MODEL_NAME, NonMatchGate


class _ScriptedModel:
    """Sklearn-shaped fake returning a caller-supplied P(plausible) per row,
    in feature-frame order."""

    def __init__(self, probas, as_1d: bool = False):
        self.probas = list(probas)
        self.as_1d = as_1d

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = np.asarray(self.probas[: len(X)], dtype=float)
        if self.as_1d:
            return p
        return np.column_stack([1.0 - p, p])


class _FakeFeatureBuilder:
    def build_features(self, candidate_pairs, df_clean, fs_features=None):
        out = candidate_pairs[["PATID_A", "PATID_B"]].copy()
        out["feat1"] = 1.0
        return out


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PATID_A": ["A", "C", "E"],
            "PATID_B": ["B", "D", "F"],
            "source_blocks": ["B1|B3", "B5", "B8"],
            "n_blocks": [2, 1, 1],
        }
    )


def _clean() -> pd.DataFrame:
    return pd.DataFrame({"PATID": ["A", "B", "C", "D", "E", "F"]})


def _gate(probas, threshold: float = 0.30, **kwargs) -> NonMatchGate:
    return NonMatchGate(
        model=_ScriptedModel(probas, **kwargs),
        feature_builder=_FakeFeatureBuilder(),
        threshold=threshold,
    )


# ─── scoring + thresholding ───────────────────────────────────────────────────
def test_score_splits_on_the_threshold():
    out = _gate([0.9, 0.1, 0.5]).score(_pairs(), _clean())
    assert list(out["classification_tier"]) == [
        TIER_HUMAN_REVIEW, TIER_NO_MATCH, TIER_HUMAN_REVIEW,
    ]


def test_threshold_boundary_is_inclusive():
    """A pair scoring exactly at the threshold PASSES — the gate errs toward
    keeping pairs, since a dropped pair is unrecoverable downstream."""
    out = _gate([0.30, 0.2999]).score(_pairs().head(2), _clean())
    assert list(out["classification_tier"]) == [TIER_HUMAN_REVIEW, TIER_NO_MATCH]


def test_gate_emits_only_two_tiers_never_auto_merge():
    """The gate makes no merge decision — that is the ML matcher's job."""
    tiers = set(_gate([0.99, 0.01, 0.5]).score(_pairs(), _clean())["classification_tier"])
    assert tiers <= {TIER_HUMAN_REVIEW, TIER_NO_MATCH}


def test_predict_accepts_1d_proba():
    out = _gate([0.9, 0.1, 0.5], as_1d=True).predict(
        _FakeFeatureBuilder().build_features(_pairs(), _clean())
    )
    assert list(out["plausible_probability"]) == [0.9, 0.1, 0.5]


def test_predict_without_model_raises():
    gate = NonMatchGate(model=None, feature_builder=_FakeFeatureBuilder())
    with pytest.raises(RuntimeError, match="no model attached"):
        gate.predict(_FakeFeatureBuilder().build_features(_pairs(), _clean()))


# ─── projection ───────────────────────────────────────────────────────────────
def test_to_evaluation_schema_satisfies_the_shared_contract():
    gate = _gate([0.9, 0.1, 0.5])
    ev = gate.to_evaluation_schema(gate.score(_pairs(), _clean()))
    validate(ev, ClassificationResults)
    assert list(ev.columns) == ["PATID_A", "PATID_B", "model_name", "score", "predicted_tier"]
    assert (ev["model_name"] == MODEL_NAME).all()


def test_to_evaluation_schema_rejects_a_non_classified_frame():
    with pytest.raises(ValueError, match="missing columns"):
        _gate([0.9]).to_evaluation_schema(_pairs())


# ─── apply() — the pipeline's one call ────────────────────────────────────────
def test_apply_returns_survivors_and_a_full_audit_frame():
    result = _gate([0.9, 0.1, 0.5]).apply(_pairs(), _clean())
    assert list(zip(result.survivors["PATID_A"], result.survivors["PATID_B"])) == [
        ("A", "B"), ("E", "F"),
    ]
    # The audit frame covers EVERY scored pair, not just the survivors.
    assert len(result.evaluation) == 3
    # Explanations are opt-in — a caller that doesn't ask pays nothing.
    assert result.explanations is None


def test_apply_preserves_passthrough_columns():
    """Survivors are rows of the INPUT frame, so `source_blocks`/`n_blocks`
    survive the gate — the ML matcher and the review UI still see them."""
    survivors = _gate([0.9, 0.1, 0.5]).apply(_pairs(), _clean()).survivors
    assert list(survivors.columns) == list(_pairs().columns)
    assert list(survivors["source_blocks"]) == ["B1|B3", "B8"]


def test_apply_can_drop_everything():
    result = _gate([0.01, 0.02, 0.03]).apply(_pairs(), _clean())
    assert result.survivors.empty
    assert (result.evaluation["predicted_tier"] == TIER_NO_MATCH).all()


def test_apply_on_an_empty_pool_is_a_no_op():
    empty = _pairs().iloc[:0]
    result = _gate([]).apply(empty, _clean())
    assert result.survivors.empty and result.evaluation.empty


# ─── feature-order alignment ──────────────────────────────────────────────────
def test_predict_reorders_columns_to_the_models_training_order():
    """A model exposing `feature_name_` gets its columns in training order —
    a shuffled feature frame must not silently corrupt predictions."""

    class _OrderSensitive:
        feature_name_ = ["b", "a"]

        def predict_proba(self, X):
            assert list(X.columns) == ["b", "a"]
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    feats = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["B"], "a": [1.0], "b": [2.0]})
    out = NonMatchGate(model=_OrderSensitive()).predict(feats)
    assert list(out["plausible_probability"]) == [1.0]


def test_default_feature_builder_is_the_v3_builder():
    """The gate model is trained on the ML matcher's 12 features — sharing the
    builder is what keeps the two stages' inputs identical."""
    from src.models.ml_matcher.lightgbm_v3 import V3FeatureBuilder

    assert isinstance(NonMatchGate(model=_ScriptedModel([])).feature_builder, V3FeatureBuilder)
