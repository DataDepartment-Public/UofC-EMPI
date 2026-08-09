"""Unit tests for MLMatcher (src/models/ml_matcher/matcher.py).

Mirrors test_fs_matcher_base.py's classify-threshold pattern (same cutoffs,
same inclusive-boundary behavior) plus the BYOM/BYOF plumbing specific to the
ML matcher: predict()'s no-model guard, run()'s 5-col projection, and the
train() stub.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts import ClassificationResults, validate
from src.models.ml_matcher.base import MLClassificationConfig
from src.models.ml_matcher.matcher import MLMatcher, MODEL_NAME


class _FakeModel:
    """Sklearn-shaped fake — predict_proba returns a fixed probability."""

    def __init__(self, positive_proba: float = 0.9, as_1d: bool = False):
        self.positive_proba = positive_proba
        self.as_1d = as_1d

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        if self.as_1d:
            return np.full(n, self.positive_proba)
        return np.column_stack([np.full(n, 1 - self.positive_proba), np.full(n, self.positive_proba)])


class _FakeFeatureBuilder:
    def build_features(self, candidate_pairs, df_clean, fs_features=None):
        out = candidate_pairs[["PATID_A", "PATID_B"]].copy()
        out["feat1"] = 1.0
        return out


def _pairs() -> pd.DataFrame:
    return pd.DataFrame({"PATID_A": ["A", "C"], "PATID_B": ["B", "D"]})


def _clean() -> pd.DataFrame:
    return pd.DataFrame({"PATID": ["A", "B", "C", "D"]})


# ─── classify: one threshold, two tiers ──────────────────────────────────────
@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, "human_review"),
        (0.39, "human_review"),
        (0.699, "human_review"),
        (0.70, "auto_merge"),     # auto_merge_threshold inclusive
        (0.999, "auto_merge"),
    ],
)
def test_classify_thresholds_inclusive(score, expected):
    out = MLMatcher().classify(pd.DataFrame({"match_probability": [score]}))
    assert out["classification_tier"].iloc[0] == expected


def test_classify_never_emits_no_match():
    """THE structural guarantee of this stage: it cannot discard a pair. Only
    the Stage-4.25 gate drops pairs, and only it records what it dropped."""
    scores = pd.DataFrame({"match_probability": [0.0, 0.01, 0.5, 0.9, 1.0]})
    tiers = MLMatcher().classify(scores)["classification_tier"]
    assert set(tiers) <= {"auto_merge", "human_review"}


def test_classify_respects_a_custom_threshold():
    cfg = MLClassificationConfig(auto_merge_threshold=0.8)
    out = MLMatcher(classification_config=cfg).classify(
        pd.DataFrame({"match_probability": [0.5]})
    )
    assert out["classification_tier"].iloc[0] == "human_review"


# ─── predict() ──────────────────────────────────────────────────────────────────
def test_predict_raises_without_attached_model():
    features = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["B"], "feat1": [1.0]})
    with pytest.raises(RuntimeError, match="no model attached"):
        MLMatcher().predict(features)


def test_predict_uses_positive_class_column_from_2d_proba():
    features = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["B"], "feat1": [1.0]})
    out = MLMatcher(model=_FakeModel(0.73)).predict(features)
    assert out["match_probability"].iloc[0] == pytest.approx(0.73)


def test_predict_accepts_1d_proba_output():
    features = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["B"], "feat1": [1.0]})
    out = MLMatcher(model=_FakeModel(0.42, as_1d=True)).predict(features)
    assert out["match_probability"].iloc[0] == pytest.approx(0.42)


# ─── run() — the PairClassifier entry point ────────────────────────────────────
def test_run_produces_classification_results_shape():
    m = MLMatcher(model=_FakeModel(0.99), feature_builder=_FakeFeatureBuilder())
    out = m.run(_pairs(), _clean())
    assert list(out.columns) == ["PATID_A", "PATID_B", "model_name", "score", "predicted_tier"]
    assert (out["model_name"] == MODEL_NAME).all()
    assert (out["predicted_tier"] == "auto_merge").all()
    validate(out, ClassificationResults, allow_empty=False)


def test_run_passes_fs_features_through_to_builder():
    seen = {}

    class RecordingBuilder:
        def build_features(self, candidate_pairs, df_clean, fs_features=None):
            seen["fs_features"] = fs_features
            out = candidate_pairs[["PATID_A", "PATID_B"]].copy()
            out["feat1"] = 1.0
            return out

    fs_feats = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["B"], "gamma_dob": [2]})
    m = MLMatcher(model=_FakeModel(0.5), feature_builder=RecordingBuilder())
    m.run(_pairs(), _clean(), fs_features=fs_feats)
    assert seen["fs_features"] is fs_feats


def test_predict_carries_feature_columns_through():
    """MLFeatures (docs/Data-Contract.md §4.5b) specifies the candidate parquet
    as pair keys + score + tier + EVERY feature column. `to_ml_features` can
    only pass through what `predict` hands it, so the features have to survive
    scoring — they were previously dropped here, and the tests below missed it
    by hand-building a frame that already had them."""
    features = pd.DataFrame(
        {"PATID_A": ["A"], "PATID_B": ["B"], "feat1": [1.0], "feat2": [2.0]}
    )
    out = MLMatcher(model=_FakeModel(0.9)).predict(features)
    assert {"feat1", "feat2"} <= set(out.columns)


def test_score_output_reaches_to_ml_features_with_features_intact():
    """The end-to-end version of the above: the real path from score() to the
    candidate parquet must retain the feature columns."""
    matcher = MLMatcher(model=_FakeModel(0.9), feature_builder=_FakeFeatureBuilder())
    classified = matcher.score(_pairs(), _clean())
    assert "feat1" in matcher.to_ml_features(classified).columns


# ─── to_ml_features() ───────────────────────────────────────────────────────────
def _classified() -> pd.DataFrame:
    return pd.DataFrame({
        "PATID_A": ["a", "a"], "PATID_B": ["b", "c"],
        "match_probability": [0.98, 0.10],
        "classification_tier": ["auto_merge", "no_match"],
        "feat1": [1.0, 2.0],
    })


def test_to_ml_features_keeps_feature_columns():
    out = MLMatcher().to_ml_features(_classified())
    assert "feat1" in out.columns
    assert len(out) == 2


def test_to_ml_features_keeps_every_scored_pair():
    """No cutoff: the parquet is the complete record of the stage's scores,
    which is what an offline threshold sweep reads. A filtered file would make
    any threshold below the cutoff unevaluable."""
    classified = _classified()
    out = MLMatcher().to_ml_features(classified)
    assert len(out) == len(classified)
    assert out["match_probability"].min() == classified["match_probability"].min()


def test_to_ml_features_raises_on_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        MLMatcher().to_ml_features(pd.DataFrame({"PATID_A": ["a"]}))


# ─── train() — deliberate stub ─────────────────────────────────────────────────
def test_train_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="scaffold stub"):
        MLMatcher().train(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
