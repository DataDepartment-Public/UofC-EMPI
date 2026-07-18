"""Unit tests for the ML matcher's pluggable contracts (src/models/ml_matcher/base.py).

Covers the BYOF/BYOM Protocols and the default `NotImplementedFeatureBuilder`
stub. Splink-free (ml_matcher has no Splink dependency at all).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.fs_matcher.base import ClassificationConfig as FSClassificationConfig
from src.models.ml_matcher.base import (
    ClassificationConfig,
    FeatureBuilder,
    MLModel,
    NotImplementedFeatureBuilder,
)


def test_classification_config_is_reexported_from_fs_matcher():
    # Reuses fs_matcher's dataclass rather than re-declaring a near-duplicate.
    assert ClassificationConfig is FSClassificationConfig


def test_not_implemented_feature_builder_raises():
    builder = NotImplementedFeatureBuilder()
    with pytest.raises(NotImplementedError, match="feature builder"):
        builder.build_features(
            pd.DataFrame({"PATID_A": ["a"], "PATID_B": ["b"]}), pd.DataFrame()
        )


def test_not_implemented_feature_builder_satisfies_protocol():
    assert isinstance(NotImplementedFeatureBuilder(), FeatureBuilder)


def test_arbitrary_sklearn_like_estimator_satisfies_mlmodel_protocol():
    class FakeEstimator:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return [[0.1, 0.9]] * len(X)

    assert isinstance(FakeEstimator(), MLModel)


def test_object_missing_predict_proba_fails_mlmodel_protocol():
    class Incomplete:
        def fit(self, X, y):
            return self

    assert not isinstance(Incomplete(), MLModel)


def test_custom_feature_builder_satisfies_protocol():
    class CustomBuilder:
        def build_features(self, candidate_pairs, df_clean, fs_features=None):
            return candidate_pairs

    assert isinstance(CustomBuilder(), FeatureBuilder)
