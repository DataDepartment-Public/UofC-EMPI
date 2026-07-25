"""Unit tests for the LightGBM v3 concrete BYOF + BYOM implementation
(src/models/ml_matcher/lightgbm_v3.py) and the joblib loader
(registry.load_model_artifact).

Covers: V3FeatureBuilder output shape/dtypes, a few known feature values,
graceful handling of a missing MiddleNM_clean column, the serve-time
column-swap in MatchProbabilityAdapter, and a joblib round-trip through
load_model_artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts import TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH
from src.models.ml_matcher.base import ClassificationConfig
from src.models.ml_matcher.lightgbm_v3 import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    MatchProbabilityAdapter,
    V3FeatureBuilder,
)
from src.models.ml_matcher.matcher import MLMatcher
from src.models.ml_matcher.registry import load_model_artifact


def _clean() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PATID": ["A", "B", "C", "D"],
            "FirstNM_clean": ["john", "john", "shawn", "mary"],
            "LastNM_clean": ["smith", "smith", "smith", "jones"],
            "MiddleNM_clean": ["q", np.nan, np.nan, np.nan],
            "BirthDT_clean": pd.to_datetime(
                ["1990-01-02", "1990-01-02", "1990-01-02", "1985-06-15"]
            ),
            "SSN_clean": ["123456789", "123456789", np.nan, "987654321"],
            "Email_clean": ["j@x.com", "j@x.com", np.nan, "m@y.com"],
            "AddressLine1_clean": ["10 main st", "10 main st", "12 oak ave", "5 elm rd"],
            "Phones_set": [{"3125551212"}, {"3125551212"}, {"7735550000"}, set()],
        }
    )


def _pairs() -> pd.DataFrame:
    # A<->B exact-ish match; A<->C phonetic first-name (shawn/sean-ish) + last match; A<->D disagree.
    return pd.DataFrame(
        {
            "PATID_A": ["A", "A", "A"],
            "PATID_B": ["B", "C", "D"],
            "source_blocks": ["B1", "B3", "B4"],
            "n_blocks": [1, 1, 1],
        }
    )


# ─── V3FeatureBuilder ─────────────────────────────────────────────────────────
def test_build_features_shape_and_dtypes():
    out = V3FeatureBuilder().build_features(_pairs(), _clean())

    assert list(out.columns) == ["PATID_A", "PATID_B", *FEATURE_COLS]
    assert len(out) == 3
    for c in CATEGORICAL_FEATURES:
        assert isinstance(out[c].dtype, pd.CategoricalDtype)
        assert list(out[c].cat.categories) == ["missing", "same", "different"]
    numeric = [c for c in FEATURE_COLS if c not in CATEGORICAL_FEATURES]
    for c in numeric:
        assert out[c].dtype == float


def test_build_features_known_values():
    out = V3FeatureBuilder().build_features(_pairs(), _clean()).set_index("PATID_B")

    # A<->B: identical first/last/dob/ssn/email/address/phone -> exact agreement.
    ab = out.loc["B"]
    assert ab["sim_jw_first"] == 1.0
    assert ab["sim_jw_last"] == 1.0
    assert ab["sim_dob"] == 1.0
    assert ab["ssn_digit_frac"] == 1.0
    assert ab["sim_phones"] == 1.0
    assert ab["sound_first"] == "same"
    assert ab["sound_last"] == "same"
    assert ab["cmp_street_num"] == "same"

    # A<->D: different names/dob/address -> street numbers differ.
    ad = out.loc["D"]
    assert ad["sim_jw_last"] < 1.0
    assert ad["cmp_street_num"] == "different"


def test_missing_middle_name_column_is_tolerated():
    clean = _clean().drop(columns=["MiddleNM_clean"])
    out = V3FeatureBuilder().build_features(_pairs(), clean)
    # Column still present, all NaN — never raises KeyError.
    assert "sim_jw_middle" in out.columns
    assert out["sim_jw_middle"].isna().all()


def test_missing_patid_yields_nan_row_not_crash():
    pairs = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["ZZZ"]})  # ZZZ not in clean
    out = V3FeatureBuilder().build_features(pairs, _clean())
    assert len(out) == 1
    assert out.iloc[0]["sim_jw_first"] != out.iloc[0]["sim_jw_first"]  # NaN


# ─── MatchProbabilityAdapter ──────────────────────────────────────────────────
class _InnerModel:
    """Sklearn-shaped stub: class 1 = ambiguous. Picklable (module-level)."""

    feature_name_ = list(FEATURE_COLS)

    def predict_proba(self, X):
        n = len(X)
        # [P(match)=0.8, P(ambiguous)=0.2]
        return np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])


def test_adapter_swaps_probability_columns():
    adapter = MatchProbabilityAdapter(_InnerModel())
    X = pd.DataFrame({c: [0.0, 0.0] for c in FEATURE_COLS})
    proba = adapter.predict_proba(X)
    # column 1 must now be P(confident match) = 1 - P(ambiguous) = 0.8
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba[:, 1], 0.8)
    np.testing.assert_allclose(proba[:, 0], 0.2)


def test_adapter_reorders_columns_to_training_order():
    adapter = MatchProbabilityAdapter(_InnerModel())
    # shuffle columns — adapter should realign to feature_name_ before predict
    X = pd.DataFrame({c: [1.0] for c in reversed(FEATURE_COLS)})
    proba = adapter.predict_proba(X)
    np.testing.assert_allclose(proba[:, 1], 0.8)


# ─── 2-tier classify (review_floor = 0.0) ─────────────────────────────────────
def test_classify_two_tier_when_review_floor_zero():
    """With the FS gate removing non-matches upstream, the ML matcher runs with
    review_floor=0.0 and must emit ONLY auto_merge / human_review (no no_match)."""
    ml = MLMatcher(classification_config=ClassificationConfig(auto_merge_threshold=0.70, review_floor=0.0))
    preds = pd.DataFrame({
        "PATID_A": ["A", "C", "E", "G"],
        "PATID_B": ["B", "D", "F", "H"],
        "match_probability": [0.0, 0.40, 0.699, 0.95],
    })
    tiers = ml.classify(preds)["classification_tier"].tolist()
    assert TIER_NO_MATCH not in tiers
    assert tiers == [TIER_HUMAN_REVIEW, TIER_HUMAN_REVIEW, TIER_HUMAN_REVIEW, TIER_AUTO_MERGE]


# ─── load_model_artifact round-trip ───────────────────────────────────────────
def test_load_model_artifact_roundtrip(tmp_path):
    joblib = pytest.importorskip("joblib")
    artifact = tmp_path / "ml_model_20260101T000000Z.pkl"
    joblib.dump(MatchProbabilityAdapter(_InnerModel()), artifact)

    loaded = load_model_artifact(artifact)
    proba = loaded.predict_proba(pd.DataFrame({c: [0.0] for c in FEATURE_COLS}))
    np.testing.assert_allclose(proba[:, 1], 0.8)
