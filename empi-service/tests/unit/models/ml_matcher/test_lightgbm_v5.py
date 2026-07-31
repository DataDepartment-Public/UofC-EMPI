"""Unit tests for the LightGBM v5 concrete BYOF + BYOM implementation
(src/models/ml_matcher/lightgbm_v5.py) and the joblib loader
(registry.load_model_artifact).

Covers: FeatureBuilderV5 output shape/dtypes, a few known feature values,
graceful handling of a missing MiddleNM_clean column, the pass-through
semantics of DirectMatchAdapter, and a joblib round-trip through
load_model_artifact.

v5's defining property is that it inverts *nothing*: its class 1 is already
the confident match, so `predict_proba` passes through and SHAP contributions
keep their sign. That is pinned below, because a future "fix" reintroducing an
inversion here would produce scores and waterfalls that are silently, exactly
backwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts import TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH
from src.models.ml_matcher.base import ClassificationConfig
from src.models.ml_matcher.lightgbm_v5 import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    DirectMatchAdapter,
    FeatureBuilderV5,
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


# ─── FeatureBuilderV5 ─────────────────────────────────────────────────────────
def test_build_features_shape_and_dtypes():
    out = FeatureBuilderV5().build_features(_pairs(), _clean())

    assert list(out.columns) == ["PATID_A", "PATID_B", *FEATURE_COLS]
    assert len(out) == 3
    for c in CATEGORICAL_FEATURES:
        assert isinstance(out[c].dtype, pd.CategoricalDtype)
        assert list(out[c].cat.categories) == ["missing", "same", "different"]
    numeric = [c for c in FEATURE_COLS if c not in CATEGORICAL_FEATURES]
    for c in numeric:
        assert out[c].dtype == float


def test_feature_roster_is_twelve():
    assert len(FEATURE_COLS) == 12
    assert set(CATEGORICAL_FEATURES) <= set(FEATURE_COLS)


def test_build_features_known_values():
    out = FeatureBuilderV5().build_features(_pairs(), _clean()).set_index("PATID_B")

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
    out = FeatureBuilderV5().build_features(_pairs(), clean)
    # Column still present, all NaN — never raises KeyError.
    assert "sim_jw_middle" in out.columns
    assert out["sim_jw_middle"].isna().all()


def test_missing_patid_yields_nan_row_not_crash():
    pairs = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["ZZZ"]})  # ZZZ not in clean
    out = FeatureBuilderV5().build_features(pairs, _clean())
    assert len(out) == 1
    assert out.iloc[0]["sim_jw_first"] != out.iloc[0]["sim_jw_first"]  # NaN


# ─── DirectMatchAdapter ───────────────────────────────────────────────────────
class _InnerModel:
    """Sklearn-shaped stub: class 1 = confident match. Picklable (module-level)."""

    feature_name_ = list(FEATURE_COLS)

    def predict_proba(self, X, pred_contrib=False):
        n = len(X)
        if pred_contrib:
            # (n, n_features + 1), base value last; deterministic and asymmetric
            # so a negation or a reordering cannot pass unnoticed.
            row = np.arange(1, len(FEATURE_COLS) + 2, dtype=float)
            return np.tile(row, (n, 1))
        # [P(not confident match)=0.2, P(confident match)=0.8]
        return np.column_stack([np.full(n, 0.2), np.full(n, 0.8)])


def _X(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({c: [0.0] * n for c in FEATURE_COLS})


def test_adapter_passes_probabilities_through_unswapped():
    """Column 1 must stay P(confident match) — the pipeline reads it as
    match_probability and maps high -> auto_merge."""
    proba = DirectMatchAdapter(_InnerModel()).predict_proba(_X())
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba[:, 1], 0.8)
    np.testing.assert_allclose(proba[:, 0], 0.2)


def test_adapter_reorders_columns_to_training_order():
    adapter = DirectMatchAdapter(_InnerModel())
    # Shuffled columns — the adapter realigns to feature_name_ before predict.
    X = pd.DataFrame({c: [1.0] for c in reversed(FEATURE_COLS)})
    np.testing.assert_allclose(adapter.predict_proba(X)[:, 1], 0.8)


def test_contributions_are_not_negated():
    """THE regression test for v5.

    The served margin IS the inner margin, so contributions pass through with
    their sign intact. Negating them would render a feature that pushes toward
    auto-merge as pushing toward review — plausible-looking and backwards.
    """
    inner = _InnerModel()
    served = DirectMatchAdapter(inner).contributions(_X())
    np.testing.assert_allclose(served, inner.predict_proba(_X(), pred_contrib=True))
    assert served.shape == (2, len(FEATURE_COLS) + 1)


def test_contributions_none_when_inner_model_cannot_produce_them():
    class _NoContrib:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    assert DirectMatchAdapter(_NoContrib()).contributions(_X()) is None


def test_fit_is_refused():
    with pytest.raises(NotImplementedError):
        DirectMatchAdapter(_InnerModel()).fit(_X(), [0, 1])


# ─── 2-tier classify (review_floor = 0.0) ─────────────────────────────────────
def test_classify_two_tier_when_review_floor_zero():
    """With the gate removing non-matches upstream, the ML matcher runs with
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
    # The notebook names artifacts ml_model_confident_match_v5_<ts>.pkl, which
    # the registry's ml_model_*.pkl glob still discovers.
    artifact = tmp_path / "ml_model_confident_match_v5_20260101T000000Z.pkl"
    joblib.dump(DirectMatchAdapter(_InnerModel()), artifact)

    loaded = load_model_artifact(artifact)
    np.testing.assert_allclose(loaded.predict_proba(_X(1))[:, 1], 0.8)
