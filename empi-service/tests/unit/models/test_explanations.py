"""Unit tests for per-pair SHAP explanations (src/models/explanations.py).

The properties that matter here are not "does it run" but:

* contributions + base value reconstruct the model's raw margin, and the
  sigmoid of that margin is the score the pipeline recorded;
* the **sign convention** is normalized across both models — the ML matcher's
  inner model is trained with class 1 = ambiguous, so un-negated contributions
  would render a waterfall that reads exactly backwards while looking
  entirely plausible;
* the payload's precomputed geometry is internally consistent, so the UI can
  draw rectangles without doing any arithmetic of its own.

Uses a real LightGBM model, because `pred_contrib` is the thing under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import explanations as E
from src.models.ml_matcher.lightgbm_v3 import (
    CATEGORICAL_FEATURES,
    COMPARE_LEVELS,
    FEATURE_COLS,
    MatchProbabilityAdapter,
)

pytest.importorskip("lightgbm")


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for col in FEATURE_COLS:
        if col in CATEGORICAL_FEATURES:
            data[col] = pd.Categorical(
                rng.choice(COMPARE_LEVELS, n), categories=COMPARE_LEVELS
            )
        else:
            values = rng.random(n)
            values[rng.random(n) < 0.25] = np.nan   # NaN like the real features
            data[col] = values
    return pd.DataFrame(data)[FEATURE_COLS]


@pytest.fixture(scope="module")
def model():
    import lightgbm as lgb

    X = _frame(2000)
    y = (X["sim_dob"].fillna(0) + X["sim_jw_last"].fillna(0) > 1.0).astype(int)
    m = lgb.LGBMClassifier(n_estimators=40, num_leaves=7, random_state=42, verbose=-1)
    m.fit(X, y, categorical_feature=CATEGORICAL_FEATURES)
    return m


@pytest.fixture(scope="module")
def X():
    return _frame(50, seed=7)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ─── contributions ────────────────────────────────────────────────────────────
def test_contributions_reconstruct_the_models_score(model, X):
    """base_value + Σ contributions == raw margin, and sigmoid(margin) is the
    probability the pipeline records. If this drifts, the waterfall no longer
    adds up to the number shown next to it."""
    contrib = E.compute_contributions(model, X)
    margin = contrib.sum(axis=1)
    assert np.allclose(_sigmoid(margin), model.predict_proba(X)[:, 1], atol=1e-6)


def test_contributions_shape_is_features_plus_base(model, X):
    contrib = E.compute_contributions(model, X)
    assert contrib.shape == (len(X), len(FEATURE_COLS) + 1)


def test_adapter_negates_so_positive_means_confident_match(model, X):
    """THE regression test for the ML matcher.

    The inner model's class 1 is *ambiguous*; the served score is
    `P(confident match) = 1 - P(ambiguous)`. So the served margin is the
    negation of the inner margin, and every contribution must flip with it.
    Without the flip, a feature pushing a pair toward review renders as
    pushing it toward auto-merge.
    """
    adapter = MatchProbabilityAdapter(model)
    inner = np.asarray(model.predict_proba(X, pred_contrib=True))
    served = E.compute_contributions(adapter, X)

    assert np.allclose(served, -inner)
    # And the reconstruction still holds against the *served* probability.
    assert np.allclose(
        _sigmoid(served.sum(axis=1)), adapter.predict_proba(X)[:, 1], atol=1e-6,
    )


def test_gate_contributions_are_not_negated(model, X):
    """The gate's class 1 is already P(plausible) — flipping it too would be
    just as wrong, in the other direction."""
    assert np.allclose(
        E.compute_contributions(model, X),
        np.asarray(model.predict_proba(X, pred_contrib=True)),
    )


def test_model_without_contribution_support_returns_none():
    class _Plain:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    assert E.compute_contributions(_Plain(), pd.DataFrame({"a": [1.0]})) is None
    assert E.compute_contributions(None, pd.DataFrame({"a": [1.0]})) is None


def test_supports_contributions_detects_both_shapes(model):
    assert E.supports_contributions(model)
    assert E.supports_contributions(MatchProbabilityAdapter(model))
    assert not E.supports_contributions(object())


# ─── the persisted frame ──────────────────────────────────────────────────────
def _explanation_frame(model, X):
    pairs = pd.DataFrame(
        {"PATID_A": [f"{i:05d}" for i in range(len(X))],
         "PATID_B": [f"{i + 500:05d}" for i in range(len(X))]}
    )
    proba = model.predict_proba(X)[:, 1]
    return E.build_explanation_frame(
        pairs, X, E.compute_contributions(model, X),
        model_name="nonmatch_gate", scores=proba,
        tiers=np.where(proba >= 0.3, "human_review", "no_match"),
    )


def test_frame_carries_a_contribution_and_a_value_per_feature(model, X):
    frame = _explanation_frame(model, X)
    for col in FEATURE_COLS:
        assert f"{E.SHAP_PREFIX}{col}" in frame.columns
        assert f"{E.FEAT_PREFIX}{col}" in frame.columns


def test_frame_satisfies_the_contract(model, X):
    from src.contracts import validate_pair_explanations

    validate_pair_explanations(_explanation_frame(model, X))


def test_contract_rejects_contributions_without_values(model, X):
    from src.contracts import validate_pair_explanations

    frame = _explanation_frame(model, X).drop(columns=[f"{E.FEAT_PREFIX}sim_dob"])
    with pytest.raises(ValueError, match="shap_.*/feat_.* columns disagree"):
        validate_pair_explanations(frame)


def test_frame_is_sorted_by_pair_key(model, X):
    """Sorted at write time so the endpoint's single-pair filtered read is a
    row-group pushdown rather than a full scan."""
    frame = _explanation_frame(model, X)
    keys = list(zip(frame["PATID_A"], frame["PATID_B"]))
    assert keys == sorted(keys)


def test_categorical_values_survive_as_labels(model, X):
    frame = _explanation_frame(model, X)
    values = set(frame[f"{E.FEAT_PREFIX}sound_first"].dropna())
    assert values <= set(COMPARE_LEVELS)


# ─── the payload ──────────────────────────────────────────────────────────────
def test_payload_geometry_is_a_connected_waterfall(model, X):
    """Each bar starts where the previous ended, the first starts at the base
    value, and the last ends at the margin — so the UI never computes an
    offset."""
    frame = _explanation_frame(model, X)
    payload = E.build_payload(frame.iloc[0])

    assert payload["features"][0]["start"] == pytest.approx(payload["base_value"])
    for prev, nxt in zip(payload["features"], payload["features"][1:]):
        assert nxt["start"] == pytest.approx(prev["end"])
    assert payload["features"][-1]["end"] == pytest.approx(payload["final_margin"])


def test_payload_margin_reconstructs_the_recorded_score(model, X):
    frame = _explanation_frame(model, X)
    payload = E.build_payload(frame.iloc[0])
    assert _sigmoid(payload["final_margin"]) == pytest.approx(
        payload["decision"]["score"], abs=1e-5
    )


def test_payload_features_are_ranked_by_absolute_contribution(model, X):
    frame = _explanation_frame(model, X)
    magnitudes = [abs(f["shap"]) for f in E.build_payload(frame.iloc[0])["features"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_payload_direction_matches_the_sign(model, X):
    frame = _explanation_frame(model, X)
    for feature in E.build_payload(frame.iloc[0])["features"]:
        expected = "positive" if feature["shap"] >= 0 else "negative"
        assert feature["direction"] == expected


def test_payload_axis_contains_every_bar(model, X):
    frame = _explanation_frame(model, X)
    payload = E.build_payload(frame.iloc[0])
    lo, hi = payload["axis"]["min"], payload["axis"]["max"]
    for feature in payload["features"]:
        assert lo <= feature["start"] <= hi
        assert lo <= feature["end"] <= hi


def test_payload_cumulative_prob_tracks_the_running_total(model, X):
    frame = _explanation_frame(model, X)
    for feature in E.build_payload(frame.iloc[0])["features"]:
        assert feature["cumulative_prob"] == pytest.approx(_sigmoid(feature["end"]))


def test_payload_labels_every_feature(model, X):
    frame = _explanation_frame(model, X)
    for feature in E.build_payload(frame.iloc[0])["features"]:
        assert feature["label"] and feature["label"] != feature["name"]


def test_unlabelled_feature_degrades_to_a_readable_name():
    assert E.feature_label("some_new_feature") == "Some new feature"


def test_payload_renders_missing_feature_values_as_null(model):
    """NaN is meaningful — it is what the model branched on — so it must reach
    the UI as an explicit null, not as the string 'nan'."""
    X = _frame(5, seed=3)
    X.loc[:, "sim_jw_middle"] = np.nan
    frame = _explanation_frame(model, X)
    payload = E.build_payload(frame.iloc[0])
    middle = next(f for f in payload["features"] if f["name"] == "sim_jw_middle")
    assert middle["value"] is None and middle["display_value"] is None
