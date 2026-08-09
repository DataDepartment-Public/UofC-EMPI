"""Unit tests for FSMatcher.to_fs_features (the GBT feature projection).

Operates on a synthetic 'classified' frame — no Splink training required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.contracts import validate_fs_features
from src.models.fs_matcher.base import ClassificationConfig
from src.models.fs_matcher.matcher import FSMatcher


def _matcher(review_floor=0.40):
    return FSMatcher(
        classification_config=ClassificationConfig(
            auto_merge_threshold=0.95, review_floor=review_floor
        )
    )


def _classified() -> pd.DataFrame:
    # 4 pairs spanning the tiers; two per-field feature columns.
    return pd.DataFrame({
        "PATID_A": ["a", "a", "c", "e"],
        "PATID_B": ["b", "d", "f", "g"],
        "match_probability": [0.98, 0.55, 0.30, 0.05],
        "match_weight": [6.0, 0.3, -1.2, -4.0],
        "classification_tier": ["auto_merge", "human_review", "no_match", "no_match"],
        "gamma_SSN_clean": [2, 0, 0, 1],
        "bf_SSN_clean": [30.0, 1.0, 1.0, 0.2],
        "gamma_FirstNM_clean": [2, 2, 0, 0],
        "bf_FirstNM_clean": [5.0, 5.0, 1.0, 1.0],
        # a non-feature column that must NOT be carried over
        "match_key": ["0", "0", "0", "0"],
    })


def test_carries_base_and_feature_columns_only():
    out = _matcher().to_fs_features(_classified(), candidates_only=False)
    for c in ["PATID_A", "PATID_B", "match_probability", "match_weight",
              "classification_tier", "gamma_SSN_clean", "bf_SSN_clean",
              "gamma_FirstNM_clean", "bf_FirstNM_clean"]:
        assert c in out.columns
    assert "match_key" not in out.columns  # non gamma_/bf_ extras dropped
    validate_fs_features(out)


def test_candidates_only_filters_below_review_floor():
    out = _matcher(review_floor=0.40).to_fs_features(_classified(), candidates_only=True)
    # keeps 0.98 and 0.55 (>= 0.40); drops 0.30 and 0.05
    assert len(out) == 2
    assert (out["match_probability"] >= 0.40).all()


def test_candidates_only_false_keeps_all():
    out = _matcher().to_fs_features(_classified(), candidates_only=False)
    assert len(out) == 4


def test_label_join_populates_known_pairs_only():
    labels = pd.DataFrame({
        "PATID_A": ["a", "c"], "PATID_B": ["b", "f"], "silver_label": [1, 0],
    })
    out = _matcher().to_fs_features(
        _classified(), labels_df=labels, label_col="silver_label", candidates_only=False,
    )
    lab = dict(zip(zip(out["PATID_A"], out["PATID_B"]), out["label"]))
    assert lab[("a", "b")] == 1.0
    assert lab[("c", "f")] == 0.0
    assert pd.isna(lab[("a", "d")])  # unlabeled -> null


def test_raises_on_missing_base_columns():
    with pytest.raises(ValueError, match="missing columns"):
        _matcher().to_fs_features(pd.DataFrame({"PATID_A": ["a"], "PATID_B": ["b"]}))
