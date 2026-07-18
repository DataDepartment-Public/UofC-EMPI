"""Unit tests for the shared classifier interface (src/models/base.py).

Covers the `PairClassifier` Protocol itself (structural conformance,
including across the real deterministic-rules and ML-matcher
implementations — splink-free, so the FS matcher isn't constructed here; its
own conformance is implicit in test_fs_matcher_base.py's shared-shape
assertions) and the `to_edges()` projection.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.contracts import ClassificationResults, Edges, validate
from src.models.base import PairClassifier, to_edges
from src.models.deterministic_rules import DeterministicRulesClassifier
from src.models.ml_matcher import MLMatcher


# ─── Protocol conformance ──────────────────────────────────────────────────────
def test_deterministic_rules_classifier_satisfies_protocol():
    assert isinstance(DeterministicRulesClassifier(), PairClassifier)


def test_ml_matcher_satisfies_protocol():
    assert isinstance(MLMatcher(), PairClassifier)


def test_non_conforming_object_fails_protocol_check():
    class NotAClassifier:
        pass

    assert not isinstance(NotAClassifier(), PairClassifier)


# ─── to_edges() projection ─────────────────────────────────────────────────────
def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PATID_A": ["a", "a", "b"],
            "PATID_B": ["b", "c", "d"],
            "model_name": ["m", "m", "m"],
            "score": [0.99, 0.5, 0.10],
            "predicted_tier": ["auto_merge", "human_review", "no_match"],
        }
    )


def test_to_edges_keeps_only_auto_merge_rows():
    out = to_edges(_results(), match_source="ml")
    assert len(out) == 1
    assert out.iloc[0]["PATID_A"] == "a"
    assert out.iloc[0]["PATID_B"] == "b"


def test_to_edges_maps_columns_correctly():
    out = to_edges(_results(), match_source="ml")
    row = out.iloc[0]
    assert row["confidence"] == pytest.approx(0.99)
    assert row["match_source"] == "ml"
    assert row["evidence"] == "m"


def test_to_edges_output_validates_against_edges_contract():
    out = to_edges(_results(), match_source="model")
    validate(out, Edges, allow_empty=False)


def test_to_edges_empty_when_no_auto_merge_rows():
    results = _results()
    results = results[results["predicted_tier"] != "auto_merge"]
    out = to_edges(results, match_source="ml")
    assert out.empty
    assert list(out.columns) == ["PATID_A", "PATID_B", "confidence", "match_source", "evidence"]


# ─── ClassificationResults contract sanity ─────────────────────────────────────
def test_classification_results_rejects_unknown_tier():
    bad = pd.DataFrame({
        "PATID_A": ["a"], "PATID_B": ["b"], "model_name": ["m"],
        "score": [0.5], "predicted_tier": ["maybe"],
    })
    with pytest.raises(Exception):
        validate(bad, ClassificationResults, allow_empty=False)


def test_classification_results_allows_null_score():
    ok = pd.DataFrame({
        "PATID_A": ["a"], "PATID_B": ["b"], "model_name": ["deterministic_rules"],
        "score": [None], "predicted_tier": ["human_review"],
    })
    validate(ok, ClassificationResults, allow_empty=False)
