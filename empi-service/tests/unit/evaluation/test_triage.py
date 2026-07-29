"""Tests for the three-class routing evaluation.

The property under test throughout is that **routing an ambiguous pair to
review is a success, not a miss**. That is the whole reason this module exists
alongside the binary headline, and it is the thing a future refactor is most
likely to quietly break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cluster_eval import canonical_key
from src.evaluation.triage import (
    ROUTES,
    classification_report,
    confusion_matrix,
    expected_route,
    stage_flow,
    system_route,
    triage_evaluation,
)


def _labels(rows) -> pd.DataFrame:
    """rows: (a, b, match, ambiguous)"""
    return pd.DataFrame(
        [{"PATID_A": a, "PATID_B": b, "label": m, "ambiguous_pair": amb}
         for a, b, m, amb in rows],
        columns=["PATID_A", "PATID_B", "label", "ambiguous_pair"],
    )


# ── expected_route ───────────────────────────────────────────────────────────
def test_three_classes_map_to_the_three_routes():
    labeled = _labels([("1", "2", False, False),
                       ("3", "4", False, True),
                       ("5", "6", True, False)])
    assert expected_route(labeled, "label", "ambiguous_pair").tolist() == [
        "no_match", "human_review", "auto_merge"]


def test_ambiguous_outranks_match_by_default():
    """A pair flagged both is undecidable evidence — it belongs in review."""
    labeled = _labels([("1", "2", True, True)])
    assert expected_route(labeled, "label", "ambiguous_pair").tolist() == ["human_review"]


def test_ambiguous_precedence_is_overridable():
    labeled = _labels([("1", "2", True, True)])
    got = expected_route(labeled, "label", "ambiguous_pair", ambiguous_precedence=False)
    assert got.tolist() == ["auto_merge"]


def test_label_source_without_an_ambiguous_column_is_two_class():
    """Synthetic/silver make no undecidability claim, so they must not
    manufacture a human_review expectation out of nothing."""
    labeled = _labels([("1", "2", True, False), ("3", "4", False, False)])
    got = expected_route(labeled, "label", ambiguous_col=None)
    assert got.tolist() == ["auto_merge", "no_match"]
    assert "human_review" not in set(got)


def test_a_missing_ambiguous_column_degrades_rather_than_raising():
    labeled = _labels([("1", "2", True, False)]).drop(columns=["ambiguous_pair"])
    assert expected_route(labeled, "label", "ambiguous_pair").tolist() == ["auto_merge"]


# ── system_route ─────────────────────────────────────────────────────────────
def _keys(pairs):
    return [canonical_key(a, b) for a, b in pairs]


def test_each_drop_reason_routes_to_no_match():
    keys = _keys([("1", "2"), ("3", "4"), ("5", "6"), ("7", "8")])
    got = system_route(
        keys,
        partition={p: i for i, p in enumerate("12345678")},   # nothing merged
        candidates=set(keys[:3]),                              # ("7","8") never blocked
        rejects={keys[0]},
        gate_drop={keys[1]},
        ml_reject={keys[2]},
    )
    assert got.tolist() == ["no_match"] * 4


def test_same_cluster_is_auto_merge_even_via_transitive_closure():
    """Clustering is the shipped decision — a pair merged only by closure was
    still merged, and scoring it as review would hide over-merges."""
    keys = _keys([("1", "3")])
    got = system_route(keys, partition={"1": 7, "2": 7, "3": 7},
                       candidates=set())      # never a direct candidate
    assert got.tolist() == ["auto_merge"]


def test_survivors_that_were_not_merged_are_human_review():
    keys = _keys([("1", "2")])
    got = system_route(keys, partition={"1": 1, "2": 2}, candidates=set(keys))
    assert got.tolist() == ["human_review"]


def test_ml_auto_merge_tier_alone_does_not_count_as_a_merge():
    """`ml_feeds_clustering` is off, so the matcher's auto_merge tier changes
    no output. The shipped route for those pairs is review."""
    keys = _keys([("1", "2")])
    got = system_route(keys, partition={"1": 1, "2": 2}, candidates=set(keys))
    assert got.tolist() == ["human_review"]


def test_a_pair_touching_an_unclustered_record_is_not_a_merge():
    keys = _keys([("1", "9")])   # "9" was an invalid record, never clustered
    got = system_route(keys, partition={"1": 1}, candidates=set(keys))
    assert got.tolist() == ["human_review"]


# ── matrix and report ────────────────────────────────────────────────────────
def test_matrix_is_always_the_full_three_by_three():
    cm = confusion_matrix(["no_match"], ["no_match"])
    assert list(cm.index) == list(ROUTES) == list(cm.columns)
    assert cm.to_numpy().sum() == 1


def test_perfect_routing_is_a_clean_diagonal():
    rep = classification_report(ROUTES, ROUTES)
    assert rep["accuracy"] == 1.0
    assert all(rep["per_class"][r]["f1"] == 1.0 for r in ROUTES)


def test_ambiguous_routed_to_review_scores_as_correct():
    """The headline property: the binary view calls this a miss, triage does not."""
    expected = ["human_review"] * 10
    actual = ["human_review"] * 10
    rep = classification_report(expected, actual)
    assert rep["per_class"]["human_review"]["recall"] == 1.0
    assert rep["accuracy"] == 1.0


def test_recall_of_a_class_with_no_support_is_none_not_zero():
    """`None` and 0.0 mean different things and only one drags the macro
    average down — a two-class label source must not be penalized for the
    review route it never defined."""
    rep = classification_report(["no_match", "auto_merge"], ["no_match", "auto_merge"])
    assert rep["per_class"]["human_review"]["recall"] is None
    assert rep["macro_avg"]["recall"] == 1.0


def test_precision_counts_the_column_and_recall_counts_the_row():
    # two true auto_merges, one of which we sent to review; plus a non-match we merged
    expected = ["auto_merge", "auto_merge", "no_match"]
    actual = ["auto_merge", "human_review", "auto_merge"]
    m = classification_report(expected, actual)["per_class"]["auto_merge"]
    assert m["support"] == 2 and m["predicted"] == 2
    assert m["recall"] == 0.5 and m["precision"] == 0.5


def test_weighted_average_follows_support():
    expected = ["no_match"] * 9 + ["auto_merge"]
    actual = ["no_match"] * 9 + ["human_review"]      # the lone match misrouted
    rep = classification_report(expected, actual)
    assert rep["weighted_avg"]["recall"] > rep["macro_avg"]["recall"]


def test_matrix_and_report_are_derived_from_the_same_counts():
    """They are rendered separately; they must never disagree."""
    expected = ["auto_merge", "auto_merge", "no_match", "human_review"]
    actual = ["auto_merge", "human_review", "no_match", "human_review"]
    cm = confusion_matrix(expected, actual)
    rep = classification_report(expected, actual)
    for route in ROUTES:
        assert rep["per_class"][route]["support"] == int(cm.loc[route].sum())
        assert rep["per_class"][route]["predicted"] == int(cm[route].sum())


# ── the assembled block ──────────────────────────────────────────────────────
def test_triage_evaluation_serializes_to_plain_json_types():
    labeled = _labels([("1", "2", True, False), ("3", "4", False, True)])
    block = triage_evaluation(
        labeled, "label", _keys([("1", "2"), ("3", "4")]),
        ambiguous_col="ambiguous_pair",
        partition={"1": 1, "2": 1, "3": 3, "4": 4},
        candidates=set(_keys([("1", "2"), ("3", "4")])),
    )
    import json
    json.loads(json.dumps(block))            # must not raise
    assert block["confusion_matrix"]["auto_merge"]["auto_merge"] == 1
    assert block["confusion_matrix"]["human_review"]["human_review"] == 1
    assert block["accuracy"] == 1.0


def test_triage_evaluation_records_the_precedence_it_used():
    """The number is uninterpretable without knowing which rule produced it."""
    labeled = _labels([("1", "2", True, True)])
    block = triage_evaluation(labeled, "label", _keys([("1", "2")]),
                              ambiguous_col="ambiguous_pair",
                              partition={"1": 1, "2": 2}, candidates=set())
    assert block["ambiguous_precedence"] is True
    assert block["ambiguous_col"] == "ambiguous_pair"


def test_two_class_source_reports_no_ambiguous_column():
    labeled = _labels([("1", "2", True, False)])
    block = triage_evaluation(labeled, "label", _keys([("1", "2")]),
                              partition={"1": 1, "2": 1},
                              candidates=set(_keys([("1", "2")])))
    assert block["ambiguous_col"] is None
    assert block["per_class"]["human_review"]["support"] == 0


@pytest.mark.parametrize("n", [0, 1, 50])
def test_support_totals_always_equal_the_labeled_pairs(n):
    labeled = _labels([(str(i), str(i + 1000), i % 2 == 0, i % 3 == 0) for i in range(n)])
    keys = _keys([(str(i), str(i + 1000)) for i in range(n)])
    block = triage_evaluation(labeled, "label", keys, ambiguous_col="ambiguous_pair",
                              partition={}, candidates=set())
    assert sum(block["per_class"][r]["support"] for r in ROUTES) == n
    assert block["n_pairs"] == n


# ── stage flow ───────────────────────────────────────────────────────────────
def _flow_setup():
    """4 labeled pairs, 2 of them true matches."""
    keys = _keys([("1", "2"), ("3", "4"), ("5", "6"), ("7", "8")])
    y = np.array([True, True, False, False])
    return keys, y


def test_flow_rows_follow_pipeline_order():
    keys, y = _flow_setup()
    rows = stage_flow(keys, y, candidates=set(keys), matches=set(), rejects=set(),
                      gate_scored=set(keys), gate_drop=set(),
                      ml_scored=set(keys), partition={})
    assert [r["stage"] for r in rows] == [
        "blocking", "deterministic rules", "non-match gate", "ml matcher",
        "clustering", "-> human review queue"]


def test_stages_that_did_not_run_are_absent_not_zeroed():
    """A zero row for a stage that never ran reads as 'it dropped nothing',
    which is a different claim from 'it was not in this pipeline'."""
    keys, y = _flow_setup()
    rows = stage_flow(keys, y, candidates=set(keys), matches=set(), rejects=set(),
                      partition={})
    assert "non-match gate" not in {r["stage"] for r in rows}
    assert "ml matcher" not in {r["stage"] for r in rows}


def test_each_stage_conserves_its_input():
    """merged + rejected + passed on must equal what the stage saw, or a pair
    has silently vanished from the accounting."""
    keys, y = _flow_setup()
    rows = stage_flow(keys, y, candidates=set(keys[:3]), matches={keys[0]},
                      rejects={keys[2]}, gate_scored={keys[1]}, gate_drop=set(),
                      partition={})
    for r in rows:
        if r["stage"] in ("clustering", "-> human review queue"):
            continue
        assert r["auto_merge"] + r["no_match"] + r["to_next"] == r["saw"], r["stage"]


def test_true_lost_counts_only_unrecoverable_drops():
    keys, y = _flow_setup()
    rows = {r["stage"]: r for r in stage_flow(
        keys, y, candidates=set(keys), matches=set(),
        rejects={keys[0]},                    # a true match, rejected by rules
        gate_scored=set(keys[1:]), gate_drop={keys[1]},   # another, gate-dropped
        partition={})}
    assert rows["deterministic rules"]["true_lost"] == 1
    assert rows["non-match gate"]["true_lost"] == 1
    assert rows["blocking"]["true_lost"] == 0


def test_a_pair_that_never_blocked_is_lost_at_blocking():
    keys, y = _flow_setup()
    rows = {r["stage"]: r for r in stage_flow(
        keys, y, candidates=set(keys[1:]), matches=set(), rejects=set(), partition={})}
    assert rows["blocking"]["no_match"] == 1
    assert rows["blocking"]["true_lost"] == 1


def test_advisory_ml_tier_is_flagged_and_loses_nothing():
    """With ml_feeds_clustering off the matcher's verdict changes no output, so
    it must not be credited with merges nor blamed for losses."""
    keys, y = _flow_setup()
    rows = {r["stage"]: r for r in stage_flow(
        keys, y, candidates=set(keys), matches=set(), rejects=set(),
        ml_scored=set(keys), ml_auto={keys[0]}, ml_reject={keys[1]},
        partition={}, ml_binding=False)}
    ml = rows["ml matcher"]
    assert ml["binding"] is False and ml["auto_merge"] == 1
    assert ml["true_lost"] == 0
    assert "ADVISORY" in ml["note"]


def test_binding_ml_tier_is_blamed_for_its_rejects():
    keys, y = _flow_setup()
    rows = {r["stage"]: r for r in stage_flow(
        keys, y, candidates=set(keys), matches=set(), rejects=set(),
        ml_scored=set(keys), ml_reject={keys[0]}, partition={}, ml_binding=True)}
    assert rows["ml matcher"]["true_lost"] == 1


def test_clustering_row_counts_the_shipped_merges():
    keys, y = _flow_setup()
    rows = {r["stage"]: r for r in stage_flow(
        keys, y, candidates=set(keys), matches=set(), rejects=set(),
        partition={"1": 1, "2": 1, "3": 2, "4": 3})}
    assert rows["clustering"]["auto_merge"] == 1
    assert rows["clustering"]["auto_merge_true"] == 1


def test_review_queue_is_whatever_the_last_filter_passed_on():
    keys, y = _flow_setup()
    rows = {r["stage"]: r for r in stage_flow(
        keys, y, candidates=set(keys), matches={keys[0]}, rejects={keys[2]},
        partition={})}
    assert rows["-> human review queue"]["saw"] == rows["deterministic rules"]["to_next"] == 2
