"""Unit tests for the pair-labels-to-clusters metric core."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.cluster_eval import (
    bcubed,
    binary_metrics,
    canonical_key,
    cluster_map,
    cluster_recovery,
    induce,
    pair_confusion,
    pair_keys,
    pairwise_against_clusters,
    predict_same_cluster,
    size_distribution,
    truth_clusters_from_pairs,
)


def _labeled(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["PATID_A", "PATID_B", "label"])


# ── primitives ───────────────────────────────────────────────────────────────
def test_canonical_key_is_order_independent():
    assert canonical_key("b", "a") == canonical_key("a", "b") == ("a", "b")


def test_pair_keys_canonicalizes_rows_out_of_order():
    df = _labeled([("p2", "p1", True), ("p1", "p3", True)])
    assert pair_keys(df) == [("p1", "p2"), ("p1", "p3")]


def test_cluster_map_coerces_types():
    clusters = pd.DataFrame({"PATID": ["p1", "p2"], "cluster_id": [0, 1]})
    assert cluster_map(clusters) == {"p1": 0, "p2": 1}


def test_induce_keeps_only_covered_members_and_preserves_ids():
    assert induce({"a": 7, "b": 8, "c": 9}, ["a", "c", "zz"]) == {"a": 7, "c": 9}


def test_size_distribution_counts_singletons():
    dist = size_distribution({"a": 0, "b": 0, "c": 1})
    assert dist["n_records"] == 3
    assert dist["n_clusters"] == 2
    assert dist["n_singletons"] == 1
    assert dist["max_size"] == 2


# ── restricted pairwise ──────────────────────────────────────────────────────
def test_binary_metrics_basic_counts():
    m = binary_metrics([True, True, False, False], [True, False, True, False])
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5


def test_predict_same_cluster_flags_uncovered_records():
    keys = [("a", "b"), ("a", "zz")]
    same, covered = predict_same_cluster(keys, {"a": 0, "b": 0})
    assert list(same) == [True, False]
    assert list(covered) == [True, False]


def test_pairwise_against_clusters_credits_transitive_merges():
    """A pair no classifier ever saw still counts if clustering merged it."""
    labeled = _labeled([("p1", "p3", True)])
    # p1 and p3 are in one cluster only via p2 — nothing scored (p1, p3).
    metrics = pairwise_against_clusters(labeled, {"p1": 0, "p2": 0, "p3": 0}, "label")
    assert metrics["TP"] == 1 and metrics["recall"] == 1.0


def test_pairwise_against_clusters_separates_uncovered_from_negative():
    """An unclustered record is not the same thing as a scored non-merge."""
    labeled = _labeled([("p1", "p2", True), ("p3", "p4", True)])
    partition = {"p1": 0, "p2": 0}  # p3/p4 invalid -> absent from the run
    m = pairwise_against_clusters(labeled, partition, "label")

    assert m["n_pairs"] == 1 and m["recall"] == 1.0  # covered pairs only
    assert m["coverage"]["uncovered_pairs"] == 1
    assert m["coverage"]["uncovered_positives"] == 1
    assert m["uncovered_as_negative"]["recall"] == 0.5  # the reviewer's view


# ── truth closure ────────────────────────────────────────────────────────────
def test_truth_clusters_chain_positive_pairs():
    labeled = _labeled([("p1", "p2", True), ("p2", "p3", True), ("p4", "p5", False)])
    truth, diag = truth_clusters_from_pairs(labeled, "label")

    assert truth["p1"] == truth["p2"] == truth["p3"]
    assert truth["p4"] != truth["p5"]  # negatives only -> singletons
    assert diag.n_truth_clusters == 3
    assert diag.n_implied_pairs == 3  # C(3,2) from the chained cluster
    assert diag.n_implied_unlabeled == 1  # (p1, p3) was never labeled


def test_closure_contradiction_is_detected_and_counted():
    """A~B, B~C positive but A~C explicitly negative — the hazard the
    cluster-level metrics inherit and `ClosureDiagnostics` exists to surface."""
    labeled = _labeled([
        ("p1", "p2", True), ("p2", "p3", True), ("p1", "p3", False),
    ])
    _truth, diag = truth_clusters_from_pairs(labeled, "label")

    assert diag.n_contradicted == 1
    assert diag.contradiction_rate == 1.0  # the only negative pair


def test_truth_closure_has_no_contradictions_when_labels_are_consistent():
    labeled = _labeled([("p1", "p2", True), ("p3", "p4", False)])
    _truth, diag = truth_clusters_from_pairs(labeled, "label")
    assert diag.n_contradicted == 0


# ── B-cubed ──────────────────────────────────────────────────────────────────
def test_bcubed_perfect_partition():
    truth = pred = {"a": 0, "b": 0, "c": 1}
    m = bcubed(truth, pred)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_bcubed_penalizes_over_merge_on_precision_only():
    truth = {"a": 0, "b": 0, "c": 1}
    pred = {"a": 0, "b": 0, "c": 0}  # everything merged into one blob
    m = bcubed(truth, pred)
    assert m["precision"] == pytest.approx(5 / 9, abs=1e-4)
    assert m["recall"] == 1.0


def test_bcubed_penalizes_under_merge_on_recall_only():
    truth = {"a": 0, "b": 0, "c": 1}
    pred = {"a": 0, "b": 1, "c": 2}  # all singletons
    m = bcubed(truth, pred)
    assert m["precision"] == 1.0
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)


def test_bcubed_ignores_records_the_prediction_does_not_cover():
    truth = {"a": 0, "b": 0, "c": 1}
    m = bcubed(truth, {"a": 0, "b": 0})
    assert m["n_records"] == 2


# ── pair-counting confusion ──────────────────────────────────────────────────
def test_pair_confusion_perfect_partition():
    truth = pred = {"a": 0, "b": 0, "c": 1}
    m = pair_confusion(truth, pred)
    # 3 records -> 3 pairs; only (a, b) is a positive in either partition.
    assert (m["n_pairs"], m["TP"], m["FP"], m["FN"], m["TN"]) == (3, 1, 0, 0, 2)
    assert m["precision"] == 1.0 and m["recall"] == 1.0


def test_pair_confusion_counts_an_over_merge_as_fp():
    truth = {"a": 0, "b": 0, "c": 1}
    pred = {"a": 0, "b": 0, "c": 0}  # c wrongly pulled in
    m = pair_confusion(truth, pred)
    assert (m["TP"], m["FP"], m["FN"]) == (1, 2, 0)  # (a,c) and (b,c) are wrong
    assert m["precision"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["recall"] == 1.0


def test_pair_confusion_counts_an_under_merge_as_fn():
    truth = {"a": 0, "b": 0, "c": 1}
    pred = {"a": 0, "b": 1, "c": 2}  # all singletons
    m = pair_confusion(truth, pred)
    assert (m["TP"], m["FP"], m["FN"]) == (0, 0, 1)
    assert m["recall"] == 0.0


def test_pair_confusion_sees_an_unlabeled_over_merge_that_pairwise_misses():
    """The reason this function exists.

    Truth says a~b and c~d; the run welds all four into one cluster. Only the
    two positive pairs were ever labeled, so `pairwise_against_clusters` scores
    a perfect 2/2 and reports no error at all — the four cross pairs it never
    saw are exactly the over-merge.
    """
    labeled = _labeled([("a", "b", True), ("c", "d", True)])
    truth, _ = truth_clusters_from_pairs(labeled, "label")
    pred = {"a": 0, "b": 0, "c": 0, "d": 0}

    restricted = pairwise_against_clusters(labeled, pred, "label")
    assert restricted["FP"] == 0 and restricted["precision"] == 1.0

    m = pair_confusion(truth, pred)
    assert (m["TP"], m["FP"]) == (2, 4)
    assert m["precision"] == pytest.approx(1 / 3, abs=1e-4)


def test_pair_confusion_ignores_records_the_prediction_does_not_cover():
    truth = {"a": 0, "b": 0, "c": 1}
    m = pair_confusion(truth, {"a": 0, "b": 0})
    assert m["n_records"] == 2 and m["n_pairs"] == 1


def test_pair_confusion_cells_sum_to_every_pair_in_the_universe():
    truth = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 2}
    pred = {"a": 0, "b": 0, "c": 0, "d": 1, "e": 1}
    m = pair_confusion(truth, pred)
    assert m["TP"] + m["FP"] + m["FN"] + m["TN"] == m["n_pairs"] == 10


# ── cluster recovery ─────────────────────────────────────────────────────────
def test_cluster_recovery_classifies_split_merged_and_exact():
    truth = {
        "a": "T1", "b": "T1",          # split across two predicted clusters
        "c": "T2", "d": "T2",          # exact
        "e": "T3",                     # merged with outsiders
        "f": "T4",
    }
    pred = {
        "a": 0, "b": 1,
        "c": 2, "d": 2,
        "e": 3, "f": 3,
    }
    r = cluster_recovery(truth, pred)
    assert r["split"] == 1      # T1
    assert r["exact"] == 1      # T2
    assert r["merged"] == 2     # T3 and T4 share predicted cluster 3
    assert r["n_truth_clusters"] == 4


def test_cluster_recovery_reports_non_singletons_separately():
    """In a mostly-singleton population the overall rate is uninformative."""
    truth = {"a": "T1", "b": "T1", **{f"s{i}": f"S{i}" for i in range(10)}}
    pred = {"a": 0, "b": 1, **{f"s{i}": 10 + i for i in range(10)}}
    r = cluster_recovery(truth, pred)

    assert r["exact_rate"] == pytest.approx(10 / 11, abs=1e-4)  # flattered
    assert r["non_singleton"]["n_truth_clusters"] == 1
    assert r["non_singleton"]["exact_rate"] == 0.0              # the real story
