"""Cluster-quality metrics — the bridge from *pair* labels to *cluster* output.

The label sets in this project (`data/gold_labels/`, `data/silver_labels/`,
`data/synthetic_data/`) are all **pairwise**: one row per candidate pair with
a True/False adjudication. Stage 5 (`src/models/clustering.py`) emits
**clusters**: one `cluster_id` per record. Scoring the pipeline end to end means
reconciling those two shapes, and there are exactly two defensible ways to do
it — this module implements both, because they answer different questions and
have different blind spots.

1. **Restricted pairwise** (`pairwise_against_clusters`) — for every *labeled*
   pair, ask whether the run put the two records in the same cluster, and score
   that against the label. This is the honest headline number: it needs no
   assumption beyond the labels themselves, and it automatically credits (or
   blames) **transitive** merges, which is precisely where clustering differs
   from the pair classifiers upstream. Its blind spot is that it can only see
   pairs someone labeled — a false merge between two records that were never a
   candidate pair is invisible here.

2. **Cluster-level** (`bcubed`, `cluster_recovery`) — lift the pair labels to
   truth *clusters* (connected components of the positive pairs,
   `truth_clusters_from_pairs`) and compare partition to partition with B-cubed
   precision/recall. This does see over-merging that restricted pairwise
   misses, but it inherits a real hazard: **transitive closure of the positive
   pairs**. If the labels say A~B and B~C, the closure asserts A~C even when the
   labeler explicitly marked A~C as a non-match. `ClosureDiagnostics` measures
   exactly that (`n_contradicted` / `contradiction_rate`); when it is not near
   zero, trust (1) and treat (2) as directional.

   `pair_confusion` is (2) reported in (1)'s shape — the familiar TP/FP/FN/TN,
   but counted over every `C(n, 2)` pair in the universe instead of only the
   labeled ones. It is the one view here in which an over-merge between two
   never-adjudicated records is visible at all; read its docstring for what
   that coverage costs.

Both restrict themselves to a **universe** of PATIDs — normally the records
that appear in the label set — and *induce* the predicted partition onto it
(`induce`), since truth is undefined for records nobody labeled.

PHI / HIPAA: aggregate counts and metrics only. No PATIDs, no field values, and
no example rows are returned by anything here.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Mapping

import numpy as np
import pandas as pd

from src.contracts import PATID, PATID_A, PATID_B

logger = logging.getLogger(__name__)

__all__ = [
    "canonical_key",
    "pair_keys",
    "cluster_map",
    "induce",
    "binary_metrics",
    "predict_same_cluster",
    "pairwise_against_clusters",
    "truth_clusters_from_pairs",
    "ClosureDiagnostics",
    "bcubed",
    "pair_confusion",
    "cluster_recovery",
    "size_distribution",
]

PairKey = tuple[str, str]


# ── Pair / partition primitives ──────────────────────────────────────────────
def canonical_key(a: str, b: str) -> PairKey:
    """Order-independent pair key, matching the pipeline's `PATID_A < PATID_B`."""
    return (a, b) if a <= b else (b, a)


def pair_keys(pairs: pd.DataFrame, a_col: str = PATID_A, b_col: str = PATID_B) -> list[PairKey]:
    """Canonical keys for every row of a pair frame, in row order."""
    return [canonical_key(str(a), str(b)) for a, b in zip(pairs[a_col], pairs[b_col])]


def cluster_map(
    clusters: pd.DataFrame, id_col: str = PATID, cluster_col: str = "cluster_id"
) -> dict[str, int]:
    """`ClusterAssignments` frame -> `{PATID: cluster_id}`."""
    return {str(p): int(c) for p, c in zip(clusters[id_col], clusters[cluster_col])}


def induce(partition: Mapping[str, Hashable], universe: Iterable[str]) -> dict[str, Hashable]:
    """Restrict a partition to `universe`, keeping only members it covers.

    Cluster ids are *not* renumbered — every metric here compares labels for
    equality only, so contiguity is irrelevant and preserving the run's ids
    keeps the numbers traceable back to the artifact.
    """
    return {p: partition[p] for p in universe if p in partition}


def size_distribution(partition: Mapping[str, Hashable]) -> dict[str, object]:
    """Aggregate shape of a partition (no ids, no members — safe to log)."""
    sizes = Counter(Counter(partition.values()).values())
    n_clusters = sum(sizes.values())
    return {
        "n_records": len(partition),
        "n_clusters": n_clusters,
        "n_singletons": int(sizes.get(1, 0)),
        "max_size": max(sizes) if sizes else 0,
        "size_histogram": {int(k): int(v) for k, v in sorted(sizes.items())},
    }


# ── (1) Restricted pairwise ──────────────────────────────────────────────────
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """TP/FP/FN/TN + precision/recall/F1 for two boolean arrays."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {
        "n_pairs": int(len(y_true)),
        "positives": int(y_true.sum()),
        "predicted_positive": int(y_pred.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def predict_same_cluster(
    keys: Iterable[PairKey], partition: Mapping[str, Hashable]
) -> tuple[np.ndarray, np.ndarray]:
    """Per pair: (same-cluster?, covered?).

    `covered` is False when either PATID is absent from the partition. That is
    a real and expected state, not a bug: `ClusterAssignments` covers only
    **valid** records, so a labeled pair touching a record that failed the
    Stage-1 validity filter has no prediction at all. Callers decide whether to
    drop those rows or score them as non-merges — `pairwise_against_clusters`
    reports both, because conflating them silently understates recall.
    """
    same, covered = [], []
    for a, b in keys:
        ca, cb = partition.get(a), partition.get(b)
        ok = ca is not None and cb is not None
        covered.append(ok)
        same.append(bool(ok and ca == cb))
    return np.asarray(same, dtype=bool), np.asarray(covered, dtype=bool)


def pairwise_against_clusters(
    labeled: pd.DataFrame,
    partition: Mapping[str, Hashable],
    label_col: str,
) -> dict:
    """Headline end-to-end metric: labeled pairs vs. the run's clustering.

    Returns metrics over the **covered** pairs (both records clustered) plus a
    `coverage` block, and `uncovered_as_negative` — the same metrics with
    uncovered pairs scored as "not merged", which is what a reviewer actually
    experiences (an unclustered record is never surfaced as a duplicate).
    """
    keys = pair_keys(labeled)
    y_true = labeled[label_col].to_numpy().astype(bool)
    same, covered = predict_same_cluster(keys, partition)

    out = binary_metrics(y_true[covered], same[covered])
    out["coverage"] = {
        "labeled_pairs": int(len(keys)),
        "covered_pairs": int(covered.sum()),
        "uncovered_pairs": int((~covered).sum()),
        "uncovered_positives": int((y_true & ~covered).sum()),
        "note": "uncovered = at least one PATID missing from ClusterAssignments "
                "(invalid record, or not present in this run's input).",
    }
    out["uncovered_as_negative"] = binary_metrics(y_true, same)
    return out


# ── (2) Cluster-level ────────────────────────────────────────────────────────
@dataclass
class ClosureDiagnostics:
    """How much the truth partition was *invented* by transitive closure.

    `n_contradicted` is the number that matters: labeled **negative** pairs
    that the closure nonetheless places in the same truth cluster. Every one of
    those is a pair the labeler said is not a match while the closure insists
    it is — each will be counted as a truth positive by any cluster-level
    metric, inflating recall and deflating precision. A non-trivial
    `contradiction_rate` means the label set is not transitively consistent and
    the cluster-level numbers should be read as directional only.
    """

    n_positive_pairs: int
    n_truth_clusters: int
    n_records: int
    max_cluster_size: int
    size_histogram: dict[int, int] = field(default_factory=dict)
    n_implied_pairs: int = 0
    n_labeled_within: int = 0
    n_implied_unlabeled: int = 0
    n_contradicted: int = 0
    contradiction_rate: float | None = None

    def as_dict(self) -> dict:
        return {
            "n_positive_pairs": self.n_positive_pairs,
            "n_truth_clusters": self.n_truth_clusters,
            "n_records": self.n_records,
            "max_cluster_size": self.max_cluster_size,
            "size_histogram": self.size_histogram,
            "n_implied_pairs": self.n_implied_pairs,
            "n_labeled_within": self.n_labeled_within,
            "n_implied_unlabeled": self.n_implied_unlabeled,
            "n_contradicted": self.n_contradicted,
            "contradiction_rate": self.contradiction_rate,
        }


def truth_clusters_from_pairs(
    labeled: pd.DataFrame, label_col: str
) -> tuple[dict[str, int], ClosureDiagnostics]:
    """Lift pair labels to a truth partition via connected components.

    Every record appearing anywhere in `labeled` gets a truth cluster id —
    records only ever seen in negative pairs become singletons. Uses the same
    union-find shape as `src.models.clustering.assign_clusters` so truth and
    prediction are built the same way.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        parent[hi] = lo

    y = labeled[label_col].to_numpy().astype(bool)
    keys = pair_keys(labeled)
    for a, b in keys:  # seed every record so negatives-only records exist
        find(a)
        find(b)
    for (a, b), is_pos in zip(keys, y):
        if is_pos:
            union(a, b)

    roots = sorted({find(p) for p in parent})
    root_to_id = {root: i for i, root in enumerate(roots)}
    truth = {p: root_to_id[find(p)] for p in parent}

    sizes = Counter(Counter(truth.values()).values())
    # sum C(size, 2) over clusters — the pairs the closure asserts.
    implied = sum(n * (size * (size - 1)) // 2 for size, n in sizes.items())
    within = int(sum(1 for a, b in keys if truth[a] == truth[b]))
    contradicted = int(sum(1 for (a, b), pos in zip(keys, y) if not pos and truth[a] == truth[b]))
    n_neg = int((~y).sum())

    diag = ClosureDiagnostics(
        n_positive_pairs=int(y.sum()),
        n_truth_clusters=len(roots),
        n_records=len(truth),
        max_cluster_size=max(sizes) if sizes else 0,
        size_histogram={int(k): int(v) for k, v in sorted(sizes.items())},
        n_implied_pairs=int(implied),
        n_labeled_within=within,
        n_implied_unlabeled=int(implied - within),
        n_contradicted=contradicted,
        contradiction_rate=round(contradicted / n_neg, 6) if n_neg else None,
    )
    if contradicted:
        logger.warning(
            "Truth closure contradicts %d labeled non-match pairs (%.4f%% of "
            "negatives) — cluster-level metrics are directional; prefer the "
            "restricted-pairwise numbers.",
            contradicted, 100 * contradicted / max(n_neg, 1),
        )
    return truth, diag


def _contingency(
    truth: Mapping[str, Hashable], pred: Mapping[str, Hashable]
) -> tuple[dict[tuple, int], dict[Hashable, int], dict[Hashable, int], int]:
    """Joint and marginal counts over the records both partitions cover."""
    joint: dict[tuple, int] = defaultdict(int)
    t_marg: dict[Hashable, int] = defaultdict(int)
    p_marg: dict[Hashable, int] = defaultdict(int)
    n = 0
    for record, t in truth.items():
        p = pred.get(record)
        if p is None:
            continue
        joint[(t, p)] += 1
        t_marg[t] += 1
        p_marg[p] += 1
        n += 1
    return joint, t_marg, p_marg, n


def bcubed(truth: Mapping[str, Hashable], pred: Mapping[str, Hashable]) -> dict:
    """B-cubed precision/recall/F1 over the records both partitions cover.

    Per record `i`, precision is the fraction of `i`'s *predicted* cluster that
    shares `i`'s truth cluster, recall the fraction of `i`'s *truth* cluster
    that landed in `i`'s predicted cluster; both are averaged over records.
    Computed from the contingency table rather than per-record set
    intersections — same numbers, linear instead of quadratic.

    B-cubed is the right partition metric here because it is not dominated by
    the singletons the way pairwise counting is, and it degrades gracefully:
    one over-merge costs precision proportional to the size of the blob it
    created, rather than all-or-nothing as in exact cluster recovery.
    """
    joint, t_marg, p_marg, n = _contingency(truth, pred)
    if not n:
        return {"n_records": 0, "precision": None, "recall": None, "f1": None}

    prec = sum(c * c / p_marg[p] for (_t, p), c in joint.items()) / n
    rec = sum(c * c / t_marg[t] for (t, _p), c in joint.items()) / n
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else None
    return {
        "n_records": n,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4) if f1 is not None else None,
    }


def pair_confusion(truth: Mapping[str, Hashable], pred: Mapping[str, Hashable]) -> dict:
    """Pair-counting confusion over **every** pair of records both partitions cover.

    The same four cells as `binary_metrics`, but the population is all
    `C(n, 2)` record pairs in the universe rather than the pairs somebody
    labeled. That difference is the whole point: `pairwise_against_clusters`
    can only see labeled pairs, so a wrong merge between two records nobody
    adjudicated is invisible to it — and transitive closure merges precisely
    those. Here an over-merge shows up as `FP` whether or not it was labeled.

    Computed from the contingency table, not by enumerating pairs: `TP` is
    `sum C(n_ij, 2)` over cells, and the predicted/truth marginals give the
    row and column totals. O(cells) instead of O(n^2), so a 30k-record
    universe (~450M pairs) costs microseconds.

    The cost of that coverage is that **`FP` now absorbs label
    incompleteness**. Truth here is whatever partition the caller passes; when
    it is the closure of a *sampled* pair-label set, a record whose true links
    were never labeled is a truth singleton, and the pipeline correctly merging
    it reads as a false positive. Precision is therefore a lower bound. Recall
    is unaffected — every link truth asserts is a real one — so read recall as
    an estimate and precision as a floor. With a declared truth partition (the
    synthetic set's `entity_id`) neither caveat applies and both are exact.
    """
    joint, t_marg, p_marg, n = _contingency(truth, pred)

    def n_choose_2(k: int) -> int:
        return k * (k - 1) // 2

    tp = sum(n_choose_2(c) for c in joint.values())
    same_pred = sum(n_choose_2(c) for c in p_marg.values())
    same_truth = sum(n_choose_2(c) for c in t_marg.values())
    fp = same_pred - tp
    fn = same_truth - tp
    tn = n_choose_2(n) - tp - fp - fn

    prec = tp / same_pred if same_pred else None
    rec = tp / same_truth if same_truth else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {
        "n_records": n,
        "n_pairs": n_choose_2(n),
        "positives": same_truth,
        "predicted_positive": same_pred,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def cluster_recovery(truth: Mapping[str, Hashable], pred: Mapping[str, Hashable]) -> dict:
    """Strict, whole-cluster agreement — and where the disagreements go.

    A truth cluster counts as `exact` only if some predicted cluster contains
    exactly its members. The `split` / `merged` / `mixed` breakdown localizes
    the failure mode: `split` = the pipeline under-merged (truth cluster spread
    over several pure predicted clusters), `merged` = it over-merged (truth
    cluster intact but sharing its predicted cluster with outsiders), `mixed` =
    both at once. Non-singleton counts are reported separately because in a
    population that is mostly singletons the overall rate is ~1.0 by default
    and tells you nothing.
    """
    members_t: dict[Hashable, set[str]] = defaultdict(set)
    for record, t in truth.items():
        if record in pred:
            members_t[t].add(record)
    members_p: dict[Hashable, set[str]] = defaultdict(set)
    for record, p in pred.items():
        if record in truth:
            members_p[p].add(record)

    counts = Counter()
    counts_multi = Counter()
    for t, members in members_t.items():
        if not members:
            continue
        pred_ids = {pred[m] for m in members}
        split = len(pred_ids) > 1
        merged = any(len(members_p[p] - members) > 0 for p in pred_ids)
        kind = ("exact" if not split and not merged
                else "mixed" if split and merged
                else "split" if split else "merged")
        counts[kind] += 1
        if len(members) > 1:
            counts_multi[kind] += 1

    total = sum(counts.values())
    total_multi = sum(counts_multi.values())
    return {
        "n_truth_clusters": total,
        "exact": int(counts["exact"]),
        "split": int(counts["split"]),
        "merged": int(counts["merged"]),
        "mixed": int(counts["mixed"]),
        "exact_rate": round(counts["exact"] / total, 4) if total else None,
        "non_singleton": {
            "n_truth_clusters": total_multi,
            "exact": int(counts_multi["exact"]),
            "split": int(counts_multi["split"]),
            "merged": int(counts_multi["merged"]),
            "mixed": int(counts_multi["mixed"]),
            "exact_rate": round(counts_multi["exact"] / total_multi, 4) if total_multi else None,
        },
    }
