"""Three-class triage evaluation: where a pair *should* be routed vs. where the
pipeline actually routed it.

The binary "same cluster or not" headline in `cluster_eval` answers *did we
merge correctly*, and it is the right number for trust. It is the wrong number
for **recall**, because it scores the system against a target it was never asked
to hit: a pair the gold labeler marked `ambiguous_pair` is one no human could
resolve from the record alone, and the correct behavior for the pipeline is to
route it to a reviewer — not to merge it. The binary view counts that correct
routing as a miss, so it understates recall by exactly the ambiguous population.

This module scores the decision the pipeline actually makes. Both sides use one
vocabulary, `src.contracts.CLASSIFICATION_TIERS`:

    gold class                      expected route      system route derived from
    ─────────────────────────────────────────────────────────────────────────────
    non-match  (neither flag)       no_match            rules reject, gate drop,
                                                        or never blocked
    ambiguous  (ambiguous_pair)     human_review        survived to the end,
                                                        not merged
    match      (final_gold_label)   auto_merge          in the same cluster

so the confusion matrix is square, its diagonal is "routed correctly", and a
standard per-class precision/recall/F1 report applies.

**`ambiguous` takes precedence over `match`** when a pair carries both flags.
The class means "the labeler could not decide", which is a statement about the
*evidence*, and evidence that does not support a confident call should reach a
human whatever the eventual adjudication was. Scoring such a pair as a required
auto-merge would penalize the pipeline for correct caution. `AMBIGUOUS_WINS`
documents the choice; flip `ambiguous_precedence=False` to score gold's match
flag as authoritative instead.

What this measures is the **shipped** decision, whatever the configuration. With
`ml_feeds_clustering` on (the default) the matcher's `auto_merge` tier forms real
merge edges and shows up under `auto_merge`; with it off those pairs land in
`human_review`, correctly, because that is where they end up. The
`stage_pairwise` block is where a stage's own decision is scored in isolation.

PHI / HIPAA: counts and rates only, never pair ids.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.contracts import TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH
from src.evaluation.cluster_eval import PairKey, predict_same_cluster

logger = logging.getLogger(__name__)

__all__ = [
    "stage_flow",
    "ROUTES",
    "GOLD_CLASSES",
    "GOLD_TO_ROUTE",
    "AMBIGUOUS_WINS",
    "expected_route",
    "system_route",
    "confusion_matrix",
    "classification_report",
    "triage_evaluation",
]

#: Display order, least to most confident. The matrix then reads with the
#: "system was more aggressive than the label" errors above the diagonal.
ROUTES: tuple[str, str, str] = (TIER_NO_MATCH, TIER_HUMAN_REVIEW, TIER_AUTO_MERGE)

#: The adjudicated classes in the gold file, in the same order.
GOLD_CLASSES: tuple[str, str, str] = ("non-match", "ambiguous", "match")

#: Which route each adjudicated class should have taken.
GOLD_TO_ROUTE: dict[str, str] = dict(zip(GOLD_CLASSES, ROUTES))

#: See the module docstring — a pair flagged both ambiguous and match is scored
#: as `human_review`, not `auto_merge`.
AMBIGUOUS_WINS = True


def expected_route(
    labeled: pd.DataFrame,
    label_col: str,
    ambiguous_col: str | None = None,
    *,
    ambiguous_precedence: bool = AMBIGUOUS_WINS,
) -> pd.Series:
    """Where each labeled pair *should* be routed.

    With no `ambiguous_col` the label set is two-class (e.g. synthetic, silver),
    and the result contains no `human_review` — that is honest rather than
    degenerate: such a source makes no claim about which pairs are undecidable,
    so it cannot score the review route at all.
    """
    match = labeled[label_col].to_numpy(dtype=bool)
    if ambiguous_col is None or ambiguous_col not in labeled.columns:
        route = np.where(match, TIER_AUTO_MERGE, TIER_NO_MATCH)
        return pd.Series(route, index=labeled.index, name="expected")

    ambiguous = labeled[ambiguous_col].to_numpy(dtype=bool)
    if ambiguous_precedence:
        route = np.where(ambiguous, TIER_HUMAN_REVIEW,
                         np.where(match, TIER_AUTO_MERGE, TIER_NO_MATCH))
    else:
        route = np.where(match, TIER_AUTO_MERGE,
                         np.where(ambiguous, TIER_HUMAN_REVIEW, TIER_NO_MATCH))
    return pd.Series(route, index=labeled.index, name="expected")


def system_route(
    keys: Sequence[PairKey],
    *,
    partition: Mapping[str, object],
    candidates: set[PairKey],
    rejects: set[PairKey] = frozenset(),
    gate_drop: set[PairKey] = frozenset(),
    ml_reject: set[PairKey] = frozenset(),
    index: pd.Index | None = None,
) -> pd.Series:
    """Where the pipeline actually routed each pair, in precedence order.

    1. In the same cluster -> `auto_merge`. This is checked first and is the
       *shipped* merge decision, so it correctly captures pairs merged by
       transitive closure rather than by a direct edge.
    2. Never a candidate -> `no_match`. Blocking never emitted the pair, so no
       stage ever saw it. Scoring it as anything else would hide blocking
       misses, which are unrecoverable.
    3. Rejected by the deterministic rules, dropped by the Stage-4.25 gate, or
       tiered `no_match` by the ML matcher -> `no_match`.
    4. Anything else that survived -> `human_review`. This is the reviewer
       queue: the pipeline declined to decide.
    """
    same, covered = predict_same_cluster(keys, partition)
    merged = same & covered

    route = np.full(len(keys), TIER_HUMAN_REVIEW, dtype=object)
    dropped = rejects | gate_drop | ml_reject
    for i, key in enumerate(keys):
        if merged[i]:
            route[i] = TIER_AUTO_MERGE
        elif key not in candidates or key in dropped:
            route[i] = TIER_NO_MATCH
    return pd.Series(route, index=index if index is not None else pd.RangeIndex(len(keys)),
                     name="system")


def confusion_matrix(expected: Iterable[str], actual: Iterable[str]) -> pd.DataFrame:
    """Rows = expected route (from the labels), columns = the system's route.

    Always the full 3x3 in `ROUTES` order even when a class is unpopulated, so
    two reports are always directly comparable cell for cell.
    """
    df = pd.DataFrame({"expected": list(expected), "system": list(actual)})
    return (
        df.groupby(["expected", "system"]).size().unstack(fill_value=0)
        .reindex(index=ROUTES, columns=ROUTES, fill_value=0)
        .astype(int)
    )


def classification_report(expected: Iterable[str], actual: Iterable[str]) -> dict:
    """Per-class precision / recall / F1 / support, plus accuracy and averages.

    Computed straight from the confusion matrix rather than via sklearn: the
    matrix is already the quantity of interest, the arithmetic is three lines,
    and deriving both from one place means the printed table and the metrics
    can never disagree.

    A class with no support gets `None` for recall (not 0.0) — "we never had one
    to route" and "we routed every one wrong" are different facts, and folding
    them together silently drags the macro average down.
    """
    cm = confusion_matrix(expected, actual)
    total = int(cm.to_numpy().sum())
    correct = int(np.trace(cm.to_numpy()))

    per_class: dict[str, dict] = {}
    for route in ROUTES:
        tp = int(cm.loc[route, route])
        support = int(cm.loc[route].sum())          # actually this class
        predicted = int(cm[route].sum())            # called this class
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else None)
        per_class[route] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "support": support,
            "predicted": predicted,
        }

    def _avg(metric: str, weighted: bool) -> float | None:
        pairs = [(per_class[r][metric], per_class[r]["support"]) for r in ROUTES
                 if per_class[r][metric] is not None and per_class[r]["support"]]
        if not pairs:
            return None
        if weighted:
            denom = sum(w for _, w in pairs)
            return round(sum(v * w for v, w in pairs) / denom, 4)
        return round(sum(v for v, _ in pairs) / len(pairs), 4)

    return {
        "per_class": per_class,
        "accuracy": round(correct / total, 4) if total else None,
        "macro_avg": {m: _avg(m, False) for m in ("precision", "recall", "f1")},
        "weighted_avg": {m: _avg(m, True) for m in ("precision", "recall", "f1")},
        "n_pairs": total,
    }


def triage_evaluation(
    labeled: pd.DataFrame,
    label_col: str,
    keys: Sequence[PairKey],
    *,
    ambiguous_col: str | None = None,
    ambiguous_precedence: bool = AMBIGUOUS_WINS,
    **routing,
) -> dict:
    """The whole three-class block for one report: matrix + classification report.

    `routing` is forwarded to `system_route`. The returned matrix is nested
    dicts rather than a frame so the report serializes to JSON unchanged.
    """
    expected = expected_route(labeled, label_col, ambiguous_col,
                              ambiguous_precedence=ambiguous_precedence)
    actual = system_route(keys, index=labeled.index, **routing)
    cm = confusion_matrix(expected, actual)

    three_class = ambiguous_col is not None and ambiguous_col in labeled.columns
    if not three_class:
        logger.info(
            "Triage scored two-class: label source has no ambiguous column, so "
            "the human_review route has no expected population."
        )
    return {
        "ambiguous_col": ambiguous_col if three_class else None,
        "ambiguous_precedence": ambiguous_precedence if three_class else None,
        "classes": list(ROUTES),
        "gold_class_names": dict(zip(ROUTES, GOLD_CLASSES)),
        "confusion_matrix": {row: cm.loc[row].to_dict() for row in ROUTES},
        **classification_report(expected, actual),
    }


# ── Stage flow ───────────────────────────────────────────────────────────────
def _split(keys: Sequence[PairKey], y_true: np.ndarray,
           member: set[PairKey]) -> tuple[int, int]:
    """(count, true-matches-among-them) for the labeled pairs in `member`."""
    mask = np.array([k in member for k in keys], dtype=bool)
    return int(mask.sum()), int((mask & y_true).sum())


def stage_flow(
    keys: Sequence[PairKey],
    y_true: np.ndarray,
    *,
    candidates: set[PairKey],
    matches: set[PairKey],
    rejects: set[PairKey],
    gate_scored: set[PairKey] = frozenset(),
    gate_drop: set[PairKey] = frozenset(),
    ml_scored: set[PairKey] = frozenset(),
    ml_auto: set[PairKey] = frozenset(),
    ml_reject: set[PairKey] = frozenset(),
    partition: Mapping[str, object] | None = None,
    ml_binding: bool = False,
) -> list[dict]:
    """Per stage: what came in, and what the stage did with it.

    The per-stage precision/recall table answers "how good is this stage's
    judgement". This answers the more basic question people actually ask first
    — *how many pairs did it merge, reject, and pass on* — which the P/R view
    obscures badly, because "false positive" means "kept this pair alive" at a
    filter stage and "wrongly merged two patients" at the output. Those are not
    comparable numbers and putting them in one column invites reading a
    recall-first funnel as if it were broken.

    Every row is counted over the **labeled pairs only**, so the columns are
    directly comparable to the rest of the report rather than to the run's
    global counts (which `RunManifest.counts` already carries).

    `true_lost` is the column that matters: true matches this stage discarded
    where no later stage can recover them. A stage that passes work forward has
    `true_lost = 0` by construction however imprecise it looks.
    """
    all_keys = set(keys)
    n, n_true = len(keys), int(y_true.sum())

    def row(stage, saw, saw_true, auto=0, auto_true=0, no_match=0, lost=0,
            to_next=0, to_next_true=0, binding=True, note="") -> dict:
        return {
            "stage": stage, "saw": saw, "saw_true": saw_true,
            "auto_merge": auto, "auto_merge_true": auto_true,
            "no_match": no_match, "true_lost": lost,
            "to_next": to_next, "to_next_true": to_next_true,
            "binding": binding, "note": note,
        }

    rows: list[dict] = []

    # Blocking — emits candidates or the pair is never considered again.
    n_cand, n_cand_true = _split(keys, y_true, candidates)
    rows.append(row(
        "blocking", n, n_true, no_match=n - n_cand, lost=n_true - n_cand_true,
        to_next=n_cand, to_next_true=n_cand_true,
        note="over-generates by design; only recall matters here",
    ))

    # Deterministic rules — the only binding auto-merge in the default config.
    n_auto, n_auto_true = _split(keys, y_true, matches & candidates)
    n_rej, n_rej_true = _split(keys, y_true, rejects & candidates)
    passed = (candidates & all_keys) - matches - rejects
    n_pass, n_pass_true = _split(keys, y_true, passed)
    rows.append(row(
        "deterministic rules", n_cand, n_cand_true,
        auto=n_auto, auto_true=n_auto_true, no_match=n_rej, lost=n_rej_true,
        to_next=n_pass, to_next_true=n_pass_true,
        note="auto_merge tier is binding — this is what clustering unions",
    ))

    # Stage 4.25 gate — a pure filter; its drops are unrecoverable.
    if gate_scored:
        n_saw, n_saw_true = _split(keys, y_true, gate_scored)
        n_drop, n_drop_true = _split(keys, y_true, gate_drop)
        rows.append(row(
            "non-match gate", n_saw, n_saw_true,
            no_match=n_drop, lost=n_drop_true,
            to_next=n_saw - n_drop, to_next_true=n_saw_true - n_drop_true,
            note="drops are unrecoverable — watch true_lost, not precision",
        ))

    # Stage 4.5 matcher — advisory unless ml_feeds_clustering is on.
    if ml_scored:
        n_saw, n_saw_true = _split(keys, y_true, ml_scored)
        n_ml, n_ml_true = _split(keys, y_true, ml_auto)
        n_mlrej, n_mlrej_true = _split(keys, y_true, ml_reject)
        rows.append(row(
            "ml matcher", n_saw, n_saw_true,
            auto=n_ml, auto_true=n_ml_true,
            no_match=n_mlrej, lost=n_mlrej_true if ml_binding else 0,
            to_next=n_saw - n_ml - n_mlrej,
            to_next_true=n_saw_true - n_ml_true - n_mlrej_true,
            binding=ml_binding,
            note=("auto_merge tier is binding" if ml_binding else
                  "ADVISORY — ml_feeds_clustering is off, so this changes no output"),
        ))

    # Clustering — terminal aggregator, not a filter.
    if partition is not None:
        same, covered = predict_same_cluster(keys, partition)
        merged = same & covered
        rows.append(row(
            "clustering", n, n_true,
            auto=int(merged.sum()), auto_true=int((merged & y_true).sum()),
            binding=True,
            note="terminal: unions binding auto_merge edges + transitive closure",
        ))

    # Everything still alive at the end is the reviewer's queue.
    last = next((r for r in reversed(rows) if r["stage"] != "clustering"), None)
    if last is not None:
        rows.append(row(
            "-> human review queue", last["to_next"], last["to_next_true"],
            binding=False,
            note="survived every stage without a binding merge",
        ))
    return rows
