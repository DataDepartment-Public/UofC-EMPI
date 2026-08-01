"""Per-pair stage diagnostics: one frame carrying every stage's decision.

`pipeline_eval.py` answers *how good is each stage* and serializes the answer
as counts — the stored JSON reports are deliberately PHI-free, so they cannot
say **which** pairs a stage got wrong. This module is the other half: it
rebuilds, for one completed run and one label set, a single row per labeled
pair carrying

* the pair's adjudicated class (non-match / ambiguous / match),
* what **each** stage did with it (blocked?, rules tier, gate tier + score,
  matcher tier + score, same cluster?), and
* the pipeline's **cumulative route** after each stage — where the pair stood
  once that stage had spoken.

Two notebooks read it. `end_to_end_eval.ipynb` aggregates it into per-stage
confusion matrices and classification reports; the misclassified-pairs notebook
filters it to the off-diagonal cells and prints the pairs themselves. Deriving
both from one frame is the point: a matrix cell and the rows you can list from
it can never disagree about what the run did.

**Cumulative routing.** `route_after_rules` / `_gate` / `_ml` / `route_final`
all use the shared `CLASSIFICATION_TIERS` vocabulary, over the *whole* labeled
set, so the four matrices are directly comparable cell for cell and you can
watch the population settle stage by stage. A pair blocking never emitted is
`no_match` from the first column onward — that is where the pipeline has in
fact left it, and hiding it would make blocking misses invisible. `route_final`
is computed by `triage.system_route`, the same function the stored report uses,
so §3.5 of the notebook reproduces the report's triage matrix exactly.

**PHI / HIPAA.** Unlike every other module in `src/evaluation/`, the frame this
one returns *contains PATIDs and, once `with_attributes` is applied, cleaned
identity fields.* It is analysis scaffolding for a notebook running where the
data lives — nothing here is logged, written to `data/evaluations/`, or
returned by the API. Keep it that way: log counts, never rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import Settings, settings as default_settings
from src.contracts import (
    ADDRESS1,
    BIRTH_DT,
    EMAIL,
    FIRST_NM,
    LAST_NM,
    PATID,
    PATID_A,
    PATID_B,
    PHONES,
    SEX,
    SSN,
    RunManifest,
    TIER_AUTO_MERGE,
    TIER_HUMAN_REVIEW,
    TIER_NO_MATCH,
    VALID_RECORD,
    ZIP_BASE,
)
from src.evaluation.cluster_eval import PairKey, pair_keys, cluster_map, predict_same_cluster
from src.evaluation.holdout import (
    DEFAULT_GOLD_LABELS,
    GOLD_AMBIGUOUS_COL,
    GOLD_LABEL_COL,
    holdout_keys,
    load_labels,
)
from src.evaluation.pipeline_eval import _read, _tier_keys, load_manifest  # noqa: PLC2701
from src.evaluation.triage import (
    GOLD_CLASSES,
    GOLD_TO_ROUTE,
    ROUTES,
    expected_route,
    system_route,
)

logger = logging.getLogger(__name__)

__all__ = [
    "StageDiagnostics",
    "BinarySpec",
    "BINARY_STAGES",
    "ROUTE_COLUMNS",
    "build_diagnostics",
    "diagnostics_for_report",
    "binary_confusion",
    "binary_report",
    "binary_errors",
    "route_confusion",
    "route_report",
    "route_errors",
    "batch_name",
    "review_batches",
    "export_review_batches",
    "BATCH_COLUMNS",
    "load_cleaned",
    "with_attributes",
    "stacked_pairs",
    "style_pairs",
    "ATTRIBUTE_COLUMNS",
    "STACKED_CONTEXT",
]

#: Cumulative-route column per stage, in pipeline order. `blocking` has no entry
#: on purpose: it is a two-class filter and its routing view is `rules`' first
#: column (everything it dropped is already `no_match` there).
ROUTE_COLUMNS: dict[str, str] = {
    "rules": "route_after_rules",
    "gate": "route_after_gate",
    "ml_matcher": "route_after_ml",
    "clustering": "route_final",
}

#: Cleaned columns worth eyeballing when judging a misclassification. Order is
#: strongest identifier first — how a reviewer actually reads a pair.
ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    LAST_NM, FIRST_NM, BIRTH_DT, SSN, EMAIL, PHONES, ADDRESS1, ZIP_BASE, SEX,
    VALID_RECORD,
)


# ── The per-stage binary decisions ───────────────────────────────────────────
@dataclass(frozen=True)
class BinarySpec:
    """One stage's two-class decision: what it predicts, against what target.

    `truth` differs per stage on purpose. The gate is scored against
    *plausible* (match ∪ ambiguous) because keeping an ambiguous non-match
    alive is correct behavior for a filter, and the matcher against *confident
    match* because merging an ambiguous pair is not what it was trained to do.
    Scoring either against the bare match label reports a failure that is
    actually the design.
    """

    stage: str
    title: str
    #: Boolean column restricting the population to the pairs this stage saw.
    #: None = every labeled pair.
    population: str | None
    truth: str
    pred: str
    #: Row labels (truth) then column labels (prediction), negative class first,
    #: so the matrix reads TN, FP / FN, TP and the report's positive class is
    #: the second row.
    truth_labels: tuple[str, str]
    pred_labels: tuple[str, str]
    note: str


BINARY_STAGES: dict[str, BinarySpec] = {
    "blocking": BinarySpec(
        stage="blocking",
        title="Blocking — was the pair ever considered?",
        population=None,
        truth="gold_match",
        pred="blocked",
        truth_labels=("non-match", "true match"),
        pred_labels=("not blocked", "blocked"),
        note="the false-negative cell is the one that matters: a true pair that "
             "never blocked is unrecoverable. False positives are free — "
             "blocking over-generates by design.",
    ),
    "gate": BinarySpec(
        stage="gate",
        title="Stage-4.25 gate — confident non-match filter",
        population="gate_scored",
        truth="gold_plausible",
        pred="gate_pass",
        truth_labels=("confident non-match", "plausible"),
        pred_labels=("dropped", "passed"),
        note="scored only on the pool the gate saw (the rules' non_matches). "
             "Its drops are unrecoverable, so recall on 'plausible' is the "
             "number to watch, not precision.",
    ),
    "ml_matcher": BinarySpec(
        stage="ml_matcher",
        title="Stage-4.5 matcher — confident match classifier",
        population="ml_scored",
        truth="gold_confident",
        pred="ml_auto",
        truth_labels=("not a confident match", "confident match"),
        pred_labels=("left in review", "auto_merge"),
        note="scored only on the gate's survivors. With ml_feeds_clustering on, "
             "the false-positive cell is a real wrong merge.",
    ),
    "clustering": BinarySpec(
        stage="clustering",
        title="Stage-5 clustering — the shipped merge decision",
        population=None,
        truth="gold_match",
        pred="clustered",
        truth_labels=("non-match", "true match"),
        pred_labels=("not merged", "merged"),
        note="the headline. Includes pairs merged only by transitive closure, "
             "which no classifier ever scored.",
    ),
}


@dataclass
class StageDiagnostics:
    """One run scored against one label set, at pair granularity."""

    run_id: str
    label_source: str
    label_col: str
    holdout_name: str
    ambiguous_col: str | None
    #: One row per labeled pair (post-holdout). See `build_diagnostics`.
    pairs: pd.DataFrame
    #: Which optional stages this run actually has artifacts for.
    present: dict[str, bool] = field(default_factory=dict)

    def population(self, stage: str) -> pd.DataFrame:
        """The labeled pairs `stage` actually saw."""
        spec = BINARY_STAGES[stage]
        if spec.population is None:
            return self.pairs
        return self.pairs[self.pairs[spec.population]]

    def header(self) -> str:
        return (
            f"run {self.run_id}  |  {self.label_source}  |  holdout: "
            f"{self.holdout_name}  |  {len(self.pairs):,} labeled pairs "
            f"({int(self.pairs['gold_match'].sum()):,} match, "
            f"{int(self.pairs['gold_ambiguous'].sum()):,} ambiguous)"
        )


# ── Building the frame ───────────────────────────────────────────────────────
def _key_column(df: pd.DataFrame | None, col: str) -> dict[PairKey, object]:
    """`{pair key: value}` for one column of a pair frame; {} when absent."""
    if df is None or not len(df) or col not in df.columns:
        return {}
    return dict(zip(pair_keys(df), df[col]))


def build_diagnostics(
    manifest: RunManifest,
    labeled: pd.DataFrame,
    label_col: str,
    *,
    settings: Settings = default_settings,
    label_source: str = "labels",
    holdout: set[PairKey] | None = None,
    holdout_name: str = "none",
    ambiguous_col: str | None = None,
) -> StageDiagnostics:
    """Rebuild every stage's per-pair decision for one run.

    The holdout is applied exactly as `pipeline_eval.evaluate_run` applies it,
    so a matrix built here and a stored report built there describe the same
    population — that equivalence is what lets the notebook's numbers be quoted
    against the report's.
    """
    labeled = labeled.reset_index(drop=True).copy()
    labeled[label_col] = labeled[label_col].astype(bool)

    keys_all = pair_keys(labeled)
    if holdout is not None:
        mask = np.array([k in holdout for k in keys_all], dtype=bool)
        labeled = labeled.loc[mask].reset_index(drop=True)
        if labeled.empty:
            raise ValueError(
                f"Holdout '{holdout_name}' left no labeled pairs — the label set "
                "and the holdout do not overlap."
            )
    keys: Sequence[PairKey] = pair_keys(labeled)

    # ── artifacts ────────────────────────────────────────────────────────────
    clusters = _read(manifest.clusters, settings)
    if clusters is None:
        raise FileNotFoundError(
            "This run has no cluster assignments — stage diagnostics need Stage-5 "
            "output."
        )
    partition = cluster_map(clusters)

    cand_df = _read(manifest.candidate_pairs, settings)
    match_df = _read(manifest.matches, settings)
    reject_df = _read(manifest.rejects, settings)
    review_df = _read(manifest.review_evidence, settings)
    gate_df = _read(manifest.gate_results, settings)
    ml_df = _read(manifest.matches_ml, settings)

    candidates = set(pair_keys(cand_df)) if cand_df is not None else set()
    matches = set(pair_keys(match_df)) if match_df is not None else set()
    rejects = set(pair_keys(reject_df)) if reject_df is not None else set()
    gate_scored = set(pair_keys(gate_df)) if gate_df is not None else set()
    gate_drop = _tier_keys(gate_df, TIER_NO_MATCH)
    ml_scored = set(pair_keys(ml_df)) if ml_df is not None else set()
    ml_auto = _tier_keys(ml_df, TIER_AUTO_MERGE)
    ml_reject = _tier_keys(ml_df, TIER_NO_MATCH)

    src_blocks = _key_column(cand_df, "source_blocks")
    match_rule = _key_column(match_df, "match_rule")
    reject_rule = _key_column(reject_df, "reject_rule")
    n_contra = _key_column(reject_df, "n_contradictions")
    review_rule = _key_column(review_df, "match_rule")
    gate_score = _key_column(gate_df, "score")
    ml_score = _key_column(ml_df, "score")

    # ── the frame ────────────────────────────────────────────────────────────
    out = pd.DataFrame({
        PATID_A: labeled[PATID_A].astype(str),
        PATID_B: labeled[PATID_B].astype(str),
    })
    out["gold_match"] = labeled[label_col].to_numpy(dtype=bool)
    has_ambiguous = ambiguous_col is not None and ambiguous_col in labeled.columns
    out["gold_ambiguous"] = (
        labeled[ambiguous_col].astype(bool).to_numpy() if has_ambiguous
        else np.zeros(len(labeled), dtype=bool)
    )
    if not has_ambiguous:
        logger.info(
            "No ambiguous column: the gate and matcher fall back to the match "
            "label as their target, and the routing matrices are two-class."
        )
    # `ambiguous` outranks `match` — the same precedence `triage` documents.
    out["gold_class"] = np.where(
        out["gold_ambiguous"], GOLD_CLASSES[1],
        np.where(out["gold_match"], GOLD_CLASSES[2], GOLD_CLASSES[0]),
    )
    out["gold_plausible"] = out["gold_match"] | out["gold_ambiguous"]
    out["gold_confident"] = out["gold_match"] & ~out["gold_ambiguous"]
    out["expected_route"] = expected_route(
        labeled, label_col, ambiguous_col if has_ambiguous else None
    ).to_numpy()

    def flags(member: set[PairKey]) -> np.ndarray:
        return np.array([k in member for k in keys], dtype=bool)

    def lookup(mapping: dict[PairKey, object]) -> list:
        return [mapping.get(k) for k in keys]

    out["blocked"] = flags(candidates)
    out["source_blocks"] = lookup(src_blocks)

    out["rules_decision"] = np.where(
        ~out["blocked"], None,
        np.where(flags(matches), TIER_AUTO_MERGE,
                 np.where(flags(rejects), TIER_NO_MATCH, TIER_HUMAN_REVIEW)),
    )
    out["rules_rule"] = [
        m or rv or rj for m, rv, rj in
        zip(lookup(match_rule), lookup(review_rule), lookup(reject_rule))
    ]
    out["rules_n_contradictions"] = lookup(n_contra)

    out["gate_scored"] = flags(gate_scored)
    out["gate_pass"] = out["gate_scored"] & ~flags(gate_drop)
    out["gate_score"] = pd.to_numeric(pd.Series(lookup(gate_score)), errors="coerce")

    out["ml_scored"] = flags(ml_scored)
    out["ml_auto"] = flags(ml_auto)
    out["ml_score"] = pd.to_numeric(pd.Series(lookup(ml_score)), errors="coerce")

    same, covered = predict_same_cluster(keys, partition)
    out["clustered"] = same & covered
    out["clustered_covered"] = covered
    out["cluster_a"] = [partition.get(a) for a in out[PATID_A]]
    out["cluster_b"] = [partition.get(b) for b in out[PATID_B]]

    # ── cumulative routing ───────────────────────────────────────────────────
    after_rules = np.where(
        ~out["blocked"], TIER_NO_MATCH, out["rules_decision"].to_numpy(),
    ).astype(object)
    out["route_after_rules"] = after_rules

    after_gate = after_rules.copy()
    dropped_by_gate = flags(gate_drop)
    after_gate[(after_gate == TIER_HUMAN_REVIEW) & dropped_by_gate] = TIER_NO_MATCH
    out["route_after_gate"] = after_gate

    after_ml = after_gate.copy()
    open_now = after_ml == TIER_HUMAN_REVIEW
    after_ml[open_now & flags(ml_auto)] = TIER_AUTO_MERGE
    after_ml[open_now & flags(ml_reject)] = TIER_NO_MATCH
    out["route_after_ml"] = after_ml

    # The shipped route, from the same function the stored report uses — so it
    # picks up transitive-closure merges the per-stage columns cannot see.
    out["route_final"] = system_route(
        keys, partition=partition, candidates=candidates, rejects=rejects,
        gate_drop=gate_drop, ml_reject=ml_reject,
    ).to_numpy()

    present = {
        "blocking": cand_df is not None,
        "rules": match_df is not None,
        "gate": gate_df is not None,
        "ml_matcher": ml_df is not None,
        "clustering": True,
    }
    logger.info(
        "Stage diagnostics: %d labeled pairs, stages present: %s",
        len(out), ", ".join(k for k, v in present.items() if v),
    )
    return StageDiagnostics(
        run_id=manifest.run_id,
        label_source=label_source,
        label_col=label_col,
        holdout_name=holdout_name,
        ambiguous_col=ambiguous_col if has_ambiguous else None,
        pairs=out,
        present=present,
    )


def diagnostics_for_report(
    report: dict,
    *,
    labels=DEFAULT_GOLD_LABELS,
    settings: Settings = default_settings,
) -> StageDiagnostics:
    """Rebuild the pair-level detail behind a **stored** evaluation report.

    The notebooks pick a report in `data/evaluations/` and then want the rows
    behind its numbers. This takes that report's `run_id`, `label_col` and
    holdout restriction and reproduces exactly the population it was scored on,
    so the matrices built here can be quoted against the report's counts.

    `labels` defaults to the gold file; pass the label file explicitly when the
    focused report is a synthetic or silver one.
    """
    labeled = load_labels(labels, report["label_col"])
    holdout_name = report.get("leakage", {}).get("restriction", "none")
    is_gold = GOLD_LABEL_COL in labeled.columns and GOLD_AMBIGUOUS_COL in labeled.columns
    holdout = holdout_keys(labeled) if holdout_name != "none" and is_gold else None
    if holdout_name != "none" and not is_gold:
        logger.warning(
            "Report says holdout=%r but %s is not the gold file — scoring every "
            "labeled pair instead.", holdout_name, getattr(labels, "name", labels),
        )
        holdout_name = "none"
    return build_diagnostics(
        load_manifest(report["run_id"], settings), labeled, report["label_col"],
        settings=settings,
        label_source=report.get("label_source", "labels"),
        holdout=holdout,
        holdout_name=holdout_name,
        ambiguous_col=GOLD_AMBIGUOUS_COL if is_gold else None,
    )


# ── Matrices and reports ─────────────────────────────────────────────────────
def _report_from_matrix(cm: pd.DataFrame) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support from any square matrix.

    Rows are truth, columns are prediction, and row *i* corresponds to column
    *i* — so one implementation serves the two-class stage views and the
    three-class routing views alike, and a printed matrix and its report can
    never disagree.

    A class with no support gets NaN recall rather than 0.0: "we never had one"
    and "we got every one wrong" are different facts.
    """
    counts = cm.to_numpy(dtype=float)
    total = counts.sum()
    rows = []
    for i, name in enumerate(cm.index):
        tp = counts[i, i]
        support = counts[i].sum()
        predicted = counts[:, i].sum()
        precision = tp / predicted if predicted else np.nan
        recall = tp / support if support else np.nan
        # From the counts rather than from P and R: F1 = 2TP/(2TP+FP+FN) stays
        # defined when precision is not (nothing predicted) — a class the model
        # never predicted but did have has an F1 of 0, not "unknown". Only a
        # class with neither support nor predictions is NaN.
        denom = support + predicted
        f1 = 2 * tp / denom if denom else np.nan
        rows.append({"class": name, "precision": precision, "recall": recall,
                     "f1": f1, "support": int(support), "predicted": int(predicted)})
    report = pd.DataFrame(rows).set_index("class")

    supported = report[report["support"] > 0]
    macro = supported[["precision", "recall", "f1"]].mean()
    weighted = (
        supported[["precision", "recall", "f1"]]
        .mul(supported["support"], axis=0).sum() / supported["support"].sum()
        if len(supported) else pd.Series(np.nan, index=["precision", "recall", "f1"])
    )
    accuracy = np.trace(counts) / total if total else np.nan

    tail = pd.DataFrame(
        [
            {**macro.to_dict(), "support": int(total), "predicted": int(total)},
            {**weighted.to_dict(), "support": int(total), "predicted": int(total)},
            {"precision": np.nan, "recall": np.nan, "f1": accuracy,
             "support": int(total), "predicted": int(total)},
        ],
        index=["macro avg", "weighted avg", "accuracy"],
    )
    tail.index.name = "class"
    return pd.concat([report, tail]).round(4)


def _square(truth: pd.Series, pred: pd.Series,
            truth_labels: Sequence[str], pred_labels: Sequence[str]) -> pd.DataFrame:
    """Cross-tab reindexed to the full label set, so an empty class still shows."""
    truth, pred = pd.Series(list(truth), dtype=object), pd.Series(list(pred), dtype=object)
    if truth.empty:
        cm = pd.DataFrame(0, index=list(truth_labels), columns=list(pred_labels))
    else:
        cm = (
            pd.crosstab(truth, pred)
            .reindex(index=list(truth_labels), columns=list(pred_labels), fill_value=0)
            .fillna(0).astype(int)
        )
    cm.index.name, cm.columns.name = "actual", "predicted"
    return cm


def _with_total(cm: pd.DataFrame, normalize: bool) -> pd.DataFrame:
    if normalize:
        pct = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100
        return pct.round(2)
    out = cm.copy()
    out["total"] = cm.sum(axis=1)
    return out


def binary_confusion(diag: StageDiagnostics, stage: str, *,
                     normalize: bool = False) -> pd.DataFrame:
    """2x2 for one stage, over the pairs that stage actually saw.

    `normalize=True` gives row percentages (per-class recall), which is the
    only readable form when non-matches outnumber matches several to one.
    """
    spec = BINARY_STAGES[stage]
    pop = diag.population(stage)
    truth = np.where(pop[spec.truth], spec.truth_labels[1], spec.truth_labels[0])
    pred = np.where(pop[spec.pred], spec.pred_labels[1], spec.pred_labels[0])
    cm = _square(truth, pred, spec.truth_labels, spec.pred_labels)
    return _with_total(cm, normalize)


def binary_report(diag: StageDiagnostics, stage: str) -> pd.DataFrame:
    """Classification report for one stage's two-class decision."""
    spec = BINARY_STAGES[stage]
    cm = binary_confusion(diag, stage).drop(columns="total")
    # `_report_from_matrix` pairs row i with column i; the labels differ in
    # wording (truth vs. decision) but the classes are the same two.
    cm.columns = list(spec.truth_labels)
    return _report_from_matrix(cm)


def route_confusion(diag: StageDiagnostics, stage: str, *,
                    normalize: bool = False,
                    population: str | None = None) -> pd.DataFrame:
    """3x3 routing matrix: where each pair belonged vs. where the pipeline had
    it after `stage`.

    Rows are the adjudicated class, columns the route. `population` restricts
    to a boolean column (e.g. `"blocked"` to see the rules' own decision rather
    than the cumulative one, which folds blocking misses into `no_match`).
    """
    col = ROUTE_COLUMNS[stage]
    pairs = diag.pairs if population is None else diag.pairs[diag.pairs[population]]
    truth_labels = [f"{name} -> {route}" for name, route in GOLD_TO_ROUTE.items()]
    truth = pairs["gold_class"].map(lambda c: f"{c} -> {GOLD_TO_ROUTE[c]}")
    cm = _square(truth, pairs[col], truth_labels, ROUTES)
    return _with_total(cm, normalize)


def route_report(diag: StageDiagnostics, stage: str, *,
                 population: str | None = None) -> pd.DataFrame:
    """Classification report for a routing matrix (three classes)."""
    cm = route_confusion(diag, stage, population=population).drop(columns="total")
    cm.index = list(ROUTES)
    return _report_from_matrix(cm)


# ── Error selection (the misclassified-pairs notebook) ───────────────────────
#: Columns carried on every error listing, before the identity attributes.
_CONTEXT_COLUMNS: tuple[str, ...] = (
    PATID_A, PATID_B, "gold_class", "source_blocks", "rules_decision", "rules_rule",
    "gate_score", "ml_score", "route_final",
)


def _order(df: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    cols: list[str] = []
    for c in (*_CONTEXT_COLUMNS, *extra):
        if c in df.columns and c not in cols:
            cols.append(c)
    rest = [c for c in df.columns if c not in cols]
    return df[[*cols, *rest]]


def binary_errors(diag: StageDiagnostics, stage: str, kind: str = "all") -> pd.DataFrame:
    """The pairs in a stage's off-diagonal cells.

    `kind` is `"FN"` (truth positive, stage said no), `"FP"` (truth negative,
    stage said yes), or `"all"`. An `error` column names the cell, so a
    concatenation of several stages stays readable.
    """
    spec = BINARY_STAGES[stage]
    pop = diag.population(stage)
    truth, pred = pop[spec.truth].astype(bool), pop[spec.pred].astype(bool)
    fn, fp = truth & ~pred, ~truth & pred
    if kind == "FN":
        mask = fn
    elif kind == "FP":
        mask = fp
    elif kind == "all":
        mask = fn | fp
    else:
        raise ValueError(f"kind must be 'FN', 'FP' or 'all'; got {kind!r}")

    out = pop[mask].copy()
    out["error"] = np.where(
        fn[mask],
        f"{spec.truth_labels[1]} -> {spec.pred_labels[0]}",
        f"{spec.truth_labels[0]} -> {spec.pred_labels[1]}",
    )
    out["stage"] = stage
    return _order(out, extra=("error",))


def _as_route(value: str | None, argument: str) -> str | None:
    """Accept either a route (`auto_merge`) or the gold class that maps to it
    (`match`), and reject anything else.

    The two vocabularies sit side by side in the matrix's row labels
    (`ambiguous -> human_review`), so reaching for the wrong one is the obvious
    mistake. Silently matching nothing would read as "no pairs in that cell",
    which is the one wrong answer this function must never give.
    """
    if value is None or value in ROUTES:
        return value
    if value in GOLD_TO_ROUTE:
        return GOLD_TO_ROUTE[value]
    raise ValueError(
        f"{argument}={value!r} is neither a route {list(ROUTES)} nor a gold "
        f"class {list(GOLD_TO_ROUTE)}."
    )


def route_errors(diag: StageDiagnostics, stage: str, *,
                 expected: str | None = None, actual: str | None = None,
                 population: str | None = None) -> pd.DataFrame:
    """The misrouted pairs after `stage` — every off-diagonal cell, or one.

    Pass `expected` / `actual` to pull a single cell, e.g.
    `expected="auto_merge", actual="no_match"` for true matches the pipeline has
    already thrown away. Both accept a route name (`no_match` / `human_review` /
    `auto_merge`) or the gold class that maps to it (`non-match` / `ambiguous` /
    `match`) — `expected="ambiguous"` and `expected="human_review"` select the
    same pairs. An unknown value raises rather than matching nothing.
    """
    expected = _as_route(expected, "expected")
    actual = _as_route(actual, "actual")
    col = ROUTE_COLUMNS[stage]
    pairs = diag.pairs if population is None else diag.pairs[diag.pairs[population]]
    exp = pairs["gold_class"].map(GOLD_TO_ROUTE)
    mask = exp != pairs[col]
    if expected is not None:
        mask &= exp == expected
    if actual is not None:
        mask &= pairs[col] == actual

    out = pairs[mask].copy()
    out["error"] = exp[mask] + " -> " + out[col].astype(str)
    out["stage"] = stage
    return _order(out, extra=("error", col))


# ── Review batches (hand-off to the gold labeler) ────────────────────────────
#: Filename-safe stem per gold class. `non-match` loses its hyphen so a batch
#: name is a valid identifier wherever it is pasted.
_CLASS_SLUG = {"non-match": "nonmatch", "ambiguous": "ambiguous", "match": "match"}

#: What travels with a batch. Deliberately **not** the identity fields: the
#: labeling notebook rejoins these ids to its own record frame, so a batch file
#: carries pair ids and provenance, and the fields come from the source of truth
#: on the other side rather than from a copy that can go stale.
BATCH_COLUMNS: tuple[str, ...] = (
    PATID_A, PATID_B, "gold_class", "expected_route", "actual_route", "error",
    "source_blocks", "rules_decision", "rules_rule", "gate_score", "ml_score",
)


def batch_name(expected: str, actual: str) -> str:
    """`match_to_no_match`, `ambiguous_to_auto_merge`, … — the batch's stem."""
    expected = _as_route(expected, "expected")
    actual = _as_route(actual, "actual")
    gold_class = next(c for c, r in GOLD_TO_ROUTE.items() if r == expected)
    return f"{_CLASS_SLUG[gold_class]}_to_{actual}"


def review_batches(diag: StageDiagnostics, *, stage: str = "clustering",
                   population: str | None = None) -> dict[str, pd.DataFrame]:
    """One frame per error type — the six off-diagonal cells of `stage`'s
    routing matrix, keyed by `batch_name`.

    Empty cells are included (as empty frames) on purpose: "we made none of
    this error" is a result, and a batch silently missing from the mapping
    reads as a bug in the export instead.
    """
    col = ROUTE_COLUMNS[stage]
    out: dict[str, pd.DataFrame] = {}
    for gold_class, expected in GOLD_TO_ROUTE.items():
        for actual in ROUTES:
            if actual == expected:
                continue                     # the diagonal is correct routing
            cell = route_errors(diag, stage, expected=expected, actual=actual,
                                population=population).copy()
            cell["expected_route"] = expected
            cell["actual_route"] = actual
            cell["gold_class"] = gold_class
            out[batch_name(expected, actual)] = cell
    return out


def export_review_batches(
    diag: StageDiagnostics,
    out_dir,
    *,
    stage: str = "clustering",
    population: str | None = None,
    columns: Sequence[str] = BATCH_COLUMNS,
    max_per_batch: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Write one CSV per error type plus a `_batches.csv` index; return the index.

    The batches are the review queue: each file is one *kind* of mistake, so a
    reviewer adjudicates a homogeneous list and their judgement stays calibrated
    — rather than context-switching between "did we wrongly merge these?" and
    "should this have been merged?" every few rows.

    `max_per_batch` takes a reproducible random sample of an oversized cell.
    Sampling rather than `head` matters: the frame is ordered by the label file,
    and the first *n* rows of it are not a fair look at the error.

    **The files are PHI** (pair ids). Write them somewhere gitignored.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for name, cell in review_batches(diag, stage=stage, population=population).items():
        total = len(cell)
        if max_per_batch is not None and total > max_per_batch:
            cell = cell.sample(max_per_batch, random_state=random_state)
        cell = cell[[c for c in columns if c in cell.columns]]
        path = out_dir / f"{name}.csv"
        cell.to_csv(path, index=False)
        index_rows.append({
            "batch": name,
            "expected_route": name.split("_to_")[0],
            "actual_route": name.split("_to_")[1],
            "pairs": total,
            "exported": len(cell),
            "file": path.name,
        })

    index = pd.DataFrame(index_rows)
    index.insert(0, "run_id", diag.run_id)
    index.insert(1, "holdout", diag.holdout_name)
    index.insert(2, "stage", stage)
    index.to_csv(out_dir / "_batches.csv", index=False)
    logger.info("Wrote %d review batches (%d pairs) to %s",
                len(index), int(index["exported"].sum()), out_dir)
    return index


# ── Identity attributes ──────────────────────────────────────────────────────
def load_cleaned(manifest: RunManifest,
                 settings: Settings = default_settings) -> pd.DataFrame:
    """The run's cleaned records, indexed by PATID."""
    cleaned = _read(manifest.cleaned, settings)
    if cleaned is None:
        raise FileNotFoundError("The manifest's cleaned artifact is missing.")
    return cleaned.set_index(cleaned[PATID].astype(str))


def _as_set(value) -> set[str] | None:
    """Set-valued cleaned fields, whatever form they survived Parquet in.

    `Phones_set` round-trips as the legacy stringified form `"{'a', 'b'}"`
    (Arrow has no set type), and a list/array compares unequal on ordering
    alone. Both are normalized here so the comparison is membership, not
    formatting.
    """
    if isinstance(value, (set, frozenset)):
        return {str(v) for v in value}
    if isinstance(value, (list, tuple, np.ndarray)):
        return {str(v) for v in value}
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            inner = text[1:-1].strip()
            if not inner:
                return set()
            return {t.strip().strip("'\"") for t in inner.split(",")}
    return None


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return len(value) == 0
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):      # array-likes pandas can't reduce
        return False
    return isinstance(value, str) and not value.strip()


def _format(value) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    as_set = _as_set(value)
    if as_set is not None:
        return ", ".join(sorted(as_set))
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


#: Cell backgrounds for the stacked view — the same vocabulary
#: `gold_labeling.ipynb` uses, so a pair reads the same way in both notebooks.
AGREE_COLORS = {
    "equal": "#a8e6a3",       # green  — both present and the same
    "differ": "#f4a8a8",      # red    — both present and different
    "one_missing": "#fff3a3",  # yellow — only one side has it
    "both_missing": "#d3d3d3",  # grey   — neither side has it
}

#: The adjudicated class, called out strongly (white text on a solid fill) so
#: the eye lands on it before reading any field.
CLASS_COLORS = {
    "match": "#00c853",
    "ambiguous": "#ffab00",
    "non-match": "#d50000",
}

#: Context columns repeated on both rows of a pair in the stacked view.
STACKED_CONTEXT: tuple[str, ...] = (
    "gold_class", "error", "source_blocks", "rules_decision", "rules_rule",
    "rules_n_contradictions", "gate_score", "ml_score", "route_final",
)


def _agreement(a, b) -> str:
    am, bm = _is_missing(a), _is_missing(b)
    if am and bm:
        return "both_missing"
    if am or bm:
        return "one_missing"
    sa, sb = _as_set(a), _as_set(b)
    if sa is not None and sb is not None:
        return "equal" if sa & sb else "differ"
    return "equal" if _format(a) == _format(b) else "differ"


def stacked_pairs(
    pairs: pd.DataFrame,
    cleaned: pd.DataFrame,
    *,
    fields: Sequence[str] = ATTRIBUTE_COLUMNS,
    context: Sequence[str] = STACKED_CONTEXT,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One pair per block of three rows: record A, record B, a blank spacer.

    Returns `(display, colors)` — same shape, `colors` holding a CSS string per
    cell. Kept separate from the styling so the layout and the comparison logic
    can be tested without going through pandas' `Styler`.

    Two records stacked with their fields aligned column-wise is how a reviewer
    actually reads a pair: the eye compares down, not across, and the colour
    says which fields agree before any of them are read.
    """
    if limit is not None:
        pairs = pairs.head(limit)
    fields = [f for f in fields if f"{f}_A" in pairs.columns or f in cleaned.columns]
    context = [c for c in context if c in pairs.columns]
    display_names = {f: f.replace("_clean", "") for f in fields}

    a_side = cleaned.reindex(pairs[PATID_A].astype(str))
    b_side = cleaned.reindex(pairs[PATID_B].astype(str))

    rows: list[dict] = []
    cells: list[dict] = []
    for i in range(len(pairs)):
        pair = pairs.iloc[i]
        ctx = {c: _format(pair[c]) for c in context}
        klass = str(pair.get("gold_class", ""))
        class_style = (f"background-color: {CLASS_COLORS[klass]}; color: white"
                       if klass in CLASS_COLORS else "")

        for side, frame in (("A", a_side), ("B", b_side)):
            values = {f: frame.iloc[i][f] if f in frame.columns else None for f in fields}
            # PATID is a column, not the index: two records of a pair and the
            # spacers repeat, and `Styler.apply` rejects a non-unique index.
            rows.append({PATID: str(pair[PATID_A if side == "A" else PATID_B]),
                         **ctx,
                         **{display_names[f]: _format(v) for f, v in values.items()}})
            style = {c: "" for c in context}
            if "gold_class" in ctx:
                style["gold_class"] = class_style
            for f in fields:
                a = a_side.iloc[i][f] if f in a_side.columns else None
                b = b_side.iloc[i][f] if f in b_side.columns else None
                style[display_names[f]] = (
                    f"background-color: {AGREE_COLORS[_agreement(a, b)]}"
                )
            cells.append(style)

        rows.append({})          # spacer between pairs
        cells.append({})

    columns = [PATID, *context, *(display_names[f] for f in fields)]
    display = pd.DataFrame(rows, columns=columns).fillna("")
    colors = pd.DataFrame(cells, columns=columns).fillna("")
    return display, colors


def style_pairs(pairs: pd.DataFrame, cleaned: pd.DataFrame, **kwargs):
    """`stacked_pairs` rendered as a colour-coded pandas `Styler`.

    Green = the two records agree on this field, red = they disagree, yellow =
    only one side has it, grey = neither does. The `gold_class` cell carries the
    adjudicated class in solid colour.
    """
    display, colors = stacked_pairs(pairs, cleaned, **kwargs)
    if display.empty:
        return display
    return (
        display.style
        .apply(lambda _: colors, axis=None)
        .hide(axis="index")      # the row number carries nothing; PATID is a column
        .set_properties(**{"font-size": "11px", "white-space": "nowrap"})
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "11px"),
                                         ("text-align", "left")]},
            {"selector": "td", "props": [("padding", "2px 6px")]},
        ])
    )


def with_attributes(pairs: pd.DataFrame, cleaned: pd.DataFrame, *,
                    columns: Sequence[str] = ATTRIBUTE_COLUMNS) -> pd.DataFrame:
    """Attach each side's cleaned fields, interleaved `<field>_A`, `<field>_B`.

    Interleaved rather than blocked so the two values of a field sit next to
    each other — you are comparing A to B field by field, not reading two
    records in sequence.
    """
    cols = [c for c in columns if c in cleaned.columns]
    missing = [c for c in columns if c not in cleaned.columns]
    if missing:
        logger.info("Cleaned frame has no %s — skipped.", ", ".join(missing))

    out = pairs.copy()
    a = cleaned.reindex(out[PATID_A].astype(str))[cols]
    b = cleaned.reindex(out[PATID_B].astype(str))[cols]
    for col in cols:
        out[f"{col}_A"] = a[col].to_numpy()
        out[f"{col}_B"] = b[col].to_numpy()
    return out
