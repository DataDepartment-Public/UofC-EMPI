"""Load stored end-to-end reports and reshape them into tidy frames.

Every `scripts/evaluate_all.py` / `scripts/eval_end_to_end.py` invocation writes
self-contained JSON reports to `settings.evaluations_dir` (kept out of
`runs_dir`, which holds pipeline `RunManifest`s — immutable per-run lineage —
while these are measurements *about* runs). Those files **are** the history — this module is the read side, so a
notebook or dashboard can compare runs over time without re-running anything
(re-running is both slow and, for a past `run_id`, not reproducible once the
artifacts age out).

The naming scheme is `eval_<session_id>__<source>__<holdout>.json`. The unit of
comparison is the **session**, not the run: one `evaluate_all.py` invocation
scores the real-data run *and* the synthetic run, which necessarily have
different run ids but are one measurement of one state of the system, so they
share a session and land on the same timeline point.

Re-evaluating the same (session, source, holdout) triple deliberately
**overwrites** rather than accumulating: that is the same measurement, and
keeping both would put a duplicate point on every trend.

Frames returned here are display-oriented, not contracts — column sets follow
whatever the reports contain, so an older report missing a newer field yields
`NaN` rather than an error. Nothing here validates against pandera on purpose.

PHI / HIPAA: reports carry aggregate metrics only, so everything in this module
is safe to render, export, or screenshot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

__all__ = [
    "REPORT_GLOB",
    "load_report",
    "load_reports",
    "summary_frame",
    "triage_matrix",
    "triage_report",
    "triage_history",
    "flow_frame",
    "stage_frame",
    "funnel_frame",
    "loss_frame",
    "transitivity_frame",
    "cluster_frame",
    "metric_history",
]

REPORT_GLOB = "eval_*.json"

#: Headline columns lifted to the top level of `summary_frame`, in display order.
_SUMMARY_METRICS = ("precision", "recall", "f1", "TP", "FP", "FN")


def load_report(path: str | Path) -> dict:
    """Read one stored report, tagging it with the file it came from."""
    path = Path(path)
    report = json.loads(path.read_text())
    report["_path"] = str(path)
    return report


def load_reports(
    evaluations_dir: Path | None = None,
    *,
    settings: Settings = default_settings,
    sources: Sequence[str] | None = None,
    holdouts: Sequence[str] | None = None,
) -> list[dict]:
    """Load every stored report, newest evaluation first.

    `sources` / `holdouts` filter on the report's own fields rather than the
    filename, so a report is never mis-bucketed by a filename that was renamed
    by hand.
    """
    evaluations_dir = (Path(evaluations_dir) if evaluations_dir is not None
                       else settings.evaluations_dir)
    if not evaluations_dir.exists():
        logger.warning("No evaluations directory at %s", evaluations_dir)
        return []

    reports = []
    for path in sorted(evaluations_dir.glob(REPORT_GLOB)):
        try:
            reports.append(load_report(path))
        except (json.JSONDecodeError, OSError) as exc:
            # One unreadable file must not sink a whole comparison.
            logger.warning("Skipping unreadable report %s: %s", path.name, exc)

    if sources is not None:
        reports = [r for r in reports if r.get("label_source") in sources]
    if holdouts is not None:
        reports = [r for r in reports
                   if (r.get("leakage") or {}).get("restriction") in holdouts]

    reports.sort(key=lambda r: (r.get("evaluated_utc") or "", r.get("run_id", "")),
                 reverse=True)
    logger.info("Loaded %d stored end-to-end reports from %s",
                len(reports), evaluations_dir)
    return reports


def _session(report: dict) -> str:
    """Timeline key for a report. Falls back to `run_id` for reports written
    before sessions existed, so old files still plot."""
    return report.get("session_id") or report.get("run_id") or "?"


def _label(report: dict) -> str:
    """Short identity for one report — what a legend or an index shows."""
    holdout = (report.get("leakage") or {}).get("restriction", "?")
    return f"{report.get('label_source', '?')} / {holdout}"


def summary_frame(reports: Iterable[dict]) -> pd.DataFrame:
    """One row per report: the headline clustering metrics + provenance.

    This is the table to look at first — it answers "which run/label/holdout
    combinations do I have, and how did the clustering score on each".
    """
    rows = []
    for r in reports:
        clustering = r.get("clustering") or {}
        universe = r.get("universe") or {}
        rows.append({
            "session_id": _session(r),
            "evaluated_utc": r.get("evaluated_utc"),
            "run_id": r.get("run_id"),
            "source": r.get("label_source"),
            "holdout": (r.get("leakage") or {}).get("restriction"),
            "labeled_pairs": universe.get("labeled_pairs"),
            "positives": universe.get("positives"),
            **{m: clustering.get(m) for m in _SUMMARY_METRICS},
            "git_sha": (r.get("git_sha") or "")[:8] or None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["evaluated_utc", "session_id"], ascending=False,
                            ignore_index=True)
    return df


#: Route order for the triage frames, matching `src.evaluation.triage.ROUTES`.
#: Duplicated as a literal rather than imported so this module stays a pure
#: reader — it must be able to render a report written by an older or newer
#: version of the writer without importing its code.
_ROUTES = ("no_match", "human_review", "auto_merge")


def triage_matrix(report: dict, *, normalize: bool = False) -> pd.DataFrame:
    """3x3 routing confusion matrix: rows expected, columns actual.

    Row labels carry the gold class name alongside the route it should have
    taken (`non-match -> no_match`), because the two vocabularies are what the
    table is reconciling and showing only one of them makes the diagonal look
    tautological.

    `normalize=True` divides by the row total, giving per-class recall down the
    diagonal — the right view when the classes are wildly imbalanced, which
    they are (non-matches outnumber matches several to one).
    """
    t = report.get("triage") or {}
    cm = t.get("confusion_matrix") or {}
    if not cm:
        return pd.DataFrame()

    names = t.get("gold_class_names") or {}
    rows = {}
    for route in _ROUTES:
        row = cm.get(route) or {}
        label = f"{names.get(route, route)} -> {route}"
        rows[label] = {c: int(row.get(c, 0)) for c in _ROUTES}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "expected"
    df.columns.name = "system routed to"
    if normalize:
        totals = df.sum(axis=1).replace(0, pd.NA)
        return (df.div(totals, axis=0) * 100).round(1)
    df["total"] = df.sum(axis=1)
    return df


def triage_report(report: dict) -> pd.DataFrame:
    """Per-route precision / recall / F1 / support, plus averages and accuracy.

    Shaped like sklearn's `classification_report` so it reads the way anyone
    expects, with the averages appended as trailing rows rather than returned
    separately — they belong in the same table or they get quoted without it.
    """
    t = report.get("triage") or {}
    per_class = t.get("per_class") or {}
    if not per_class:
        return pd.DataFrame()

    rows = []
    for route in _ROUTES:
        m = per_class.get(route) or {}
        rows.append({"route": route, "precision": m.get("precision"),
                     "recall": m.get("recall"), "f1": m.get("f1"),
                     "support": m.get("support"), "predicted": m.get("predicted")})
    for avg in ("macro_avg", "weighted_avg"):
        m = t.get(avg) or {}
        rows.append({"route": avg, "precision": m.get("precision"),
                     "recall": m.get("recall"), "f1": m.get("f1"),
                     "support": t.get("n_pairs"), "predicted": None})
    rows.append({"route": "accuracy", "precision": None, "recall": None,
                 "f1": t.get("accuracy"), "support": t.get("n_pairs"),
                 "predicted": None})
    return pd.DataFrame(rows)


def triage_history(reports: Iterable[dict], metric: str = "recall") -> pd.DataFrame:
    """One row per (report, route) for trending a single triage metric.

    Long format keyed on `session_id`, same shape as `metric_history`, so the
    notebook's plotting helper takes either without a branch.
    """
    rows = []
    for r in reports:
        per_class = ((r.get("triage") or {}).get("per_class")) or {}
        for route in _ROUTES:
            value = (per_class.get(route) or {}).get(metric)
            if value is None:
                continue
            rows.append({
                "session_id": _session(r),
                "evaluated_utc": r.get("evaluated_utc"),
                "run_id": r.get("run_id"),
                "series": route,
                "label": _label(r),
                "metric": metric,
                "value": value,
                "support": (per_class.get(route) or {}).get("support"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["series", "evaluated_utc"], ignore_index=True)
    return df


def flow_frame(report: dict) -> pd.DataFrame:
    """Per stage: pairs in, and what the stage did with them.

    Counts rather than rates, and the counts a person asks for first — merged,
    rejected, passed on — plus `true_lost`, the only column where a number
    above zero is unambiguously bad. `binding` says whether the stage's
    auto_merge tier actually reaches the output or is scored and discarded.
    """
    rows = report.get("stage_flow") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["stage", "saw", "auto_merge", "no_match", "to_next", "true_lost",
            "saw_true", "auto_merge_true", "to_next_true", "binding", "note"]
    return df[[c for c in cols if c in df.columns]]


def stage_frame(report: dict) -> pd.DataFrame:
    """Per-stage metrics for one report, in pipeline order.

    Skipped stages are kept as rows with NaN metrics rather than dropped — "the
    gate did not run" is information, and silently omitting the row makes a
    comparison across reports mis-align.
    """
    rows = []
    for stage, m in (report.get("stage_pairwise") or {}).items():
        if m.get("skipped"):
            rows.append({"stage": stage, "status": "not present in this run"})
            continue
        rows.append({
            "stage": stage,
            "status": "",
            "target": m.get("target"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "TP": m.get("TP"), "FP": m.get("FP"), "FN": m.get("FN"),
            "scored_pairs": m.get("scored_pairs"),
            "labeled_pairs": m.get("labeled_pairs"),
        })
    return pd.DataFrame(rows)


def funnel_frame(report: dict) -> pd.DataFrame:
    """Labeled pairs surviving each stage boundary."""
    rows = [{"stage": stage, **counts}
            for stage, counts in (report.get("funnel") or {}).items()]
    return pd.DataFrame(rows)


def loss_frame(report: dict) -> pd.DataFrame:
    """Where true pairs were lost, with each stage's share of the misses."""
    loss = report.get("loss_attribution") or {}
    total = loss.get("total_missed") or 0
    rows = [
        {"stage": stage, "missed": n,
         "share_pct": round(100 * n / total, 1) if total else None}
        for stage, n in (loss.get("by_stage") or {}).items()
    ]
    return pd.DataFrame(rows).sort_values("missed", ascending=False,
                                          ignore_index=True)


def transitivity_frame(report: dict) -> pd.DataFrame:
    """Direct vs. transitive-closure-only merges, split TP/FP."""
    t = report.get("transitivity") or {}
    rows = []
    for kind in ("direct", "transitive_only"):
        d = t.get(kind) or {}
        rows.append({"merge_kind": kind, "n": d.get("n"), "TP": d.get("TP"),
                     "FP": d.get("FP"), "precision": d.get("precision")})
    return pd.DataFrame(rows)


def cluster_frame(reports: Iterable[dict]) -> pd.DataFrame:
    """Cluster-level metrics across reports, with the closure caveat attached.

    `closure_contradictions` travels next to the B-cubed numbers on purpose: a
    non-zero value means the truth partition asserts pairs the labeler called
    non-matches, and the row should be read as directional. Splitting them into
    separate tables is how that caveat gets lost.
    """
    rows = []
    for r in reports:
        cl = r.get("cluster_level") or {}
        b = cl.get("bcubed") or {}
        rec = cl.get("recovery") or {}
        nonsing = rec.get("non_singleton") or {}
        diag = cl.get("closure_diagnostics") or {}
        rows.append({
            "session_id": _session(r),
            "run_id": r.get("run_id"),
            "source": r.get("label_source"),
            "holdout": (r.get("leakage") or {}).get("restriction"),
            "bcubed_precision": b.get("precision"),
            "bcubed_recall": b.get("recall"),
            "bcubed_f1": b.get("f1"),
            "exact_rate": rec.get("exact_rate"),
            "nonsingleton_exact_rate": nonsing.get("exact_rate"),
            "nonsingleton_split": nonsing.get("split"),
            "nonsingleton_merged": nonsing.get("merged"),
            "truth_source": diag.get("source", "transitive closure of labels"),
            "closure_contradictions": diag.get("n_contradicted"),
        })
    return pd.DataFrame(rows)


def metric_history(
    reports: Iterable[dict],
    metric: str = "recall",
    stage: str = "clustering",
) -> pd.DataFrame:
    """Long-format history for plotting: one row per (report, series).

    Columns: `session_id`, `evaluated_utc`, `run_id`, `series`
    (source / holdout), `value`. Keyed on the session so the real-data and
    synthetic reports of one evaluation share an x-position. Long rather than
    wide because the set of series varies between comparisons, and a wide frame
    would need re-pivoting every time one is added.
    """
    rows = []
    for r in reports:
        stages = r.get("stage_pairwise") or {}
        m = (r.get("clustering") if stage == "clustering" else stages.get(stage)) or {}
        if m.get("skipped") or m.get(metric) is None:
            continue
        rows.append({
            "session_id": _session(r),
            "evaluated_utc": r.get("evaluated_utc"),
            "run_id": r.get("run_id"),
            "series": _label(r),
            "metric": metric,
            "stage": stage,
            "value": m.get(metric),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["series", "evaluated_utc"], ignore_index=True)
    return df
