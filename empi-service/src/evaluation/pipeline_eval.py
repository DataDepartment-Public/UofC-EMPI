"""End-to-end pipeline evaluation against a pairwise label set.

Everything in `src/evaluation/` before this module scored **one stage**:
`blocking_eval.py` measures blocking against the deterministic rules,
`rule_eval.py` measures the rules against heuristic adjudication. Neither
reads `data/clusters/`, so until now nothing answered the question this module
exists for — *how good is the thing the pipeline actually produces?*

It takes a completed run's `RunManifest` plus a labeled pair set and reports:

* **the headline** — labeled pairs vs. the run's final clustering
  (`cluster_eval.pairwise_against_clusters`);
* **a funnel** — where every labeled pair sits at each stage boundary
  (blocked → rules → gate → matcher → clustered), so a bad headline can be
  attributed to a stage instead of guessed at;
* **loss attribution** — for each true pair the run failed to cluster, the
  *first* stage that dropped it. This is the actionable output: it converts
  "recall is 0.83" into "6,100 of the misses were gate drops";
* **transitivity** — merges that exist only because clustering takes the
  transitive closure of the edges. These are invisible to every pair-level
  metric upstream and are the one way clustering can be worse than its inputs;
* **cluster-level metrics** — B-cubed and exact-cluster recovery against the
  truth partition implied by the labels, with the closure diagnostics that say
  how far to trust them.

**Leakage.** For gold labels this is not optional: the Stage-4.25 gate and the
served Stage-4.5 matcher were both trained on the gold file. Pass a holdout
from `src.evaluation.holdout` (the CLI's `--holdout strict` does) or the ML
stages' numbers are reporting memorization. The `leakage` block in the report
records which restriction was applied so a number can never be read without it.

PHI / HIPAA: counts and metrics only — no PATIDs, no field values, no examples.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.config import Settings, settings as default_settings
from src.contracts import (
    RunManifest,
    TIER_AUTO_MERGE,
    TIER_HUMAN_REVIEW,
    TIER_NO_MATCH,
)
from src.evaluation.cluster_eval import (
    PairKey,
    bcubed,
    binary_metrics,
    cluster_map,
    cluster_recovery,
    induce,
    pair_keys,
    pairwise_against_clusters,
    predict_same_cluster,
    size_distribution,
    truth_clusters_from_pairs,
)
from src.evaluation.triage import ROUTES, stage_flow, triage_evaluation

logger = logging.getLogger(__name__)

__all__ = ["EndToEndReport", "evaluate_run", "load_manifest", "write_report"]

#: Ordered stage gates a labeled pair passes through. Order is load-bearing:
#: `loss_attribution` credits the FIRST stage that dropped a pair, so a pair
#: rejected by the rules is never also blamed on the gate.
_FUNNEL_STAGES = ("blocked", "rules_kept", "gate_passed", "ml_kept", "clustered")


# ── Artifact loading ─────────────────────────────────────────────────────────
def load_manifest(run_id: str, settings: Settings = default_settings) -> RunManifest:
    path = settings.runs_dir / f"run_{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No run manifest at {path}")
    return RunManifest.model_validate_json(path.read_text())


def _resolve(rel: str, settings: Settings) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else settings.project_root / path


def _read(ref, settings: Settings) -> pd.DataFrame | None:
    """Load an optional `ArtifactRef`; None when the stage was skipped."""
    if ref is None:
        return None
    path = _resolve(ref.path, settings)
    if not path.exists():
        logger.warning("Manifest references a missing artifact: %s", ref.path)
        return None
    return pd.read_parquet(path)


def _keyset(df: pd.DataFrame | None) -> set[PairKey]:
    return set(pair_keys(df)) if df is not None and len(df) else set()


#: The tier column is named differently by different producers: the shared
#: `ClassificationResults` shape (gate, ML matcher) calls it `predicted_tier`,
#: while the FS matcher's persisted `ProbabilisticMatches` artifact calls it
#: `classification_tier`. The tier *vocabulary* is the same either way.
_TIER_COLUMNS = ("predicted_tier", "classification_tier")


def _tier_keys(df: pd.DataFrame | None, tier: str) -> set[PairKey]:
    if df is None or not len(df):
        return set()
    for col in _TIER_COLUMNS:
        if col in df.columns:
            return set(pair_keys(df[df[col] == tier]))
    raise KeyError(
        f"No tier column in the scored frame (looked for {list(_TIER_COLUMNS)}; "
        f"found {list(df.columns)})."
    )


# ── Report ───────────────────────────────────────────────────────────────────
@dataclass
class EndToEndReport:
    #: Groups every report produced by one `scripts/evaluate_all.py` invocation.
    #: This — not `run_id` — is the unit of comparison over time: one session
    #: evaluates the real-data run AND the synthetic run, which necessarily have
    #: different run ids, and putting them on the same timeline point is the
    #: whole reason the field exists.
    session_id: str
    run_id: str
    #: When this evaluation ran, and the code state the *pipeline run* was
    #: produced at. Both exist so a stored report can be placed on a timeline
    #: and tied back to a commit — comparing two reports is meaningless if you
    #: can't tell whether the pipeline or only the labels changed.
    evaluated_utc: str
    git_sha: str | None
    label_source: str
    label_col: str
    leakage: dict
    universe: dict
    clustering: dict
    #: Three-class routing: where each pair *should* have gone (no_match /
    #: human_review / auto_merge) vs. where the pipeline sent it. This is the
    #: fair recall view — `clustering` above scores an ambiguous pair routed to
    #: review as a miss, when routing it there is the correct behavior.
    triage: dict
    #: Per stage: what came in and what the stage did with it (merged /
    #: rejected / passed on). Counts, not rates — the P/R view below is
    #: routinely misread because "false positive" means "kept this pair alive"
    #: at a filter stage and "wrongly merged two patients" at the output.
    stage_flow: list
    stage_pairwise: dict
    funnel: dict
    loss_attribution: dict
    transitivity: dict
    cluster_level: dict
    counts: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        lines: list[str] = []

        def hdr(title: str) -> None:
            lines.append("")
            lines.append("=" * 72)
            lines.append(f"  {title}")
            lines.append("=" * 72)

        def prf(label: str, m: dict) -> None:
            p = "N/A" if m.get("precision") is None else f"{m['precision']:.4f}"
            r = "N/A" if m.get("recall") is None else f"{m['recall']:.4f}"
            f = "N/A" if m.get("f1") is None else f"{m['f1']:.4f}"
            lines.append(
                f"  {label:<26} P={p:<9} R={r:<9} F1={f:<9} "
                f"TP={m.get('TP')} FP={m.get('FP')} FN={m.get('FN')}"
            )

        hdr(f"END-TO-END EVALUATION — run {self.run_id} vs {self.label_source} labels")
        lines.append(f"  label column       : {self.label_col}")
        lines.append(f"  labeled pairs      : {self.universe['labeled_pairs']}"
                     f"  ({self.universe['positives']} positive)")
        lines.append(f"  leakage restriction: {self.leakage['restriction']}")
        if self.leakage.get("note"):
            lines.append(f"  {self.leakage['note']}")

        hdr("HEADLINE — labeled pairs vs. final clustering (Stage 5)")
        prf("clustering", self.clustering)
        cov = self.clustering["coverage"]
        lines.append(f"  covered {cov['covered_pairs']}/{cov['labeled_pairs']} pairs "
                     f"({cov['uncovered_pairs']} uncovered, "
                     f"{cov['uncovered_positives']} of them true matches)")
        prf("  (uncovered as non-merge)", self.clustering["uncovered_as_negative"])

        if self.triage:
            t = self.triage
            names = t.get("gold_class_names", {})
            hdr("TRIAGE — 3-class routing (the decision the pipeline actually makes)")
            if t.get("ambiguous_col") is None:
                lines.append("  2-class only: this label source marks no ambiguous "
                             "pairs, so the human_review route has no expected "
                             "population.")
            else:
                lines.append(f"  ambiguous column: {t['ambiguous_col']}   "
                             f"(ambiguous outranks match: {t['ambiguous_precedence']})")
            lines.append("")
            lines.append("  confusion matrix — rows: expected route, cols: actual")
            header = "".join(f"{c:>14}" for c in ROUTES)
            lines.append(f"  {'':<26}{header}{'total':>10}")
            cm = t["confusion_matrix"]
            for route in ROUTES:
                row = cm.get(route, {})
                label = f"{names.get(route, route)} -> {route}"
                cells = "".join(f"{int(row.get(c, 0)):>14,}" for c in ROUTES)
                lines.append(f"  {label:<26}{cells}{int(sum(row.values())):>10,}")
            lines.append("")
            lines.append("  classification report")
            lines.append(f"  {'route':<22}{'precision':>12}{'recall':>10}"
                         f"{'f1':>10}{'support':>12}")
            for route in ROUTES:
                m = t["per_class"][route]
                fmt = lambda v: "N/A" if v is None else f"{v:.4f}"  # noqa: E731
                lines.append(
                    f"  {route:<22}{fmt(m['precision']):>12}{fmt(m['recall']):>10}"
                    f"{fmt(m['f1']):>10}{m['support']:>12,}"
                )
            for avg in ("macro_avg", "weighted_avg"):
                m = t[avg]
                fmt = lambda v: "N/A" if v is None else f"{v:.4f}"  # noqa: E731
                lines.append(f"  {avg:<22}{fmt(m['precision']):>12}"
                             f"{fmt(m['recall']):>10}{fmt(m['f1']):>10}")
            lines.append(f"  {'accuracy':<22}{fmt(t['accuracy']):>12}"
                         f"{'':>20}{t['n_pairs']:>12,}")

        if self.stage_flow:
            hdr("STAGE FLOW — what each stage did with the pairs it received")
            lines.append(f"  {'stage':<22}{'saw':>9}{'merged':>9}{'rejected':>10}"
                         f"{'passed on':>11}{'true lost':>11}")
            def cell(n: int) -> str:
                return f"{n:,}" if n else "-"

            for r in self.stage_flow:
                merged = cell(r["auto_merge"])
                if r["auto_merge"] and not r["binding"]:
                    merged += "*"      # scored but wired to nothing
                lines.append(
                    f"  {r['stage']:<22}{r['saw']:>9,}{merged:>9}"
                    f"{cell(r['no_match']):>10}{cell(r['to_next']):>11}"
                    f"{r['true_lost']:>11,}"
                )
            lines.append("")
            lines.append("  of which true matches:")
            for r in self.stage_flow:
                lines.append(f"    {r['stage']:<22}saw {r['saw_true']:>7,}   "
                             f"merged {r['auto_merge_true']:>7,}   "
                             f"passed on {r['to_next_true']:>7,}")
            lines.append("")
            for r in self.stage_flow:
                if r.get("note"):
                    lines.append(f"  {r['stage']:<22}{r['note']}")
            if any(r["auto_merge"] and not r["binding"] for r in self.stage_flow):
                lines.append("  * advisory — this tier feeds nothing downstream")

        hdr("PER-STAGE — each stage's own decision, on the pairs it actually saw")
        for name, m in self.stage_pairwise.items():
            if m.get("skipped"):
                lines.append(f"  {name:<26} (stage not present in this run)")
                continue
            prf(name, m)
            detail = f"target = {m['target']}" if m.get("target") else ""
            scored, total = m.get("scored_pairs"), m.get("labeled_pairs")
            if scored is not None and total is not None and scored < total:
                detail += (f"   |   scored on {scored}/{total} labeled pairs "
                           f"(the rest never reached this stage)")
            if detail:
                lines.append(f"  {'':<26} {detail}")

        hdr("FUNNEL — labeled pairs surviving each stage")
        lines.append(f"  {'stage':<16}{'positives':>12}{'negatives':>12}{'total':>10}")
        for stage in _FUNNEL_STAGES:
            d = self.funnel[stage]
            lines.append(f"  {stage:<16}{d['positives']:>12}{d['negatives']:>12}{d['total']:>10}")
        lines.append("  note: 'clustered' is not a subset of the row above it — "
                     "transitivity can merge a pair that never even blocked.")

        hdr("LOSS ATTRIBUTION — where true pairs were dropped (first stage to blame)")
        total = self.loss_attribution["total_missed"]
        lines.append(f"  missed true pairs: {total}")
        for stage, n in self.loss_attribution["by_stage"].items():
            pct = f"{100 * n / total:.1f}%" if total else "—"
            lines.append(f"    {stage:<40}{n:>8}{pct:>9}")

        hdr("TRANSITIVITY — merges clustering created beyond its direct edges")
        t = self.transitivity
        lines.append(f"  labeled pairs in same cluster : {t['same_cluster']}")
        lines.append(f"    via a stage's auto_merge edge: {t['direct']['n']} "
                     f"(TP={t['direct']['TP']} FP={t['direct']['FP']})")
        lines.append(f"    transitive closure only      : {t['transitive_only']['n']} "
                     f"(TP={t['transitive_only']['TP']} FP={t['transitive_only']['FP']})")
        if t["transitive_only"]["precision"] is not None:
            lines.append(f"    precision of transitive-only merges: "
                         f"{t['transitive_only']['precision']:.4f}")
        lines.append(f"  {t['note']}")

        hdr("CLUSTER-LEVEL — truth partition from the labels vs. the run's")
        c = self.cluster_level
        d = c["closure_diagnostics"]
        lines.append(f"  truth clusters: {d['n_truth_clusters']} over {d['n_records']} "
                     f"records (max size {d['max_cluster_size']})")
        if d.get("source"):
            lines.append(f"  truth source  : {d['source']} — these metrics are exact.")
        else:
            lines.append(f"  closure asserts {d['n_implied_pairs']} pairs; "
                         f"{d['n_implied_unlabeled']} of them were never labeled")
            lines.append(f"  closure CONTRADICTS {d['n_contradicted']} labeled "
                         f"non-matches (rate {d['contradiction_rate']})")
            if d["n_contradicted"]:
                lines.append("  ^ labels are not transitively consistent — read the "
                             "numbers below as directional, and prefer the headline.")
        b = c["bcubed"]
        lines.append(f"  B-cubed        P={b['precision']}  R={b['recall']}  F1={b['f1']} "
                     f"(n={b['n_records']})")
        r = c["recovery"]
        lines.append(f"  exact clusters {r['exact']}/{r['n_truth_clusters']} "
                     f"(rate {r['exact_rate']}) — split={r['split']} "
                     f"merged={r['merged']} mixed={r['mixed']}")
        nm = r["non_singleton"]
        lines.append(f"    non-singleton  {nm['exact']}/{nm['n_truth_clusters']} "
                     f"(rate {nm['exact_rate']}) — split={nm['split']} "
                     f"merged={nm['merged']} mixed={nm['mixed']}")
        lines.append(f"  predicted partition (induced): {c['predicted_sizes']}")
        lines.append("")
        return "\n".join(lines)


# ── Core evaluation ──────────────────────────────────────────────────────────
def evaluate_run(
    manifest: RunManifest,
    labeled: pd.DataFrame,
    label_col: str,
    *,
    settings: Settings = default_settings,
    label_source: str = "labels",
    holdout: set[PairKey] | None = None,
    holdout_name: str = "none",
    leakage_note: str | None = None,
    session_id: str | None = None,
    plausible_col: str | None = None,
    confident_match_col: str | None = None,
    ambiguous_col: str | None = None,
    truth_partition: Mapping[str, object] | None = None,
) -> EndToEndReport:
    """Score one completed run against a labeled pair set.

    Parameters
    ----------
    labeled
        Pair frame with `PATID_A`, `PATID_B` and a boolean-ish `label_col`.
    holdout
        Restrict every metric to these pair keys. For gold labels this is how
        the gate/matcher training leakage is excluded — see `holdout.py`.
    plausible_col
        Optional column marking pairs the Stage-4.25 gate is *supposed* to keep
        (gold: `final_gold_label | ambiguous_pair`). The gate is scored against
        this rather than `label_col`, because keeping an ambiguous non-match is
        correct behavior for it and would otherwise be counted as a false
        positive.
    confident_match_col
        Optional column for the Stage-4.5 matcher's target (gold: label and not
        ambiguous). Same reasoning in the other direction.
    ambiguous_col
        Optional column marking pairs the labeler could not resolve (gold:
        `ambiguous_pair`). Enables the three-class `triage` block: with it, a
        pair the pipeline routed to review is scored as *correct* rather than
        as a missed merge. Without it the block degrades to two classes.
    truth_partition
        Real `{PATID: entity_id}` ground truth, when the label source has it
        (the synthetic set's `entity_id_*` columns do). Skips the transitive
        closure entirely, which makes the cluster-level metrics exact instead
        of directional.
    """
    labeled = labeled.reset_index(drop=True).copy()
    labeled[label_col] = labeled[label_col].astype(bool)

    keys_all = pair_keys(labeled)
    if holdout is not None:
        mask = np.array([k in holdout for k in keys_all], dtype=bool)
        n_before = len(labeled)
        labeled = labeled.loc[mask].reset_index(drop=True)
        logger.info(
            "Holdout '%s': %d/%d labeled pairs retained.", holdout_name,
            len(labeled), n_before,
        )
        if labeled.empty:
            raise ValueError(
                f"Holdout '{holdout_name}' left no labeled pairs — the label set "
                "and the holdout do not overlap."
            )

    keys = pair_keys(labeled)
    y_true = labeled[label_col].to_numpy().astype(bool)

    # ── artifacts ────────────────────────────────────────────────────────────
    clusters = _read(manifest.clusters, settings)
    if clusters is None:
        raise FileNotFoundError(
            "This run has no cluster assignments — end-to-end evaluation needs "
            "Stage 5 output."
        )
    partition = cluster_map(clusters)

    candidates = _keyset(_read(manifest.candidate_pairs, settings))
    matches = _keyset(_read(manifest.matches, settings))
    rejects = _keyset(_read(manifest.rejects, settings))
    review_evidence = _keyset(_read(manifest.review_evidence, settings))

    gate_df = _read(manifest.gate_results, settings)
    gate_pass = _tier_keys(gate_df, TIER_HUMAN_REVIEW)
    gate_drop = _tier_keys(gate_df, TIER_NO_MATCH)

    ml_df = _read(manifest.matches_ml, settings)
    ml_auto = _tier_keys(ml_df, TIER_AUTO_MERGE)
    ml_reject = _tier_keys(ml_df, TIER_NO_MATCH)

    fs_df = _read(manifest.matches_model, settings)

    same, covered = predict_same_cluster(keys, partition)

    # ── headline ─────────────────────────────────────────────────────────────
    clustering = pairwise_against_clusters(labeled, partition, label_col)

    # ── three-class routing ──────────────────────────────────────────────────
    # Scored against the pipeline's *shipped* decision, so `ml_reject` is the
    # matcher's own no_match tier (empty while ml_review_floor = 0.0) and the
    # matcher's auto_merge tier is deliberately NOT treated as a merge — it
    # feeds nothing while `ml_feeds_clustering` is off.
    triage = triage_evaluation(
        labeled, label_col, keys,
        ambiguous_col=ambiguous_col,
        partition=partition,
        candidates=candidates,
        rejects=rejects,
        gate_drop=gate_drop,
        ml_reject=ml_reject,
    )

    flow = stage_flow(
        keys, y_true,
        candidates=candidates, matches=matches, rejects=rejects,
        gate_scored=_keyset(gate_df), gate_drop=gate_drop,
        ml_scored=_keyset(ml_df), ml_auto=ml_auto, ml_reject=ml_reject,
        partition=partition, ml_binding=settings.ml_feeds_clustering,
    )

    # ── per-stage pairwise ───────────────────────────────────────────────────
    def _stage(pred_keys: set[PairKey], target: np.ndarray, target_name: str,
               present: bool = True, scored: set[PairKey] | None = None) -> dict:
        """Metrics for one stage's decision.

        `scored` is the stage's actual input population. It matters: the gate
        only sees the rules' `non_matches` pool and the ML matcher only sees
        the gate's survivors, so a true pair the rules already auto-merged
        never reaches either. Scoring it against the full labeled set would
        count those as stage misses and understate recall by exactly the
        number of pairs the previous stage resolved — which is most of them.
        """
        if not present:
            return {"skipped": True}
        if scored is None:
            mask = np.ones(len(keys), dtype=bool)
        else:
            mask = np.array([k in scored for k in keys], dtype=bool)
        pred = np.array([k in pred_keys for k in keys], dtype=bool)
        m = binary_metrics(target[mask], pred[mask])
        m["target"] = target_name
        m["scored_pairs"] = int(mask.sum())
        m["labeled_pairs"] = int(len(keys))
        return m

    def _col(name: str | None) -> np.ndarray | None:
        if name and name in labeled.columns:
            return labeled[name].astype(bool).to_numpy()
        return None

    y_plausible = _col(plausible_col)
    y_confident = _col(confident_match_col)

    stage_pairwise = {
        "blocking": _stage(candidates, y_true, label_col),
        "rules_auto_merge": _stage(matches, y_true, label_col),
        "rules_any_confirmed": _stage(matches | review_evidence, y_true, label_col),
        "gate_pass": _stage(
            gate_pass,
            y_plausible if y_plausible is not None else y_true,
            plausible_col if y_plausible is not None else label_col,
            present=gate_df is not None,
            scored=_keyset(gate_df),
        ),
        "ml_auto_merge": _stage(
            ml_auto,
            y_confident if y_confident is not None else y_true,
            confident_match_col if y_confident is not None else label_col,
            present=ml_df is not None,
            scored=_keyset(ml_df),
        ),
        "clustering": {**{k: v for k, v in clustering.items()
                          if k not in ("coverage", "uncovered_as_negative")},
                       "target": label_col},
    }
    if ml_df is not None and y_confident is not None:
        # The decision-relevant view for `ml_feeds_clustering`. Scored against
        # the *match* label rather than `confident_match`: merging a pair that
        # is a true match but was labeled ambiguous is a correct merge, and
        # counting it as a false positive (as the row above does) understates
        # what turning the toggle on would actually cost.
        stage_pairwise["ml_auto_merge (vs match label)"] = _stage(
            ml_auto, y_true, label_col, scored=_keyset(ml_df)
        )
    if fs_df is not None:
        stage_pairwise["fs_auto_merge (audit-only)"] = _stage(
            _tier_keys(fs_df, TIER_AUTO_MERGE), y_true, label_col,
            scored=_keyset(fs_df),
        )

    # ── funnel ───────────────────────────────────────────────────────────────
    # `rules_kept` = not rejected by REJECT_RULES; a rules auto-merge counts as
    # kept too. `gate_passed`/`ml_kept` treat "never reached this stage" as
    # kept-if-already-merged so an auto-merged pair is not miscounted as dropped.
    def _survives(idx: int) -> dict[str, bool]:
        k = keys[idx]
        blocked = k in candidates
        merged_by_rules = k in matches
        kept_rules = blocked and not (k in rejects)
        if gate_df is None:
            gate_ok = kept_rules
        else:
            gate_ok = kept_rules and (merged_by_rules or k not in gate_drop)
        if ml_df is None:
            ml_ok = gate_ok
        else:
            ml_ok = gate_ok and (merged_by_rules or k not in ml_reject)
        return {
            "blocked": blocked,
            "rules_kept": kept_rules,
            "gate_passed": gate_ok,
            "ml_kept": ml_ok,
            "clustered": bool(same[idx]),
        }

    survival = [_survives(i) for i in range(len(keys))]
    funnel = {}
    for stage in _FUNNEL_STAGES:
        flags = np.array([s[stage] for s in survival], dtype=bool)
        funnel[stage] = {
            "positives": int((flags & y_true).sum()),
            "negatives": int((flags & ~y_true).sum()),
            "total": int(flags.sum()),
        }

    # ── loss attribution ─────────────────────────────────────────────────────
    reasons: dict[str, int] = {
        "not blocked": 0,
        "rejected by deterministic rules": 0,
        "dropped by non-match gate": 0,
        "ml matcher tiered no_match": 0,
        "left in review (no auto_merge edge)": 0,
        "record not clustered (invalid/absent)": 0,
    }
    for i, k in enumerate(keys):
        if not y_true[i] or same[i]:
            continue
        s = survival[i]
        if not covered[i]:
            reasons["record not clustered (invalid/absent)"] += 1
        elif not s["blocked"]:
            reasons["not blocked"] += 1
        elif not s["rules_kept"]:
            reasons["rejected by deterministic rules"] += 1
        elif not s["gate_passed"]:
            reasons["dropped by non-match gate"] += 1
        elif not s["ml_kept"]:
            reasons["ml matcher tiered no_match"] += 1
        else:
            reasons["left in review (no auto_merge edge)"] += 1
    loss_attribution = {
        "total_missed": int((y_true & ~same).sum()),
        "by_stage": reasons,
        "note": "each missed pair is attributed to the FIRST stage that dropped "
                "it; 'left in review' means every stage kept the pair but no "
                "stage emitted an auto_merge edge for it (expected while "
                "ml_feeds_clustering is off).",
    }

    # ── transitivity ─────────────────────────────────────────────────────────
    auto_edges = matches | ml_auto | _tier_keys(fs_df, TIER_AUTO_MERGE)
    direct = np.array([k in auto_edges for k in keys], dtype=bool) & same
    trans = same & ~direct

    def _split(mask: np.ndarray) -> dict:
        tp = int((mask & y_true).sum())
        fp = int((mask & ~y_true).sum())
        n = tp + fp
        return {"n": n, "TP": tp, "FP": fp,
                "precision": round(tp / n, 4) if n else None}

    transitivity = {
        "same_cluster": int(same.sum()),
        "direct": _split(direct),
        "transitive_only": _split(trans),
        "note": "transitive-only merges exist because clustering takes the "
                "connected components of its edges — no pair classifier ever "
                "scored them, so their FPs are invisible upstream.",
    }

    # ── cluster level ────────────────────────────────────────────────────────
    if truth_partition is not None:
        truth = dict(truth_partition)
        sizes = size_distribution(truth)
        diagnostics = {
            "source": "declared ground truth — no transitive closure applied",
            "n_truth_clusters": sizes["n_clusters"],
            "n_records": sizes["n_records"],
            "max_cluster_size": sizes["max_size"],
            "n_singletons": sizes["n_singletons"],
            # No closure was taken, so none of the closure hazards apply.
            "n_implied_pairs": None,
            "n_implied_unlabeled": None,
            "n_contradicted": 0,
            "contradiction_rate": 0.0,
        }
    else:
        truth, diag = truth_clusters_from_pairs(labeled, label_col)
        diagnostics = diag.as_dict()
    pred_induced = induce(partition, truth.keys())
    cluster_level = {
        "closure_diagnostics": diagnostics,
        "bcubed": bcubed(truth, pred_induced),
        "recovery": cluster_recovery(truth, pred_induced),
        "truth_sizes": {k: v for k, v in size_distribution(truth).items()
                        if k != "size_histogram"},
        "predicted_sizes": {k: v for k, v in size_distribution(pred_induced).items()
                            if k != "size_histogram"},
    }

    # `restriction` stays short — it is the series label in every comparison
    # table and chart legend. Explanations belong in `note`.
    # Whether the *headline* is leakage-free at --holdout none depends on the
    # run's configuration, not just on the holdout: it is clean only while no
    # gold-trained stage feeds clustering. With a feed toggle on, the clusters
    # themselves are partly a memorized reconstruction of the training labels,
    # and saying only "the ML-stage rows are optimistic" would understate it.
    gold_trained_feeds = settings.ml_feeds_clustering or settings.fs_feeds_clustering
    if holdout_name == "none":
        default_note = (
            "NO holdout applied — the Stage-4.25 gate and the Stage-4.5 matcher "
            "were trained on ~80% of the gold pairs, so their rows are optimistic."
        )
        default_note += (
            " ml/fs_feeds_clustering is ON, so a gold-trained stage forms clusters "
            "and the HEADLINE is memorized too. Use --holdout strict."
            if gold_trained_feeds else
            " The headline itself is clean: with both feed toggles off, clustering "
            "unions only deterministic-rule edges, which were never fit on gold."
        )
    else:
        default_note = None
    leakage = {
        "restriction": holdout_name,
        "note": leakage_note if leakage_note is not None else default_note,
        "gold_trained_stage_feeds_clustering": bool(gold_trained_feeds),
    }

    return EndToEndReport(
        session_id=session_id or manifest.run_id,
        run_id=manifest.run_id,
        evaluated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=manifest.git_sha,
        label_source=label_source,
        label_col=label_col,
        leakage=leakage,
        universe={
            "labeled_pairs": int(len(labeled)),
            "positives": int(y_true.sum()),
            "records_in_labels": int(len({p for k in keys for p in k})),
        },
        clustering=clustering,
        triage=triage,
        stage_flow=flow,
        stage_pairwise=stage_pairwise,
        funnel=funnel,
        loss_attribution=loss_attribution,
        transitivity=transitivity,
        cluster_level=cluster_level,
        counts=dict(manifest.counts),
    )


# ── Persistence ──────────────────────────────────────────────────────────────
def write_report(
    report: EndToEndReport,
    settings: Settings = default_settings,
    out: Path | None = None,
) -> Path:
    """Persist a report as JSON (+ a human-readable .txt sibling).

    Reports live in `settings.evaluations_dir`, deliberately NOT in `runs_dir`:
    that directory holds pipeline `RunManifest`s — immutable per-run lineage —
    while these are measurements *about* runs and accumulate on their own
    timeline. Mixing them makes both harder to glob.

    The filename is keyed on (session, source, holdout), so re-evaluating the
    same session under the same settings overwrites rather than accumulating —
    that is the same measurement, and keeping both would put a duplicate point
    on every trend.
    """
    holdout = report.leakage.get("restriction", "none").replace("/", "-")
    out = out or (
        settings.evaluations_dir
        / f"eval_{report.session_id}__{report.label_source}__{holdout}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2, default=str))
    out.with_suffix(".txt").write_text(report.to_text())
    logger.info("Wrote evaluation report → %s", out)
    return out
