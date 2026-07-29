"""End-to-end evaluation over a hand-built run: funnel, attribution, transitivity.

The fixture run is deliberately tiny and fully hand-checked so every number in
the report has one obvious right answer:

    p1─p2─p7   clustered together (p1-p2 and p2-p7 are auto-merge edges)
    p3  p4     a true pair the rules never confirmed — stuck in review
    p5  p6     a true non-match the reject rules dropped
    p1  p7     labeled a NON-match, but transitivity merged them anyway
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from src.config import Settings
from src.contracts import (
    ArtifactRef,
    RunManifest,
    TIER_AUTO_MERGE,
    TIER_HUMAN_REVIEW,
    TIER_NO_MATCH,
)
from src.evaluation.pipeline_eval import evaluate_run, load_manifest


def _write(df: pd.DataFrame, root, rel: str) -> ArtifactRef:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return ArtifactRef(
        path=rel, rows=len(df), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def _pairs(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["PATID_A", "PATID_B"])


def _results(rows) -> pd.DataFrame:
    """`ClassificationResults`-shaped frame (gate / ML matcher)."""
    return pd.DataFrame(
        [(a, b, "m", 0.5, tier) for a, b, tier in rows],
        columns=["PATID_A", "PATID_B", "model_name", "score", "predicted_tier"],
    )


def _fs_results(rows) -> pd.DataFrame:
    """`ProbabilisticMatches`-shaped frame — the FS matcher names its tier
    column `classification_tier`, not `predicted_tier`."""
    return pd.DataFrame(
        [(a, b, "model", 0.5, 1.0, tier) for a, b, tier in rows],
        columns=["PATID_A", "PATID_B", "match_source", "score", "match_weight",
                 "classification_tier"],
    )


@pytest.fixture
def run(tmp_path):
    """A complete run's artifacts + manifest under an isolated project root."""
    clusters = pd.DataFrame({
        "PATID": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        "cluster_id": [0, 0, 1, 2, 3, 4, 0],
    })
    manifest = RunManifest(
        run_id="TESTRUN",
        created_utc="2026-07-28T00:00:00Z",
        git_sha=None,
        raw_input=_write(pd.DataFrame({"x": [1]}), tmp_path, "data/raw/in.parquet"),
        cleaned=_write(clusters[["PATID"]], tmp_path, "data/processed/c.parquet"),
        candidate_pairs=_write(
            _pairs([("p1", "p2"), ("p2", "p7"), ("p3", "p4"), ("p5", "p6")]),
            tmp_path, "data/blocking/cp.parquet"),
        matches=_write(_pairs([("p1", "p2"), ("p2", "p7")]),
                       tmp_path, "data/auto_merge/m.parquet"),
        non_matches=_write(_pairs([("p3", "p4")]),
                           tmp_path, "data/non_matches/nm.parquet"),
        rejects=_write(_pairs([("p5", "p6")]), tmp_path, "data/no_match/r.parquet"),
        gate_results=_write(_results([("p3", "p4", TIER_HUMAN_REVIEW)]),
                            tmp_path, "data/gate_output/g.parquet"),
        matches_ml=_write(_results([("p3", "p4", TIER_HUMAN_REVIEW)]),
                          tmp_path, "data/ml_output/ml.parquet"),
        clusters=_write(clusters, tmp_path, "data/clusters/cl.parquet"),
        counts={"cleaned_rows": 7},
    )
    return manifest, Settings(project_root=tmp_path), tmp_path


@pytest.fixture
def labeled():
    return pd.DataFrame(
        [("p1", "p2", True), ("p3", "p4", True), ("p5", "p6", False),
         ("p1", "p7", False)],
        columns=["PATID_A", "PATID_B", "label"],
    )


def _report(run, labeled, **kw):
    manifest, settings, _ = run
    return evaluate_run(manifest, labeled, "label", settings=settings, **kw)


# ── headline ─────────────────────────────────────────────────────────────────
def test_headline_scores_the_clustering_not_the_classifiers(run, labeled):
    r = _report(run, labeled)
    c = r.clustering
    # p1-p2 merged (TP), p1-p7 merged but labeled non-match (FP),
    # p3-p4 not merged (FN), p5-p6 not merged (TN).
    assert (c["TP"], c["FP"], c["FN"], c["TN"]) == (1, 1, 1, 1)
    assert c["precision"] == 0.5 and c["recall"] == 0.5
    assert c["coverage"]["uncovered_pairs"] == 0


def test_rules_stage_is_scored_separately_from_clustering(run, labeled):
    r = _report(run, labeled)
    rules = r.stage_pairwise["rules_auto_merge"]
    assert rules["TP"] == 1 and rules["FN"] == 1


def test_gate_is_scored_only_on_the_pool_it_actually_saw(run, labeled):
    """p1-p2 is a true pair the rules auto-merged, so it never enters the
    gate's `non_matches` pool. Counting it as a gate miss would understate
    gate recall by exactly the number of pairs the rules already resolved."""
    r = _report(run, labeled)
    gate = r.stage_pairwise["gate_pass"]

    assert gate["scored_pairs"] == 1          # only p3-p4 reached the gate
    assert gate["labeled_pairs"] == 4
    assert gate["FN"] == 0                    # not blamed for p1-p2
    assert gate["recall"] == 1.0


def test_ml_matcher_is_scored_only_on_gate_survivors(run, labeled):
    r = _report(run, labeled)
    assert r.stage_pairwise["ml_auto_merge"]["scored_pairs"] == 1


def test_full_population_stages_are_scored_on_every_labeled_pair(run, labeled):
    r = _report(run, labeled)
    for name in ("blocking", "rules_auto_merge"):
        assert r.stage_pairwise[name]["scored_pairs"] == 4


def test_ml_is_also_scored_against_the_match_label_for_the_feed_decision(run, labeled):
    """`confident_match` penalizes merging a true-but-ambiguous pair; the
    match-label view is what actually answers 'turn ml_feeds_clustering on?'"""
    labeled = labeled.copy()
    labeled["confident_match"] = [True, False, False, False]  # p3-p4 ambiguous
    r = _report(run, labeled, confident_match_col="confident_match")
    assert "ml_auto_merge (vs match label)" in r.stage_pairwise
    assert r.stage_pairwise["ml_auto_merge"]["target"] == "confident_match"
    assert r.stage_pairwise["ml_auto_merge (vs match label)"]["target"] == "label"


def test_fs_artifact_tier_column_is_read_despite_its_different_name(run, labeled):
    """`ProbabilisticMatches` calls the tier `classification_tier` while every
    other stage calls it `predicted_tier` — reading FS must not crash."""
    manifest, settings, tmp_path = run
    manifest.matches_model = _write(
        _fs_results([("p1", "p2", TIER_AUTO_MERGE), ("p3", "p4", TIER_NO_MATCH)]),
        tmp_path, "data/fs_output/fs.parquet")
    r = evaluate_run(manifest, labeled, "label", settings=settings)

    fs = r.stage_pairwise["fs_auto_merge (audit-only)"]
    assert fs["TP"] == 1 and fs["FP"] == 0


def test_scored_frame_without_any_tier_column_fails_loudly(run, labeled):
    manifest, settings, tmp_path = run
    manifest.matches_ml = _write(_pairs([("p3", "p4")]),
                                 tmp_path, "data/ml_output/bad.parquet")
    with pytest.raises(KeyError, match="No tier column"):
        evaluate_run(manifest, labeled, "label", settings=settings)


def test_skipped_stages_are_reported_as_absent(run, labeled):
    manifest, settings, _ = run
    manifest.gate_results = None
    manifest.matches_ml = None
    r = evaluate_run(manifest, labeled, "label", settings=settings)
    assert r.stage_pairwise["gate_pass"] == {"skipped": True}
    assert r.stage_pairwise["ml_auto_merge"] == {"skipped": True}


# ── funnel + attribution ─────────────────────────────────────────────────────
def test_funnel_narrows_through_the_classifier_stages(run, labeled):
    r = _report(run, labeled)
    # (p1, p7) is labeled but was never a candidate pair, so only 3 of the 4
    # labeled pairs enter the funnel at all.
    assert r.funnel["blocked"]["total"] == 3
    assert r.funnel["rules_kept"]["total"] == 2      # p5-p6 rejected
    assert r.funnel["gate_passed"]["total"] == 2
    assert r.funnel["ml_kept"]["total"] == 2


def test_clustered_row_is_not_a_subset_of_the_stage_above_it(run, labeled):
    """Transitivity lets clustering merge a pair that never even blocked —
    the funnel's last row is a different population, not a further filter."""
    r = _report(run, labeled)
    assert r.funnel["clustered"]["total"] == 2       # p1-p2 and p1-p7
    assert r.transitivity["transitive_only"]["n"] == 1


def test_loss_attribution_blames_the_review_backlog(run, labeled):
    """p3-p4 survived every stage but no stage emitted an auto_merge edge —
    the expected failure mode while ml_feeds_clustering is off."""
    r = _report(run, labeled)
    assert r.loss_attribution["total_missed"] == 1
    assert r.loss_attribution["by_stage"]["left in review (no auto_merge edge)"] == 1
    assert sum(r.loss_attribution["by_stage"].values()) == 1


def test_loss_attribution_blames_the_gate_when_the_gate_drops_a_true_pair(run, labeled):
    manifest, settings, tmp_path = run
    manifest.gate_results = _write(
        _results([("p3", "p4", TIER_NO_MATCH)]), tmp_path, "data/gate_output/g2.parquet")
    r = evaluate_run(manifest, labeled, "label", settings=settings)
    assert r.loss_attribution["by_stage"]["dropped by non-match gate"] == 1
    assert r.loss_attribution["by_stage"]["left in review (no auto_merge edge)"] == 0


def test_loss_attribution_credits_only_the_first_stage_to_drop_a_pair(run, labeled):
    """A pair the rules rejected is never also blamed on the gate."""
    manifest, settings, tmp_path = run
    labeled = labeled.copy()
    labeled.loc[labeled["PATID_A"] == "p5", "label"] = True  # p5-p6 now a true pair
    manifest.gate_results = _write(
        _results([("p5", "p6", TIER_NO_MATCH)]), tmp_path, "data/gate_output/g3.parquet")
    r = evaluate_run(manifest, labeled, "label", settings=settings)
    assert r.loss_attribution["by_stage"]["rejected by deterministic rules"] == 1
    assert r.loss_attribution["by_stage"]["dropped by non-match gate"] == 0


def test_uncovered_records_are_attributed_not_silently_dropped(run, labeled):
    manifest, settings, tmp_path = run
    manifest.clusters = _write(
        pd.DataFrame({"PATID": ["p1", "p2", "p7"], "cluster_id": [0, 0, 0]}),
        tmp_path, "data/clusters/cl2.parquet")
    r = evaluate_run(manifest, labeled, "label", settings=settings)
    assert r.loss_attribution["by_stage"]["record not clustered (invalid/absent)"] == 1


# ── transitivity ─────────────────────────────────────────────────────────────
def test_transitive_only_merges_are_separated_from_direct_edges(run, labeled):
    """The one thing clustering can get wrong that no upstream metric sees."""
    r = _report(run, labeled)
    t = r.transitivity
    assert t["same_cluster"] == 2
    assert t["direct"]["n"] == 1 and t["direct"]["TP"] == 1      # p1-p2
    assert t["transitive_only"]["n"] == 1                        # p1-p7
    assert t["transitive_only"]["FP"] == 1
    assert t["transitive_only"]["precision"] == 0.0


# ── cluster level ────────────────────────────────────────────────────────────
def test_declared_truth_partition_skips_the_closure(run, labeled):
    truth = {"p1": "E1", "p2": "E1", "p3": "E2", "p4": "E2",
             "p5": "E3", "p6": "E4", "p7": "E5"}
    r = _report(run, labeled, truth_partition=truth)
    d = r.cluster_level["closure_diagnostics"]
    assert d["source"].startswith("declared ground truth")
    assert d["n_truth_clusters"] == 5
    assert d["n_contradicted"] == 0
    # p7 merged into E1's cluster costs precision; E2 split across two clusters
    # costs recall on p3 and p4 (1/2 each) -> (5 * 1 + 2 * 0.5) / 7.
    assert r.cluster_level["bcubed"]["recall"] == pytest.approx(6 / 7, abs=1e-4)
    assert r.cluster_level["bcubed"]["precision"] < 1.0


def test_closure_is_used_when_no_truth_partition_is_given(run, labeled):
    r = _report(run, labeled)
    d = r.cluster_level["closure_diagnostics"]
    assert "source" not in d
    assert d["n_truth_clusters"] == 5  # {p1,p2} {p3,p4} {p5} {p6} {p7}


# ── leakage guard ────────────────────────────────────────────────────────────
def test_no_holdout_is_recorded_as_a_warning_in_the_report(run, labeled):
    r = _report(run, labeled)
    assert r.leakage["restriction"] == "none"
    assert "optimistic" in r.leakage["note"]


def test_holdout_restricts_the_universe(run, labeled):
    r = _report(run, labeled, holdout={("p1", "p2")}, holdout_name="strict")
    assert r.universe["labeled_pairs"] == 1
    assert r.leakage["restriction"] == "strict"
    assert r.leakage["note"] is None


def test_non_overlapping_holdout_fails_loudly(run, labeled):
    with pytest.raises(ValueError, match="left no labeled pairs"):
        _report(run, labeled, holdout={("zz", "zzz")}, holdout_name="strict")


def test_run_without_clusters_cannot_be_scored_end_to_end(run, labeled):
    manifest, settings, _ = run
    manifest.clusters = None
    with pytest.raises(FileNotFoundError, match="no cluster assignments"):
        evaluate_run(manifest, labeled, "label", settings=settings)


# ── manifest loading ─────────────────────────────────────────────────────────
def test_load_manifest_round_trips(run, tmp_path):
    manifest, settings, root = run
    runs_dir = root / "runs"
    runs_dir.mkdir()
    (runs_dir / "run_TESTRUN.json").write_text(manifest.model_dump_json())
    settings = Settings(project_root=root, runs_dir=runs_dir)
    assert load_manifest("TESTRUN", settings).run_id == "TESTRUN"


def test_load_manifest_missing_run_is_explicit(run):
    _manifest, settings, _ = run
    with pytest.raises(FileNotFoundError, match="No run manifest"):
        load_manifest("NOPE", settings)
