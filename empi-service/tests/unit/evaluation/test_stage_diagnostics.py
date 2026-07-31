"""Per-pair stage diagnostics over a hand-built run.

The fixture is small enough that every cell of every matrix has one obvious
right answer, and it exercises each disposition exactly once:

    p1─p2   true match, auto-merged by the rules            -> correct
    p3 p4   true match blocking never emitted               -> unrecoverable miss
    p5 p6   ambiguous, gate kept it, matcher auto-merged it -> over-merge
    p7 p8   ambiguous, the gate dropped it                  -> gate recall miss
    p9 p10  non-match the reject rules dropped              -> correct
    p11 p12 non-match still sitting in review               -> correct route
"""

from __future__ import annotations

import hashlib

import numpy as np
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
from src.evaluation import stage_diagnostics as sd


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
    return pd.DataFrame(
        [(a, b, "m", score, tier) for a, b, score, tier in rows],
        columns=["PATID_A", "PATID_B", "model_name", "score", "predicted_tier"],
    )


@pytest.fixture
def run(tmp_path):
    patids = [f"p{i}" for i in range(1, 13)]
    clusters = pd.DataFrame({
        "PATID": patids,
        # p1-p2 (rules) and p5-p6 (matcher) are the only merged pairs.
        "cluster_id": [0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9],
    })
    cleaned = pd.DataFrame({
        "PATID": patids,
        "LastNM_clean": ["SMITH"] * 12,
        "FirstNM_clean": ["JOHN"] * 12,
        "valid_record": [True] * 12,
    })
    manifest = RunManifest(
        run_id="TESTRUN",
        created_utc="2026-07-31T00:00:00Z",
        raw_input=_write(cleaned, tmp_path, "data/raw/in.parquet"),
        cleaned=_write(cleaned, tmp_path, "data/processed/c.parquet"),
        # p3-p4 is absent: blocking never emitted it.
        candidate_pairs=_write(
            pd.DataFrame(
                [("p1", "p2"), ("p5", "p6"), ("p7", "p8"), ("p9", "p10"),
                 ("p11", "p12")],
                columns=["PATID_A", "PATID_B"],
            ).assign(source_blocks="B3", n_blocks=1),
            tmp_path, "data/blocking/cp.parquet"),
        matches=_write(_pairs([("p1", "p2")]).assign(match_rule="SSN_DOB"),
                       tmp_path, "data/auto_merge/m.parquet"),
        non_matches=_write(_pairs([("p5", "p6")]),
                           tmp_path, "data/non_matches/nm.parquet"),
        rejects=_write(_pairs([("p9", "p10")]).assign(reject_rule="SSN_CONFLICT",
                                                      n_contradictions=3),
                       tmp_path, "data/no_match/r.parquet"),
        gate_results=_write(
            _results([("p5", "p6", 0.9, TIER_HUMAN_REVIEW),
                      ("p7", "p8", 0.1, TIER_NO_MATCH),
                      ("p11", "p12", 0.6, TIER_HUMAN_REVIEW)]),
            tmp_path, "data/gate_output/g.parquet"),
        matches_ml=_write(
            _results([("p5", "p6", 0.95, TIER_AUTO_MERGE),
                      ("p11", "p12", 0.2, TIER_HUMAN_REVIEW)]),
            tmp_path, "data/ml_output/ml.parquet"),
        clusters=_write(clusters, tmp_path, "data/clusters/cl.parquet"),
    )
    return manifest, Settings(project_root=tmp_path)


@pytest.fixture
def labeled():
    return pd.DataFrame(
        [("p1", "p2", True, False),
         ("p3", "p4", True, False),
         ("p5", "p6", False, True),
         ("p7", "p8", False, True),
         ("p9", "p10", False, False),
         ("p11", "p12", False, False)],
        columns=["PATID_A", "PATID_B", "label", "ambiguous_pair"],
    )


@pytest.fixture
def diag(run, labeled):
    manifest, settings = run
    return sd.build_diagnostics(
        manifest, labeled, "label", settings=settings, label_source="test",
        ambiguous_col="ambiguous_pair",
    )


# ── the frame ────────────────────────────────────────────────────────────────
def test_every_stage_decision_lands_on_the_right_pair(diag):
    p = diag.pairs.set_index([diag.pairs["PATID_A"], diag.pairs["PATID_B"]])
    assert p.loc[("p1", "p2"), "rules_decision"] == TIER_AUTO_MERGE
    assert not p.loc[("p3", "p4"), "blocked"]
    assert p.loc[("p3", "p4"), "rules_decision"] is None   # never reached them
    assert p.loc[("p9", "p10"), "rules_decision"] == TIER_NO_MATCH
    assert p.loc[("p7", "p8"), "gate_scored"] and not p.loc[("p7", "p8"), "gate_pass"]
    assert p.loc[("p5", "p6"), "ml_auto"] and p.loc[("p5", "p6"), "clustered"]


def test_ambiguous_outranks_match_in_the_gold_class(run):
    """The same precedence `triage` documents: weak evidence belongs with a
    human whatever the eventual adjudication was."""
    manifest, settings = run
    both = pd.DataFrame([("p1", "p2", True, True)],
                        columns=["PATID_A", "PATID_B", "label", "ambiguous_pair"])
    d = sd.build_diagnostics(manifest, both, "label", settings=settings,
                             ambiguous_col="ambiguous_pair")
    assert d.pairs.loc[0, "gold_class"] == "ambiguous"
    assert not d.pairs.loc[0, "gold_confident"]
    assert d.pairs.loc[0, "gold_plausible"]


def test_routes_only_ever_settle_forward(diag):
    """Each cumulative column refines the previous one; a pair the rules
    auto-merged is never re-opened by a later stage."""
    p = diag.pairs.set_index([diag.pairs["PATID_A"], diag.pairs["PATID_B"]])
    assert list(p.loc[("p1", "p2"), ["route_after_rules", "route_after_gate",
                                     "route_after_ml", "route_final"]]) == \
        [TIER_AUTO_MERGE] * 4
    # p7-p8 is open after the rules and closed by the gate.
    assert p.loc[("p7", "p8"), "route_after_rules"] == TIER_HUMAN_REVIEW
    assert p.loc[("p7", "p8"), "route_after_gate"] == TIER_NO_MATCH
    # p5-p6 is open until the matcher merges it.
    assert p.loc[("p5", "p6"), "route_after_gate"] == TIER_HUMAN_REVIEW
    assert p.loc[("p5", "p6"), "route_after_ml"] == TIER_AUTO_MERGE


def test_a_pair_blocking_never_emitted_is_no_match_from_the_first_column(diag):
    """Not `human_review`: no stage ever saw it, and the pipeline has in fact
    left it unmerged with nothing downstream able to recover it."""
    p = diag.pairs.set_index([diag.pairs["PATID_A"], diag.pairs["PATID_B"]])
    assert p.loc[("p3", "p4"), "route_after_rules"] == TIER_NO_MATCH
    assert p.loc[("p3", "p4"), "route_final"] == TIER_NO_MATCH


# ── binary views ─────────────────────────────────────────────────────────────
def test_blocking_matrix_counts_the_unrecoverable_miss(diag):
    cm = sd.binary_confusion(diag, "blocking")
    assert cm.loc["true match", "not blocked"] == 1     # p3-p4
    assert cm.loc["true match", "blocked"] == 1         # p1-p2
    assert cm.loc["non-match", "blocked"] == 4
    assert cm["total"].sum() == 6


def test_the_gate_is_scored_on_its_own_pool_against_plausible(diag):
    """Only the three pairs the gate scored, and its target is plausible
    (match ∪ ambiguous) — dropping an ambiguous pair is a recall miss."""
    cm = sd.binary_confusion(diag, "gate")
    assert cm["total"].sum() == 3
    assert cm.loc["plausible", "dropped"] == 1          # p7-p8
    assert cm.loc["plausible", "passed"] == 1           # p5-p6
    report = sd.binary_report(diag, "gate")
    assert report.loc["plausible", "recall"] == 0.5


def test_the_matcher_is_scored_against_confident_match(diag):
    """p5-p6 is ambiguous, so auto-merging it is a false positive for the
    matcher even though the pair may well be a match."""
    cm = sd.binary_confusion(diag, "ml_matcher")
    assert cm["total"].sum() == 2
    assert cm.loc["not a confident match", "auto_merge"] == 1


def test_clustering_matrix_is_the_headline(diag):
    cm = sd.binary_confusion(diag, "clustering")
    assert cm.loc["true match", "merged"] == 1          # p1-p2
    assert cm.loc["true match", "not merged"] == 1      # p3-p4
    assert cm.loc["non-match", "merged"] == 1           # p5-p6, ambiguous


def test_normalized_matrix_is_row_percentages(diag):
    pct = sd.binary_confusion(diag, "blocking", normalize=True)
    assert "total" not in pct.columns
    assert pct.loc["true match"].sum() == pytest.approx(100.0)


def test_report_is_derived_from_the_matrix_it_prints(diag):
    """One implementation behind both, so a cell and its metric cannot
    disagree."""
    cm = sd.binary_confusion(diag, "blocking").drop(columns="total")
    report = sd.binary_report(diag, "blocking")
    tp = cm.loc["true match", "blocked"]
    predicted = cm["blocked"].sum()
    assert report.loc["true match", "precision"] == pytest.approx(tp / predicted, abs=1e-4)
    assert report.loc["true match", "support"] == cm.loc["true match"].sum()


def test_an_unpredicted_but_supported_class_scores_zero_not_nan(run):
    """Precision is undefined when nothing was predicted, but recall and F1
    are honestly zero — folding the two together hides a total failure."""
    manifest, settings = run
    only_miss = pd.DataFrame([("p3", "p4", True, False)],
                             columns=["PATID_A", "PATID_B", "label", "ambiguous_pair"])
    d = sd.build_diagnostics(manifest, only_miss, "label", settings=settings,
                             ambiguous_col="ambiguous_pair")
    report = sd.binary_report(d, "blocking")
    assert report.loc["true match", "recall"] == 0.0
    assert report.loc["true match", "f1"] == 0.0


# ── routing views ────────────────────────────────────────────────────────────
def test_routing_matrix_is_always_the_full_three_by_three(diag):
    cm = sd.route_confusion(diag, "rules")
    assert list(cm.columns) == [TIER_NO_MATCH, TIER_HUMAN_REVIEW, TIER_AUTO_MERGE,
                                "total"]
    assert cm["total"].sum() == len(diag.pairs)


def test_routing_after_the_matcher_shows_the_over_merge(diag):
    cm = sd.route_confusion(diag, "ml_matcher")
    assert cm.loc["ambiguous -> human_review", TIER_AUTO_MERGE] == 1   # p5-p6
    assert cm.loc["match -> auto_merge", TIER_AUTO_MERGE] == 1         # p1-p2
    assert cm.loc["match -> auto_merge", TIER_NO_MATCH] == 1           # p3-p4


def test_restricting_to_the_blocked_pool_gives_the_rules_own_decision(diag):
    """The cumulative view folds blocking misses into `no_match`; passing
    `population="blocked"` isolates what the rules themselves decided."""
    cumulative = sd.route_confusion(diag, "rules")
    own = sd.route_confusion(diag, "rules", population="blocked")
    assert cumulative.loc["match -> auto_merge", TIER_NO_MATCH] == 1
    assert own.loc["match -> auto_merge", TIER_NO_MATCH] == 0
    assert own["total"].sum() == 5


def test_route_report_classes_are_the_routes(diag):
    report = sd.route_report(diag, "clustering")
    assert list(report.index[:3]) == [TIER_NO_MATCH, TIER_HUMAN_REVIEW,
                                      TIER_AUTO_MERGE]
    assert set(report.index[3:]) == {"macro avg", "weighted avg", "accuracy"}


# ── error listings ───────────────────────────────────────────────────────────
def test_binary_errors_returns_exactly_the_off_diagonal_pairs(diag):
    fn = sd.binary_errors(diag, "blocking", kind="FN")
    assert list(zip(fn["PATID_A"], fn["PATID_B"])) == [("p3", "p4")]
    assert fn["error"].iloc[0] == "true match -> not blocked"

    both = sd.binary_errors(diag, "gate")
    cm = sd.binary_confusion(diag, "gate")
    off_diagonal = cm.loc["plausible", "dropped"] + cm.loc["confident non-match", "passed"]
    assert len(both) == off_diagonal


def test_binary_errors_rejects_an_unknown_kind(diag):
    with pytest.raises(ValueError, match="kind must be"):
        sd.binary_errors(diag, "blocking", kind="wrong")


def test_route_errors_can_pull_one_cell(diag):
    cell = sd.route_errors(diag, "clustering", expected=TIER_AUTO_MERGE,
                           actual=TIER_NO_MATCH)
    assert list(zip(cell["PATID_A"], cell["PATID_B"])) == [("p3", "p4")]

    everything = sd.route_errors(diag, "clustering")
    cm = sd.route_confusion(diag, "clustering").drop(columns="total")
    assert len(everything) == cm.to_numpy().sum() - np.trace(cm.to_numpy())


def test_error_listings_carry_no_duplicate_columns(diag):
    errors = sd.route_errors(diag, "clustering")
    assert not errors.columns.duplicated().any()


# ── attributes ───────────────────────────────────────────────────────────────
def test_attributes_are_attached_per_side(run, diag):
    manifest, settings = run
    cleaned = sd.load_cleaned(manifest, settings)
    out = sd.with_attributes(sd.binary_errors(diag, "blocking"), cleaned)
    assert out["LastNM_clean_A"].iloc[0] == "SMITH"
    assert out["LastNM_clean_B"].iloc[0] == "SMITH"


def test_missing_cleaned_columns_are_skipped_not_fatal(run, diag):
    manifest, settings = run
    cleaned = sd.load_cleaned(manifest, settings)
    out = sd.with_attributes(sd.binary_errors(diag, "blocking"), cleaned)
    assert "SSN_clean_A" not in out.columns      # the fixture has no SSN column
