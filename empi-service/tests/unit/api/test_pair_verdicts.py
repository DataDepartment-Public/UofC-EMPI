"""`src/api/pair_verdicts.py` — the verdict/bucket vocabulary the publish
step, both SQL backends and the Parquet backend all resolve pairs through.

The SQL backends inline `bucket_for` as a CASE expression rather than calling
it, so these tests are the readable definition the three implementations are
supposed to agree with. `tests/integration/test_api.py::TestReviewQueueBuckets`
is what actually checks that they do, end to end.
"""

from __future__ import annotations

import pytest

from src.api.pair_verdicts import (
    BUCKET_AUTO_MERGED,
    BUCKET_AUTO_REJECTED,
    BUCKET_NEEDS_REVIEW,
    BUCKET_REVIEWED,
    DECISION_MERGED,
    DECISION_NOT_A_MATCH,
    MERGE_VERDICTS,
    QUEUE_BUCKETS,
    REJECT_VERDICTS,
    VERDICT_AUTO_MERGE_RULE,
    VERDICT_GATE_DROPPED,
    VERDICT_ML_AUTO_MERGE,
    VERDICT_ML_HUMAN_REVIEW,
    VERDICT_REJECT,
    VERDICT_UNDECIDED,
    bucket_for,
    verdict_for_pair,
)


class TestVerdictForPair:
    def test_deterministic_stages_win_over_the_models(self):
        # A rule-confirmed pair was never model-scored, so this only pins the
        # ordering, not a real conflict.
        assert verdict_for_pair(
            in_matches=True, in_rejects=False, ml_tier=None, gate_tier=None
        ) == VERDICT_AUTO_MERGE_RULE
        assert verdict_for_pair(
            in_matches=False, in_rejects=True, ml_tier=None, gate_tier=None
        ) == VERDICT_REJECT

    @pytest.mark.parametrize(
        "ml_tier,expected",
        [
            ("auto_merge", VERDICT_ML_AUTO_MERGE),
            ("human_review", VERDICT_ML_HUMAN_REVIEW),
        ],
    )
    def test_ml_tier_maps_to_its_verdict(self, ml_tier, expected):
        assert verdict_for_pair(
            in_matches=False, in_rejects=False,
            ml_tier=ml_tier, gate_tier="human_review",
        ) == expected

    def test_ml_beats_the_gate_for_a_pair_the_gate_passed(self):
        """A pair the gate passed *and* the matcher scored has a row in both
        artifacts. The ML decision is the one that determined whether an edge
        formed, so reading the gate's tier off it would be wrong."""
        assert verdict_for_pair(
            in_matches=False, in_rejects=False,
            ml_tier="auto_merge", gate_tier="human_review",
        ) == VERDICT_ML_AUTO_MERGE

    def test_gate_drop_is_reached_only_where_the_pair_stopped(self):
        assert verdict_for_pair(
            in_matches=False, in_rejects=False,
            ml_tier=None, gate_tier="no_match",
        ) == VERDICT_GATE_DROPPED

    def test_a_pair_no_stage_decided_is_undecided(self):
        # An ungated run (no gate model, no FS fallback) leaves every
        # non-match here, and it is still open work.
        assert verdict_for_pair(
            in_matches=False, in_rejects=False, ml_tier=None, gate_tier=None
        ) == VERDICT_UNDECIDED
        # As does a pair the gate passed that Stage 4.5 never scored.
        assert verdict_for_pair(
            in_matches=False, in_rejects=False,
            ml_tier=None, gate_tier="human_review",
        ) == VERDICT_UNDECIDED


class TestBucketFor:
    @pytest.mark.parametrize("decision", [DECISION_MERGED, DECISION_NOT_A_MATCH])
    @pytest.mark.parametrize(
        "verdict",
        [VERDICT_AUTO_MERGE_RULE, VERDICT_GATE_DROPPED, VERDICT_UNDECIDED, None],
    )
    @pytest.mark.parametrize("same_entity", [True, False])
    def test_a_reviewer_decision_always_wins(self, decision, verdict, same_entity):
        assert bucket_for(
            verdict=verdict, reviewer_decision=decision, same_entity=same_entity
        ) == BUCKET_REVIEWED

    @pytest.mark.parametrize("verdict", sorted(MERGE_VERDICTS))
    def test_a_merge_verdict_the_index_did_not_apply_is_not_auto_merged(
        self, verdict
    ):
        """Publish never repoints a reviewer-locked PATID, and a reviewer can
        split a pipeline merge back apart. Either way the records are apart,
        so the section that claims they were merged would be describing an
        entity that doesn't exist — it falls back to open work."""
        assert bucket_for(
            verdict=verdict, reviewer_decision=None, same_entity=False
        ) == BUCKET_NEEDS_REVIEW
        assert bucket_for(
            verdict=verdict, reviewer_decision=None, same_entity=True
        ) == BUCKET_AUTO_MERGED

    @pytest.mark.parametrize("verdict", sorted(REJECT_VERDICTS))
    def test_reject_verdicts_are_auto_rejected(self, verdict):
        assert bucket_for(
            verdict=verdict, reviewer_decision=None, same_entity=False
        ) == BUCKET_AUTO_REJECTED

    def test_sharing_an_entity_counts_as_auto_merged(self):
        """Clustering is transitive: a pair the matcher left ambiguous can
        still be merged through a path of other pairs. Those records are
        together, so leaving the pair in "needs review" would be a lie."""
        assert bucket_for(
            verdict=VERDICT_ML_HUMAN_REVIEW,
            reviewer_decision=None,
            same_entity=True,
        ) == BUCKET_AUTO_MERGED

    def test_a_rejected_pair_that_merged_anyway_reads_as_merged(self):
        """Same reasoning, the other direction — the gate dropped this pair,
        but something else put its two records in one entity. What a reviewer
        needs to see is that they're merged."""
        assert bucket_for(
            verdict=VERDICT_GATE_DROPPED,
            reviewer_decision=None,
            same_entity=True,
        ) == BUCKET_AUTO_MERGED

    @pytest.mark.parametrize(
        "verdict", [VERDICT_ML_HUMAN_REVIEW, VERDICT_UNDECIDED, None]
    )
    def test_everything_else_needs_review(self, verdict):
        assert bucket_for(
            verdict=verdict, reviewer_decision=None, same_entity=False
        ) == BUCKET_NEEDS_REVIEW

    def test_null_verdict_is_open_work_not_an_error(self):
        """Incremental scoring runs no model stage, and publishes predating
        the column wrote no verdict at all. Both were review-queue rows."""
        assert bucket_for(
            verdict=None, reviewer_decision=None, same_entity=False
        ) == BUCKET_NEEDS_REVIEW


def test_every_bucket_is_reachable():
    """`QUEUE_BUCKETS` is what the router validates `?bucket=` against, so a
    name in it that `bucket_for` never returns would be a filter that always
    comes back empty."""
    produced = {
        bucket_for(verdict=v, reviewer_decision=d, same_entity=s)
        for v in (None, VERDICT_AUTO_MERGE_RULE, VERDICT_REJECT, VERDICT_UNDECIDED)
        for d in (None, DECISION_MERGED)
        for s in (True, False)
    }
    assert produced == set(QUEUE_BUCKETS)
