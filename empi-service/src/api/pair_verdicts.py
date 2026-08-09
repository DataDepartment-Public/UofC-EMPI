"""The pipeline's per-pair verdict vocabulary, and the four reviewer-facing
queue buckets derived from it.

A candidate pair leaves the pipeline with exactly one verdict — the stage
that decided it, resolved most-decisive-stage-first (`verdict_for_pair`).
`src/api/ingest/publish.py` stamps that verdict on every `review_candidate`
row; `routers/cluster_pairs.py` derives the same labels straight from the run
artifacts for its per-cluster trace. Both share this vocabulary so the Review
Queue's "Auto-merged" section and a cluster's pairwise trail can never
disagree about what the pipeline did with a pair.

WHY BUCKETS AREN'T JUST VERDICTS
--------------------------------
A verdict is what the *pipeline* decided. A bucket is what the *reviewer*
should see, which additionally depends on two things the verdict can't know:

  * whether a human has since ruled on the pair (`reviewer_pair_decision`) —
    that always wins, which is the whole point of the "Already reviewed"
    section, and
  * whether the pair's two records are in the same entity *right now*.

`BUCKET_NEEDS_REVIEW` is therefore the residual: nobody decided, and nothing
merged them.

WHY "AUTO-MERGED" ASKS THE INDEX, NOT THE VERDICT
-------------------------------------------------
`bucket_for` decides merged-ness from `same_entity` alone and never from
`verdict in MERGE_VERDICTS`, because the two can disagree in both directions
and the index is the one that's true:

  * A pair the matcher left at `human_review` can still be merged — clustering
    is transitive, so a path through other pairs unions them. Those records
    are together, and calling the pair undecided would be a lie.
  * A pair the pipeline merged can be apart: publish never repoints a
    reviewer-locked PATID (`backend.locked_patids`), and a reviewer can split
    one back out. Calling that pair "auto-merged" would describe an entity
    that doesn't exist.

The merge verdicts still matter — they're how `publish.py` decides a pair is
not open work — they just aren't what puts a pair in the Auto-merged section.
"""

from __future__ import annotations

from src.contracts import TIER_AUTO_MERGE, TIER_NO_MATCH

#: A deterministic auto-merge rule confirmed the pair (Stage 3, `matches`).
VERDICT_AUTO_MERGE_RULE = "auto_merge_rule"
#: >=`min_contradictions` strong-identifier conflicts (Stage 3, `rejects`).
VERDICT_REJECT = "reject"
#: The Stage-4.25 non-match gate scored it below `settings.gate_threshold`.
VERDICT_GATE_DROPPED = "gate_dropped"
#: The Stage-4.5 ML matcher tiered it `auto_merge` — a real merge edge.
VERDICT_ML_AUTO_MERGE = "ml_auto_merge"
#: The Stage-4.5 ML matcher tiered it `human_review` — genuinely ambiguous.
VERDICT_ML_HUMAN_REVIEW = "ml_human_review"
#: No rule fired and no model scored it — an ungated run (no gate model and
#: no FS fallback), or a pair that reached `non_matches` with nothing after.
VERDICT_UNDECIDED = "undecided"

#: Verdicts that formed a merge edge feeding `assign_clusters`.
MERGE_VERDICTS = frozenset({VERDICT_AUTO_MERGE_RULE, VERDICT_ML_AUTO_MERGE})
#: Verdicts where the pipeline decisively declined to merge. Note these are
#: the two stages that *discard*: the deterministic reject rules and the gate.
#: Stage 4.5 has no `no_match` tier by construction, so it never lands here.
REJECT_VERDICTS = frozenset({VERDICT_REJECT, VERDICT_GATE_DROPPED})

#: Pipeline left it for a human, and no human has ruled yet.
BUCKET_NEEDS_REVIEW = "needs_review"
#: A reviewer merged or dismissed this exact pair.
BUCKET_REVIEWED = "reviewed"
#: The pipeline merged the two records, with no reviewer involved.
BUCKET_AUTO_MERGED = "auto_merged"
#: The pipeline decided against the pair, with no reviewer involved.
BUCKET_AUTO_REJECTED = "auto_rejected"

QUEUE_BUCKETS: tuple[str, ...] = (
    BUCKET_NEEDS_REVIEW,
    BUCKET_REVIEWED,
    BUCKET_AUTO_MERGED,
    BUCKET_AUTO_REJECTED,
)

#: `reviewer_pair_decision.decision` values — what a human concluded about a
#: pair, independent of which bucket the pipeline would have put it in.
DECISION_MERGED = "merged"
DECISION_NOT_A_MATCH = "not_a_match"


def verdict_for_pair(
    *,
    in_matches: bool,
    in_rejects: bool,
    ml_tier: str | None,
    gate_tier: str | None,
) -> str:
    """The single verdict for one pair, resolved most-decisive stage first.

    Order is not arbitrary, and mirrors `routers/cluster_pairs.py::_verdict`
    (which additionally distinguishes pairs that never reached a decision at
    all — a per-cluster trace reads the full candidate pool, while this only
    ever sees pairs a stage actually produced).

    A pair a deterministic rule auto-merged was never scored by either model,
    so `auto_merge_rule` can't collide with the ML verdicts. But a pair the
    gate passed *and* the matcher scored has both a gate row and an ML row,
    and the ML decision is the one that determined whether an edge formed —
    reading `gate_dropped` off such a pair would be actively wrong. So ML is
    checked before the gate, and `gate_dropped` is reached only when the gate
    is where the pair stopped.
    """
    if in_matches:
        return VERDICT_AUTO_MERGE_RULE
    if in_rejects:
        return VERDICT_REJECT
    if ml_tier is not None:
        return (
            VERDICT_ML_AUTO_MERGE
            if ml_tier == TIER_AUTO_MERGE
            else VERDICT_ML_HUMAN_REVIEW
        )
    # The gate emits only `human_review` (passed) / `no_match` (dropped). A
    # pair it passed but that carries no ML row means Stage 4.5 never ran or
    # was skipped — still undecided, not auto-anything.
    if gate_tier == TIER_NO_MATCH:
        return VERDICT_GATE_DROPPED
    return VERDICT_UNDECIDED


def bucket_for(
    *, verdict: str | None, reviewer_decision: str | None, same_entity: bool
) -> str:
    """Which of the four Review Queue sections a pair belongs in.

    The SQL/pandas backends each inline this as a single expression so they
    can filter and count without materializing every row; this is the
    readable definition they must agree with, and what the unit tests pin.
    """
    if reviewer_decision is not None:
        return BUCKET_REVIEWED
    if same_entity:
        return BUCKET_AUTO_MERGED
    if verdict in REJECT_VERDICTS:
        return BUCKET_AUTO_REJECTED
    return BUCKET_NEEDS_REVIEW
