"""POST /audit/merge, POST /audit/unmerge, POST /audit/{id}/undo, GET /audit —
docs/API-Design.md §3.

Each write is one `backend` transaction (SQLite or Parquet local mode — see
docs/Data-Contract.md Stage 6): mutate `entity`/`entity_member`, then insert
the `audit_log` row — both land or neither does (`backend.begin()`/
`commit()`/`rollback()` here, exactly like `src/api/ingest/incremental.py`).

`_do_merge`/`_do_unmerge` hold the actual mutation logic and are shared
between the public `/merge`/`/unmerge` routes (`undo_of=None`) and `/undo`
(`undo_of=<the audit entry being reversed>`) — undoing a merge is exactly
"unmerge every patid back out", and undoing an unmerge is exactly "merge the
patid back into where it came from". `unmerge`'s audit_log row records
`prev_mid` (the entity the patid is being removed from) specifically so a
later undo has somewhere to re-merge it into — `merge` has no equivalent
single prior mid per patid (each patid could have come from a different
entity), so undoing a merge always falls back out to N singleton entities
rather than reconstructing whatever they were before.

PAIR DECISIONS
--------------
Every action that rules on a pair also writes `reviewer_pair_decision`
(`_pair_decision_rows`), which is what moves the pair into the Review
Queue's "Already reviewed" section and out of whichever pipeline section it
sat in. `audit_log` alone can't serve that: it is entity-grain and
action-grain, so answering "has a human ruled on exactly (A, B)?" from it
means reconstructing past membership per query. Deriving it once at write
time is both cheaper and the only place the answer is unambiguous.

The two sections the pipeline fills are overridable from the queue, and both
overrides reuse these same routes rather than adding new ones: promoting an
auto-rejected pair is `POST /audit/merge`, and rejecting an auto-merged pair
is `POST /audit/dismiss`, which splits the entity first (see `dismiss`).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from itertools import combinations

from src.api.deps import get_backend, get_reviewer_id
from src.api.backends.index_backend import IndexBackend
from src.api.pair_verdicts import DECISION_MERGED, DECISION_NOT_A_MATCH
from src.api.routers.records import _to_entity
from src.api.schemas import (
    AuditLogRow,
    DismissRequest,
    DismissResponse,
    MergeRequest,
    MergeResponse,
    UndoResponse,
    UnmergeRequest,
    UnmergeResponse,
)

router = APIRouter(prefix="/audit", tags=["audit"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pair_decision_rows(
    patids: list[str], decision: str, audit_id: int, ts_utc: str
) -> list[tuple]:
    """`reviewer_pair_decision` rows for every distinct pair among `patids`,
    canonicalized `a < b` to match `review_candidate`'s own ordering.

    A merge is an assertion about the whole resulting set, not just about the
    records named in the request: merging C into {A, B} says C is the same
    patient as A *and* as B. Recording only the requested patids would leave
    (A, C) or (B, C) sitting in the queue as if still undecided."""
    return [
        (a, b, decision, audit_id, ts_utc)
        for a, b in combinations(sorted(set(patids)), 2)
    ]


def _single_member_origin(backend: IndexBackend, patid: str) -> str:
    """What a lone leftover record's origin should be — 'review' if it still
    has an unresolved review-queue candidate, else 'none'. A singleton can
    never be 'deterministic'/'merge' (those require >=2 members)."""
    return "review" if backend.review_candidates_for_patid(patid) else "none"


def _do_merge(
    backend: IndexBackend,
    *,
    mid: str,
    patids: list[str],
    reviewer_id: str,
    undo_of: int | None = None,
) -> MergeResponse:
    target = backend.get_entity(mid)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Unknown mid: {mid}")

    now = _now()
    prev_state = "Merged" if target["entity"]["is_merged"] else "Needs review"
    evidence = (
        f"Manually merged by {reviewer_id}"
        if undo_of is None
        else f"Restored by undoing audit entry #{undo_of}"
    )
    # Snapshot taken BEFORE the mutation loop below -- this entity's
    # membership at the moment of the merge, for audit_log.related_patids
    # (see sql_backend.py's audit_log DDL comment). Combined with body.patids
    # (who's being added), this lets a later export derive every
    # newly-confirmed pair without reconstructing past state from
    # entity_member, which only ever holds current membership.
    prior_members = [m["patid"] for m in target["members"]]

    backend.begin()
    try:
        for patid in patids:
            backend.upsert_entity_member(
                patid, mid,
                is_primary=False, added_by=reviewer_id, updated_utc=now,
            )
        backend.upsert_entity(
            mid, target["entity"]["run_id"], "merge",
            is_merged=True, confidence=target["entity"]["confidence"],
            updated_utc=now,
            match_rule=target["entity"].get("match_rule"),
            evidence=evidence,
        )
        audit_id = backend.insert_audit_log(
            ts_utc=now, user=reviewer_id, action="merge",
            patids=",".join(patids), mid=mid,
            prev_state=prev_state, next_state="Merged",
            run_id=target["entity"]["run_id"],
            related_patids=",".join(prior_members) if prior_members else None,
            undo_of=undo_of,
        )
        # Every pair across the entity's new membership is now reviewer-
        # confirmed — the same snapshot `related_patids` records, resolved to
        # pair grain so the Review Queue can act on it directly. Not on the
        # undo path: restoring an entity a reviewer had split retracts their
        # earlier "not a match", it doesn't assert the opposite, and `undo`
        # has already cleared the original entry's rows.
        if undo_of is None:
            backend.record_pair_decisions(
                _pair_decision_rows(
                    [*prior_members, *patids], DECISION_MERGED, audit_id, now
                )
            )
        backend.commit()
    except Exception:
        backend.rollback()
        raise

    detail = backend.get_entity(mid)
    return MergeResponse(
        audit_id=audit_id,
        entity=_to_entity(backend, detail["entity"], detail["members"]),
    )


def _do_unmerge(
    backend: IndexBackend,
    *,
    mid: str,
    patid: str,
    reviewer_id: str,
    undo_of: int | None = None,
) -> UnmergeResponse:
    current_mid = backend.get_entity_mid_for_patid(patid)
    if current_mid != mid:
        raise HTTPException(
            status_code=404,
            detail=f"PATID {patid} is not a member of {mid}.",
        )

    now = _now()
    source = backend.get_entity(mid)
    new_mid = backend.next_mid()

    backend.begin()
    try:
        backend.upsert_entity(
            new_mid, source["entity"]["run_id"],
            _single_member_origin(backend, patid),
            is_merged=False, confidence=None, updated_utc=now,
        )
        backend.upsert_entity_member(
            patid, new_mid,
            is_primary=True, added_by=reviewer_id, updated_utc=now,
        )

        remaining_patids = [
            m["patid"] for m in backend.get_entity(mid)["members"]
            if m["patid"] != patid
        ]
        if len(remaining_patids) <= 1:
            leftover_origin = (
                _single_member_origin(backend, remaining_patids[0])
                if remaining_patids else source["entity"]["origin"]
            )
            backend.upsert_entity(
                mid, source["entity"]["run_id"], leftover_origin,
                is_merged=False, confidence=None, updated_utc=now,
            )

        audit_id = backend.insert_audit_log(
            ts_utc=now, user=reviewer_id, action="unmerge",
            patids=patid, mid=new_mid,
            prev_state="Merged", next_state="Unmerged",
            run_id=source["entity"]["run_id"],
            # Who stayed behind in body.mid, captured now rather than
            # reconstructed later from entity_member's current-state-only
            # table -- see audit_log.related_patids' DDL comment.
            related_patids=",".join(remaining_patids) if remaining_patids else None,
            prev_mid=mid, undo_of=undo_of,
        )
        # Pulling a record out of an entity says it doesn't belong with the
        # members left behind — one `not_a_match` per resulting pair. Not on
        # the undo path, though: undoing a merge is a retraction, not the
        # opposite assertion, and `undo` clears the original's rows instead.
        if undo_of is None:
            backend.record_pair_decisions(
                [
                    (*sorted((patid, other)), DECISION_NOT_A_MATCH, audit_id, now)
                    for other in remaining_patids
                ]
            )
        backend.commit()
    except Exception:
        backend.rollback()
        raise

    detail = backend.get_entity(new_mid)
    return UnmergeResponse(
        audit_id=audit_id, new_mid=new_mid,
        entity=_to_entity(backend, detail["entity"], detail["members"]),
    )


@router.post("/merge", response_model=MergeResponse)
def merge(
    body: MergeRequest,
    backend: IndexBackend = Depends(get_backend),
    reviewer_id: str = Depends(get_reviewer_id),
) -> MergeResponse:
    return _do_merge(backend, mid=body.mid, patids=body.patids, reviewer_id=reviewer_id)


@router.post("/unmerge", response_model=UnmergeResponse)
def unmerge(
    body: UnmergeRequest,
    backend: IndexBackend = Depends(get_backend),
    reviewer_id: str = Depends(get_reviewer_id),
) -> UnmergeResponse:
    return _do_unmerge(backend, mid=body.mid, patid=body.patid, reviewer_id=reviewer_id)


@router.post("/{audit_id}/undo", response_model=UndoResponse)
def undo(
    audit_id: int,
    backend: IndexBackend = Depends(get_backend),
    reviewer_id: str = Depends(get_reviewer_id),
) -> UndoResponse:
    row = backend.get_audit_log_row(audit_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown audit_id: {audit_id}")
    if row["action"] not in ("merge", "unmerge", "dismiss"):
        raise HTTPException(
            status_code=400,
            detail=f"Undo isn't supported for action {row['action']!r}.",
        )
    if row["undone"]:
        raise HTTPException(
            status_code=400,
            detail=f"Audit entry #{audit_id} has already been undone.",
        )

    # Retract the pair decisions this entry asserted before reversing its
    # entity mutation. Scoped to `audit_id`, so a pair some *later* action
    # has since ruled on keeps that newer decision (see
    # `sql_backend.clear_pair_decisions_for_audit`).
    backend.begin()
    try:
        backend.clear_pair_decisions_for_audit(audit_id)
        backend.commit()
    except Exception:
        backend.rollback()
        raise

    if row["action"] == "dismiss":
        # A dismiss mutated no entity, so the retraction above is the entire
        # reversal — the pair falls back to whichever section its pipeline
        # verdict puts it in. Note this does *not* re-merge a pair that
        # `dismiss` split: that split is its own `unmerge` entry, undoable on
        # its own terms, and silently reversing it from here would undo two
        # audit entries under one id.
        now = _now()
        backend.begin()
        try:
            backend.insert_audit_log(
                ts_utc=now, user=reviewer_id, action="dismiss",
                patids=row["patids"], mid=row["mid"],
                prev_state="Dismissed", next_state="Needs review",
                run_id=None, undo_of=audit_id,
            )
            backend.commit()
        except Exception:
            backend.rollback()
            raise
        return UndoResponse(audit_id=audit_id, reversed_action="dismiss")

    if row["action"] == "merge":
        # Each patid could have come from a different prior entity, so there
        # is no single mid to send them all back to — reverse a merge by
        # unmerging every patid back out into its own singleton, one
        # `_do_unmerge` transaction per patid (see module docstring).
        patids = [p for p in row["patids"].split(",") if p]
        new_mids: list[str] = []
        for patid in patids:
            result = _do_unmerge(
                backend, mid=row["mid"], patid=patid,
                reviewer_id=reviewer_id, undo_of=audit_id,
            )
            new_mids.append(result.new_mid)
        return UndoResponse(
            audit_id=audit_id, reversed_action="merge", new_mids=new_mids
        )

    # row["action"] == "unmerge"
    if not row["prev_mid"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Audit entry #{audit_id} predates undo support and has no "
                "prior mid recorded."
            ),
        )
    result = _do_merge(
        backend, mid=row["prev_mid"], patids=[row["patids"]],
        reviewer_id=reviewer_id, undo_of=audit_id,
    )
    return UndoResponse(
        audit_id=audit_id, reversed_action="unmerge",
        entity=result.entity, new_mids=[result.entity.mid],
    )


@router.post("/dismiss", response_model=DismissResponse)
def dismiss(
    body: DismissRequest,
    backend: IndexBackend = Depends(get_backend),
    reviewer_id: str = Depends(get_reviewer_id),
) -> DismissResponse:
    """Mark a candidate pair "Not a match" — the reviewer's considered
    rejection, distinct from simply leaving it unreviewed. Writes a
    `reviewer_pair_decision`, which moves the pair into the queue's "Already
    reviewed" section instead of leaving it to reappear indefinitely.

    **If the two records are currently in the same entity, this splits them
    first.** That is the override for the Auto-merged section: a reviewer
    told the pair isn't a match, on a pair the pipeline merged, has to see
    the records come apart or the decision was cosmetic. `patid_b` is the
    side that moves out, into a fresh singleton, via the ordinary
    `_do_unmerge` path — so it lands in the audit log as a real unmerge, is
    undoable like any other, and leaves `patid_a` where it was.

    Splitting is a separate transaction from the dismiss below it, and the
    two produce separate audit entries. That is deliberate: they are two
    facts (the entity changed; a reviewer ruled on a pair), and collapsing
    them would make the unmerge invisible to `/undo`, which works entry by
    entry."""
    mid_a = backend.get_entity_mid_for_patid(body.patid_a)
    mid_b = backend.get_entity_mid_for_patid(body.patid_b)
    mid = mid_a or mid_b
    if mid is None:
        raise HTTPException(
            status_code=404,
            detail=f"Neither {body.patid_a} nor {body.patid_b} has a known mid.",
        )

    unmerged_to_mid: str | None = None
    if mid_a is not None and mid_a == mid_b:
        unmerged_to_mid = _do_unmerge(
            backend, mid=mid_a, patid=body.patid_b, reviewer_id=reviewer_id
        ).new_mid

    a, b = sorted((body.patid_a, body.patid_b))
    now = _now()
    backend.begin()
    try:
        audit_id = backend.insert_audit_log(
            ts_utc=now, user=reviewer_id, action="dismiss",
            patids=f"{body.patid_a},{body.patid_b}", mid=mid,
            prev_state="Merged" if unmerged_to_mid else "Needs review",
            next_state="Dismissed",
            run_id=None,
        )
        backend.record_pair_decisions(
            [(a, b, DECISION_NOT_A_MATCH, audit_id, now)]
        )
        backend.commit()
    except Exception:
        backend.rollback()
        raise
    return DismissResponse(audit_id=audit_id, unmerged_to_mid=unmerged_to_mid)


@router.get("", response_model=list[AuditLogRow])
def list_audit(
    limit: int = 100,
    since: str | None = None,
    backend: IndexBackend = Depends(get_backend),
) -> list[AuditLogRow]:
    """The reviewer-facing audit trail (Dataset tab's Merge Audit Log). Excludes
    `view_raw`/`view_ssn_clean` entries — those are PHI-access records, not
    reviewer decisions, and have no natural place in a table built around
    undoable state transitions; they remain in `audit_log` and are queryable
    directly."""
    rows = backend.list_audit_log(limit=limit, since=since)
    return [
        AuditLogRow(**row)
        for row in rows
        if row["action"] not in ("view_raw", "view_ssn_clean")
    ]


__all__ = ["router"]
