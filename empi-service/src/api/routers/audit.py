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
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_backend, get_reviewer_id
from src.api.backends.index_backend import IndexBackend
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
            undo_of=undo_of,
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
            prev_mid=mid, undo_of=undo_of,
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
    if row["action"] not in ("merge", "unmerge"):
        raise HTTPException(
            status_code=400,
            detail=f"Undo isn't supported for action {row['action']!r}.",
        )
    if row["undone"]:
        raise HTTPException(
            status_code=400,
            detail=f"Audit entry #{audit_id} has already been undone.",
        )

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
    """Mark a review-queue candidate "Not a match" — the reviewer's
    considered rejection of a false-positive suggestion, distinct from
    simply leaving it unreviewed. Recorded as an audit_log entry only; no
    entity/member mutation, since the pair was never merged. The queue query
    (`sql_backend.list_review_candidates`) excludes any pair with a prior `dismiss`
    entry from the default "Needs review" view — it moves to "Already
    reviewed" instead of reappearing indefinitely."""
    mid = backend.get_entity_mid_for_patid(
        body.patid_a
    ) or backend.get_entity_mid_for_patid(body.patid_b)
    if mid is None:
        raise HTTPException(
            status_code=404,
            detail=f"Neither {body.patid_a} nor {body.patid_b} has a known mid.",
        )

    backend.begin()
    try:
        audit_id = backend.insert_audit_log(
            ts_utc=_now(), user=reviewer_id, action="dismiss",
            patids=f"{body.patid_a},{body.patid_b}", mid=mid,
            prev_state="Needs review", next_state="Dismissed",
            run_id=None,
        )
        backend.commit()
    except Exception:
        backend.rollback()
        raise
    return DismissResponse(audit_id=audit_id)


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
