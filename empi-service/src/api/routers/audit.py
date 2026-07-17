"""POST /audit/merge, POST /audit/unmerge, GET /audit — docs/API-Design.md §3.

Each write is one `backend` transaction (SQLite or Parquet local mode — see
docs/Data-Contract.md Stage 6): mutate `entity`/`entity_member`, then insert
the `audit_log` row — both land or neither does (`backend.begin()`/
`commit()`/`rollback()` here, exactly like `src/api/incremental.py`).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_backend, get_reviewer_id
from src.api.index_backend import IndexBackend
from src.api.routers.records import _to_entity
from src.api.schemas import (
    AuditLogRow,
    DismissRequest,
    DismissResponse,
    MergeRequest,
    MergeResponse,
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


@router.post("/merge", response_model=MergeResponse)
def merge(
    body: MergeRequest,
    backend: IndexBackend = Depends(get_backend),
    reviewer_id: str = Depends(get_reviewer_id),
) -> MergeResponse:
    target = backend.get_entity(body.mid)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Unknown mid: {body.mid}")

    now = _now()
    prev_state = "Merged" if target["entity"]["is_merged"] else "Needs review"

    backend.begin()
    try:
        for patid in body.patids:
            backend.upsert_entity_member(
                patid, body.mid,
                is_primary=False, added_by=reviewer_id, updated_utc=now,
            )
        backend.upsert_entity(
            body.mid, target["entity"]["run_id"], "merge",
            is_merged=True, confidence=target["entity"]["confidence"],
            updated_utc=now,
            match_rule=target["entity"].get("match_rule"),
            evidence=f"Manually merged by {reviewer_id}",
        )
        audit_id = backend.insert_audit_log(
            ts_utc=now, user=reviewer_id, action="merge",
            patids=",".join(body.patids), mid=body.mid,
            prev_state=prev_state, next_state="Merged",
            run_id=target["entity"]["run_id"],
        )
        backend.commit()
    except Exception:
        backend.rollback()
        raise

    detail = backend.get_entity(body.mid)
    return MergeResponse(
        audit_id=audit_id,
        entity=_to_entity(backend, detail["entity"], detail["members"]),
    )


@router.post("/unmerge", response_model=UnmergeResponse)
def unmerge(
    body: UnmergeRequest,
    backend: IndexBackend = Depends(get_backend),
    reviewer_id: str = Depends(get_reviewer_id),
) -> UnmergeResponse:
    current_mid = backend.get_entity_mid_for_patid(body.patid)
    if current_mid != body.mid:
        raise HTTPException(
            status_code=404,
            detail=f"PATID {body.patid} is not a member of {body.mid}.",
        )

    now = _now()
    source = backend.get_entity(body.mid)
    new_mid = backend.next_mid()

    backend.begin()
    try:
        backend.upsert_entity(
            new_mid, source["entity"]["run_id"],
            _single_member_origin(backend, body.patid),
            is_merged=False, confidence=None, updated_utc=now,
        )
        backend.upsert_entity_member(
            body.patid, new_mid,
            is_primary=True, added_by=reviewer_id, updated_utc=now,
        )

        remaining_patids = [
            m["patid"] for m in backend.get_entity(body.mid)["members"]
            if m["patid"] != body.patid
        ]
        if len(remaining_patids) <= 1:
            leftover_origin = (
                _single_member_origin(backend, remaining_patids[0])
                if remaining_patids else source["entity"]["origin"]
            )
            backend.upsert_entity(
                body.mid, source["entity"]["run_id"], leftover_origin,
                is_merged=False, confidence=None, updated_utc=now,
            )

        audit_id = backend.insert_audit_log(
            ts_utc=now, user=reviewer_id, action="unmerge",
            patids=body.patid, mid=new_mid,
            prev_state="Merged", next_state="Unmerged",
            run_id=source["entity"]["run_id"],
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
    (`store.list_review_candidates`) excludes any pair with a prior `dismiss`
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
    rows = backend.list_audit_log(limit=limit, since=since)
    return [AuditLogRow(**row) for row in rows]


__all__ = ["router"]
