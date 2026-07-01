"""POST /audit/merge, POST /audit/unmerge, GET /audit — docs/API-Design.md §3.

Each write is one SQLite transaction: mutate `entity`/`entity_member`, then
insert the `audit_log` row — both land or neither does (`store.py` functions
run inside a single `BEGIN`/`COMMIT` here).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from src.api import store
from src.api.deps import get_db, get_reviewer_id
from src.api.routers.records import _to_entity
from src.api.schemas import (
    AuditLogRow,
    MergeRequest,
    MergeResponse,
    UnmergeRequest,
    UnmergeResponse,
)

router = APIRouter(prefix="/audit", tags=["audit"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/merge", response_model=MergeResponse)
def merge(
    body: MergeRequest,
    conn=Depends(get_db),
    reviewer_id: str = Depends(get_reviewer_id),
) -> MergeResponse:
    target = store.get_entity(conn, body.mid)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Unknown mid: {body.mid}")

    now = _now()
    prev_state = "Merged" if target["entity"]["is_merged"] else "Needs review"

    conn.execute("BEGIN")
    try:
        for patid in body.patids:
            store.upsert_entity_member(
                conn, patid, body.mid,
                is_primary=False, added_by=reviewer_id, updated_utc=now,
            )
        store.upsert_entity(
            conn, body.mid, target["entity"]["run_id"], "merge",
            is_merged=True, confidence=target["entity"]["confidence"],
            updated_utc=now,
        )
        audit_id = store.insert_audit_log(
            conn,
            ts_utc=now, user=reviewer_id, action="merge",
            patids=",".join(body.patids), mid=body.mid,
            prev_state=prev_state, next_state="Merged",
            run_id=target["entity"]["run_id"],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    detail = store.get_entity(conn, body.mid)
    return MergeResponse(
        audit_id=audit_id, entity=_to_entity(detail["entity"], detail["members"])
    )


@router.post("/unmerge", response_model=UnmergeResponse)
def unmerge(
    body: UnmergeRequest,
    conn=Depends(get_db),
    reviewer_id: str = Depends(get_reviewer_id),
) -> UnmergeResponse:
    current_mid = store.get_entity_mid_for_patid(conn, body.patid)
    if current_mid != body.mid:
        raise HTTPException(
            status_code=404,
            detail=f"PATID {body.patid} is not a member of {body.mid}.",
        )

    now = _now()
    source = store.get_entity(conn, body.mid)
    new_mid = store.next_mid(conn)

    conn.execute("BEGIN")
    try:
        store.upsert_entity(
            conn, new_mid, source["entity"]["run_id"], "none",
            is_merged=False, confidence=None, updated_utc=now,
        )
        store.upsert_entity_member(
            conn, body.patid, new_mid,
            is_primary=True, added_by=reviewer_id, updated_utc=now,
        )

        remaining = store.member_count(conn, body.mid)
        if remaining <= 1:
            store.upsert_entity(
                conn, body.mid, source["entity"]["run_id"], source["entity"]["origin"],
                is_merged=False, confidence=source["entity"]["confidence"],
                updated_utc=now,
            )

        audit_id = store.insert_audit_log(
            conn,
            ts_utc=now, user=reviewer_id, action="unmerge",
            patids=body.patid, mid=new_mid,
            prev_state="Merged", next_state="Unmerged",
            run_id=source["entity"]["run_id"],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    detail = store.get_entity(conn, new_mid)
    return UnmergeResponse(
        audit_id=audit_id, new_mid=new_mid,
        entity=_to_entity(detail["entity"], detail["members"]),
    )


@router.get("", response_model=list[AuditLogRow])
def list_audit(
    limit: int = 100, since: str | None = None, conn=Depends(get_db)
) -> list[AuditLogRow]:
    rows = store.list_audit_log(conn, limit=limit, since=since)
    return [AuditLogRow(**row) for row in rows]


__all__ = ["router"]
