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
    UndoResponse,
    UnmergeRequest,
    UnmergeResponse,
)

router = APIRouter(prefix="/audit", tags=["audit"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _single_member_origin(conn, patid: str) -> str:
    """What a lone leftover record's origin should be — 'review' if it still
    has an unresolved review-queue candidate, else 'none'. A singleton can
    never be 'deterministic'/'merge' (those require >=2 members)."""
    return "review" if store.review_candidates_for_patid(conn, patid) else "none"


def _do_merge(
    conn, *, mid: str, patids: list[str], reviewer_id: str, undo_of: int | None = None,
) -> MergeResponse:
    """Shared by `POST /audit/merge` and the `unmerge`-undo path — merging a
    patid back into its original entity is exactly a regular merge."""
    target = store.get_entity(conn, mid)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Unknown mid: {mid}")

    now = _now()
    prev_state = "Merged" if target["entity"]["is_merged"] else "Needs review"

    conn.execute("BEGIN")
    try:
        for patid in patids:
            store.upsert_entity_member(
                conn, patid, mid,
                is_primary=False, added_by=reviewer_id, updated_utc=now,
            )
        store.upsert_entity(
            conn, mid, target["entity"]["run_id"], "merge",
            is_merged=True, confidence=target["entity"]["confidence"],
            updated_utc=now,
            match_rule=target["entity"].get("match_rule"),
            evidence=f"Manually merged by {reviewer_id}",
        )
        audit_id = store.insert_audit_log(
            conn,
            ts_utc=now, user=reviewer_id, action="merge",
            patids=",".join(patids), mid=mid,
            prev_state=prev_state, next_state="Merged",
            run_id=target["entity"]["run_id"],
            undo_of=undo_of,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    detail = store.get_entity(conn, mid)
    return MergeResponse(
        audit_id=audit_id, entity=_to_entity(conn, detail["entity"], detail["members"])
    )


def _do_unmerge(
    conn, *, mid: str, patid: str, reviewer_id: str, undo_of: int | None = None,
) -> UnmergeResponse:
    """Shared by `POST /audit/unmerge` and the `merge`-undo path — splitting a
    patid back out of the entity a merge just added it to is exactly a
    regular unmerge. Always records `prev_mid` (the entity split *from*) so a
    later `/audit/{id}/undo` on *this* action knows where to put it back."""
    current_mid = store.get_entity_mid_for_patid(conn, patid)
    if current_mid != mid:
        raise HTTPException(
            status_code=404,
            detail=f"PATID {patid} is not a member of {mid}.",
        )

    now = _now()
    source = store.get_entity(conn, mid)
    new_mid = store.next_mid(conn)

    conn.execute("BEGIN")
    try:
        store.upsert_entity(
            conn, new_mid, source["entity"]["run_id"],
            _single_member_origin(conn, patid),
            is_merged=False, confidence=None, updated_utc=now,
        )
        store.upsert_entity_member(
            conn, patid, new_mid,
            is_primary=True, added_by=reviewer_id, updated_utc=now,
        )

        remaining_patids = [
            m["patid"] for m in store.get_entity(conn, mid)["members"]
            if m["patid"] != patid
        ]
        if len(remaining_patids) <= 1:
            leftover_origin = (
                _single_member_origin(conn, remaining_patids[0])
                if remaining_patids else source["entity"]["origin"]
            )
            store.upsert_entity(
                conn, mid, source["entity"]["run_id"], leftover_origin,
                is_merged=False, confidence=None, updated_utc=now,
            )

        audit_id = store.insert_audit_log(
            conn,
            ts_utc=now, user=reviewer_id, action="unmerge",
            patids=patid, mid=new_mid,
            prev_state="Merged", next_state="Unmerged",
            run_id=source["entity"]["run_id"],
            prev_mid=mid, undo_of=undo_of,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    detail = store.get_entity(conn, new_mid)
    return UnmergeResponse(
        audit_id=audit_id, new_mid=new_mid,
        entity=_to_entity(conn, detail["entity"], detail["members"]),
    )


@router.post("/merge", response_model=MergeResponse)
def merge(
    body: MergeRequest,
    conn=Depends(get_db),
    reviewer_id: str = Depends(get_reviewer_id),
) -> MergeResponse:
    return _do_merge(conn, mid=body.mid, patids=body.patids, reviewer_id=reviewer_id)


@router.post("/unmerge", response_model=UnmergeResponse)
def unmerge(
    body: UnmergeRequest,
    conn=Depends(get_db),
    reviewer_id: str = Depends(get_reviewer_id),
) -> UnmergeResponse:
    return _do_unmerge(conn, mid=body.mid, patid=body.patid, reviewer_id=reviewer_id)


@router.post("/{audit_id}/undo", response_model=UndoResponse)
def undo(
    audit_id: int,
    conn=Depends(get_db),
    reviewer_id: str = Depends(get_reviewer_id),
) -> UndoResponse:
    row = store.get_audit_log_row(conn, audit_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown audit log entry: {audit_id}")

    already_undone = conn.execute(
        "SELECT 1 FROM audit_log WHERE undo_of = ?", (audit_id,)
    ).fetchone()
    if already_undone:
        raise HTTPException(status_code=409, detail="This action has already been undone.")

    if row["action"] == "merge":
        mid = row["mid"]
        patids = row["patids"].split(",")
        for patid in patids:
            if store.get_entity_mid_for_patid(conn, patid) != mid:
                raise HTTPException(
                    status_code=409,
                    detail=f"{patid} is no longer part of {mid} — can't undo.",
                )
        new_mids = []
        for patid in patids:
            resp = _do_unmerge(conn, mid=mid, patid=patid, reviewer_id=reviewer_id, undo_of=audit_id)
            new_mids.append(resp.new_mid)
        detail = store.get_entity(conn, mid)
        entity = (
            _to_entity(conn, detail["entity"], detail["members"])
            if detail else resp.entity
        )
        return UndoResponse(
            audit_id=resp.audit_id, reversed_action="merge",
            entity=entity, new_mids=new_mids,
        )

    if row["action"] == "unmerge":
        prev_mid = row["prev_mid"]
        patid = row["patids"]
        if prev_mid is None or store.get_entity(conn, prev_mid) is None:
            raise HTTPException(
                status_code=409,
                detail="The original entity no longer exists — can't undo.",
            )
        if store.get_entity_mid_for_patid(conn, patid) != row["mid"]:
            raise HTTPException(
                status_code=409,
                detail=f"{patid} has since moved — can't undo.",
            )
        resp = _do_merge(conn, mid=prev_mid, patids=[patid], reviewer_id=reviewer_id, undo_of=audit_id)
        return UndoResponse(
            audit_id=resp.audit_id, reversed_action="unmerge", entity=resp.entity,
        )

    raise HTTPException(
        status_code=400, detail=f"Undo isn't supported for action '{row['action']}'.",
    )


@router.get("", response_model=list[AuditLogRow])
def list_audit(
    limit: int = 100, since: str | None = None, conn=Depends(get_db)
) -> list[AuditLogRow]:
    rows = store.list_audit_log(conn, limit=limit, since=since)
    return [AuditLogRow(**row) for row in rows]


__all__ = ["router"]
