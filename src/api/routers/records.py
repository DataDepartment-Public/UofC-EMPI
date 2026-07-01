"""GET /records, GET /clusters/{mid} — docs/API-Design.md §3 "Clusters / records".

Read straight from `empi.db` (`entity` ⨝ `entity_member` ⨝ `record_attrs`) —
no Parquet I/O per request; `publish.py` denormalized the display fields at
publish time.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api import store
from src.api.deps import get_db, get_settings
from src.api.schemas import Entity, EntityMember, RecordsPage
from src.config import Settings

router = APIRouter(tags=["records"])


def _to_entity(entity_row: dict, member_rows: list[dict]) -> Entity:
    return Entity(
        mid=entity_row["mid"],
        run_id=entity_row["run_id"],
        origin=entity_row["origin"],
        is_merged=bool(entity_row["is_merged"]),
        confidence=entity_row["confidence"],
        updated_utc=entity_row["updated_utc"],
        members=[
            EntityMember(
                patid=m["patid"],
                is_primary=bool(m["is_primary"]),
                added_by=m["added_by"],
                updated_utc=m["updated_utc"],
                first_name=m.get("first_name"),
                last_name=m.get("last_name"),
                birth_date=m.get("birth_date"),
                ssn_last4=m.get("ssn_last4"),
                email=m.get("email"),
                zip_code=m.get("zip_code"),
                address1=m.get("address1"),
                sex=m.get("sex"),
            )
            for m in member_rows
        ],
    )


@router.get("/records", response_model=RecordsPage)
def list_records(
    search: str | None = None,
    status: str | None = None,
    page: int = 1,
    page_size: int | None = None,
    conn=Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RecordsPage:
    page_size = page_size or settings.records_page_size
    rows, total = store.list_entities(
        conn, search=search, status=status, page=page, page_size=page_size
    )
    items = []
    for row in rows:
        detail = store.get_entity(conn, row["mid"])
        items.append(_to_entity(detail["entity"], detail["members"]))
    return RecordsPage(total=total, page=page, page_size=page_size, items=items)


@router.get("/clusters/{mid}", response_model=Entity)
def get_cluster(mid: str, conn=Depends(get_db)) -> Entity:
    detail = store.get_entity(conn, mid)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Unknown mid: {mid}")
    return _to_entity(detail["entity"], detail["members"])


__all__ = ["router"]
