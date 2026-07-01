"""GET /records, GET /clusters/{mid}, GET /records/{patid}/raw —
docs/API-Design.md §3 "Clusters / records" + the Dashboard FR doc's FR-22/24.

Read straight from `empi.db` (`entity` ⨝ `entity_member` ⨝ `record_attrs` ⨝
`review_candidate`) — no Parquet I/O per request except the raw-data route,
which reads the one JSON blob `publish.py` denormalized per PATID.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from src.api import store
from src.api.deps import get_db, get_settings
from src.api.schemas import (
    CandidatePatient,
    Entity,
    EntityMember,
    RawRecord,
    RecordsPage,
    ReviewCandidate,
)
from src.config import Settings

router = APIRouter(tags=["records"])


def _to_entity(conn, entity_row: dict, member_rows: list[dict]) -> Entity:
    review_candidates: dict[tuple, ReviewCandidate] = {}
    for m in member_rows:
        for rc in store.review_candidates_for_patid(conn, m["patid"]):
            key = (rc["patid_a"], rc["patid_b"])
            review_candidates[key] = ReviewCandidate(
                patid_a=rc["patid_a"], patid_b=rc["patid_b"],
                match_rule=rc["match_rule"], confidence=rc["confidence"],
                evidence=rc["evidence"], source_blocks=rc["source_blocks"],
                patient_a=CandidatePatient(
                    patid=rc["patid_a"], first_name=rc["a_first_name"],
                    last_name=rc["a_last_name"], birth_date=rc["a_birth_date"],
                    ssn_last4=rc["a_ssn_last4"], email=rc["a_email"],
                    zip_code=rc["a_zip_code"], address1=rc["a_address1"],
                    sex=rc["a_sex"], phone=rc["a_phone"],
                ),
                patient_b=CandidatePatient(
                    patid=rc["patid_b"], first_name=rc["b_first_name"],
                    last_name=rc["b_last_name"], birth_date=rc["b_birth_date"],
                    ssn_last4=rc["b_ssn_last4"], email=rc["b_email"],
                    zip_code=rc["b_zip_code"], address1=rc["b_address1"],
                    sex=rc["b_sex"], phone=rc["b_phone"],
                ),
            )

    return Entity(
        mid=entity_row["mid"],
        run_id=entity_row["run_id"],
        origin=entity_row["origin"],
        is_merged=bool(entity_row["is_merged"]),
        confidence=entity_row["confidence"],
        match_rule=entity_row.get("match_rule"),
        evidence=entity_row.get("evidence"),
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
                phone=m.get("phone"),
            )
            for m in member_rows
        ],
        review_candidates=list(review_candidates.values()),
    )


@router.get("/records", response_model=RecordsPage)
def list_records(
    search: str | None = None,
    origin: str | None = None,
    is_merged: bool | None = None,
    birth_date: str | None = None,
    ssn_last4: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    page: int = 1,
    page_size: int | None = None,
    conn=Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RecordsPage:
    page_size = page_size or settings.records_page_size
    rows, total = store.list_entities(
        conn, search=search, origin=origin, is_merged=is_merged,
        birth_date=birth_date, ssn_last4=ssn_last4,
        updated_after=updated_after, updated_before=updated_before,
        page=page, page_size=page_size,
    )
    items = []
    for row in rows:
        detail = store.get_entity(conn, row["mid"])
        items.append(_to_entity(conn, detail["entity"], detail["members"]))
    return RecordsPage(total=total, page=page, page_size=page_size, items=items)


@router.get("/clusters/{mid}", response_model=Entity)
def get_cluster(mid: str, conn=Depends(get_db)) -> Entity:
    detail = store.get_entity(conn, mid)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Unknown mid: {mid}")
    return _to_entity(conn, detail["entity"], detail["members"])


@router.get("/records/{patid}/raw", response_model=RawRecord)
def get_raw_record(patid: str, conn=Depends(get_db)) -> RawRecord:
    raw_json = store.get_record_raw(conn, patid)
    if raw_json is None:
        raise HTTPException(
            status_code=404, detail=f"No raw data published for PATID: {patid}"
        )
    return RawRecord(patid=patid, fields=json.loads(raw_json))


__all__ = ["router"]
