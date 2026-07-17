"""GET /records, GET /clusters/{mid}, GET /records/{patid}/raw —
docs/API-Design.md §3 "Clusters / records" + the Dashboard FR doc's FR-22/24.

Reads go through `IndexBackend` (`entity` ⨝ `entity_member` ⨝ `record_attrs`
⨝ `review_candidate`) — see docs/Data-Contract.md Stage 6 — so these routes
work identically whether `settings.index_backend` is SQLite or the local
Parquet index.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api import jobs
from src.api.deps import get_backend, get_settings
from src.api.index_backend import IndexBackend
from src.api.schemas import (
    CandidatePatient,
    Entity,
    EntityMember,
    RawRecord,
    RecordScoreOutcome,
    RecordsPage,
    ReviewCandidate,
    ReviewQueueItem,
    ReviewQueuePage,
    ScoreCreateResponse,
    ScoreRequest,
    ScoreResult,
)
from src.config import Settings

router = APIRouter(tags=["records"])


def _to_entity(backend: IndexBackend, entity_row: dict, member_rows: list[dict]) -> Entity:
    review_candidates: dict[tuple, ReviewCandidate] = {}
    for m in member_rows:
        for rc in backend.review_candidates_for_patid(m["patid"]):
            key = (rc["patid_a"], rc["patid_b"])
            review_candidates[key] = ReviewCandidate(
                patid_a=rc["patid_a"], patid_b=rc["patid_b"],
                match_rule=rc["match_rule"], confidence=rc["confidence"],
                evidence=rc["evidence"], source_blocks=rc["source_blocks"],
                fs_match_probability=rc.get("fs_match_probability"),
                fs_classification_tier=rc.get("fs_classification_tier"),
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


def _to_review_queue_item(rc: dict) -> ReviewQueueItem:
    return ReviewQueueItem(
        patid_a=rc["patid_a"], patid_b=rc["patid_b"],
        mid_a=rc["mid_a"], mid_b=rc["mid_b"],
        member_count_a=rc["member_count_a"], member_count_b=rc["member_count_b"],
        match_rule=rc["match_rule"], confidence=rc["confidence"],
        evidence=rc["evidence"], source_blocks=rc["source_blocks"],
        fs_match_probability=rc.get("fs_match_probability"),
        fs_classification_tier=rc.get("fs_classification_tier"),
        reviewed=bool(rc["reviewed"]),
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


@router.get("/review-queue", response_model=ReviewQueuePage)
def list_review_queue(
    confidence_min: float | None = None,
    confidence_max: float | None = None,
    reviewed: bool | None = None,
    search: str | None = None,
    page: int = 1,
    page_size: int | None = None,
    backend: IndexBackend = Depends(get_backend),
    settings: Settings = Depends(get_settings),
) -> ReviewQueuePage:
    """Candidate-grain review queue — one row per pending pair, independent
    of which cluster it belongs to (docs/Dashboard-Guide.md's Review Queue
    tab). `reviewed` unset returns every candidate; pass `reviewed=false` for
    the default "Needs review" queue view, `reviewed=true` for "Already
    reviewed"."""
    page_size = page_size or settings.records_page_size
    rows, total = backend.list_review_candidates(
        confidence_min=confidence_min, confidence_max=confidence_max,
        reviewed=reviewed, search=search, page=page, page_size=page_size,
    )
    items = [_to_review_queue_item(row) for row in rows]
    return ReviewQueuePage(total=total, page=page, page_size=page_size, items=items)


@router.get("/records", response_model=RecordsPage)
def list_records(
    search: str | None = None,
    origin: str | None = None,
    is_merged: bool | None = None,
    birth_date: str | None = None,
    ssn_last4: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    confidence_min: float | None = None,
    confidence_max: float | None = None,
    page: int = 1,
    page_size: int | None = None,
    backend: IndexBackend = Depends(get_backend),
    settings: Settings = Depends(get_settings),
) -> RecordsPage:
    page_size = page_size or settings.records_page_size
    rows, total = backend.list_entities(
        search=search, origin=origin, is_merged=is_merged,
        birth_date=birth_date, ssn_last4=ssn_last4,
        updated_after=updated_after, updated_before=updated_before,
        confidence_min=confidence_min, confidence_max=confidence_max,
        page=page, page_size=page_size,
    )
    items = []
    for row in rows:
        detail = backend.get_entity(row["mid"])
        items.append(_to_entity(backend, detail["entity"], detail["members"]))
    return RecordsPage(total=total, page=page, page_size=page_size, items=items)


@router.get("/clusters/{mid}", response_model=Entity)
def get_cluster(mid: str, backend: IndexBackend = Depends(get_backend)) -> Entity:
    detail = backend.get_entity(mid)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Unknown mid: {mid}")
    return _to_entity(backend, detail["entity"], detail["members"])


@router.get("/records/{patid}/raw", response_model=RawRecord)
def get_raw_record(patid: str, backend: IndexBackend = Depends(get_backend)) -> RawRecord:
    raw_json = backend.get_record_raw(patid)
    if raw_json is None:
        raise HTTPException(
            status_code=404, detail=f"No raw data published for PATID: {patid}"
        )
    return RawRecord(patid=patid, fields=json.loads(raw_json))


@router.post("/records/score", response_model=ScoreCreateResponse, status_code=202)
def score_records(
    body: ScoreRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> ScoreCreateResponse:
    """Score one or a batch of new records against the existing population
    without re-running the full pipeline — see `docs/API-Design.md` and
    `src/api/incremental.py`. Always a background job (same 202 + poll
    pattern as `POST /runs`), regardless of batch size."""
    settings.ensure_dirs()
    run_id = jobs.new_run_id()
    jobs.mark_score_queued(run_id)
    background_tasks.add_task(
        jobs.score_records_job,
        run_id,
        [r.model_dump() for r in body.records],
        settings,
    )
    return ScoreCreateResponse(run_id=run_id, status="queued")


@router.get("/records/score/{run_id}", response_model=ScoreResult)
def get_score_result(run_id: str) -> ScoreResult:
    status = jobs.get_score_status(run_id)
    if status is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown score run_id: {run_id}"
        )
    outcomes = jobs.get_score_result(run_id) or []
    return ScoreResult(
        run_id=run_id,
        status=status["status"],
        outcomes=[RecordScoreOutcome(**o) for o in outcomes],
        error=status.get("error"),
    )


__all__ = ["router"]
