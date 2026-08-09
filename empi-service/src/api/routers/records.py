"""GET /records, GET /clusters/{mid}, GET /records/{patid}/raw —
docs/API-Design.md §3 "Clusters / records" + the Dashboard FR doc's FR-22/24.

Reads go through `IndexBackend` (`entity` ⨝ `entity_member` ⨝ `record_attrs`
⨝ `review_candidate`) — see docs/Data-Contract.md Stage 6 — so these routes
work identically whether `settings.index_backend` is SQLite or the local
Parquet index.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api import jobs
from src.api.deps import get_backend, get_reviewer_id_optional, get_settings
from src.api.backends.index_backend import IndexBackend
from src.api.schemas import (
    CandidatePatient,
    CleanSsn,
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


def _phones(raw: object) -> list[str]:
    """`record_attrs.phones` (a JSON array, stored as TEXT) -> `list[str]`.
    NULL for a row written before the column existed, or by a backend that
    round-trips a missing Parquet value as NaN — both mean "not known", which
    is the empty list, not an error."""
    if not isinstance(raw, str) or not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(p) for p in parsed] if isinstance(parsed, list) else []


def _candidate_patient(rc: dict, side: str) -> CandidatePatient:
    """One side of a review-candidate row, whose `record_attrs` columns the
    backends alias `a_*`/`b_*` (`sql_backend._PAIR_ATTR_SELECT`)."""
    return CandidatePatient(
        patid=rc[f"patid_{side}"],
        first_name=rc[f"{side}_first_name"],
        middle_name=rc.get(f"{side}_middle_name"),
        last_name=rc[f"{side}_last_name"],
        suffix=rc.get(f"{side}_suffix"),
        birth_date=rc[f"{side}_birth_date"],
        ssn_last4=rc[f"{side}_ssn_last4"],
        email=rc[f"{side}_email"],
        zip_code=rc[f"{side}_zip_code"],
        city=rc.get(f"{side}_city"),
        address1=rc[f"{side}_address1"],
        sex=rc[f"{side}_sex"],
        phone=rc[f"{side}_phone"],
        phones=_phones(rc.get(f"{side}_phones")),
    )


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
                ml_match_probability=rc.get("ml_match_probability"),
                ml_classification_tier=rc.get("ml_classification_tier"),
                patient_a=_candidate_patient(rc, "a"),
                patient_b=_candidate_patient(rc, "b"),
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
                middle_name=m.get("middle_name"),
                last_name=m.get("last_name"),
                suffix=m.get("suffix"),
                birth_date=m.get("birth_date"),
                ssn_last4=m.get("ssn_last4"),
                email=m.get("email"),
                zip_code=m.get("zip_code"),
                city=m.get("city"),
                address1=m.get("address1"),
                sex=m.get("sex"),
                phone=m.get("phone"),
                phones=_phones(m.get("phones")),
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
        ml_match_probability=rc.get("ml_match_probability"),
        ml_classification_tier=rc.get("ml_classification_tier"),
        reviewed=bool(rc["reviewed"]),
        patient_a=_candidate_patient(rc, "a"),
        patient_b=_candidate_patient(rc, "b"),
    )


@router.get("/review-queue", response_model=ReviewQueuePage)
def list_review_queue(
    confidence_min: float | None = None,
    confidence_max: float | None = None,
    reviewed: bool | None = None,
    search: str | None = None,
    patid_a: str | None = None,
    patid_b: str | None = None,
    page: int = 1,
    page_size: int | None = None,
    backend: IndexBackend = Depends(get_backend),
    settings: Settings = Depends(get_settings),
) -> ReviewQueuePage:
    """Candidate-grain review queue — one row per pending pair, independent
    of which cluster it belongs to (docs/Dashboard-Guide.md's Review Queue
    tab). `reviewed` unset returns every candidate; pass `reviewed=false` for
    the default "Needs review" queue view, `reviewed=true` for "Already
    reviewed".

    `patid_a`/`patid_b` (either order — canonicalized here) narrow to one
    exact pair: the deep link a cluster's comparison history sends a
    reviewer on knows the pair, not what confidence/search/page would
    currently surface it. They combine with any other filter passed
    alongside them by AND, same as every filter here — a caller resolving a
    specific pair should pass no other filter."""
    page_size = page_size or settings.records_page_size
    if patid_a is not None and patid_b is not None:
        patid_a, patid_b = sorted((patid_a, patid_b))
    rows, total = backend.list_review_candidates(
        confidence_min=confidence_min, confidence_max=confidence_max,
        reviewed=reviewed, search=search, patid_a=patid_a, patid_b=patid_b,
        page=page, page_size=page_size,
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
    min_members: int | None = None,
    sort: str | None = None,
    page: int = 1,
    page_size: int | None = None,
    backend: IndexBackend = Depends(get_backend),
    settings: Settings = Depends(get_settings),
) -> RecordsPage:
    """`sort` — `'confidence'`, `'name'`, or unset/anything else for the
    default most-recently-updated-first (see `IndexBackend.list_entities`).

    `min_members=2` is the Patient Registry's "clusters with something to
    compare" filter. Use it rather than `is_merged=true`, which is derived
    from the *unlocked* member count at publish time and so misses any
    multi-record cluster a reviewer has already touched.
    """
    page_size = page_size or settings.records_page_size
    rows, total = backend.list_entities(
        search=search, origin=origin, is_merged=is_merged,
        birth_date=birth_date, ssn_last4=ssn_last4,
        updated_after=updated_after, updated_before=updated_before,
        confidence_min=confidence_min, confidence_max=confidence_max,
        min_members=min_members, sort=sort, page=page, page_size=page_size,
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
def get_raw_record(
    patid: str,
    backend: IndexBackend = Depends(get_backend),
    reviewer_id: str = Depends(get_reviewer_id_optional),
) -> RawRecord:
    """Unmasked source record — includes the full SSN (`SSN_raw`), backing
    the "View raw data" drawer. The SSN-reveal toggle uses the separate
    `GET /records/{patid}/ssn-clean` below instead (the pipeline-normalized
    value, not this raw one) — see that route's docstring for why. Every
    successful call here is written to `audit_log` (action `view_raw`) so
    there's a record of who accessed a patient's unmasked PHI and when,
    matching the accountability every other reviewer action already gets —
    a 404 (nothing to view) is not logged."""
    raw_json = backend.get_record_raw(patid)
    if raw_json is None:
        raise HTTPException(
            status_code=404, detail=f"No raw data published for PATID: {patid}"
        )

    backend.begin()
    try:
        backend.insert_audit_log(
            ts_utc=datetime.now(timezone.utc).isoformat(),
            user=reviewer_id,
            action="view_raw",
            patids=patid,
            mid=backend.get_entity_mid_for_patid(patid) or patid,
            prev_state="N/A",
            next_state="Viewed",
            run_id=None,
        )
        backend.commit()
    except Exception:
        backend.rollback()
        raise

    return RawRecord(patid=patid, fields=json.loads(raw_json))


@router.get("/records/{patid}/ssn-clean", response_model=CleanSsn)
def get_clean_ssn(
    patid: str,
    backend: IndexBackend = Depends(get_backend),
    reviewer_id: str = Depends(get_reviewer_id_optional),
) -> CleanSsn:
    """The pipeline-normalized full SSN (`cleaned_attrs.ssn`) — the value
    blocking (B1) and the deterministic rules actually matched on, unlike
    `SSN_raw` from `GET /records/{patid}/raw`, which is the un-scrubbed
    source value and may contain formatting noise or fail `clean_ssn`'s
    validation entirely. Reveal-toggle UI should prefer this over the raw
    endpoint so a reviewer never sanity-checks a merge against a value the
    matching engine didn't trust. Audit-logged the same way as `view_raw`
    since this also exposes unmasked PHI; a 404 (nothing published) is not
    logged."""
    rows = backend.get_cleaned_attrs([patid])
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"No cleaned attributes published for PATID: {patid}"
        )

    backend.begin()
    try:
        backend.insert_audit_log(
            ts_utc=datetime.now(timezone.utc).isoformat(),
            user=reviewer_id,
            action="view_ssn_clean",
            patids=patid,
            mid=backend.get_entity_mid_for_patid(patid) or patid,
            prev_state="N/A",
            next_state="Viewed",
            run_id=None,
        )
        backend.commit()
    except Exception:
        backend.rollback()
        raise

    return CleanSsn(patid=patid, ssn=rows[0].get("ssn"))


@router.post("/records/score", response_model=ScoreCreateResponse, status_code=202)
def score_records(
    body: ScoreRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> ScoreCreateResponse:
    """Score one or a batch of new records against the existing population
    without re-running the full pipeline — see `docs/API-Design.md` and
    `src/api/ingest/incremental.py`. Always a background job (same 202 + poll
    pattern as `POST /runs`), regardless of batch size."""
    settings.ensure_dirs()
    run_id = jobs.new_run_id()
    jobs.mark_score_queued(run_id, settings)
    background_tasks.add_task(
        jobs.score_records_job,
        run_id,
        [r.model_dump() for r in body.records],
        settings,
    )
    return ScoreCreateResponse(run_id=run_id, status="queued")


@router.get("/records/score/{run_id}", response_model=ScoreResult)
def get_score_result(run_id: str, settings: Settings = Depends(get_settings)) -> ScoreResult:
    status = jobs.get_score_status(run_id) or jobs.read_score_status_file(run_id, settings)
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
