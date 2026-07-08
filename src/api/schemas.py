"""Request/response models for the API routes (docs/API-Design.md §3).

Responses reuse `src.contracts.RunManifest` where the doc says to; everything
else (entities, members, audit rows) is modeled here against the SQLite shape
in `src/api/store.py`.

Deviation from the doc's literal request body for `POST /audit/*`: `user` is
NOT accepted in the body. Identity comes from the trusted `X-Reviewer-Id`
header only (docs/Application-Architecture.md §3 "Identity / auth" — "the
browser never sets that header itself, so the audit trail can't be spoofed
from the client"; a body field the caller controls would defeat that).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RunStatus = Literal["queued", "running", "succeeded", "failed"]


class RunCreateResponse(BaseModel):
    run_id: str
    status: RunStatus


class RunSummary(BaseModel):
    run_id: str
    status: RunStatus
    counts: dict[str, int] = Field(default_factory=dict)
    created_utc: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyChecks(BaseModel):
    db: bool
    data_dirs: bool
    last_run_id: str | None


class ReadyResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    checks: ReadyChecks


class RecordAttrs(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    birth_date: str | None = None
    ssn_last4: str | None = None
    email: str | None = None
    zip_code: str | None = None
    address1: str | None = None
    sex: str | None = None
    phone: str | None = None


class EntityMember(RecordAttrs):
    patid: str
    is_primary: bool
    added_by: str
    updated_utc: str


class CandidatePatient(RecordAttrs):
    """A review-candidate's one side — same display fields as `EntityMember`,
    so the Model Explanation page can build a full feature-comparison table
    for review pairs too, not just confirmed members."""

    patid: str


class ReviewCandidate(BaseModel):
    """An unresolved candidate pair from the review queue — not a confirmed
    membership. See `src/api/store.py` `review_candidate` table."""

    patid_a: str
    patid_b: str
    match_rule: str | None
    confidence: float | None
    evidence: str | None
    source_blocks: str | None
    patient_a: CandidatePatient
    patient_b: CandidatePatient


class Entity(BaseModel):
    mid: str
    run_id: str
    origin: Literal["deterministic", "review", "merge", "none"]
    is_merged: bool
    confidence: float | None
    match_rule: str | None = None
    evidence: str | None = None
    updated_utc: str
    members: list[EntityMember] = Field(default_factory=list)
    review_candidates: list[ReviewCandidate] = Field(default_factory=list)


class RecordsPage(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[Entity]


class RawRecord(BaseModel):
    patid: str
    fields: dict[str, object]


class MergeRequest(BaseModel):
    mid: str
    patids: list[str] = Field(min_length=1)


class MergeResponse(BaseModel):
    audit_id: int
    entity: Entity


class UnmergeRequest(BaseModel):
    mid: str
    patid: str


class UnmergeResponse(BaseModel):
    audit_id: int
    new_mid: str
    entity: Entity


class MatchStatusCounts(BaseModel):
    """FR-11: the three-way Auto-match / Needs review / No match bar chart."""

    auto_match: int
    needs_review: int
    no_match: int


class DashboardSummary(BaseModel):
    """`GET /dashboard/summary` — KPI aggregates for the Dashboard tab
    (FR-4..FR-17). Live counts come from `empi.db`; `total_raw_rows`,
    `invalid_records`, `candidate_pairs`, and `auto_match_rate` come from the
    latest `RunManifest.counts` (empi.db only holds valid, published records)."""

    last_run_id: str | None
    last_run_created_utc: str | None
    model_version: str | None

    total_raw_rows: int
    total_records: int
    invalid_records: int
    duplicate_clusters: int
    matched_records: int
    matched_pct: float
    unmerged_pct: float
    needs_review_records: int
    manual_merge_actions: int
    manual_unmerge_actions: int
    manual_merge_pct: float
    manual_unmerge_pct: float
    auto_match_rate: float
    status_counts: MatchStatusCounts
    confidence_thresholds: dict[str, float]
    history: list[dict]


class AuditLogRow(BaseModel):
    id: int
    ts_utc: str
    user: str
    action: Literal["merge", "unmerge", "split"]
    patids: str
    mid: str
    prev_state: str
    next_state: str
    run_id: str | None
    prev_mid: str | None = None
    undo_of: int | None = None
    undone: bool = False


class UndoResponse(BaseModel):
    audit_id: int
    reversed_action: Literal["merge", "unmerge"]
    entity: Entity
    new_mids: list[str] = Field(default_factory=list)


__all__ = [
    "RunStatus",
    "RunCreateResponse",
    "RunSummary",
    "HealthResponse",
    "ReadyChecks",
    "ReadyResponse",
    "RecordAttrs",
    "EntityMember",
    "CandidatePatient",
    "ReviewCandidate",
    "Entity",
    "RecordsPage",
    "RawRecord",
    "MatchStatusCounts",
    "DashboardSummary",
    "MergeRequest",
    "MergeResponse",
    "UnmergeRequest",
    "UnmergeResponse",
    "AuditLogRow",
    "UndoResponse",
]
