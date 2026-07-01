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


class EntityMember(RecordAttrs):
    patid: str
    is_primary: bool
    added_by: str
    updated_utc: str


class Entity(BaseModel):
    mid: str
    run_id: str
    origin: Literal["deterministic", "review", "merge", "none"]
    is_merged: bool
    confidence: float | None
    updated_utc: str
    members: list[EntityMember] = Field(default_factory=list)


class RecordsPage(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[Entity]


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


__all__ = [
    "RunStatus",
    "RunCreateResponse",
    "RunSummary",
    "HealthResponse",
    "ReadyChecks",
    "ReadyResponse",
    "RecordAttrs",
    "EntityMember",
    "Entity",
    "RecordsPage",
    "MergeRequest",
    "MergeResponse",
    "UnmergeRequest",
    "UnmergeResponse",
    "AuditLogRow",
]
