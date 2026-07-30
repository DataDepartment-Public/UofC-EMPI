"""Request/response models for the API routes (docs/API-Design.md §3).

Responses reuse `src.contracts.RunManifest` where the doc says to; everything
else (entities, members, audit rows) is modeled here against the SQLite shape
in `src/api/backends/sql_backend.py`.

Deviation from the doc's literal request body for `POST /audit/*`: `user` is
NOT accepted in the body. Identity comes from the trusted `X-Reviewer-Id`
header only (docs/Application-Architecture.md §3 "Identity / auth" — "the
browser never sets that header itself, so the audit trail can't be spoofed
from the client"; a body field the caller controls would defeat that).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    membership. See `src/api/backends/sql_backend.py` `review_candidate` table."""

    patid_a: str
    patid_b: str
    match_rule: str | None
    confidence: float | None
    evidence: str | None
    source_blocks: str | None
    patient_a: CandidatePatient
    patient_b: CandidatePatient
    #: Audit-only FS matcher signal (docs/FS-Matcher-Production-Guide.md) — feeds
    #: a future GBT, not a scored decision on this pair. Populated only for
    #: candidates scored via the incremental path (src/api/ingest/incremental.py); null
    #: for candidates from a full batch publish, which doesn't run FS yet.
    fs_match_probability: float | None = None
    fs_classification_tier: str | None = None


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


class ReviewQueueItem(BaseModel):
    """One candidate-grain row of `GET /review-queue` — a pending pair, not a
    cluster. See `src/api/backends/sql_backend.py` `list_review_candidates`."""

    patid_a: str
    patid_b: str
    mid_a: str
    mid_b: str
    member_count_a: int
    member_count_b: int
    match_rule: str | None
    confidence: float | None
    evidence: str | None
    source_blocks: str | None
    fs_match_probability: float | None = None
    fs_classification_tier: str | None = None
    reviewed: bool
    patient_a: CandidatePatient
    patient_b: CandidatePatient


class ReviewQueuePage(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[ReviewQueueItem]


class DismissRequest(BaseModel):
    patid_a: str
    patid_b: str


class DismissResponse(BaseModel):
    audit_id: int


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


class IncomingRecord(BaseModel):
    """Raw input for one record in `POST /records/score` — the pipeline's raw
    CSV column names (`src.preprocessing.transformations.RENAME_TO_RAW` +
    `PATID`), not the cleaned `*_clean` names. `extra="allow"` tolerates
    passthrough columns (e.g. `CountryNM`) the cleaner doesn't require."""

    model_config = ConfigDict(extra="allow")

    PATID: str
    FirstNM: str | None = None
    LastNM: str | None = None
    MiddleNM: str | None = None
    SuffixNM: str | None = None
    BirthDT: str | None = None
    SSN: str | None = None
    AddressLine1: str | None = None
    AddressLine2: str | None = None
    CityNM: str | None = None
    ZipCD: str | None = None
    StateCD: str | None = None
    PrimaryPhoneNBR: str | None = None
    Phone01NBR: str | None = None
    Phone02NBR: str | None = None
    Phone03NBR: str | None = None
    Email: str | None = None
    SexAtBirthDSC: str | None = None


class ScoreRequest(BaseModel):
    """`POST /records/score` body — one record or a batch."""

    records: list[IncomingRecord] = Field(min_length=1)


class ScoreCreateResponse(BaseModel):
    run_id: str
    status: RunStatus


#: "invalid" (failed cleaning validity checks) is not a pair-classification
#: decision, so it's not in `contracts.CLASSIFICATION_TIERS` — added here only.
ScoreTier = Literal["auto_merge", "human_review", "no_match", "invalid"]


class RecordScoreOutcome(BaseModel):
    """One submitted record's result — see `src/api/ingest/incremental.py`."""

    patid: str
    tier: ScoreTier
    mid: str | None = None
    match_rule: str | None = None
    confidence: float | None = None
    matched_patids: list[str] = Field(default_factory=list)
    fs_match_probability: float | None = None
    fs_classification_tier: str | None = None


class ScoreResult(BaseModel):
    run_id: str
    status: RunStatus
    outcomes: list[RecordScoreOutcome] = Field(default_factory=list)
    error: str | None = None


class AuditLogRow(BaseModel):
    id: int
    ts_utc: str
    user: str
    action: Literal["merge", "unmerge", "split", "dismiss"]
    patids: str
    mid: str
    prev_state: str
    next_state: str
    run_id: str | None


# ── Explanations (GET /explanations/...) ─────────────────────────────────────
class ExplanationFeature(BaseModel):
    """One bar of the waterfall.

    `start`/`end` are precomputed cumulative positions, so the UI draws a
    rectangle per feature and does no arithmetic — it needs no notion of a
    base value, log-odds, or SHAP at all.
    """

    name: str
    label: str
    value: float | str | None = None
    display_value: str | None = None
    shap: float
    start: float
    end: float
    direction: Literal["positive", "negative"]
    cumulative_prob: float


class ExplanationDecision(BaseModel):
    score: float
    tier: str
    threshold: float | None = None


class ExplanationAxis(BaseModel):
    min: float
    max: float


class PairExplanation(BaseModel):
    """A plot-ready waterfall for one pair under one model.

    Contributions are exact TreeSHAP in **log-odds** (`units`), summing to
    `final_margin` from `base_value`. Signs are normalized so positive always
    pushes toward the model's positive decision — plausible for the gate,
    confident-match for the ML matcher. `features` is ordered by descending
    |contribution|; `top_n` is a suggestion for how many to show.
    """

    model_config = ConfigDict(protected_namespaces=())

    model: str
    run_id: str | None = None
    model_file: str | None = None
    patid_a: str
    patid_b: str
    decision: ExplanationDecision
    base_value: float
    final_margin: float
    units: Literal["log_odds"] = "log_odds"
    top_n: int
    axis: ExplanationAxis
    features: list[ExplanationFeature]


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
    "ReviewQueueItem",
    "ReviewQueuePage",
    "DismissRequest",
    "DismissResponse",
    "RawRecord",
    "MatchStatusCounts",
    "DashboardSummary",
    "MergeRequest",
    "MergeResponse",
    "UnmergeRequest",
    "UnmergeResponse",
    "AuditLogRow",
    "IncomingRecord",
    "ScoreRequest",
    "ScoreCreateResponse",
    "ScoreTier",
    "RecordScoreOutcome",
    "ScoreResult",
    "ExplanationFeature",
    "ExplanationDecision",
    "ExplanationAxis",
    "PairExplanation",
]
