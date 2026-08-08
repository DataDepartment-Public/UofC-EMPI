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


class ActiveModelInfo(BaseModel):
    """Meta of one matcher's currently-resolved active model, or None if
    neither an override nor active.json nor any model file resolves."""

    model_config = ConfigDict(extra="allow")

    model_file: str | None = None


class CachedModelInfo(BaseModel):
    path: str
    mtime: float


class ModelReloadResponse(BaseModel):
    invalidated: list[str]
    fs_active_model: ActiveModelInfo | None
    ml_active_model: ActiveModelInfo | None


class ModelStatusResponse(BaseModel):
    cached: dict[str, CachedModelInfo]
    fs_active_model: ActiveModelInfo | None
    ml_active_model: ActiveModelInfo | None


class RecordAttrs(BaseModel):
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None
    suffix: str | None = None
    birth_date: str | None = None
    ssn_last4: str | None = None
    email: str | None = None
    zip_code: str | None = None
    city: str | None = None
    address1: str | None = None
    sex: str | None = None
    #: The primary cleaned phone (`PrimaryPhoneNBR_clean`) — one number.
    phone: str | None = None
    #: Every cleaned phone on the record (`Phones_set`), the set B5 blocking
    #: and the NAME_DOB_PHONE rule actually intersect on. Empty when the
    #: record has no phone, or when `record_attrs` predates the column and the
    #: run hasn't been re-published (see `sql_backend._COLUMN_MIGRATIONS`).
    phones: list[str] = Field(default_factory=list)


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
    #: Stage 4.5 ML matcher's score (`docs/ML-Model-LightGBM-v5.md`) — the
    #: actual scored decision for pairs that reached this queue with no rule
    #: firing (`confidence` null). Null only when the ML matcher didn't score
    #: this pair (no active model, or dropped earlier by the non-match gate).
    ml_match_probability: float | None = None
    ml_classification_tier: str | None = None


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
    ml_match_probability: float | None = None
    ml_classification_tier: str | None = None
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


class CleanSsn(BaseModel):
    patid: str
    ssn: str | None


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


class UndoResponse(BaseModel):
    """`POST /audit/{audit_id}/undo` — reverses a `merge` or `unmerge` entry.

    `reversed_action` is the action that was undone (not the action taken to
    undo it). Undoing a `merge` unmerges every patid back into its own
    singleton entity (no single `entity` to return, hence `new_mids`);
    undoing an `unmerge` re-merges the one patid back into `prev_mid`
    (`entity` is that reconstituted entity, and `new_mids` is just `[mid]`
    for symmetry with the merge case).
    """

    audit_id: int
    reversed_action: Literal["merge", "unmerge"]
    entity: Entity | None = None
    new_mids: list[str] = Field(default_factory=list)


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
    action: Literal["merge", "unmerge", "split", "dismiss", "view_raw", "view_ssn_clean"]
    patids: str
    mid: str
    prev_state: str
    next_state: str
    run_id: str | None
    related_patids: str | None = None
    prev_mid: str | None = None
    undo_of: int | None = None
    undone: bool = False


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


# ── Cluster pair trace (GET /clusters/{mid}/pairs) ───────────────────────────
#: Every value `ClusterPair.verdict` can take, most decisive stage first —
#: which is also the order the route resolves them in, and the order the
#: dashboard sorts by. Exported so the UI's badge map and the tests can be
#: checked against one list rather than three copies of the same strings.
CLUSTER_PAIR_VERDICTS = (
    "auto_merge_rule",     # a deterministic auto-merge rule confirmed it
    "reject",              # the rules rejected it (>=3 strong contradictions)
    "ml_auto_merge",       # Stage 4.5 scored it a confident match
    "ml_human_review",     # Stage 4.5 scored it ambiguous
    "gate_dropped",        # Stage 4.25 dropped it as a confident non-match
    "blocked_undecided",   # blocked together, but no stage recorded a decision
    "not_compared",        # never blocked together — same cluster transitively
)


class ClusterPair(BaseModel):
    """What the pipeline did with one pair of a cluster's current members.

    Assembled from the run's Parquet artifacts, not the index: publishing
    collapses a cluster's deterministic pairs into one best-pair evidence
    string and deletes the gate's drops, so this is the only place the full
    picture survives. Every stage field is nullable — a run that skipped a
    stage, or a pair that never reached it, reports nothing rather than a
    fabricated zero.
    """

    patid_a: str
    patid_b: str
    verdict: str

    #: Stage 2 — did blocking ever put these two in the same candidate set?
    #: `False` with a non-`not_compared` verdict is impossible; `False` alone
    #: is what makes a pair transitive rather than directly compared.
    blocked: bool
    source_blocks: str | None = None
    n_blocks: int | None = None

    #: Stage 3 — deterministic rules, both directions.
    match_rule: str | None = None
    rules_fired: str | None = None
    confidence: float | None = None
    reject_rule: str | None = None
    n_contradictions: int | None = None

    #: Stage 4.25 / 4.5. A gate score is present for every pair the gate saw,
    #: including the ones it dropped; ML fields only for gate survivors.
    gate_score: float | None = None
    gate_tier: str | None = None
    ml_score: float | None = None
    ml_tier: str | None = None

    #: Reviewer provenance, from `entity_member.added_by` rather than a scan
    #: of `audit_log`: a member the pipeline placed reads `"pipeline"`, and
    #: anything else is the reviewer id that merged it in. A pair is
    #: reviewer-joined when either side was.
    joined_by: Literal["pipeline", "reviewer"] = "pipeline"
    reviewer_id: str | None = None
    reviewer_ts_utc: str | None = None


class ClusterExternalPair(ClusterPair):
    """A comparison between one of this cluster's members and a record that
    ended up somewhere else.

    Why this is a separate list from `pairs`: those explain how the cluster
    was *built*, these explain where it *stopped* — the near-misses the
    pipeline considered and declined. For a singleton it is the only thing
    there is to show, and "nothing was ever compared to this record" and
    "six records were compared and all six were rejected" are very different
    answers to "why is this patient alone?".

    Unlike `pairs`, this list never contains a `not_compared` verdict. It is
    built by reading the artifacts and keeping rows that touch a member, so a
    pair only appears if some stage actually looked at it — enumerating the
    other ~7,000 records the pipeline never considered would be noise, not
    evidence.
    """

    #: Which side of the pair belongs to this cluster; the other is outside.
    #: `patid_a`/`patid_b` stay canonically ordered, so the UI needs this to
    #: know which record to describe rather than re-deriving it.
    member_patid: str
    other_patid: str
    #: The counterpart's current cluster and display fields, resolved from the
    #: index at request time — the artifacts carry PATIDs only, and a reviewer
    #: needs a name to judge whether a rejection looks right.
    other_mid: str | None = None
    other_first_name: str | None = None
    other_last_name: str | None = None
    other_birth_date: str | None = None
    other_ssn_last4: str | None = None


class ClusterPairsResponse(BaseModel):
    """Every unordered pair of a cluster's current members, with its trace.

    Membership is read from the index (`entity_member`), not from the run's
    `cluster_assignments` — sticky-unmerge and reviewer merges make the two
    diverge, and a reviewer is looking at the cluster as it stands now.
    A pair whose two records were never in the same run therefore lands on
    `not_compared`, which is the honest answer.
    """

    mid: str
    run_id: str | None = None
    #: False when the run is unresolvable or its artifacts are gone from disk.
    #: The pairs are still enumerated (so the UI can list the cluster's
    #: members) but every stage field is null — distinguishable from "the
    #: pipeline genuinely decided nothing about this pair".
    artifacts_available: bool
    members: list[str] = Field(default_factory=list)
    #: `gate_threshold` / `ml_auto_merge_threshold`, so the UI can draw the
    #: decision boundary beside a score without a second call to /admin.
    thresholds: dict[str, float] = Field(default_factory=dict)
    #: Every unordered pair of members — how the cluster was assembled.
    pairs: list[ClusterPair] = Field(default_factory=list)
    #: Comparisons against records outside the cluster — the near-misses.
    external_pairs: list[ClusterExternalPair] = Field(default_factory=list)


# ── Admin (GET/PUT /admin/thresholds) ────────────────────────────────────────
class ThresholdSettings(BaseModel):
    """The live-tunable ML decision thresholds — see
    `src/api/threshold_store.py`. Same shape for both the GET response and
    the PUT request body."""

    gate_threshold: float = Field(
        ge=0.0, le=1.0,
        description="P(plausible) at/above which a pair passes the "
        "non-match gate and reaches the ML matcher.",
    )
    ml_auto_merge_threshold: float = Field(
        ge=0.0, le=1.0,
        description="Match-probability at/above which the ML matcher "
        "tiers a pair 'auto_merge'.",
    )
    fs_review_floor: float = Field(
        ge=0.0, le=1.0,
        description="Match-probability at/above which the FS matcher "
        "tiers a pair 'human_review' (also the candidate-inclusion floor "
        "for the FSFeatures parquet).",
    )


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
    "UndoResponse",
    "AuditLogRow",
    "IncomingRecord",
    "ScoreRequest",
    "ScoreCreateResponse",
    "ScoreTier",
    "RecordScoreOutcome",
    "ScoreResult",
    "ActiveModelInfo",
    "CachedModelInfo",
    "ModelReloadResponse",
    "ModelStatusResponse",
    "ExplanationFeature",
    "ExplanationDecision",
    "ExplanationAxis",
    "PairExplanation",
    "CLUSTER_PAIR_VERDICTS",
    "ClusterPair",
    "ClusterExternalPair",
    "ClusterPairsResponse",
    "ThresholdSettings",
]
