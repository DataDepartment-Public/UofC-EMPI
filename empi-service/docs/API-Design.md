# eMPI Service API — Design

> **Implementation status (2026-07-14):** all build-order steps (1-5) are
> implemented and in production. `src/api/` — `store.py` (SQLite),
> `parquet_backend.py` (local Parquet index), `index_backend.py` (the
> pluggable-storage seam both go through), `publish.py` / `publish_local.py`
> (batch publish), `incremental.py` / `local_score.py` (single/few-record
> scoring), five routers (`health`, `runs`, `records`, `audit`, `dashboard`),
> `jobs.py`, `deps.py`, `schemas.py`, `main.py`. The `empi-dashboard/` Next.js
> app (step 5) is built and consumes these routes via its BFF route handlers
> — see [Application-Architecture.md](Application-Architecture.md). Tests:
> `tests/unit/test_api_store.py`, `test_api_publish.py`, `test_incremental.py`,
> `test_parquet_backend.py`, `test_api_deps.py`,
> `tests/integration/test_api.py`. Run locally with
> `uvicorn src.api.main:app --reload`. Two deliberate deviations from the
> literal spec, both noted inline in the code: `POST /audit/*` takes reviewer
> identity from the `X-Reviewer-Id` header only (never the request body — see
> `schemas.py`), and `POST /runs`'s `input_path` travels as a form field, not
> a raw JSON body (FastAPI can't mix `File` and `Body` params on one route —
> see `routers/runs.py`).
>
> **Storage is fully pluggable** (`src/api/index_backend.py`,
> `EMPI_INDEX_BACKEND`): SQLite (`empi.db`, default) or a local Parquet index
> (`data/local_index/`, `EMPI_INDEX_BACKEND=parquet`) — every route below,
> including `/audit/*`, works identically against either. A `python -m
> src.api.local_score` / `python -m src.api.publish_local` CLI pair needs no
> FastAPI/uvicorn at all for the Parquet path. Full schema + per-table status:
> [Data-Contract.md](Data-Contract.md) Stage 6.

Design for wrapping the in-process eMPI pipeline (`src/pipeline.py`) in a **FastAPI**
service. The service exposes four concerns:

1. **Health** — liveness/readiness for the deploy target and the dashboard.
2. **Runs** — accept new raw data and execute `clean → block → rules → FS
   matcher → cluster` as a background job, tracked by `run_id`.
3. **Records / Dashboard** — read models for the reviewer UI, plus incremental
   single/few-record scoring against the existing published population.
4. **Audit** — the reviewer-facing merge/unmerge actions. These call the API,
   which **updates the final resolved output in the resolved-output index**
   (the system of record) and appends an immutable audit entry.

> **Scope:** backend contract only. The front-end (per
> [Dashboard-Guide.md](../../empi-dashboard/docs/Dashboard-Guide.md)) is the
> `empi-dashboard/` Next.js app and consumes these routes.

---

## 1. Architecture at a glance

```
                         ┌──────────────────────────────────────────────┐
   raw file  ──POST/runs─▶│  FastAPI (uvicorn)                            │
                         │                                              │
                         │  BackgroundTasks ─▶ run_pipeline()           │
                         │        │              (existing, unchanged)  │
                         │        ▼                                     │
   Parquet artifacts ◀───┼──  data/{processed,blocking,matches,...}     │
   RunManifest JSON  ◀───┼──  data/runs/run_<id>.json                   │
                         │        │                                     │
                         │        ▼  load final output                  │
   reviewer ──POST/audit─▶│   SQLite (empi.db)  ◀── system of record ───┼─▶ GET /clusters
                         │     entities / members / audit_log           │   GET /records
                         └──────────────────────────────────────────────┘
```

Two storage tiers, by purpose:

| Tier | Holds | Why |
|---|---|---|
| **Parquet + manifest** (today) | per-stage batch artifacts (cleaned, candidate_pairs, matches, …) | columnar, versioned, reproducible; already built |
| **SQLite `empi.db`** (new) | the **final resolved output** + audit log | transactional, concurrent-safe under a web server, queryable for the dashboard, mutable by reviewers |

The pipeline stays a pure batch function that writes Parquet + a manifest. A thin
**publish** step loads that run's final output into `empi.db`. Reviewer edits then
mutate `empi.db` directly — never the Parquet artifacts, which remain the immutable
record of "what the algorithm produced for run X."

---

## 2. The final-output data model (SQLite)

The pipeline currently derives clusters in-memory via `assign_clusters` (union-find)
and never persists a per-record assignment. The service makes that output durable so
the dashboard can read it and reviewers can edit it.

```sql
-- One row per resolved master entity (a cluster, or a singleton).
CREATE TABLE entity (
    mid          TEXT PRIMARY KEY,      -- master id, e.g. "M-20001"
    run_id       TEXT NOT NULL,         -- run that last (re)published this entity
    origin       TEXT NOT NULL,         -- 'deterministic' | 'review' | 'merge' | 'none'
    is_merged    INTEGER NOT NULL,      -- 0/1
    confidence   REAL,                  -- carried from matches when applicable
    updated_utc  TEXT NOT NULL
);

-- Membership: which source records belong to which entity. The join target of a
-- merge/unmerge action.
CREATE TABLE entity_member (
    patid        TEXT PRIMARY KEY,      -- a source PATID belongs to exactly one entity
    mid          TEXT NOT NULL REFERENCES entity(mid),
    is_primary   INTEGER NOT NULL,      -- the surviving/golden record for the entity
    added_by     TEXT NOT NULL,         -- 'pipeline' | reviewer id
    updated_utc  TEXT NOT NULL
);

-- Append-only audit trail. Never updated or deleted.
CREATE TABLE audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT NOT NULL,
    user         TEXT NOT NULL,         -- reviewer identity (see §5 auth)
    action       TEXT NOT NULL,         -- 'merge' | 'unmerge' | 'split'
    patids       TEXT NOT NULL,         -- comma-separated PATIDs acted on
    mid          TEXT NOT NULL,         -- entity affected (or created)
    prev_state   TEXT NOT NULL,         -- human-readable, e.g. "Needs review"
    next_state   TEXT NOT NULL,         -- e.g. "Merged"
    run_id       TEXT                   -- run the entity belonged to when edited
);
```

`audit_log` maps 1:1 onto the demo's `AUDIT` array
(`{user, ts, ids, prev, next, mid}`), so the dashboard's audit table renders with no
shape change.

**Incremental-scoring index** (`src/api/store.py`, feeds `POST
/records/score` — see §3): two more tables, rebuilt wholesale by every full
publish and incrementally appended to between publishes.

```sql
-- On-disk mirror of blocking.BlockingIndex — a persisted posting list so a
-- single-record score is an indexed lookup, not a full-population rebuild.
-- B1 (SSN) / B6 (email) key values are SHA-256 hashed (direct identifiers);
-- every other block's key is already a lossy derived signal.
CREATE TABLE block_key (
    block_id   TEXT NOT NULL,   -- B1 | B3 | B4 | B5 | B6 | B7 | B8 | B9
    key_value  TEXT NOT NULL,
    patid      TEXT NOT NULL,
    PRIMARY KEY (block_id, key_value, patid)
);

-- SQL-queryable mirror of the CleanedRecords contract (src/contracts.py),
-- one row per valid PATID — lets rule evaluation for a handful of candidates
-- skip a ~163k-row Parquet read.
CREATE TABLE cleaned_attrs (
    patid TEXT PRIMARY KEY, first_nm TEXT, last_nm TEXT, birth_dt TEXT,
    ssn TEXT, ssn_last4 TEXT, email TEXT, zip_base TEXT, address1 TEXT,
    sex TEXT, phones_json TEXT, run_id TEXT NOT NULL
);
```

### Reconciliation: re-runs vs. human edits

Because `empi.db` is the system of record and the pipeline is the algorithm's opinion,
a **re-run must not silently clobber a reviewer's decisions.** The publish step is a
*merge*, not a *truncate*:

- New entities the reviewer never touched → upserted from the run.
- Entities a reviewer edited (any `audit_log` row referencing their members) →
  the human decision **wins**; the run's proposed grouping for those PATIDs is recorded
  as a *suggestion* but not auto-applied.
- Brand-new PATIDs in the run → placed by the algorithm as normal.

This keeps the answer the user gave ("the API updates the final output in the
database") consistent across repeated pipeline runs. The reconciliation policy is the
one design decision worth a second look before coding — it's called out again in §6.

---

## 3. Routes

### Health

```
GET /health        → 200 {"status":"ok"}                     # liveness
GET /health/ready  → 200/503 {checks: {db, data_dirs, last_run_id}}  # readiness
```

`ready` verifies `empi.db` is reachable and the data dirs are writable; returns the
last `run_id` so the dashboard can show "data current as of …".

### Runs (pipeline on new data)

```
POST /runs
  body: multipart file upload  OR  {"input_path": "data/raw/..."}
  → 202 {"run_id": "...", "status": "queued"}
```

Saves the upload under `data/raw/`, then schedules `run_pipeline(raw_input=...)` via
`BackgroundTasks`. Returns immediately — the pipeline is minutes-long and must not
block the request.

```
GET /runs            → [{run_id, status, counts, created_utc}, ...]
GET /runs/{run_id}   → the full RunManifest (reuse src/contracts.RunManifest as the
                       response model) + status: queued|running|succeeded|failed
```

Status store: a small `run_status` dict in memory for in-flight jobs, backed by the
presence of `data/runs/run_<id>.json` for completed ones (the manifest *is* the
durable success record). On failure the background task records the traceback summary
against the `run_id`.

### Records — incremental score (single/batch, no full re-run)

```
POST /records/score
  body: {"records": [{"PATID": "...", "FirstNM": "...", ...}, ...]}  # raw column
                                                                       # names, see
                                                                       # IncomingRecord
  → 202 {"run_id": "...", "status": "queued"}

GET /records/score/{run_id}
  → {"run_id", "status", "outcomes": [{"patid", "tier", "mid", "match_rule",
     "confidence", "matched_patids", "fs_match_probability",
     "fs_classification_tier"}, ...], "error"}
```

Scores one or a batch of new records against the **existing published
population** and writes the result straight into `empi.db`, without
re-running `clean → block → rules → cluster` over the whole dataset (`POST
/runs`'s job). Always a background job — same 202 + poll pattern as `POST
/runs`, regardless of batch size (`src/api/jobs.py`'s `score_records_job`,
tracked in a registry separate from full-pipeline runs and not listed by `GET
/runs`).

**Storage backend.** Scoring is written against `src.api.index_backend.IndexBackend`,
not a raw connection — `src/api/incremental.py` never imports `store`
directly. Two implementations:

- `SqlIndexBackend` (default, `EMPI_INDEX_BACKEND=sqlite`) — thin adapter
  over `store.py` + `empi.db`. `store.py`'s own SQL stays SQLite-flavored;
  the adapter takes its store module and connection as parameters, so a
  future Postgres-flavored store module could swap in without changing the
  adapter — scaffolding for that, not a built multi-engine store today.
- `ParquetIndexBackend` (`EMPI_INDEX_BACKEND=parquet`) — nine Parquet files
  under `settings.local_index_dir` (`block_key`, `cleaned_attrs`, `entity`,
  `entity_member`, `review_candidate`, `entity_suggestion`, `record_attrs`,
  `record_raw`, `audit_log`), loaded into memory and written back on commit.
  No `empi.db` — but full parity with it, including reviewer locking
  (`locked_patids()` reads the local `audit_log` table for real) and the
  dashboard read routes below. No FastAPI/uvicorn needed for either write
  path:

  ```
  python -m src.api.local_score --input record.json [--data-dir data/local_index]
  python -m src.api.publish_local --run-id <run_id> [--data-dir data/local_index]
  ```

  `--input` is a JSON file (one record object or a list), same raw-column
  shape as `POST /records/score`'s body. `EMPI_INDEX_BACKEND=parquet` also
  lets the *live* API (including the dashboard and `/audit/*`) run against
  the same Parquet files instead of `empi.db` — see
  [Data-Contract.md](Data-Contract.md) Stage 6 for the full schema and a note
  on the process-local lock (`deps.py::_PARQUET_BACKEND_LOCK`) that
  serializes concurrent requests against this backend.

Implementation (`src/api/incremental.py`): each record is cleaned via the
same `transform_dataframe` normalization as the batch pipeline, candidates
are found via an indexed lookup against the persisted block-key table (§2)
rather than rebuilding a `blocking.BlockingIndex` from the full population,
deterministic rules run over a tiny reconstructed frame, and:

- **auto-merge tier** → joins the record into an existing entity (or bridges
  two previously-separate entities if the record's matches span more than
  one — the same union `assign_clusters` performs at batch time, done
  directly against the backend here). Reviewer-locked candidates
  (`backend.locked_patids()`, §2 Reconciliation) are never auto-merged into;
  an `entity_suggestion` row is written instead.
- **review tier** → a `review_candidate` row, optionally scored by the active
  FS model (same Stage 4 as the batch pipeline) if one is configured.
- The record's own `block_key`/`cleaned_attrs` rows are appended immediately
  after it's scored, so a later record in the same batch (or a later call)
  can find it as a candidate.

### Clusters / records (read models for the Dataset tab)

```
GET /records?search=&origin=&is_merged=&birth_date=&ssn_last4=
    &updated_after=&updated_before=&page=&page_size=
                                     → paginated master rows + their candidate members
GET /clusters/{mid}                 → one entity, its members, and field values
GET /records/{patid}/raw            → the un-scrubbed *_raw source fields ("View Raw Data")
```

These go through `IndexBackend` (`entity` ⨝ `entity_member` ⨝ `record_attrs` ⨝
`review_candidate`), not a raw connection — identical results whether the
backend is SQLite or the local Parquet index. They back the dashboard Dataset
view; `GET /records/{patid}/raw` reads the one JSON blob `publish.py` (or
`incremental.py`) denormalized per PATID.

### Audit (merge / unmerge)

```
POST /audit/merge
  body: {"mid": "M-...", "patids": ["P1","P2",...], "user": "reviewer.jclark"}
  → 200 {audit_id, entity}    # entity reflects is_merged=1, origin='merge'

POST /audit/unmerge
  body: {"mid": "M-...", "patid": "P2", "user": "..."}   # split one record out
  → 200 {audit_id, new_mid, entity}

GET /audit?limit=&since=  → [audit_log rows, newest first]
```

Each write is **one backend transaction** (`begin()`/`commit()`/`rollback()` on
whichever `IndexBackend` is active): mutate `entity`/`entity_member`, then
insert the `audit_log` row. Either both land or neither does — the audit
trail can never disagree with the stored output. `merge` collapses members
into one `mid`; `unmerge` detaches a `patid` into a fresh standalone `mid`.

---

## 4. Code layout

```
src/api/
  main.py              # FastAPI app, lifespan (init db, configure_logging once)
  deps.py              # get_settings, get_db (raw SQLite), get_backend (IndexBackend;
                       #   holds the Parquet-mode concurrency lock for its duration)
  schemas.py           # request/response models; some reuse contracts.RunManifest etc.
  jobs.py              # background-job wrappers (run/score) + in-memory status registries
  index_backend.py     # IndexBackend protocol + SqlIndexBackend + build_index_backend
  store.py             # SQLite: schema, entity/member/audit_log CRUD, dashboard aggregates
  parquet_backend.py   # ParquetIndexBackend — the same operations over local Parquet files
  publish.py           # one RunManifest's output -> IndexBackend (batch)
  publish_local.py     # publish.py's batch path with no FastAPI/uvicorn — Parquet only
  incremental.py       # one/few new records -> IndexBackend, no full pipeline re-run
  local_score.py       # incremental.py's path with no FastAPI/uvicorn — Parquet only
  routers/
    health.py  runs.py  records.py  audit.py  dashboard.py
```

`run_pipeline()` itself is **not modified** by any of this — the service calls it
as-is. `assign_clusters` is reused to seed initial entities on publish.

---

## 5. Cross-cutting concerns

- **Concurrency / `configure_logging`** — its module-level `_LOGGING_CONFIGURED`
  guard means call it **once** in the app lifespan, not per request. `run_pipeline`
  re-calls it harmlessly (idempotent). Guard against two overlapping `POST /runs`
  writing the same `run_id` (timestamp collisions) by appending a short suffix.
  Separately, the Parquet backend needs its own concurrency guard
  (`deps.py::_PARQUET_BACKEND_LOCK`) since it wasn't originally designed for a
  live multi-request server — see §2.
- **PHI / HIPAA** — keep the existing "aggregate counts only" logging rule. The audit
  endpoints handle PATIDs and reviewer identity, so they need real auth before any
  non-local deploy. For local, the `user` comes from a request header
  (`X-Reviewer-Id`), set by the dashboard's BFF from a server-side session — the
  browser never sets it directly. Do **not** log field values.
- **Validation** — reuse pandera/pydantic. Request bodies are pydantic models;
  `RunManifest` is the run response model unchanged.
- **Idempotency** — `POST /audit/*` should be safe to retry; dedupe on
  `(action, sorted(patids), mid)` within a short window, or have the client send an
  idempotency key.

---

## 6. Design decisions (resolved)

1. **Reconciliation policy (§2)** — "human edit wins on re-run." A re-run does
   **not** re-merge PATIDs a reviewer explicitly **unmerged** — an unmerge is
   sticky until the reviewer reverses it.
2. **Identity** — header-based `X-Reviewer-Id`, set by the dashboard's BFF.
3. **Singletons** — every source record gets an `entity` row (singletons
   included, matching `ClusterAssignments` in `contracts`) — this makes
   `GET /records` a single clean query.

---

## 7. Build order (once approved)

1. `store.py` + schema + `publish.py`; unit-test publish against an existing
   `data/runs/run_*.json`.
2. `health` + `runs` routes over the unmodified `run_pipeline` (the §6 "minimal slice").
3. `records`/`clusters` read models.
4. `audit` merge/unmerge transactions + the reconciliation step in `publish.py`.
5. Wire the demo dashboard's `fetch` calls to the live routes.
