# eMPI Service API — Design

> **Implementation status:** build-order steps 1–4 are implemented in `src/api/`
> — `store.py`, `publish.py`, all four routers (`health`, `runs`, `records`,
> `audit`), `jobs.py`, `deps.py`, `schemas.py`, `main.py`. Tests:
> `tests/unit/test_api_store.py`, `test_api_publish.py`,
> `tests/integration/test_api.py`. Run locally with
> `uvicorn src.api.main:app --reload`. Step 5 (wiring the demo dashboard's
> `fetch` calls) is not started — the front-end isn't built yet. Two
> deliberate deviations from the literal spec, both noted inline in the code:
> `POST /audit/*` takes reviewer identity from the `X-Reviewer-Id` header only
> (never the request body — see `schemas.py`), and `POST /runs`'s
> `input_path` travels as a form field, not a raw JSON body (FastAPI can't mix
> `File` and `Body` params on one route — see `routers/runs.py`).

Design for wrapping the in-process eMPI pipeline (`src/pipeline.py`) in a **FastAPI**
service. The service exposes three concerns:

1. **Health** — liveness/readiness for the deploy target and the dashboard.
2. **Runs** — accept new raw data and execute `clean → block → rules → cluster`
   as a background job, tracked by `run_id`.
3. **Audit** — the reviewer-facing merge/unmerge actions. These call the API,
   which **updates the final resolved output in a local database** (the system of
   record) and appends an immutable audit entry.

> **Scope:** backend contract only. The front-end (per
> [Dashboard-Guide.md](Dashboard-Guide.md), demo at
> [demo/dashboard-demo.html](../demo/dashboard-demo.html)) is built separately and
> consumes these routes.

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

### Clusters / records (read models for the Dataset tab)

```
GET /records?search=&status=&page=  → paginated master rows + their candidate members
GET /clusters/{mid}                 → one entity, its members, and field values
```

These read straight from `empi.db` (`entity` ⨝ `entity_member`), joined to the cleaned
attributes for display. They back the dashboard Dataset view.

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

Each write is **one transaction**: mutate `entity`/`entity_member`, then insert the
`audit_log` row. Either both land or neither does — the audit trail can never
disagree with the stored output. `merge` collapses members into one `mid`;
`unmerge` (the demo's `unmergeRecord`) detaches a `patid` into a fresh standalone
`mid`. Maps directly onto the demo handlers `confirmMergeBtn`, `unmerge`,
`unmergeRecord`.

---

## 4. Proposed code layout

```
src/api/
  main.py            # FastAPI app, lifespan (init db, configure_logging once)
  deps.py            # get_settings, get_db (SQLite connection per request)
  schemas.py         # request models; responses reuse contracts.RunManifest etc.
  jobs.py            # run-pipeline background wrapper + in-memory status registry
  store.py           # SQLite: schema init, entity/member upsert, audit insert
  publish.py         # load a run's final output (matches+clusters) into empi.db
  routers/
    health.py
    runs.py
    audit.py
    records.py
```

`run_pipeline()` itself is **not modified** — the service calls it as-is. The only new
pipeline-adjacent code is `publish.py` (Parquet → DB) and `store.py` (the DB layer).
`assign_clusters` is reused to seed initial entities on publish.

New dependencies: `fastapi`, `uvicorn[standard]`, `python-multipart` (uploads).
SQLite is stdlib. Add to `requirements.txt` / `pyproject`.

---

## 5. Cross-cutting concerns

- **Concurrency / `configure_logging`** — its module-level `_LOGGING_CONFIGURED`
  guard means call it **once** in the app lifespan, not per request. `run_pipeline`
  re-calls it harmlessly (idempotent). Guard against two overlapping `POST /runs`
  writing the same `run_id` (timestamp collisions) by appending a short suffix.
- **PHI / HIPAA** — keep the existing "aggregate counts only" logging rule. The audit
  endpoints handle PATIDs and reviewer identity, so they need real auth before any
  non-local deploy. For local, the `user` comes from a request header
  (`X-Reviewer-Id`); the demo hardcodes `reviewer.jclark`. Do **not** log field values.
- **Validation** — reuse pandera/pydantic. Request bodies are pydantic models;
  `RunManifest` is the run response model unchanged.
- **Idempotency** — `POST /audit/*` should be safe to retry; dedupe on
  `(action, sorted(patids), mid)` within a short window, or have the client send an
  idempotency key.

---

## 6. Open decisions (confirm before coding)

1. **Reconciliation policy (§2)** — confirmed model is "human edit wins on re-run."
   Is a re-run allowed to *re-merge* PATIDs a reviewer explicitly **unmerged**? Default
   here: no — an unmerge is sticky until the reviewer reverses it.
2. **Identity** — header-based `X-Reviewer-Id` acceptable for the local build, or do
   you want even a stub login now?
3. **Singletons** — should every source record get an `entity` row (singletons
   included, matching `ClusterAssignments` in contracts), or only multi-record
   clusters? Including singletons makes `GET /records` a single clean query.

---

## 7. Build order (once approved)

1. `store.py` + schema + `publish.py`; unit-test publish against an existing
   `data/runs/run_*.json`.
2. `health` + `runs` routes over the unmodified `run_pipeline` (the §6 "minimal slice").
3. `records`/`clusters` read models.
4. `audit` merge/unmerge transactions + the reconciliation step in `publish.py`.
5. Wire the demo dashboard's `fetch` calls to the live routes.
