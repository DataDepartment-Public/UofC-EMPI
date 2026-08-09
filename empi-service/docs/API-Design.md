# eMPI Service API — Design

> **Implementation status (2026-08-07):** every route below is implemented
> and in production, not proposed. `src/api/` — `sql_backend.py` (SQLite),
> `postgres_backend.py` (Azure Database for PostgreSQL, AAD-only auth —
> the deployed target, see `terraform/postgres.tf`), `parquet_backend.py`
> (local Parquet index), `index_backend.py` (the pluggable-storage seam all
> three go through), `publish.py` / `publish_local.py` (batch publish),
> `incremental.py` / `local_score.py` (single/few-record scoring), eight
> routers (`health`, `runs`, `records`, `audit`, `dashboard`, `admin`,
> `explanations`), `jobs.py`, `deps.py`, `schemas.py`, `main.py`. The
> `empi-dashboard/` Next.js app consumes these routes via its BFF route
> handlers — see [Application-Architecture.md](Application-Architecture.md).
> Run locally with `uvicorn src.api.main:app --reload`. Two deliberate
> deviations from a literal REST spec, both noted inline in the code:
> `POST /audit/*` takes reviewer identity from the `X-Reviewer-Id` header
> only (never the request body — see `schemas.py`), and `POST /runs`'s
> `input_path` travels as a form field, not a raw JSON body (FastAPI can't
> mix `File` and `Body` params on one route — see `routers/runs.py`).
>
> **Storage is fully pluggable** (`src/api/backends/index_backend.py`,
> `EMPI_INDEX_BACKEND`): **SQLite** (`empi.db`, the default for local/dev),
> **Postgres** (`EMPI_INDEX_BACKEND=postgres`, AAD-token auth via the
> backend App Service's own managed identity — no stored password; the
> production target), or a **local Parquet index** (`data/local_index/`,
> `EMPI_INDEX_BACKEND=parquet`) — every route below, including `/audit/*`,
> works identically against all three. A `python -m
> src.api.ingest.local_score` / `python -m src.api.ingest.publish_local`
> CLI pair needs no FastAPI/uvicorn at all for the Parquet path. Full
> schema + per-table status: [Data-Contract.md](Data-Contract.md) Stage 6.

Design for wrapping the in-process eMPI pipeline (`src/pipeline.py`) in a **FastAPI**
service. The service exposes six concerns:

1. **Health** — liveness/readiness for the deploy target and the dashboard.
2. **Runs** — accept new raw data and execute `clean → block → deterministic
   rules → FS matcher (audit-only) → non-match gate → ML matcher → cluster`
   as a background job, tracked by `run_id`.
3. **Records / Review queue / Dashboard** — read models for the reviewer
   UI, plus incremental single/few-record scoring against the existing
   published population.
4. **Audit** — the reviewer-facing merge/unmerge/dismiss/undo actions.
   These call the API, which **updates the final resolved output in the
   resolved-output index** (the system of record) and appends an immutable
   audit entry — including PHI-access reads (`view_raw`, `view_ssn_clean`),
   not just state-changing actions.
5. **Explanations** — per-pair SHAP contribution vectors for the non-match
   gate and ML matcher's decisions, served read-only.
6. **Admin** — live model hot-reload and live-tunable decision thresholds;
   not reviewer-facing, no dashboard route calls the model ones directly.

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
                         │        ▼  publish() loads final output        │
   reviewer ──POST/audit─▶│  SQLite / Postgres / local Parquet          │
                         │   entities / members / review_candidate /    │
                         │   audit_log  ◀── system of record ───────────┼─▶ GET /records
                         └──────────────────────────────────────────────┘   GET /review-queue
```

Two storage tiers, by purpose:

| Tier | Holds | Why |
|---|---|---|
| **Parquet + manifest** | per-stage batch artifacts (cleaned, candidate_pairs, matches, gate_results, ml_features, …) | columnar, versioned, reproducible |
| **Resolved-output index** (SQLite / Postgres / local Parquet) | the **final resolved output** + review queue + audit log | transactional, concurrent-safe under a web server, queryable for the dashboard, mutable by reviewers |

The pipeline stays a pure batch function that writes Parquet + a manifest. A thin
**publish** step loads that run's final output into the resolved-output index. Reviewer
edits then mutate the index directly — never the Parquet artifacts, which remain the
immutable record of "what the algorithm produced for run X."

---

## 2. The final-output data model

The pipeline derives clusters in-memory via `assign_clusters` (union-find) and
never persists a per-record assignment on its own. The service makes that
output durable so the dashboard can read it and reviewers can edit it.

```sql
-- One row per resolved master entity (a cluster, or a singleton).
CREATE TABLE entity (
    mid          TEXT PRIMARY KEY,      -- master id, e.g. "M-000034"
    run_id       TEXT NOT NULL,         -- run that last (re)published this entity
    origin       TEXT NOT NULL,         -- 'deterministic' | 'review' | 'merge' | 'none'
    is_merged    INTEGER NOT NULL,      -- 0/1
    confidence   REAL,                  -- carried from matches when applicable
    match_rule   TEXT,                  -- the rule that confirmed it, when applicable
    evidence     TEXT,                  -- rules_fired / "Manually merged by <reviewer>"
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

-- One row per candidate pair the run decided -- the WHOLE pool, not just the
-- human queue (backs GET /review-queue and all four of its sections). See
-- src/api/pair_verdicts.py.
CREATE TABLE review_candidate (
    id                      INTEGER PRIMARY KEY,
    patid_a, patid_b        TEXT NOT NULL,
    match_rule              TEXT,       -- set only for the (currently empty) review-tier rule set
    confidence               REAL,
    evidence, source_blocks TEXT,
    run_id                  TEXT NOT NULL,
    created_utc             TEXT NOT NULL,
    fs_match_probability    REAL,       -- incremental scoring only
    fs_classification_tier  TEXT,
    ml_match_probability    REAL,       -- Stage 4.5 score, threaded in at batch publish
    ml_classification_tier  TEXT,
    gate_score              REAL,       -- Stage 4.25 P(plausible), threaded in at batch
                                        -- publish. A SEPARATE AXIS from
                                        -- ml_match_probability, not a fallback for it:
                                        -- for a gate-dropped pair it is the only score
                                        -- there is (the matcher never saw the pair),
                                        -- and for a pair the gate passed it is what
                                        -- explains a near-zero match probability on a
                                        -- row that still warrants review. NULL where
                                        -- the gate never scored the pair (a
                                        -- deterministic reject, an ungated run).
    verdict                 TEXT        -- which stage decided it: auto_merge_rule |
                                        -- reject | gate_dropped | ml_auto_merge |
                                        -- ml_human_review | undecided. NULL from
                                        -- incremental scoring (no model stage runs).
);

-- What a human concluded about one specific pair. Its own table, not a column
-- on review_candidate, because publish REPLACES that table per run and a
-- reviewer's decision must survive any number of republishes -- the same
-- stickiness locked_patids gives entity membership, at pair grain. Written by
-- every /audit/* action that rules on a pair; an undo deletes by audit_id.
CREATE TABLE reviewer_pair_decision (
    patid_a, patid_b        TEXT NOT NULL,
    decision                TEXT NOT NULL,   -- 'merged' | 'not_a_match'
    audit_id                INTEGER NOT NULL,
    ts_utc                  TEXT NOT NULL,
    PRIMARY KEY (patid_a, patid_b)
);

-- Append-only audit trail. Never updated in place except the `undone` flag.
CREATE TABLE audit_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc         TEXT NOT NULL,
    user           TEXT NOT NULL,         -- reviewer identity (see §5 auth)
    action         TEXT NOT NULL,         -- 'merge'|'unmerge'|'dismiss'|'split'|'view_raw'|'view_ssn_clean'
    patids         TEXT NOT NULL,         -- comma-separated PATIDs acted on
    mid            TEXT NOT NULL,         -- entity affected (or created)
    prev_state     TEXT NOT NULL,         -- human-readable, e.g. "Needs review"
    next_state     TEXT NOT NULL,         -- e.g. "Merged"
    run_id         TEXT,                  -- run the entity belonged to when edited
    related_patids TEXT,                  -- snapshot for retraining-label export; see Data-Contract.md §6e
    prev_mid       TEXT,                  -- undo provenance: the mid this action reversed from
    undo_of        INTEGER,               -- undo provenance: the audit_log.id this row reverses
    undone         INTEGER NOT NULL DEFAULT 0  -- set true on the original row once reversed
);
```

Full column-level schema for every table (including the incremental-scoring
lookup indexes `block_key`/`cleaned_attrs` and the display-denormalization
tables `record_attrs`/`record_raw`): [Data-Contract.md](Data-Contract.md)
Stage 6.

### Reconciliation: re-runs vs. human edits

Because the resolved-output index is the system of record and the pipeline is
the algorithm's opinion, a **re-run must not silently clobber a reviewer's
decisions.** The publish step is a *merge*, not a *truncate*:

- New entities the reviewer never touched → upserted from the run.
- Entities a reviewer edited (any `audit_log` row referencing their members) →
  the human decision **wins**; the run's proposed grouping for those PATIDs is recorded
  as a *suggestion* (`entity_suggestion`) but not auto-applied.
- Brand-new PATIDs in the run → placed by the algorithm as normal.

**Caveat this reconciliation does not cover:** if a run publishes an entirely
different source population (different PATID scheme — e.g. swapping a
synthetic test dataset for a real one), old entities/review candidates whose
patids don't appear in the new run are never *deleted*, only left un-upserted
— they linger in the index (deduped correctly at read time within a table,
but not purged across tables) until something explicitly clears them. This
is a real, encountered gap, not a hypothetical: see `to-do.md`'s
2026-08-07 entry. Swapping datasets in a long-lived environment should wipe
and republish, not just publish on top.

---

## 3. Routes

### Health

```
GET /health        → 200 {"status":"ok"}                     # liveness
GET /health/ready  → 200/503 {checks: {db, data_dirs, last_run_id}}  # readiness
```

### Runs (pipeline on new data)

```
POST /runs
  body: multipart file upload  OR  form field {"input_path": "data/raw/..."}
  → 202 {"run_id": "...", "status": "queued"}
```

Saves the upload under `data/raw/`, then schedules `run_pipeline(raw_input=...)` via
`BackgroundTasks`, and **auto-publishes on success** in the same background job — no
separate publish call needed. Returns immediately — the pipeline can be minutes-long
against a real population and must not block the request.

```
GET /runs            → [{run_id, status, counts, created_utc}, ...]
GET /runs/{run_id}   → the full RunManifest (reuse src/contracts.RunManifest as the
                       response model) + status: queued|running|succeeded|failed
```

### Records / Review queue (read models for the dashboard)

```
GET /records?search=&origin=&is_merged=&birth_date=&ssn_last4=
    &updated_after=&updated_before=&page=&page_size=
                                     → paginated master rows + their candidate members
GET /clusters/{mid}                 → one entity, its members, and field values
GET /clusters/{mid}/pairs[?run_id=] → the pairwise decision trace behind a cluster,
                                       read from the run's Parquet artifacts (see
                                       src/api/routers/cluster_pairs.py). Two guards
                                       shape the response:
                                       • the trace comes from the run that produced
                                         the entity or from NOWHERE — never quietly
                                         from another run. An entity last touched by
                                         incremental scoring names a job that writes
                                         no manifest; that id comes back as
                                         `unresolved_run_id` with
                                         `artifacts_available=false`, rather than
                                         being back-filled from the newest batch run
                                         (which knows nothing about these records and
                                         would render as "the pipeline decided
                                         nothing"). The `latest_run_with` fallback
                                         applies only when the entity names no run.
                                       • `pairs` is quadratic in members and the
                                         transitive path walk is worse, so above
                                         EMPI_CLUSTER_PAIRS_MAX_MEMBERS (200) both are
                                         omitted and `pairs_truncated=true`; `members`
                                         and `external_pairs` are still complete. A
                                         several-hundred-member hub cluster would
                                         otherwise cost minutes of blocking CPU and a
                                         response measured in hundreds of MB.
GET /records/{patid}/raw            → the un-scrubbed *_raw source fields ("View Raw
                                       Data" drawer) — writes a `view_raw` audit_log
                                       entry on every successful call
GET /records/{patid}/ssn-clean      → the pipeline-normalized SSN (`cleaned_attrs.ssn`,
                                       the value blocking/rules actually matched on) —
                                       backs the SSN-reveal toggle specifically, in
                                       preference to the raw endpoint above (a
                                       reviewer should sanity-check against the value
                                       the matching engine trusted, not raw source
                                       noise). A SEPARATE endpoint from `.../raw`,
                                       logged as its own `view_ssn_clean` audit action
                                       — the two are not interchangeable and not
                                       logged identically.
GET /review-queue?confidence_min=&confidence_max=&gate_score_min=&gate_score_max=
                  &verdict=&bucket=&search=&page=&page_size=
                                     → candidate-grain review queue (one row per
                                       candidate pair, not per cluster). Sorted on
                                       COALESCE(confidence, ml_match_probability) —
                                       most pairs have no rule confidence, only an ML
                                       score.

                                       Two independent score axes, matching the two
                                       models:
                                         confidence_min/max — the matcher side
                                           (COALESCE(confidence, ml_match_probability)).
                                           Max is INCLUSIVE.
                                         gate_score_min/max — the Stage-4.25 gate's
                                           P(plausible). Max is EXCLUSIVE, so adjacent
                                           bands can't both claim a boundary value.
                                       These are not two views of one number: a
                                       gate-dropped pair has no matcher score at all
                                       and is invisible to any confidence_* bound
                                       (NULL >= x is never true), reachable only on the
                                       gate axis.

                                       `verdict` filters on src/api/pair_verdicts.py's
                                       vocabulary exactly (422 on anything else). It is
                                       the only filter that reaches pairs no model
                                       scored — a deterministic `reject` carries no
                                       number for any range to match.

                                       `bucket` is one of the four reviewer-facing
                                       sections (422 on anything else); unset returns
                                       every candidate, and each row carries its own
                                       `bucket` either way:
                                         needs_review  — nothing resolved it, nobody ruled
                                         reviewed      — a reviewer ruled on this exact pair
                                         auto_merged   — the two records share a mid today
                                         auto_rejected — the reject rules or the gate
                                                         declined it
                                       A reviewer decision always wins over the
                                       pipeline's verdict. "Merged" is read off the
                                       index, not the verdict, since the two can
                                       disagree — see src/api/pair_verdicts.py.
                                       Every response also carries `bucket_counts`,
                                       the whole-index total per section (not narrowed
                                       by this request's filters).
POST /records/score
GET /records/score/{run_id}         → incremental single/batch scoring against the
                                       existing published population, no full re-run
                                       (see §3 continued below)
```

These go through `IndexBackend` (`entity` ⨝ `entity_member` ⨝ `record_attrs` ⨝
`review_candidate`), not a raw connection — identical results whether the
backend is SQLite, Postgres, or the local Parquet index.

**Incremental scoring detail** (`POST /records/score`):

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
population**, without re-running `clean → block → rules → cluster` over the
whole dataset (`POST /runs`'s job). Always a background job — same 202 +
poll pattern as `POST /runs` (`src/api/jobs.py`'s `score_records_job`,
tracked in a registry separate from full-pipeline runs and not listed by
`GET /runs`). Implementation (`src/api/ingest/incremental.py`): each record
is cleaned via the same normalization as the batch pipeline, candidates are
found via an indexed lookup against the persisted block-key table rather
than rebuilding a `BlockingIndex` from the full population, deterministic
rules run over a tiny reconstructed frame, and:

- **auto-merge tier** → joins the record into an existing entity (or bridges
  two previously-separate entities). Reviewer-locked candidates
  (`backend.locked_patids()`) are never auto-merged into; an
  `entity_suggestion` row is written instead.
- **review tier** → a `review_candidate` row, optionally scored by the active
  FS model if one is configured (`fs_match_probability`/`fs_classification_tier`
  — the ML matcher and non-match gate are batch-publish-only today, not
  wired into the incremental path).

The Parquet backend has a CLI equivalent needing no FastAPI/uvicorn process:

```
python -m src.api.ingest.local_score --input record.json [--data-dir data/local_index]
python -m src.api.ingest.publish_local --run-id <run_id> [--data-dir data/local_index]
```

### Audit (merge / unmerge / dismiss / undo)

```
POST /audit/merge
  headers: X-Reviewer-Id: <reviewer>
  body: {"mid": "M-...", "patids": ["P1","P2",...]}
  → 200 {audit_id, entity}    # entity reflects is_merged=1, origin='merge'

POST /audit/unmerge
  headers: X-Reviewer-Id: <reviewer>
  body: {"mid": "M-...", "patid": "P2"}   # split one record out into a fresh mid
  → 200 {audit_id, new_mid, entity}

POST /audit/dismiss
  headers: X-Reviewer-Id: <reviewer>
  body: {"patid_a": "P1", "patid_b": "P2"}   # "Not a match" on a candidate pair
  → 200 {audit_id, unmerged_to_mid}
  # If the two records currently share a mid, this SPLITS them first (an
  # ordinary unmerge of patid_b, its own audit entry, undoable on its own
  # terms) and returns the new mid — that's the override for the queue's
  # Auto-merged section. `unmerged_to_mid` is null when they weren't merged.

POST /audit/{audit_id}/undo
  headers: X-Reviewer-Id: <reviewer>
  → 200 {audit_id, reversed_action, entity, new_mids, already_detached}
  # reverses a prior merge/unmerge/dismiss by id; marks the original row
  # undone=true, writes a new row with prev_mid/undo_of set, and retracts the
  # reviewer_pair_decision rows that entry wrote. Undoing a dismiss mutates no
  # entity (a dismiss never did) — it only retracts the decision, returning the
  # pair to whichever section its verdict puts it in. 404 if already undone.
  #
  # ONE transaction for the WHOLE reversal, retraction included — a merge undo
  # is N unmerges, and committing them one at a time could leave the entity
  # half-split while `undone` went true anyway (it is derived: any row pointing
  # at the entry via undo_of sets it), i.e. an audit trail claiming a clean
  # reversal that did not happen. Either every patid came back out, or nothing
  # changed and the entry is still undoable.
  #
  # `already_detached` lists patids of an undone merge that a LATER action had
  # already moved out of the merged entity. They are skipped, not treated as a
  # failure: they are already where the reversal wants them, and 404-ing over
  # one would leave the reviewer unable to reverse the rest of the merge.

GET /audit?limit=&since=  → [audit_log rows, newest first]
```

Every write body deliberately omits reviewer identity — it comes from the
`X-Reviewer-Id` header only (see the status banner at the top of this doc).
Each write is **one backend transaction** (`begin()`/`commit()`/`rollback()`
on whichever `IndexBackend` is active): mutate `entity`/`entity_member`, then
insert the `audit_log` row. Either both land or neither does.

### Explanations (per-pair SHAP)

```
GET /explanations/{model_name}/{patid_a}/{patid_b}[?run_id=<run_id>]
  → 200 PairExplanation   # feature-level SHAP waterfall for the gate's or
                            # ML matcher's decision on this pair
  → 404 if this model never scored this pair (e.g. the gate dropped it
    before the ML matcher ever saw it, or no model was active for the run)
```

`model_name` is `nonmatch_gate` or `ml_matcher`. Full contract (payload
shape, waterfall semantics, sign conventions):
[Explanations-Guide.md](Explanations-Guide.md) — front-end readers only need
its §2.

### Admin (model hot-swap + live thresholds)

```
POST /admin/models/reload
  → 200 {"invalidated": [...], "fs_active_model": {...} | null,
         "ml_active_model": {...} | null, "gate_active_model": {...} | null}

GET /admin/models/status
  → 200 {"cached": {...}, "fs_active_model": {...} | null,
         "ml_active_model": {...} | null, "gate_active_model": {...} | null}

GET /admin/thresholds   → 200 ThresholdSettings
PUT /admin/thresholds
  body: ThresholdSettings {
    gate_threshold: float,           # P(plausible) floor to reach the ML matcher
    ml_auto_merge_threshold: float,  # ML matcher's auto_merge tier floor
    fs_review_floor: float,          # FS matcher's human_review tier floor
  }
  → 200 ThresholdSettings   # persisted to data/config/thresholds.json,
                            # applied on next app startup AND live via
                            # threshold_store — no restart required
```

`/admin/models/*` is not reviewer-facing (no dashboard route calls it
directly — it's the champion-promotion CI workflow's job). `src/models/
model_cache.py` caches each model's deserialized artifact in memory, keyed
by `(path, mtime)` — a promoted model (new `active.json` pointer) is picked
up automatically the next time anything asks for it; `/admin/models/reload`
just makes that moment immediate and observable. `/admin/thresholds` **is**
reviewer-facing — it backs the dashboard's Admin tab.

---

## 4. Code layout

```
src/api/
  main.py              # FastAPI app, lifespan (init db, configure_logging once)
  deps.py              # get_settings, get_db (raw connection), get_backend (IndexBackend;
                       #   holds the Parquet-mode concurrency lock for its duration)
  schemas.py           # request/response models; some reuse contracts.RunManifest etc.
  jobs.py              # background-job wrappers (run/score) + in-memory status registries
  threshold_store.py   # live-tunable decision thresholds, backed by data/config/thresholds.json
  backends/
    index_backend.py   # IndexBackend protocol + SqlIndexBackend + build_index_backend
    sql_backend.py      # SQLite implementation
    postgres_backend.py # Postgres implementation (AAD-token auth, no stored password)
    parquet_backend.py  # local Parquet implementation
  ingest/
    publish.py / publish_local.py     # batch: run output -> index (reuses assign_clusters)
    incremental.py / local_score.py   # single/few-record scoring -> index
  routers/
    health.py  runs.py  records.py  audit.py  dashboard.py  admin.py  explanations.py

src/models/
  model_cache.py         # mtime-keyed in-memory cache for the FS/ML matcher and
                         #   non-match gate artifacts — see "Admin" above
```

`run_pipeline()` itself is **not modified** by any of this — the service calls it
as-is. `assign_clusters` is reused to seed initial entities on publish.

---

## 5. Cross-cutting concerns

- **Concurrency** — the backend is a **single-instance** service by design:
  `jobs.py`'s in-flight run/score registries are in-process dicts, so a
  second replica or `--workers > 1` would silently 404 on runs/scores it
  didn't start. The Parquet backend additionally needs its own
  process-local lock (`deps.py::_PARQUET_BACKEND_LOCK`) since it wasn't
  designed for a live multi-request server.
- **SQLite thread affinity** — `sql_backend.get_connection` opens with
  `check_same_thread=False` deliberately: FastAPI runs a sync generator
  dependency's setup, the route handler, and its teardown as separate
  `anyio` worker-thread dispatches that aren't guaranteed to land on the
  same OS thread under concurrent load, which otherwise intermittently
  raised `sqlite3.ProgrammingError` on `backend.close()`. The connection is
  still only ever used sequentially within one request's lifetime.
- **PHI / HIPAA** — keep the "aggregate counts only" logging rule. Every PHI
  read (`GET /records/{patid}/raw`, `GET /records/{patid}/ssn-clean`) and
  every mutation is written to `audit_log`. Reviewer identity comes from
  the `X-Reviewer-Id` header, set by the dashboard's BFF from the
  authenticated Entra ID principal in a real deploy (Easy Auth's
  `X-MS-CLIENT-PRINCIPAL-NAME`, see `empi-dashboard/src/lib/server-api.ts`)
  or a hardcoded fallback identity in local dev (no Azure platform in front
  of the container locally). The browser never sets this header itself.
- **Schema management is decoupled from the app lifecycle** — `lifespan`
  never calls `init_db()` on boot; the app only ever connects to an
  already-correct database and logs a loud error (without crashing) if the
  expected schema isn't there. `scripts/init_db.py` is the explicit way to
  create or update it — run once per environment, and again whenever
  `_COLUMN_MIGRATIONS` (in `sql_backend.py`/`postgres_backend.py`) gains a
  new entry. Not a versioned migration framework — the same schema-setup
  logic that always existed, just no longer implicit.

---

## 6. Design decisions (resolved)

1. **Reconciliation policy (§2)** — "human edit wins on re-run." A re-run does
   **not** re-merge PATIDs a reviewer explicitly **unmerged** — an unmerge is
   sticky until the reviewer reverses it (or an `undo`).
2. **Identity** — header-based `X-Reviewer-Id`, set by the dashboard's BFF
   from the authenticated Entra ID principal (Easy Auth) in Azure.
3. **Singletons** — every source record gets an `entity` row (singletons
   included) — this makes `GET /records` a single clean query.
4. **The ML matcher and non-match gate are batch-publish-only** — neither
   is wired into `POST /records/score`'s incremental path today. Only the
   FS matcher scores incrementally.
