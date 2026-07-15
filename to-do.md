# To Do

Ordered by priority. Research paper is last per your note — you'll follow up with venue/scope details.

---

## 0. Picking back up next session

**Operationalization — Phases 1-4 DONE (2026-07-14).** Local/Parquet mode is
now a complete second storage engine matching `store.py`'s whole surface: all
9 `data/local_index/*.parquet` tables are fully implemented on both backends
(see `docs/Data-Contract.md`'s Stage 6 status summary), covering batch
publish, incremental scoring, the full dashboard read side, and reviewer
merge/unmerge. Only Phase 5 (the real 163k-row verification run) is left —
picking back up there.

Also folded in `origin/sachin-dataflow`'s "Data-contract cleanup" commit
(cherry-picked, not merged — a true merge conflicted with this branch's
still-uncommitted `empi-dashboard/` restructuring): `contracts.Rejects`
validation and `fs_matcher/train.py`'s `RunManifest`-based lineage fix are
now in this branch too.

**What shipped, in brief** (full detail in `docs/Data-Contract.md` Stage 6 +
git diff — this is a pointer, not a duplicate spec):
- **Phase 1** — `publish_run(backend, run_id, settings)` is fully
  backend-agnostic (no more `store`/`conn` imports in `publish.py`).
  `jobs.py::run_pipeline_job` now honors `EMPI_INDEX_BACKEND`. New
  `python -m src.api.publish_local --run-id <id>` CLI materializes a batch
  run into the local Parquet index with zero FastAPI/SQLite — pair with
  `python -m src.pipeline --input file.csv` for a fully local batch
  workflow. `ParquetIndexBackend` gained `record_attrs`/`record_raw` tables
  and every `*_bulk`/`replace_*` method `publish_run` needs.
- **Phase 2** — `IndexBackend` gained `list_entities`, `dashboard_summary`,
  `get_record_raw`, `review_candidates_for_patid`. New `get_backend` FastAPI
  dependency (`deps.py`); `records.py`/`dashboard.py` are backend-agnostic
  now (`_to_entity` too). The dashboard fully works with
  `EMPI_INDEX_BACKEND=parquet` — no `empi.db` involved at all.
- **Phase 3** — new `audit_log.parquet` table + `insert_audit_log`/
  `list_audit_log`; `ParquetIndexBackend.locked_patids()` is a real read now,
  not a hardcoded empty set. `audit.py` (merge/unmerge) is backend-agnostic.
  New process-local lock (`deps.py::_PARQUET_BACKEND_LOCK`) serializes
  Parquet-backend requests within one process — needed once a live FastAPI
  app (not just a one-shot CLI) drives concurrent requests against it;
  verified it's load-bearing by disabling it and watching a concurrency test
  fail intermittently, then re-enabling.
- **Phase 4** — `docs/Data-Contract.md` Stage 6 fully flipped from
  PROPOSED to IMPLEMENTED across all 9 tables. Test coverage: parity tests
  for `publish_run`/dashboard routes/audit routes against both backends,
  direct unit tests for every new `ParquetIndexBackend` method, a
  deterministic lock-acquisition test (`test_api_deps.py`), and a real
  multi-threaded concurrency test through the FastAPI `TestClient`. Full
  suite: 594 passed. `docs/Local-Mode-Guide.md` (standalone doc) wasn't
  written — the per-table "Backends:" notes in Data-Contract.md's Stage 6
  section end up covering the same ground; revisit only if a
  workflow-narrative doc (not a schema reference) turns out to be needed.

### Phase 5 — full batch run + reserve-set verification — DONE (2026-07-14)

Ran the full pipeline on `data/raw/MDM_Population_batch.csv` (163,344 rows —
the full `MDM_Population.csv` minus a 20-row reserve set), published it to
*both* backends at real scale for the first time, then fed the reserve set
through incremental scoring against both to prove a genuinely new record
resolves sensibly against the batch-published population.

- [x] Split the reserve set: `scripts/` had no committed splitter (the earlier
  session's `/tmp/split_reserve.py` was never saved), so rewrote it —
  `data/raw/MDM_Population_batch.csv` (163,344 rows) +
  `data/raw/MDM_Population_reserved.json` (the last 20 rows, raw column
  names). Confirmed none of the 20 reserved PATIDs were already published in
  the pre-existing `empi.db` (which held a smaller, unrelated 62,609-record
  demo population, not the full dataset) before proceeding, so the
  "genuinely new record" test is real, not contaminated by prior state.
- [x] `python -m src.pipeline --input data/raw/MDM_Population_batch.csv` — run
  `20260714T151856Z`, 627.6s. 158,704 valid records (4,640 invalid) → 109,026
  stacked-blocker candidate pairs → 23,143 auto-merge / 57,642 review (21,831
  rule-confirmed) / 28,241 reject → FS matcher scored the 57,642 review pairs
  (tiers: auto_merge 40,131 / human_review 5,228 / no_match 12,283; 45,359 GBT
  candidates) → 138,415 clusters. All Parquet artifacts landed as expected;
  numbers are in the same ballpark as `Deterministic-Rules-Guide.md`'s
  `real_20260620` reference run (this one excludes 20 rows and post-dates
  Sachin's contract cleanup, so exact counts differ slightly — expected).
- [x] Published to both backends — identical counts on each
  (`clusters_seen`/`entities_upserted`: 138,415; `members_upserted`: 158,704;
  `review_candidates`: 57,642; `block_keys_indexed`: 915,585): `publish_run`
  direct call for SQLite (`empi.db` grew 86MB → 502MB), `python -m
  src.api.publish_local --run-id 20260714T151856Z` for Parquet (`data/local_index/`
  went from empty to all 9 tables populated — the first time it's been
  exercised at production scale, not just test fixtures).
- [x] Scored the 20 reserved records against both backends
  (`incremental.score_records` / `local_score.score_local` directly — the
  `POST /records/score` HTTP layer is already covered by
  `tests/integration/test_api.py`, so this exercised the same underlying
  function against real-scale data instead of standing up a live server).
  Result: **7 auto_merge / 10 review / 3 no_match, and all 20 records agree
  on tier + match_rule across both backends exactly.** The auto-merges (4
  via `NAME_DOB_PHONE`, 1 via `NAME_DOB_EMAIL`, 1 via `SSN_DOB`) confirm the
  reserved rows include genuine real-world duplicates of records in the
  batch population, not just synthetic singletons — a stronger proof than a
  contrived test would have given. mids differ between backends (expected —
  each backend assigns sequence numbers from its own independent history);
  patid/tier/rule agreement is the thing that actually matters and it's exact.
  Bonus sanity check: `dashboard_summary()`/`list_entities()` also ran clean
  on both backends at this scale (SQLite's larger totals are just its
  pre-existing 62,609-record demo history layered under the new run — not a
  discrepancy).

**Environment note.** This machine's `~/Desktop/...` path is iCloud-synced;
mid-session, Docker image/build-cache growth (~28GB at peak, from this
project's builds plus several older unrelated projects — qdrant, marquez,
evidently-ui, astro-nyc-taxi-triggers, fastapi-app, opensearch) triggered
macOS to evict local copies of iCloud-synced files to "dataless" placeholders
— including files inside `.venv/` and even `.git/`, which broke `git` and
`pytest` twice mid-session until sync caught back up. Cleaned up: this
session's own two images + all build cache, plus (with explicit go-ahead)
stopped containers and unused images from other projects — freed disk from
~11GB avail back up to ~26GB avail. `qdrant` (the only running container)
was left untouched. Worth knowing about if it recurs: check `df -h /` and
`ls -lO <file>` for a `dataless` flag before assuming something's actually
broken — `docker system df` + a prune is usually the fix.

---

## 1. FastAPI: incremental single/batch record scoring — DONE

Shipped on `jason/operationalize`. `POST /records/score` (+ `GET
/records/score/{run_id}` to poll, always a background job) scores new records
against the existing population and writes straight into `empi.db`, without a
full `clean → block → rules → cluster` re-run.

Ended up as the "large overhaul" version rather than a per-call rebuild: two
new persisted tables (`store.block_key`, the on-disk mirror of
`blocking.BlockingIndex`; `store.cleaned_attrs`, a SQL-queryable mirror of the
`CleanedRecords` contract), both rebuilt wholesale by every full-pipeline
publish (`publish.py`) and incrementally appended to by each incremental call
(`src/api/incremental.py`) — so candidate lookup is an indexed SQL query, not
an in-memory index rebuilt from the whole population. Auto-merge bridges
previously-separate entities when a new record's matches span more than one
(mirrors `assign_clusters`' union, done directly against `empi.db`), respects
sticky-unmerge (locked candidates get an `entity_suggestion`, not a merge),
and the FS matcher scores the review tier when a model is active.

Tests: `tests/unit/test_incremental.py` (6 cases: auto-merge, bridging,
no-match, review-tier, sticky-unmerge, same-batch discoverability),
`tests/integration/test_api.py::TestRecordsScore`. Docs:
`docs/API-Design.md` §2/§3, `docs/Blocking-Guide.md`.

**Follow-up (also done):** storage is pluggable behind
`src.api.index_backend.IndexBackend` — `SqlIndexBackend` (default,
`empi.db`, adapter takes the store module + connection as parameters so a
future Postgres store module can swap in without touching the adapter — the
db-agnostic scaffolding, not a built multi-engine store) or
`ParquetIndexBackend` (`EMPI_INDEX_BACKEND=parquet`, nine Parquet files under
`local_index_dir`, no DB required — see the operationalization work below,
which brought this from incremental-scoring-only up to full parity). Local-mode
CLI: `python -m src.api.local_score --input record.json`. Tests:
`tests/unit/test_parquet_backend.py`, including the same auto-merge/review
scenarios run against both backends.

---

## 2. Clean up `empi-dashboard/` — DONE

Flattened `web/` up and adopted Next.js's `src/` directory convention via `git mv`
(history preserved — 45 renames):

```
empi-dashboard/
  README.md  package.json  package-lock.json  next.config.ts
  tsconfig.json  eslint.config.mjs  postcss.config.mjs  .gitignore
  src/{app,components,lib}/
  docs/
```

`demo/dashboard-demo.html` deleted. `tsconfig.json`'s `@/*` alias repointed to
`./src/*`; `eslint.config.mjs`/`next.config.ts`/`postcss.config.mjs` needed no
changes (Tailwind v4's zero-config content detection scans the whole project,
no explicit globs to update). No internal imports used relative parent-directory
paths, so the whole `app/components/lib` subtree moved as a unit with nothing to
fix. `package.json`'s `name` updated `web` → `empi-dashboard`.

Root `README.md`, `empi-dashboard/README.md`, and `.claude/launch.json`'s
`empi-web` npm prefix updated off the old `web/` path.
`empi-dashboard/docs/*.md` had no `web/` references to begin with.

Verified: `npm install`, `npm run build` (all 13 routes compiled — App Router
correctly auto-detected `src/app`), `npx tsc --noEmit`, `npm run lint` — all
clean. Did not restart the live dev servers (pre-existing, long-running
processes on 3000/8000 from outside this session) — build/typecheck/lint
passing is sufficient proof for a pure structural move with no logic changes.

---

## 3. Dockerize `empi-service` and `empi-dashboard` (production-ready) — DONE

Deploy target stayed generic/cloud-agnostic per your answer.

**`empi-service/Dockerfile`** — multi-stage (`builder` installs deps with `uv`
into `/opt/venv`, `runtime` copies just the venv + `src/`), `python:3.13-slim`,
non-root `empi` user, `data/`/`models/`/`logs/` created empty (mount volumes
over them — `empi.db`/`local_index/` need to persist), config via env vars
only (nothing `.env`-shaped baked in), `HEALTHCHECK` on `GET /health`. `postal`
(libpostal binding) deliberately excluded from the image — needs the native
library + ~2GB of data files, and `transformations.py` already tolerates its
absence (`Address_normalized` stays NaN). Flagged in the Dockerfile's own
docstring: `jobs.py`'s in-process run/score registries mean this is a
single-replica image today — multi-worker/multi-replica would need a shared
store for run-status polling.

**`empi-dashboard/Dockerfile`** — multi-stage (`deps`/`builder`/`runner`),
`next.config.ts` now sets `output: "standalone"`, `node:22-slim`, non-root
`empi` user, `HEALTHCHECK` on `GET /api/health` (the BFF route, which itself
proxies to the backend's `/health/ready`). No `public/` directory exists in
this project, so that COPY step is omitted (noted inline for whoever adds one
later). Backend URL via `EMPI_API_URL` (already read by
`src/lib/server-api.ts`), never baked in.

**`docker-compose.yml`** at the repo root — named volumes for
`empi-service`'s data/models/logs, dashboard's `depends_on: condition:
service_healthy` on the backend, `EMPI_API_URL=http://empi-service:8000` for
compose-network service-name resolution.

**Verified live**, not just build success: built + ran both images
standalone (`docker run`), confirmed non-root (`whoami` → `empi`) and
`/health`, `/health/ready` return 200 for the backend; then `docker compose
up --build`, confirmed the dashboard container correctly *waited* for the
backend's healthcheck before starting, and hit the dashboard's BFF routes
(`/api/health`, `/api/dashboard/summary`) over the compose network to prove
service-name resolution and the full proxy chain work end-to-end. Full
cleanup after (`docker compose down -v`, image removal).

---

## 4. Research paper — draft sections

**Blocked on you — said this is the last task and you'll share venue/scope
details separately.** Leaving the existing outline as-is below so it's ready
when you're ready.

- [ ] **Introduction / problem statement** — eMPI patient deduplication, motivation, scale (163k records, AllianceChicago). *Needs user context.*
- [ ] **Related work** — record linkage; blocking (phonetic, q-gram, meta-blocking, SNM); deterministic vs probabilistic (Fellegi–Sunter) entity resolution. *Needs user context / reference list.*
- [ ] **Data** — `MDM_Population` schema, field completeness, real-world messiness. Source: `docs/Data-Cleaning-Guide.md`.
- [ ] **Methods — cleaning & normalization.** Source: `docs/Data-Cleaning-Guide.md`.
- [ ] **Methods — blocking** (8-block scheme + recall evaluation). Source: `docs/Blocking-Guide.md`, `Blocking-Recall-RCA.md`.
- [ ] **Methods — deterministic matching** (five rules + three-way match/review/reject). Source: `docs/Deterministic-Rules-Guide.md`.
- [ ] **Methods — evaluation framework** (rules-as-ground-truth, deployment artifact). Source: `src/evaluation/`, `docs/Data-Contract.md`.
- [ ] **Results** — rule precision/calibration; blocking recall (99.5% valid population); rejection-rule analysis (`Rejection-Rules-Analysis.md`); embedding/graph blocking research (`Blocking-Research-Embedding-Graph.md`).
- [ ] **Limitations** — no gold-standard labels; rules-as-ground-truth caveat.
- [ ] **Future work** — Fellegi–Sunter probabilistic matching stage (scoped above); q-gram / graph blocking; SNM; super-string ANN.
- [ ] **Conclusion.**
