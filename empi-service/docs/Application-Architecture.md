# eMPI Application Architecture

> **Status (2026-07-14):** this describes the built, running system — both
> the FastAPI backend and the `empi-dashboard/` Next.js front-end are
> implemented, not just designed. Where this doc says "will"/"proposed" for
> something that has since shipped differently, [API-Design.md](API-Design.md)
> and the code are authoritative.

End-to-end architecture for the eMPI entity-resolution pipeline as a running
application: a **Python/FastAPI backend** (`empi-service/`) that wraps the
batch pipeline and owns the resolved-output index, and a **Node.js/Next.js
front-end** (`empi-dashboard/`) that gives reviewers the dashboard for running
data, reviewing matches, and merging/unmerging records.

> **Companion docs:**
> - [API-Design.md](API-Design.md) — full route + storage-schema detail for the backend.
> - [Dashboard-Guide.md](../../empi-dashboard/docs/Dashboard-Guide.md) — functional/UX spec the front-end implements.

---

## 1. System overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Browser                                                                     │
│    React UI (Dashboard tab · Dataset tab · Model Explanation sub-page)       │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │ HTTPS (JSON)
┌───────────────┴──────────────────────────────────────────────────────────────┐
│  Node.js  —  Next.js front-end + BFF                                          │
│    • SSR/CSR React app          • Auth/session (reviewer identity)            │
│    • Route Handlers = thin BFF  • proxies to FastAPI, injects X-Reviewer-Id   │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │ HTTP (JSON)  — internal network only
┌───────────────┴──────────────────────────────────────────────────────────────┐
│  Python  —  FastAPI (uvicorn)                                                 │
│    • /health   • /runs (BackgroundTasks → run_pipeline)   • /audit  • /records│
│         │                                  │                                  │
│         ▼ batch artifacts                  ▼ system of record                 │
│   data/*.parquet + RunManifest    SQLite empi.db OR local Parquet index      │
│   (immutable, reproducible)       (entity · entity_member · audit_log · ...) │
└──────────────────────────────────────────────────────────────────────────────┘
```

Three tiers, each with one job:

| Tier | Tech | Responsibility |
|---|---|---|
| **Front-end** | Node.js · Next.js · React · TypeScript | Reviewer UI, identity/session, BFF proxy |
| **Backend API** | Python · FastAPI · uvicorn | Pipeline orchestration, resolved-output DB, audit |
| **Pipeline** (existing) | Python · pandas · pandera | `clean → block → rules → [FS matcher] → [ML matcher] → cluster` (see `docs/Data-Contract.md`) |

The split keeps the data-science code (Python) and the interactive app code
(Node/React) in their native ecosystems, talking over a versioned JSON contract.

---

## 2. Backend architecture (FastAPI)

The backend wraps `src/pipeline.run_pipeline()` (unmodified) and owns the resolved
output. Full detail is in [API-Design.md](API-Design.md); the essentials:

### Storage — two tiers by purpose

| Tier | Holds | Why |
|---|---|---|
| **Parquet + `RunManifest`** | per-stage batch artifacts | columnar, versioned, reproducible |
| **Resolved-output index** | final resolved output + audit log | transactional, mutable by reviewers |

The pipeline stays a pure batch function writing Parquet + a manifest. A **publish**
step loads each run's final output into the resolved-output index. Reviewer edits
mutate that index directly — never the Parquet artifacts, which remain the
immutable record of "what the algorithm produced for run X."

The resolved-output index itself is **backend-pluggable**
(`src/api/backends/index_backend.py`, `EMPI_INDEX_BACKEND`): **SQLite** (`empi.db`,
default) or a **local Parquet index** (`data/local_index/`,
`EMPI_INDEX_BACKEND=parquet`) — every route below works identically against
either, including `/audit/*`. The Parquet option needs no database server at
all, which is what makes `python -m src.api.ingest.local_score` / `python -m
src.api.ingest.publish_local` possible with no FastAPI process running either. Full
schema for both: [Data-Contract.md](Data-Contract.md) Stage 6.

```
entity(mid, run_id, origin, is_merged, confidence, updated_utc)
entity_member(patid, mid, is_primary, added_by, updated_utc)
audit_log(id, ts_utc, user, action, patids, mid, prev_state, next_state, run_id)
```

(Plus `record_attrs`/`record_raw` for display, `review_candidate`/
`entity_suggestion` for the review queue and sticky-unmerge, and `block_key`/
`cleaned_attrs` for incremental-scoring lookups — see Data-Contract.md Stage 6
for the full picture.)

### Routes

| Route | Purpose |
|---|---|
| `GET /health`, `/health/ready` | liveness; readiness (index reachable + dirs writable, last run id) |
| `POST /runs` | save upload → `run_pipeline` via **BackgroundTasks** → `202 {run_id}` |
| `GET /runs`, `/runs/{id}` | poll status; return `RunManifest` |
| `GET /records`, `/clusters/{mid}`, `/records/{patid}/raw` | read models for the Dataset tab |
| `POST /records/score`, `GET /records/score/{run_id}` | incremental single/few-record scoring, no full re-run |
| `GET /dashboard/summary` | KPIs for the Dashboard tab |
| `POST /audit/merge`, `/audit/unmerge` | mutate `entity`/`entity_member` + insert audit row, **one transaction** |
| `GET /audit` | audit feed |

### Module layout

```
src/api/
  main.py              # app, lifespan (init db, configure_logging once)
  deps.py              # get_settings, get_db, get_backend (+ the Parquet-mode lock)
  schemas.py           # request/response models; some reuse contracts.RunManifest
  jobs.py              # background job wrappers + status registries
  backends/
    index_backend.py  # the pluggable-storage seam (IndexBackend protocol)
    sql_backend.py     # SQLite implementation
    parquet_backend.py # local Parquet implementation
  ingest/
    publish.py / publish_local.py     # batch: run output -> index (reuses assign_clusters)
    incremental.py / local_score.py   # single/few-record scoring -> index
  routers/{health,runs,records,audit,dashboard}.py
```

`run_pipeline()` is **not modified** by any of this. Deps: `fastapi`,
`uvicorn[standard]`, `python-multipart`; SQLite is stdlib, Parquet via `pyarrow`.

### Reconciliation (sticky unmerge)

Because the resolved-output index is the system of record and the pipeline is
the algorithm's opinion, **a re-run must not clobber reviewer edits.** Publish
is a *merge*, not a *truncate*: untouched entities upsert from the run;
reviewer-edited entities keep the human decision (the run's grouping becomes a
non-binding suggestion in `entity_suggestion`); new PATIDs are placed by the
algorithm. An explicit unmerge is sticky until reversed — this holds on both
backends.

---

## 3. Front-end architecture (Node.js)

A **Next.js** (App Router, React, TypeScript) application running on Node.js, in
`empi-dashboard/`. Next.js gives us one Node process that serves the React UI
**and** a thin Backend-for-Frontend (BFF) layer via Route Handlers — so the
browser never talks to FastAPI directly. That BFF is where reviewer identity
and session live, which cleanly answers the "who is the `user` in the audit
log" question.

### Layout

```
empi-dashboard/src/
  app/
    layout.tsx, providers.tsx   # shell, branding (Alliance-Chicago-Branding.md)
    page.tsx                    # Dashboard tab (KPIs, charts)
    dataset/page.tsx            # Dataset tab (records, clusters, merge/unmerge)
    dataset/[mid]/explain/      # Model Explanation sub-page
    api/                        # ── BFF: Route Handlers proxy to FastAPI ──
      runs/route.ts, runs/[runId]/route.ts
      records/route.ts, records/[patid]/raw/route.ts
      clusters/[mid]/route.ts
      audit/route.ts, audit/merge/route.ts, audit/unmerge/route.ts
      dashboard/summary/route.ts
      health/route.ts
  lib/
    server-api.ts                # server-only fetch wrapper (BFF -> FastAPI)
    api-client.ts                # browser-side fetch wrapper
    schemas.ts                   # zod models mirroring src/api/schemas.py
    hooks.ts                     # TanStack Query hooks
    compare.ts, explain.ts, format.ts
  components/
    TopNav.tsx  KpiCard.tsx  MatchStatusChart.tsx  TrendChart.tsx
    DatasetFilters.tsx  DatasetRow.tsx  MergeModal.tsx  StatusBadge.tsx
    RawDataDrawer.tsx  FeatureComparisonTable.tsx  ModelInfoPanel.tsx  Toast.tsx
```

### Responsibilities by layer

| Layer | Does | Does not |
|---|---|---|
| **React components** | render tables/KPIs/modals, optimistic UI on merge | hold business truth, call FastAPI directly |
| **TanStack Query** (`lib/hooks.ts`) | fetch/cache, poll run status until `succeeded`, invalidate `records`/`audit` after a merge | — |
| **BFF Route Handlers** (`app/api/*`) | attach reviewer identity, proxy to FastAPI via `server-api.ts`, hide the internal URL | run the pipeline, store data |
| **FastAPI** | all pipeline + data + audit logic | rendering, sessions |

### Request flows

**Run new data** — upload → `POST /api/runs` (BFF) → `POST /runs` (FastAPI) →
`202 {run_id}`. UI polls the run-status route; TanStack Query refetches every
few seconds until status is `succeeded`, then invalidates the records query so
the Dataset tab refreshes.

**Merge a cluster** — reviewer confirms in `MergeModal` → `POST
/api/audit/merge` (BFF injects `X-Reviewer-Id` from session) → `POST
/audit/merge` (FastAPI: one backend transaction, entity write + audit row) →
on success, invalidate `records` + `audit`; on error, toast. `unmerge` mirrors
this via `audit/unmerge/route.ts`.

**Model Explanation** — clicking a candidate's name inside an expanded
Dataset row navigates to `dataset/[mid]/explain/`, which shows the real
per-pair deterministic-rule feature comparison (`FeatureComparisonTable.tsx`,
`lib/explain.ts`/`compare.ts`). It deliberately does **not** show a
Fellegi-Sunter waterfall or probability — the FS matcher (Stage 4) is built
and in production, but runs as an audit-only candidate/feature generator for
a future GBT, not a scored decision on any reviewed pair (see
`docs/FS-Matcher-Production-Guide.md`) — so the page shows what's actually
true instead of fabricating a number.

### Identity / auth

The BFF owns identity. A lightweight session supplies the reviewer id, which
the BFF forwards as `X-Reviewer-Id` on every `/audit/*` call. The browser
never sets that header itself — so the audit trail can't be spoofed from the
client. For a real deploy this slot takes the org's SSO/OIDC with no change
to the FastAPI contract.

---

## 4. Cross-cutting concerns

- **Contract sync** — the backend is the source of truth. Mirror its request/response
  shapes as `zod` schemas in `empi-dashboard/src/lib/schemas.ts` (or generate a typed
  client from the FastAPI OpenAPI doc) so a backend change surfaces as a TypeScript error.
- **PHI / HIPAA** — keep the backend's "aggregate counts only" logging rule.
  PHI flows browser ↔ BFF ↔ FastAPI over TLS only; do not log field values on the Node
  side either. Identity is server-derived, never client-asserted.
- **Long-running runs** — never block a request: backend uses `BackgroundTasks`, the
  front-end polls. `configure_logging` is called **once** in the FastAPI lifespan
  (its `_LOGGING_CONFIGURED` guard makes `run_pipeline`'s re-call a no-op).
- **Config** — backend via `EMPI_`-prefixed env (existing `pydantic-settings`);
  front-end via `.env` (`EMPI_API_URL` for the BFF → FastAPI base URL).

---

## 5. Deployment topology

```
            ┌────────────────┐      ┌──────────────┐      ┌────────────────────┐
  browser ──▶│ Node (Next)    │ ────▶│ FastAPI      │ ────▶│ empi.db (or local  │
   :3000     │  empi-dashboard│ :8000│  src/api/    │      │ Parquet index) +   │
            └────────────────┘      └──────────────┘      │ data/*.parquet     │
                                                            └────────────────────┘
```

Two services (Node + FastAPI) plus the resolved-output index (SQLite file or
local Parquet directory) and the `data/` tree on a shared volume — see
`docker-compose.yml` at the repo root. Local: two `docker compose` services,
or `npm run dev` + `uvicorn` side by side. Single-node is sufficient for the
capstone; the only stateful piece is the `data/` volume.

---

## 6. Build order (as built)

1. **Backend slice** — `sql_backend.py` + `publish.py`, then `health` + `runs` routes
   over the unmodified pipeline.
2. **Read models** — `records`/`clusters`/`dashboard` routes.
3. **Front-end shell** — Next.js app (`empi-dashboard/`), branding, two tabs
   reading `GET /records` / `GET /dashboard/summary`.
4. **Audit** — backend `audit/merge|unmerge` transactions + sticky-unmerge
   reconciliation in `publish.py`; front-end `MergeModal` + audit table.
5. **Model Explanation** sub-page (`dataset/[mid]/explain/`) — ships against
   the deterministic-rule feature comparison; see §3's note on why it doesn't
   show an FS/Fellegi-Sunter waterfall.
6. **Operationalization** — incremental single/few-record scoring
   (`POST /records/score`), and a fully pluggable local-Parquet storage
   backend covering batch publish, incremental scoring, the dashboard read
   side, and audit — see [API-Design.md](API-Design.md) and
   [Data-Contract.md](Data-Contract.md) Stage 6.
