# eMPI Application Architecture

End-to-end architecture for productizing the eMPI entity-resolution pipeline as a
running application: a **Python/FastAPI backend** that wraps the existing batch
pipeline and owns the resolved-output database, and a **Node.js/Next.js front-end**
that gives reviewers the dashboard for running data, reviewing matches, and
merging/unmerging records.

> **Companion docs:**
> - [API-Design.md](API-Design.md) — full route + SQLite schema detail for the backend.
> - [Dashboard-Guide.md](Dashboard-Guide.md) — functional/UX spec the front-end implements.
> - [demo/dashboard-demo.html](../demo/dashboard-demo.html) — interactive mock of the UI.

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
│   data/*.parquet + RunManifest        SQLite  empi.db                         │
│   (immutable, reproducible)           (entity · entity_member · audit_log)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

Three tiers, each with one job:

| Tier | Tech | Responsibility |
|---|---|---|
| **Front-end** | Node.js · Next.js · React · TypeScript | Reviewer UI, identity/session, BFF proxy |
| **Backend API** | Python · FastAPI · uvicorn | Pipeline orchestration, resolved-output DB, audit |
| **Pipeline** (existing) | Python · pandas · pandera | `clean → block → rules → cluster`, unchanged |

The split keeps the data-science code (Python) and the interactive app code
(Node/React) in their native ecosystems, talking over a versioned JSON contract.

---

## 2. Backend architecture (FastAPI)

The backend wraps `src/pipeline.run_pipeline()` (unmodified) and owns the resolved
output. Full detail is in [API-Design.md](API-Design.md); the essentials:

### Storage — two tiers by purpose

| Tier | Holds | Why |
|---|---|---|
| **Parquet + `RunManifest`** (today) | per-stage batch artifacts | columnar, versioned, reproducible |
| **SQLite `empi.db`** (new) | final resolved output + audit log | transactional, concurrent-safe, mutable by reviewers |

The pipeline stays a pure batch function writing Parquet + a manifest. A **publish**
step loads each run's final output into `empi.db`. Reviewer edits mutate `empi.db`
directly — never the Parquet artifacts, which remain the immutable record of "what
the algorithm produced for run X."

```
entity(mid, run_id, origin, is_merged, confidence, updated_utc)
entity_member(patid, mid, is_primary, added_by, updated_utc)
audit_log(id, ts_utc, user, action, patids, mid, prev_state, next_state, run_id)
```

`audit_log` maps 1:1 onto the demo's `AUDIT` array, so the dashboard's audit table
renders unchanged.

### Routes

| Route | Purpose |
|---|---|
| `GET /health`, `/health/ready` | liveness; readiness (db + dirs writable, last run id) |
| `POST /runs` | save upload → `run_pipeline` via **BackgroundTasks** → `202 {run_id}` |
| `GET /runs`, `/runs/{id}` | poll status; return `RunManifest` |
| `GET /records`, `/clusters/{mid}` | read models for the Dataset tab |
| `POST /audit/merge`, `/audit/unmerge` | mutate `entity`/`entity_member` + insert audit row, **one transaction** |
| `GET /audit` | audit feed |

### Module layout

```
src/api/
  main.py        # app, lifespan (init db, configure_logging once)
  deps.py        # get_settings, get_db
  schemas.py     # request models; responses reuse contracts.RunManifest
  jobs.py        # background run wrapper + status registry
  store.py       # SQLite layer (entity/member upsert, audit insert)
  publish.py     # Parquet run output → empi.db (reuses assign_clusters)
  routers/{health,runs,audit,records}.py
```

`run_pipeline()` is **not modified**. New deps: `fastapi`, `uvicorn[standard]`,
`python-multipart`. SQLite is stdlib.

### Reconciliation (the one real design risk)

Because `empi.db` is the system of record and the pipeline is the algorithm's
opinion, **a re-run must not clobber reviewer edits.** Publish is a *merge*, not a
*truncate*: untouched entities upsert from the run; reviewer-edited entities keep the
human decision (the run's grouping becomes a non-binding suggestion); new PATIDs are
placed by the algorithm. An explicit unmerge is sticky until reversed.

---

## 3. Front-end architecture (Node.js)

A **Next.js** (App Router, React, TypeScript) application running on Node.js. Next.js
gives us one Node process that serves the React UI **and** a thin
Backend-for-Frontend (BFF) layer via Route Handlers — so the browser never talks to
FastAPI directly. That BFF is where reviewer identity and session live, which cleanly
answers the "who is the `user` in the audit log" question the backend leaves open.

### Why Next.js over the vanilla demo

The demo (`dashboard-demo.html`) is one static file with in-memory state — perfect as
a mock, but it can't persist, authenticate, or talk to an API. Next.js keeps the same
two-tab IA while adding: server-side auth/session, typed API calls, data caching with
background polling for long-running runs, and code-splitting for the Model
Explanation sub-page.

### Layout

```
web/
  package.json                 # next, react, typescript, @tanstack/react-query, zod
  app/
    layout.tsx                 # top nav, branding (Alliance-Chicago-Branding.md)
    page.tsx                   # Dashboard tab (KPIs, charts)
    dataset/page.tsx           # Dataset tab (records, clusters, merge/unmerge)
    dataset/[mid]/page.tsx     # Model Explanation sub-page
    api/                       # ── BFF: Route Handlers proxy to FastAPI ──
      runs/route.ts            #   POST/GET  → FastAPI /runs
      audit/route.ts           #   POST      → FastAPI /audit/* (+ X-Reviewer-Id)
      records/route.ts         #   GET       → FastAPI /records
      health/route.ts
  lib/
    api-client.ts              # typed fetch wrapper, base URL from env
    schemas.ts                 # zod models mirroring the backend contract
    auth.ts                    # session → reviewer identity
  components/
    DatasetTable.tsx  MergeModal.tsx  AuditLog.tsx  KpiCards.tsx  ShapPanel.tsx
```

### Responsibilities by layer

| Layer | Does | Does not |
|---|---|---|
| **React components** | render tables/KPIs/modals, optimistic UI on merge | hold business truth, call FastAPI directly |
| **TanStack Query** | fetch/cache, poll `GET /runs/{id}` until `succeeded`, invalidate `records`/`audit` after a merge | — |
| **BFF Route Handlers** | attach reviewer identity, proxy to FastAPI, hide the internal URL | run the pipeline, store data |
| **FastAPI** | all pipeline + data + audit logic | rendering, sessions |

### Request flows

**Run new data** — upload → `POST /api/runs` (BFF) → `POST /runs` (FastAPI) →
`202 {run_id}`. UI polls `GET /api/runs/{id}`; TanStack Query refetches every few
seconds until status is `succeeded`, then invalidates the records query so the Dataset
tab refreshes.

**Merge a cluster** — reviewer confirms in `MergeModal` → optimistic UI update →
`POST /api/audit/merge` (BFF injects `X-Reviewer-Id` from session) →
`POST /audit/merge` (FastAPI: transactional DB write + audit row) → on success,
invalidate `records` + `audit`; on error, roll back the optimistic update and toast.
`unmerge` mirrors this. These map directly onto the demo's `confirmMergeBtn`,
`unmerge`, and `unmergeRecord` handlers.

### Identity / auth

The BFF owns identity. For the **local build**, a lightweight session (e.g. a signed
cookie, or NextAuth with a dev credentials provider) supplies the reviewer id, which
the BFF forwards as `X-Reviewer-Id` on every `/audit/*` call. The browser never sets
that header itself — so the audit trail can't be spoofed from the client. For a real
deploy this slot takes the org's SSO/OIDC with no change to the FastAPI contract.

---

## 4. Cross-cutting concerns

- **Contract sync** — the backend is the source of truth. Mirror its request/response
  shapes as `zod` schemas in `web/lib/schemas.ts` (or generate a typed client from the
  FastAPI OpenAPI doc) so a backend change surfaces as a TypeScript error.
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
            ┌─────────────┐        ┌──────────────┐        ┌──────────────┐
  browser ──▶│ Node (Next) │ ─────▶ │ FastAPI      │ ─────▶ │ empi.db +    │
   :3000     │  web/       │  :8000 │  src/api/    │        │ data/*.parquet│
            └─────────────┘        └──────────────┘        └──────────────┘
```

Two services (Node + FastAPI) plus a SQLite file and the `data/` tree on a shared
volume. Local: two `docker compose` services, or `npm run dev` + `uvicorn` side by
side. Single-node is sufficient for the capstone; the only stateful piece is the
`data/` volume + `empi.db`.

---

## 6. Build order

1. **Backend slice** — `store.py` + `publish.py` (test against an existing
   `data/runs/run_*.json`), then `health` + `runs` routes over the unmodified pipeline.
2. **Read models** — `records`/`clusters` routes; backend ready to render.
3. **Front-end shell** — Next.js app, branding, two tabs reading `GET /records`.
4. **Audit** — backend `audit/merge|unmerge` transactions + reconciliation in
   `publish.py`; front-end `MergeModal` + optimistic mutations + audit table.
5. **Model Explanation** sub-page once the probabilistic stage lands.
