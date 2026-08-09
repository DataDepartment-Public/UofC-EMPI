# eMPI Application Architecture

> **Status (2026-08-07):** this describes the built, running system — both
> the FastAPI backend and the `empi-dashboard/` Next.js front-end are
> implemented, not just designed. Where this doc says "will"/"proposed" for
> something that has since shipped differently, [API-Design.md](API-Design.md)
> and the code are authoritative.

End-to-end architecture for the eMPI entity-resolution pipeline as a running
application: a **Python/FastAPI backend** (`empi-service/`) that wraps the
batch pipeline and owns the resolved-output index, and a **Node.js/Next.js
front-end** (`empi-dashboard/`) that gives reviewers the Review Queue,
Patient Registry, Dashboard, and Admin tabs.

> **Companion docs:**
> - [API-Design.md](API-Design.md) — full route + storage-schema detail for the backend.
> - [Dashboard-Guide.md](../../empi-dashboard/docs/Dashboard-Guide.md) — functional/UX spec the front-end implements.

---

## 1. System overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Browser                                                                     │
│    React UI (Review Queue · Dashboard · Patient Registry · Admin)            │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │ HTTPS (JSON), Entra ID SSO in Azure (Easy Auth)               │
┌───────────────┴──────────────────────────────────────────────────────────────┐
│  Node.js  —  Next.js front-end + BFF                                          │
│    • SSR/CSR React app          • Reviewer identity from Easy Auth header     │
│    • Route Handlers = thin BFF  • proxies to FastAPI, injects X-Reviewer-Id   │
└───────────────▲──────────────────────────────────────────────────────────────┘
                │ HTTP (JSON)  — internal network only
┌───────────────┴──────────────────────────────────────────────────────────────┐
│  Python  —  FastAPI (uvicorn)                                                 │
│    • /health  • /runs  • /review-queue  • /records  • /audit                  │
│    • /explanations  • /admin/thresholds • /admin/models                       │
│         │                                  │                                  │
│         ▼ batch artifacts                  ▼ system of record                 │
│   data/*.parquet + RunManifest    SQLite / Postgres / local Parquet index    │
│   (immutable, reproducible)       (entity · review_candidate · audit_log …)  │
└──────────────────────────────────────────────────────────────────────────────┘
```

Three tiers, each with one job:

| Tier | Tech | Responsibility |
|---|---|---|
| **Front-end** | Node.js · Next.js · React · TypeScript | Reviewer UI, Entra ID SSO identity, BFF proxy |
| **Backend API** | Python · FastAPI · uvicorn | Pipeline orchestration, resolved-output DB, audit |
| **Pipeline** (existing) | Python · pandas · pandera | `clean → block → rules → FS(audit) → gate → ML → cluster` (see `docs/Data-Contract.md`) |

The split keeps the data-science code (Python) and the interactive app code
(Node/React) in their native ecosystems, talking over a versioned JSON contract.

---

## 2. Backend architecture (FastAPI)

The backend wraps `src/pipeline.run_pipeline()` (unmodified) and owns the resolved
output. Full detail is in [API-Design.md](API-Design.md); the essentials:

### The pipeline, as it actually runs today

```
1. clean → 2. block → 3. deterministic rules ─┬─► matches (auto-merge)
                                                └─► non_matches (uncertain)
                                                      │
                                                      ▼
                                    4. FS matcher — audit-only: emits
                                       features/score, routes nothing
                                    (fallback gate only when no gate model
                                     is active, or EMPI_GATE_SUPERSEDES_FS=false)
                                                      │
                                                      ▼
                                    4.25. non-match gate — drops confident
                                          non-matches (the only stage that
                                          records what it discarded)
                                                      │
                                                      ▼  plausible survivors only
                                    4.5. ML matcher — classifies: auto_merge
                                         vs human_review (2-tier; cannot emit
                                         no_match — that's the gate's job)
                                                      │
                                                      ▼  ml_feeds_clustering=True
                                                      │  by default: auto_merge
                                                      │  edges union in for real
                                          5. clustering (terminal)
```

**The ML matcher is a real decision stage today, not audit-only** —
`ml_feeds_clustering` defaults `True` (`src/config.py`), so its
`auto_merge`-tier verdicts union directly into Stage 5's clustering edges
alongside the deterministic rules' matches. (The FS matcher, Stage 4, is
the one that's genuinely audit-only — `fs_feeds_clustering` defaults
`False`.) Tune `ml_auto_merge_threshold` against a held-out precision
target accordingly; set `EMPI_ML_FEEDS_CLUSTERING=false` to restore
deterministic-edges-only clustering.

Log lines tag these `[1/7]` through `[7/7]` (`CLEAN`, `BLOCK`, `RULES`,
`MODEL(FS)`, `GATE`, `MODEL(ML)`, `CLUSTER`) — the fractional `4.25`/`4.5`
labels above are the documentation convention, not what appears in logs.
See [ML-Matcher-Integration-Guide.md](ML-Matcher-Integration-Guide.md) §1
and [Nonmatch-Gate-Guide.md](Nonmatch-Gate-Guide.md) for why the gate exists
as a separate stage rather than folding into the FS matcher or ML matcher.

### Storage — two tiers by purpose

| Tier | Holds | Why |
|---|---|---|
| **Parquet + `RunManifest`** | per-stage batch artifacts | columnar, versioned, reproducible |
| **Resolved-output index** | final resolved output, review queue, and audit log | transactional, mutable by reviewers |

The pipeline stays a pure batch function writing Parquet + a manifest. A **publish**
step loads each run's final output into the resolved-output index automatically
(`POST /runs` triggers both in one background job). Reviewer edits
mutate that index directly — never the Parquet artifacts, which remain the
immutable record of "what the algorithm produced for run X."

The resolved-output index itself is **backend-pluggable**
(`src/api/backends/index_backend.py`, `EMPI_INDEX_BACKEND`): **SQLite**
(`empi.db`, local/dev default), **Postgres** (`EMPI_INDEX_BACKEND=postgres`,
AAD-token auth via the backend's own managed identity — the deployed Azure
target, see `terraform/postgres.tf`), or a **local Parquet index**
(`data/local_index/`, `EMPI_INDEX_BACKEND=parquet`) — every route below
works identically against all three, including `/audit/*`. The Parquet
option needs no database server at all. Full schema for all three:
[Data-Contract.md](Data-Contract.md) Stage 6.

```
entity(mid, run_id, origin, is_merged, confidence, match_rule, evidence, updated_utc)
entity_member(patid, mid, is_primary, added_by, updated_utc)
review_candidate(patid_a, patid_b, match_rule, confidence, ..., ml_match_probability,
                  ml_classification_tier)
audit_log(id, ts_utc, user, action, patids, mid, prev_state, next_state, run_id,
          related_patids, prev_mid, undo_of, undone)
```

(Plus `record_attrs`/`record_raw` for display, `entity_suggestion` for
sticky-unmerge, and `block_key`/`cleaned_attrs` for incremental-scoring
lookups — see Data-Contract.md Stage 6 for the full picture.)

### Routes

| Route | Purpose |
|---|---|
| `GET /health`, `/health/ready` | liveness; readiness (index reachable + dirs writable, last run id) |
| `POST /runs` | save upload → `run_pipeline` + publish, both via **BackgroundTasks** → `202 {run_id}` |
| `GET /runs`, `/runs/{id}` | poll status; return `RunManifest` |
| `GET /review-queue` | candidate-grain pending-pair queue, sorted/filtered on rule confidence falling back to ML score |
| `GET /records`, `/clusters/{mid}`, `/records/{patid}/raw`, `/records/{patid}/ssn-clean` | read models + PHI reads for the Patient Registry (raw and SSN-reveal are two separate endpoints, logged as two separate audit actions) |
| `POST /records/score`, `GET /records/score/{run_id}` | incremental single/few-record scoring, no full re-run |
| `GET /dashboard/summary` | KPIs for the Dashboard tab |
| `POST /audit/merge`, `/unmerge`, `/dismiss`, `/{id}/undo` | mutate `entity`/`entity_member`/`review_candidate` + insert audit row, **one transaction** |
| `GET /audit` | audit feed |
| `GET /explanations/{model}/{a}/{b}` | per-pair SHAP waterfall for the gate's or ML matcher's decision |
| `GET`/`PUT /admin/thresholds` | live-tunable gate/ML/FS decision thresholds |
| `POST /admin/models/reload`, `GET /admin/models/status` | model hot-swap (CI-facing, not reviewer-facing) |

### Module layout

```
src/api/
  main.py              # app, lifespan (init db, configure_logging once)
  deps.py              # get_settings, get_db, get_backend (+ the Parquet-mode lock)
  schemas.py           # request/response models; some reuse contracts.RunManifest
  jobs.py              # background job wrappers + status registries
  threshold_store.py   # live-tunable decision thresholds
  backends/
    index_backend.py   # the pluggable-storage seam (IndexBackend protocol)
    sql_backend.py      # SQLite implementation
    postgres_backend.py # Postgres implementation (AAD auth, deployed target)
    parquet_backend.py  # local Parquet implementation
  ingest/
    publish.py / publish_local.py     # batch: run output -> index (reuses assign_clusters)
    incremental.py / local_score.py   # single/few-record scoring -> index
  routers/{health,runs,records,audit,dashboard,admin,explanations}.py

src/models/
  deterministic_rules/  # rule engine (3 active rules — see Deterministic-Rules-Guide.md)
  fs_matcher/            # Fellegi-Sunter — audit-only (Stage 4)
  nonmatch_gate/         # confident-non-match filter (Stage 4.25)
  ml_matcher/             # LightGBM v5 confident-match classifier (Stage 4.5)
  model_cache.py          # mtime-keyed cache for all three models' artifacts
```

`run_pipeline()` is **not modified** by any of this. Deps: `fastapi`,
`uvicorn[standard]`, `python-multipart`; SQLite is stdlib, Postgres via
`psycopg`, Parquet via `pyarrow`.

### Reconciliation (sticky unmerge)

Because the resolved-output index is the system of record and the pipeline is
the algorithm's opinion, **a re-run must not clobber reviewer edits.** Publish
is a *merge*, not a *truncate*: untouched entities upsert from the run;
reviewer-edited entities keep the human decision (the run's grouping becomes a
non-binding suggestion in `entity_suggestion`); new PATIDs are placed by the
algorithm. An explicit unmerge is sticky until reversed via a later merge or
`POST /audit/{id}/undo` — this holds on all three backends.

This reconciliation is per-PATID upsert, not a dataset-level replace — it
does not detect or clean up a wholesale population swap (e.g. a different
source file with an entirely different PATID scheme published on top of an
existing index). See `API-Design.md` §2's reconciliation caveat.

---

## 3. Front-end architecture (Node.js)

A **Next.js** (App Router, React, TypeScript) application running on Node.js, in
`empi-dashboard/`. Next.js gives us one Node process that serves the React UI
**and** a thin Backend-for-Frontend (BFF) layer via Route Handlers — so the
browser never talks to FastAPI directly. That BFF is where reviewer identity
lives, which cleanly answers the "who is the `user` in the audit log" question.

### Layout

```
empi-dashboard/src/
  app/
    layout.tsx, providers.tsx   # shell, branding (Alliance-Chicago-Branding.md)
    page.tsx                    # Dashboard tab (KPIs, charts)
    review/page.tsx             # Review Queue tab (list + detail, merge/undo, inline SHAP)
    dataset/page.tsx            # Patient Registry tab (records, clusters, unmerge)
    dataset/[mid]/explain/      # standalone per-pair explanation view, reached from
                                 #   the Patient Registry — same SHAP waterfall + feature
                                 #   comparison as the Review Queue's inline panel
    admin/page.tsx               # Admin tab (live thresholds, model/system status)
    api/                        # ── BFF: Route Handlers proxy to FastAPI ──
      runs/, records/[patid]/{raw,ssn-clean}/, clusters/[mid]/,
      review-queue/, audit/{merge,unmerge,dismiss,[id]/undo}/,
      explanations/[model]/[a]/[b]/, admin/thresholds/, dashboard/summary/, health/
  lib/
    server-api.ts                # server-only fetch wrapper (BFF -> FastAPI); reviewerId()
                                 #   reads Easy Auth's X-MS-CLIENT-PRINCIPAL-NAME in Azure,
                                 #   falls back to a hardcoded identity in local dev
    api-client.ts                # browser-side fetch wrapper
    schemas.ts                   # zod models mirroring src/api/schemas.py
    hooks.ts                     # TanStack Query hooks
    compare.ts, explain.ts, format.ts
  components/
    dashboard/    KpiCard.tsx  MatchStatusChart.tsx  TrendChart.tsx  ModelInfoPanel.tsx
    review/       ReviewQueueList.tsx  ReviewCandidateDetail.tsx  PipelineTrail.tsx
                  MergeModal.tsx  ManualMatchModal.tsx
    dataset/      DatasetFilters.tsx  DatasetRow.tsx  UnmergeModal.tsx  StatusBadge.tsx
    admin/        SystemStatusPanel.tsx
    shared/       TopNav.tsx  AuditLog.tsx  RawDataDrawer.tsx  SsnReveal.tsx
                  FeatureComparisonTable.tsx  ShapWaterfall.tsx  Toast.tsx
```

### Responsibilities by layer

| Layer | Does | Does not |
|---|---|---|
| **React components** | render tables/KPIs/modals, optimistic UI on merge | hold business truth, call FastAPI directly |
| **TanStack Query** (`lib/hooks.ts`) | fetch/cache, poll run status until `succeeded`, invalidate `records`/`review-queue`/`audit` after a merge | — |
| **BFF Route Handlers** (`app/api/*`) | attach reviewer identity, proxy to FastAPI via `server-api.ts`, hide the internal URL | run the pipeline, store data |
| **FastAPI** | all pipeline + data + audit logic | rendering, sessions |

### Request flows

**Run new data** — upload → `POST /api/runs` (BFF) → `POST /runs` (FastAPI) →
`202 {run_id}`. UI polls the run-status route; TanStack Query refetches every
few seconds until status is `succeeded`, then invalidates the review-queue/records
queries.

**Merge a pair** — reviewer confirms in `MergeModal` (Review Queue) → `POST
/api/audit/merge` (BFF injects `X-Reviewer-Id` from the authenticated session) →
`POST /audit/merge` (FastAPI: one backend transaction, entity write + audit row) →
on success, invalidate `review-queue` + `records` + `audit`; on error, toast.
`unmerge`/`dismiss`/`undo` mirror this via their own route handlers.

**Model Explanation** — the ML matcher's and non-match gate's real,
persisted SHAP contribution vectors render in two places: inline in the
Review Queue's `ReviewCandidateDetail`/`PipelineTrail` panel as a reviewer
steps through the queue, and on a standalone page
(`dataset/[mid]/explain/`) reached by clicking a match from the Patient
Registry. Both show the deterministic-rule feature comparison
(`FeatureComparisonTable.tsx`) plus, when a model actually scored the pair,
a real `ShapWaterfall.tsx` waterfall and match-probability/tier readout
from `GET /explanations/{model}/{a}/{b}` — `null` (shown honestly, not
fabricated) only for a pair no model ever scored, e.g. one resolved purely
by a deterministic rule, or one the gate dropped before the ML matcher saw
it.

### Identity / auth

The BFF owns identity. In a real Azure deploy, App Service's built-in
authentication ("Easy Auth", `auth_settings_v2` — see `terraform/auth.tf`)
requires Entra ID sign-in before any request reaches the container; the BFF
reads the authenticated principal from Easy Auth's
`X-MS-CLIENT-PRINCIPAL-NAME` header (`server-api.ts`'s `reviewerId()`) and
forwards it as `X-Reviewer-Id` on every `/audit/*` call and every PHI-read
call (`.../raw`, `.../ssn-clean`). Locally (`docker-compose`, `npm run
dev`), there is no Azure platform in front of the container, so that header
is never present and the code falls back to a hardcoded identity — no local
dev workflow change required. The browser never sets `X-Reviewer-Id`
itself either way, so the audit trail can't be spoofed from the client.
Note this is app-level: the FastAPI `/admin/*` routes themselves carry no
role-based authorization internally — the backend has no public ingress at
all in Azure (network isolation is its access control), which is why it
doesn't need its own sign-in flow.

---

## 4. Cross-cutting concerns

- **Contract sync** — the backend is the source of truth. Mirror its request/response
  shapes as `zod` schemas in `empi-dashboard/src/lib/schemas.ts` (or generate a typed
  client from the FastAPI OpenAPI doc) so a backend change surfaces as a TypeScript error.
- **PHI / HIPAA** — keep the backend's "aggregate counts only" logging rule.
  PHI flows browser ↔ BFF ↔ FastAPI over TLS only; do not log field values on the Node
  side either. Identity is server-derived, never client-asserted. Every PHI read
  (`.../raw`, `.../ssn-clean`) is audit-logged, not just mutations.
- **Long-running runs** — never block a request: backend uses `BackgroundTasks`, the
  front-end polls. `configure_logging` is called **once** in the FastAPI lifespan
  (its `_LOGGING_CONFIGURED` guard makes `run_pipeline`'s re-call a no-op).
- **Config** — backend via `EMPI_`-prefixed env (existing `pydantic-settings`);
  front-end via `.env` (`EMPI_API_URL` for the BFF → FastAPI base URL).

---

## 5. Deployment topology

```
            ┌────────────────┐      ┌──────────────┐      ┌────────────────────┐
  browser ──▶│ Node (Next)    │ ────▶│ FastAPI      │ ────▶│ Postgres (Azure) or │
   Entra ID  │  empi-dashboard│ VNet │  src/api/    │      │ empi.db, plus       │
   sign-in   │  public HTTPS  │priv. │  no public   │      │ data/*.parquet on   │
            └────────────────┘ endpt│  ingress      │      │ Azure Files         │
                                    └──────────────┘      └────────────────────┘
```

Two services (Node + FastAPI) plus the resolved-output index (Postgres in
Azure; SQLite file or local Parquet directory locally) and the `data/` tree
on a shared volume/file share. In Azure: the backend has **no public
ingress at all**, reachable only from the dashboard's own VNet integration
— see `terraform/README.md`'s "Network isolation & encryption" section for
the full private-networking picture. Locally: two `docker compose`
services, or `npm run dev` + `uvicorn` side by side.

---

## 6. Build order (as built, historical)

1. **Backend slice** — `sql_backend.py` + `publish.py`, then `health` + `runs` routes
   over the unmodified pipeline.
2. **Read models** — `records`/`clusters`/`dashboard` routes.
3. **Front-end shell** — Next.js app (`empi-dashboard/`), branding, tabs
   reading `GET /records` / `GET /dashboard/summary`.
4. **Audit** — backend `audit/merge|unmerge` transactions + sticky-unmerge
   reconciliation in `publish.py`; front-end `MergeModal` + audit table.
5. **Model Explanation** — real SHAP-backed explanations once the non-match
   gate and ML matcher shipped, superseding the earlier deterministic-rule-
   only comparison page.
6. **Operationalization** — incremental single/few-record scoring, a fully
   pluggable storage backend (SQLite/Postgres/Parquet) covering batch
   publish, incremental scoring, the dashboard read side, and audit, plus
   the Review Queue tab, live-tunable admin thresholds, and undo — see
   [API-Design.md](API-Design.md) and [Data-Contract.md](Data-Contract.md)
   Stage 6.
