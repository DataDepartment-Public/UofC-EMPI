# eMPI — Entity Resolution / Master Patient Index

Capstone project for **AllianceChicago**: group records from the `MDM_Population`
dataset that belong to the same patient across source systems, and give reviewers
a dashboard to inspect and correct the matches.

This repo combines the two halves of the application into one place:

```
empi-service/     Python backend — pipeline + FastAPI service
empi-dashboard/    Next.js frontend — reviewer dashboard
```

## Architecture at a glance

> Full Azure deployment topology + entity-resolution pipeline diagrams (Mermaid, presentation-ready): [`docs/Architecture-Diagram.md`](docs/Architecture-Diagram.md).

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser                                                            │
│    React UI (Dashboard tab · Dataset tab · Model Explanation)       │
└───────────────▲──────────────────────────────────────────────────────┘
                │ HTTPS (JSON)
┌───────────────┴──────────────────────────────────────────────────────┐
│  empi-dashboard  —  Next.js frontend + BFF                            │
│    • SSR/CSR React app          • Route Handlers (app/api/*) proxy    │
│    • reviewer identity/session    to FastAPI, inject X-Reviewer-Id    │
└───────────────▲──────────────────────────────────────────────────────┘
                │ HTTP (JSON) — internal only
┌───────────────┴──────────────────────────────────────────────────────┐
│  empi-service/src/api  —  FastAPI (uvicorn)                           │
│    • /health   • /runs (BackgroundTasks → run_pipeline)               │
│    • /records  • /audit (merge/unmerge, system of record)             │
│         │                                  │                          │
│         ▼ batch artifacts                  ▼ resolved output          │
│   data/*.parquet + RunManifest    SQLite empi.db  OR  local Parquet   │
│                                    (EMPI_INDEX_BACKEND, pluggable)    │
└──────────────────────────────────────────────────────────────────────┘
                │
                ▼
  empi-service/src — clean → block → rules → cluster (the entity-resolution
  pipeline itself, orchestrated by src/pipeline.py, unchanged by the API layer)
```

## What's in each folder

### `empi-service/`

The Python backend: the entity-resolution pipeline and the FastAPI service that
wraps it.

- `src/pipeline.py` — orchestrates the full `clean → block → rules → cluster` run.
- `src/preprocessing/` — data cleaning and blocking (q-gram, stacked, meta-blocking).
- `src/models/` — deterministic matching rules and clustering.
- `src/evaluation/` — blocking/rule evaluation harnesses.
- `src/api/` — FastAPI app (`main.py`), routers (`health`, `runs`, `records`,
  `audit`, `dashboard`), the resolved-output store (`store.py`), and job
  orchestration (`jobs.py`). This is what the dashboard talks to.
- `data/`, `models/`, `logs/` — pipeline inputs/outputs and run artifacts
  (mostly gitignored; structure kept via `.gitkeep`).
- `notebooks/` — exploratory analysis.
- `docs/` — pipeline- and API-facing documentation: `Data-Contract.md` (schema
  of every artifact at every stage, including the resolved-output index),
  `Data-Cleaning-Guide.md`, `Blocking-Guide.md` (+ the frozen
  `Blocking-Research-Embedding-Graph.md` research writeup behind it),
  `Deterministic-Rules-Guide.md`, `FS-Matcher-Production-Guide.md`,
  `API-Design.md`, `Application-Architecture.md`.
- `tests/` — unit/integration/regression tests.

Run locally: see [`empi-service/README.md`](empi-service/README.md).

### `empi-dashboard/`

The reviewer-facing frontend and its supporting docs.

- `src/` — the Next.js app (App Router, Next's `src/` directory convention).
  `src/app/` holds pages and the Backend-for-Frontend route handlers under
  `src/app/api/*` that proxy to `empi-service`'s FastAPI; `src/components/`
  and `src/lib/` hold the UI and typed API client/schemas.
- `docs/` — dashboard-facing documentation (`Dashboard-Guide.md`,
  `Alliance-Chicago-Branding.md`, plus copies of `API-Design.md` and
  `Application-Architecture.md` shared with the backend docs).

Run locally: see [`empi-dashboard/README.md`](empi-dashboard/README.md).

## Running the whole app locally

```bash
# Terminal 1 — backend
cd empi-service
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — frontend
cd empi-dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).
