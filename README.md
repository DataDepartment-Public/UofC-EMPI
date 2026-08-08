# eMPI — Entity Resolution / Master Patient Index

Capstone project for **AllianceChicago**: group records from the `MDM_Population`
dataset that belong to the same patient across source systems, and give reviewers
a dashboard to inspect and correct the matches.

This repo combines the application, its infrastructure, and its ML training
pipeline into one place:

```
empi-service/          Python backend — pipeline + FastAPI service
empi-dashboard/         Next.js frontend — reviewer dashboard
empi-model-training/     Azure ML training pipeline for all three classifier
                        models (a logically independent codebase — see below)
terraform/               Azure infrastructure as code
docs/                    Cross-cutting architecture docs + the client deck
.github/workflows/       CI/CD — deploy, terraform plan/apply, model promotion
```

## Architecture at a glance

> Full Azure deployment topology + entity-resolution pipeline diagrams (Mermaid, presentation-ready): [`docs/Architecture-Diagram.md`](docs/Architecture-Diagram.md).

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser                                                            │
│    React UI (Dashboard · Review Queue · Patient Registry · Admin)   │
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
│   data/*.parquet + RunManifest   SQLite / Postgres / local Parquet    │
│                                   (EMPI_INDEX_BACKEND, pluggable —    │
│                                    Postgres is what Azure runs)       │
└──────────────────────────────────────────────────────────────────────┘
                │
                ▼
  empi-service/src — clean → block → rules → FS(audit) → gate → ML → cluster
  (the entity-resolution pipeline itself, orchestrated by src/pipeline.py,
  unchanged by the API layer)
```

## What's in each folder

### `empi-service/`

The Python backend: the entity-resolution pipeline and the FastAPI service that
wraps it.

- `src/pipeline.py` — orchestrates the full `clean → block → rules →
  FS matcher (audit-only) → non-match gate → ML matcher → cluster` run.
- `src/preprocessing/` — data cleaning and blocking (q-gram, stacked, meta-blocking).
- `src/models/` — deterministic matching rules, the FS matcher, the
  non-match gate, the ML matcher, and clustering.
- `src/evaluation/` — blocking/rule evaluation harnesses.
- `src/api/` — FastAPI app (`main.py`), routers (`health`, `runs`, `records`,
  `audit`, `dashboard`, `admin`, `explanations`), the pluggable resolved-output
  store (`backends/` — SQLite, Postgres, or local Parquet behind one
  `IndexBackend` interface), and job orchestration/retry (`jobs.py`). This
  is what the dashboard talks to.
- `data/`, `models/`, `logs/` — pipeline inputs/outputs and run artifacts
  (mostly gitignored; structure kept via `.gitkeep`).
- `notebooks/` — exploratory analysis.
- `docs/` — pipeline- and API-facing documentation: `Data-Contract.md` (schema
  of every artifact at every stage, including the resolved-output index),
  `Data-Cleaning-Guide.md`, `Blocking-Guide.md` (+ the frozen
  `Blocking-Research-Embedding-Graph.md` research writeup behind it),
  `Deterministic-Rules-Guide.md`, `FS-Matcher-Production-Guide.md`,
  `Nonmatch-Gate-Guide.md`, `ML-Model-LightGBM-v5.md`,
  `ML-Matcher-Integration-Guide.md`, `Explanations-Guide.md`,
  `End-to-End-Evaluation-Guide.md`, `API-Design.md`, `Application-Architecture.md`.
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

### `terraform/`

Azure infrastructure as code — one resource group holding: a Container
Registry; two App Services running `empi-service`/`empi-dashboard` as
containers; an Azure Database for PostgreSQL Flexible Server (the
resolved-output index in this environment, replacing the local SQLite
file); a Storage Account (Azure Files, mounted into the backend for
`data/`, `models/`, `logs/`); a VNet with private endpoints (no public
ingress to the backend, database, or storage — only the dashboard is
public); an Azure ML workspace; Key Vault; and Entra ID (Azure AD) auth
wiring for both App Services.

This is a one-time bootstrap for whoever has Azure subscription access —
see [`terraform/README.md`](terraform/README.md) for prerequisites and the
full apply flow. Day-to-day changes go through the GitHub Actions workflows
below, not a manual `terraform apply`.

### `empi-model-training/`

Azure ML training pipeline for the three classifier models (the Splink-based
Fellegi-Sunter matcher, the LightGBM v5 confident-match classifier, and the
LightGBM non-match gate) — experiment tracking, reproducible training jobs,
and a model registry. It's pushed as part of this repo but is a **logically
independent codebase**: no shared imports with `empi-service`, by design.
Where the two need to agree on something (a Splink comparison structure, a
feature's exact definition), the logic is faithfully reimplemented here
rather than shared — if `empi-service`'s version changes, this copy needs a
deliberate, matching update; nothing keeps them in sync automatically.

Scope is training + experiment tracking + model registry only — serving is
unchanged, `empi-service` still loads a model artifact from disk and scores
in-process. All three training scripts are plain Python CLIs with no Azure
dependency, so this runs and tests fully locally; Azure ML (workspace,
compute cluster, registry) is an additive layer for real training runs, not
a hard requirement. See [`empi-model-training/README.md`](empi-model-training/README.md).

### `docs/`

Cross-cutting documentation that spans the whole system — the application,
the Azure deployment, and the ML pipeline together — as opposed to
`empi-service/docs/`/`empi-dashboard/docs/`, which are scoped to their own
half of the app.

- [`Architecture-Diagram.md`](docs/Architecture-Diagram.md) — the Mermaid
  source of truth: deployment topology, the entity-resolution pipeline,
  CI/CD, scalability/resiliency, observability, and cost, all in one doc.
- `eMPI-Architecture-Deck.pptx` — the branded client presentation built
  from the same facts as the diagram doc above. `deck-build/` is its build
  script, brand assets, and QA tooling (see `deck-build/README.md`) — the
  deck itself has since been hand-edited directly in PowerPoint/Google
  Slides, so treat `build_deck.py` as drifted from the shipped file, not a
  script to blindly regenerate from.

### `.github/workflows/`

CI/CD, all authenticating to Azure via OIDC federated identity — no
long-lived Azure secret is ever stored in GitHub:

- `deploy-backend.yml` / `deploy-dashboard.yml` — build, push to ACR,
  repoint the App Service, verify it comes back up healthy. Automatic on
  every push to `main` that touches that service.
- `terraform-plan.yml` / `terraform-apply.yml` — plan runs and comments on
  any PR touching `terraform/**`; apply is manual-dispatch-only, gated
  behind a `production` GitHub Environment approval.
- `promote-model.yml` — manual dispatch: shows a candidate model's metrics
  for review, then (on approval) tags it champion in the Azure ML registry
  and calls the backend's `POST /admin/models/reload` so it's actually
  live — no restart, no redeploy.

## Running the whole app with Docker (recommended)

`docker-compose.yml` at the repo root brings up both halves — FastAPI on
`:8000`, the dashboard on `:3000` — and wires them together. This is the
quickest path from a fresh clone to a working UI.

### 1. Prerequisites on the host

Two things live on your machine and are mounted into the container rather than
baked into the image (both are gitignored — PHI and model artifacts never go
into the repo or the build context):

**Input data** — put the source CSV in `empi-service/data/raw/`:

```
empi-service/data/raw/MDM_Population.csv
```

Mounted read-only, so a run can never modify the source data. See
[`empi-service/data/raw/SCHEMA.md`](empi-service/data/raw/SCHEMA.md) for the
expected columns.

**Model artifacts** — the two served LightGBM models go in `empi-service/models/`:

```
empi-service/models/nonmatch_gate/   nonmatch_gate_<ts>.pkl        (Stage 4.25 — confident non-matches)
empi-service/models/ml/              ml_model_confident_match_v5_<ts>.pkl  (Stage 4.5 — confident matches)
```

Each directory's `active.json` is tracked in git and names the exact `.pkl`
file it expects — make sure the filenames on disk match, or promote your own
with the registry. Without an active gate model the pipeline falls back to the
FS gate (or runs ungated with a warning); without an active ML model Stage 4.5
is skipped entirely and only deterministic matches feed clustering.

Make sure Docker Desktop (or the Docker daemon) is actually running.

### 2. Bring the stack up

```bash
docker compose up --build -d
```

### 3. Create the database schema — once per volume

The app connects to the database but does not create its own schema. Run this
once against a fresh `empi-data` volume (i.e. the first `up`, and again after
any `docker compose down -v`):

```bash
docker compose exec empi-service python scripts/init_db.py
```

### 4. Kick off a pipeline run

`POST /runs` runs the full pipeline **and** publishes the result into the
resolved-output index that the dashboard reads.

```bash
# macOS / Linux
curl -s -X POST http://localhost:8000/runs -F "input_path=data/raw/MDM_Population.csv"
```

```powershell
# Windows (PowerShell) — use curl.exe, not the `curl` alias for Invoke-WebRequest
curl.exe -s -X POST http://localhost:8000/runs -F "input_path=data/raw/MDM_Population.csv"
```

The path is container-relative (`/app/data/raw/…`), so pass it exactly as
above regardless of where the file sits on your host.

### 5. Watch it work

```bash
docker compose logs -f empi-service
```

Stages are tagged `[n/7]`. The run finishes when clustering and the publish
step are done — on the full dataset this takes a while.

### 6. Open the dashboard

[http://localhost:3000](http://localhost:3000)

### Useful extras

```bash
docker compose down          # stop, keep the data volume
docker compose down -v       # stop and wipe the index (init_db.py needed again)
docker compose ps            # service + health status
```

Local dev overrides (feature flags, thresholds) come from
`empi-service/.env` if present — see `empi-service/.env.example`.

## Running the whole app without Docker

```bash
# Terminal 1 — backend
cd empi-service
python scripts/init_db.py            # once per environment — creates the schema
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — frontend
cd empi-dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).
