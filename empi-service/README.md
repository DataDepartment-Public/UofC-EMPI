# EMPI — Entity Resolution / Master Patient Index (backend)

Capstone project for **AllianceChicago**: group records from the `MDM_Population`
dataset that belong to the same patient across source systems. The Python backend
in this directory has two parts:

1. **The entity-resolution pipeline** — clean → block → deterministic rules →
   Fellegi-Sunter (FS) matcher (audit-only) → non-match gate → ML matcher
   (LightGBM v5) → cluster. See [`docs/Data-Contract.md`](docs/Data-Contract.md)
   for the full stage-by-stage contract.
2. **A FastAPI service** (`src/api/`) that wraps the pipeline, publishes its
   output into a resolved-output store, serves the reviewer dashboard
   (`empi-dashboard/`), and supports incremental single/few-record scoring
   without a full pipeline re-run. See [`docs/API-Design.md`](docs/API-Design.md).

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # or your preferred venv tool
pip install -r requirements.txt
```

`postal` (the libpostal binding, used for address normalization) needs the
native `libpostal` library installed separately (`brew install libpostal` on
macOS) — if it's missing, cleaning still runs and `Address_normalized` stays
`NaN`. Copy `.env.example` to `.env` to override any `EMPI_`-prefixed setting
(see `src/config.py`).

## Running the batch pipeline

```bash
python -m src.pipeline --input data/raw/MDM_Population.csv
```

Runs all seven stages in process (`src/pipeline.py`; log lines tag them
`[1/7]`-`[7/7]`), validates every boundary against `src/contracts.py`, and
writes Parquet artifacts under
`data/{processed,blocking,auto_merge,non_matches,no_match,fs_output,
gate_output,ml_output,clusters}/` plus a `RunManifest` to
`data/runs/run_<run_id>.json`. Stage 4 (FS, audit-only), Stage 4.25 (the
non-match gate), and Stage 4.5 (the ML matcher) are each skipped with a log
line if no model is active yet — see
[`docs/FS-Matcher-Production-Guide.md`](docs/FS-Matcher-Production-Guide.md),
[`docs/Nonmatch-Gate-Guide.md`](docs/Nonmatch-Gate-Guide.md), and
[`docs/ML-Model-LightGBM-v5.md`](docs/ML-Model-LightGBM-v5.md) to train and
promote them (or run `scripts/train_synthetic_models.py` for non-PHI local
dev models covering all three).

The pipeline never publishes its own output — that's a separate step (below),
so a batch run can be inspected/validated before it becomes visible to the API
or dashboard.

## Running the API + publishing a run

```bash
python scripts/init_db.py            # once per environment — creates the schema; see its own docstring
uvicorn src.api.main:app --reload --port 8000
```

The app only ever *connects* to the database — it doesn't create or alter
its schema on startup or per-request anymore. Run `scripts/init_db.py`
once for a new environment (a fresh local SQLite file, or a new Azure
Postgres instance) and again whenever a code change adds a new column to
`_COLUMN_MIGRATIONS` (`src/api/backends/sql_backend.py`/
`postgres_backend.py`). If you skip it, the app still starts, but logs a
loud, clear error instead of silently creating the schema for you.

`POST /runs` triggers a pipeline run and publishes it automatically in one
background job. For a fully local workflow with no server running:

```bash
python -m src.pipeline --input data/raw/MDM_Population.csv   # writes Parquet + manifest
python -m src.api.ingest.publish_local --run-id <run_id>              # -> data/local_index/*.parquet
```

Storage is pluggable (`src/api/backends/index_backend.py`, `EMPI_INDEX_BACKEND`):
**SQLite** (`data/empi.db`, the default), **Postgres**
(`EMPI_INDEX_BACKEND=postgres`, AAD-token auth, no stored password — the
deployed Azure target), or a **local Parquet index** (`data/local_index/`,
`EMPI_INDEX_BACKEND=parquet`) — the same FastAPI app, dashboard, and
incremental-scoring CLI (`python -m src.api.ingest.local_score`) all work
identically against any of the three. See
[`docs/Data-Contract.md`](docs/Data-Contract.md)'s Stage 6 for the full schema
and [`docs/API-Design.md`](docs/API-Design.md) for the route contract.

## Tests

```bash
pytest tests/            # unit + integration + regression
ruff check src/ tests/    # lint
```

`tests/integration/test_postgres_backend.py` (the `EMPI_INDEX_BACKEND=postgres`
store — see `src/api/backends/postgres_backend.py`) additionally needs a real
Postgres reachable via `EMPI_TEST_POSTGRES_DSN`; it skips itself otherwise. A
local one for this alone:

```bash
initdb -D /tmp/empi_pg_test/data -U postgres -A trust --encoding=UTF8
pg_ctl -D /tmp/empi_pg_test/data -l /tmp/empi_pg_test/server.log -o "-p 55432 -k /tmp/empi_pg_test" start
createdb -h /tmp/empi_pg_test -p 55432 -U postgres empi_test
EMPI_TEST_POSTGRES_DSN="host=/tmp/empi_pg_test port=55432 dbname=empi_test user=postgres" \
  pytest tests/integration/test_postgres_backend.py
```

## Project layout

```
src/
  pipeline.py             # orchestrator: clean -> block -> rules -> FS matcher
                          # (audit-only) -> non-match gate -> ML matcher -> cluster
  contracts.py             # pandera/pydantic contracts for every stage boundary
  config.py                # Settings (EMPI_-prefixed env vars) + logging setup
  preprocessing/            # cleaning (clean.py, transformations.py) + blocking
                          # (blocking.py, qgram_blocking.py, meta_blocking.py,
                          #  stacked_blocking.py, run_blocking.py)
  models/
    base.py                 # PairClassifier — the shared interface every
                          # classifier stage below satisfies
    clustering.py            # connected-component clustering (terminal stage)
    deterministic_rules/     # rules.py (rule engine, 3 active rules), classifier.py
                          # (three-way decision + the PairClassifier adapter)
    run_rules.py             # dev/debug CLI for deterministic_rules
    fs_matcher/               # Fellegi-Sunter matcher — audit-only; train/serve/registry
    nonmatch_gate/             # confident-non-match filter (Stage 4.25) — train/serve/registry
    ml_matcher/                # LightGBM v5 confident-match classifier (Stage 4.5) —
                          # served model, see docs/ML-Model-LightGBM-v5.md
  evaluation/               # blocking + rule evaluation harnesses
  api/
    main.py, deps.py, schemas.py, jobs.py   # FastAPI app wiring
    backends/                 # index_backend.py (the pluggable-storage seam),
                          # sql_backend.py (SQLite), postgres_backend.py (Postgres),
                          # parquet_backend.py (local mode)
    ingest/                    # publish.py / publish_local.py (batch),
                          # incremental.py / local_score.py (single/few-record)
    routers/                   # health, runs, records, audit, dashboard, admin, explanations
data/, models/, logs/       # pipeline inputs/outputs, gitignored (structure kept via .gitkeep)
notebooks/                  # exploratory analysis
scripts/                    # operational eval/data CLIs (build_eval_workbook.py, eval_against_labels.py, ...)
                          # research/ — concluded one-off investigations (blocking rounds, FS/splink rounds)
tests/                      # unit/{api,models,preprocessing,evaluation}/, integration/, regression/
                          # (unit/ subpackages mirror src/'s layout)
docs/                       # see below
```

## Docs

- [`Data-Contract.md`](docs/Data-Contract.md) — the schema of every artifact at
  every stage boundary, including the resolved-output index (Stage 6). Start
  here for "what does the data look like at point X."
- [`Data-Cleaning-Guide.md`](docs/Data-Cleaning-Guide.md) — field-by-field
  cleaning/transformation rules.
- [`Blocking-Guide.md`](docs/Blocking-Guide.md) — the stacked blocking scheme
  (8-block ∪ q-gram → meta-blocking prune) and its recall evaluation.
- [`Blocking-Research-Embedding-Graph.md`](docs/Blocking-Research-Embedding-Graph.md)
  — the research synthesis behind the stacked blocker (frozen; historical record).
- [`Deterministic-Rules-Guide.md`](docs/Deterministic-Rules-Guide.md) — the three
  active matching rules, their tiers, and precision/recall evaluation (two
  former review-tier rules, `NAME_DOB_SEX`/`NAME_DOB_ADDRESS`, were removed
  for low precision — see that doc's "Removed rules" section).
- [`FS-Matcher-Production-Guide.md`](docs/FS-Matcher-Production-Guide.md) — the
  Fellegi-Sunter matcher (Stage 4, audit-only)'s train/promote/serve/swap lifecycle.
- [`Nonmatch-Gate-Guide.md`](docs/Nonmatch-Gate-Guide.md) — the confident-non-match
  filter (Stage 4.25) that replaced FS as the pool gate.
- [`ML-Model-LightGBM-v5.md`](docs/ML-Model-LightGBM-v5.md) — the served ML
  matcher (Stage 4.5): model semantics, the v3→v5 migration, and why it matters.
- [`ML-Matcher-Integration-Guide.md`](docs/ML-Matcher-Integration-Guide.md) —
  the original build-order spec for Stage 4.5 (`src/models/ml_matcher/`);
  historical now that it's shipped, kept as the feature-contract reference.
- [`Explanations-Guide.md`](docs/Explanations-Guide.md) — the per-pair SHAP
  waterfall the gate and ML matcher persist and serve via `GET /explanations/...`.
- [`API-Design.md`](docs/API-Design.md) — the FastAPI route contract.
- [`Application-Architecture.md`](docs/Application-Architecture.md) — how the
  backend and `empi-dashboard/` fit together end to end.
- [`End-to-End-Evaluation-Guide.md`](docs/End-to-End-Evaluation-Guide.md) — how
  to score the pipeline's final cluster output against a label set.

See the [repo-root README](../README.md) for how this fits with `empi-dashboard/`.
