# EMPI — Entity Resolution / Master Patient Index (backend)

Capstone project for **AllianceChicago**: group records from the `MDM_Population`
dataset that belong to the same patient across source systems. The Python backend
in this directory has two parts:

1. **The entity-resolution pipeline** — clean → block → deterministic rules →
   Fellegi-Sunter (FS) probabilistic matcher → cluster. See
   [`docs/Data-Contract.md`](docs/Data-Contract.md) for the full stage-by-stage
   contract.
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

Runs all five stages in process (`src/pipeline.py`), validates every boundary
against `src/contracts.py`, and writes Parquet artifacts under
`data/{processed,blocking,matches,non_matches,rejects,matches_model,FS_output,
clusters}/` plus a `RunManifest` to `data/runs/run_<run_id>.json`. Stage 4 (the
FS matcher) is skipped with a log line if no model is active yet — see
[`docs/FS-Matcher-Production-Guide.md`](docs/FS-Matcher-Production-Guide.md)
to train and promote one.

The pipeline never publishes its own output — that's a separate step (below),
so a batch run can be inspected/validated before it becomes visible to the API
or dashboard.

## Running the API + publishing a run

```bash
uvicorn src.api.main:app --reload --port 8000
```

`POST /runs` triggers a pipeline run and publishes it automatically in one
background job. For a fully local workflow with no server running:

```bash
python -m src.pipeline --input data/raw/MDM_Population.csv   # writes Parquet + manifest
python -m src.api.ingest.publish_local --run-id <run_id>              # -> data/local_index/*.parquet
```

Storage is pluggable (`src/api/backends/index_backend.py`, `EMPI_INDEX_BACKEND`):
**SQLite** (`data/empi.db`, the default) or a **local Parquet index**
(`data/local_index/`, `EMPI_INDEX_BACKEND=parquet`) — the same FastAPI app,
dashboard, and incremental-scoring CLI (`python -m src.api.ingest.local_score`) all
work identically against either. See
[`docs/Data-Contract.md`](docs/Data-Contract.md)'s Stage 6 for the full schema
and [`docs/API-Design.md`](docs/API-Design.md) for the route contract.

## Tests

```bash
pytest tests/            # unit + integration + regression
ruff check src/ tests/    # lint
```

## Project layout

```
src/
  pipeline.py             # orchestrator: clean -> block -> rules -> FS matcher
                          # -> ML matcher -> cluster
  contracts.py             # pandera/pydantic contracts for every stage boundary
  config.py                # Settings (EMPI_-prefixed env vars) + logging setup
  preprocessing/            # cleaning (clean.py, transformations.py) + blocking
                          # (blocking.py, qgram_blocking.py, meta_blocking.py,
                          #  stacked_blocking.py, run_blocking.py)
  models/
    base.py                 # PairClassifier — the shared interface every
                          # classifier stage below satisfies
    clustering.py            # connected-component clustering (terminal stage)
    deterministic_rules/     # rules.py (rule engine), classifier.py (three-way
                          # decision + the PairClassifier adapter)
    run_rules.py             # dev/debug CLI for deterministic_rules
    fs_matcher/               # Fellegi-Sunter matcher — train/serve/registry
    ml_matcher/                # pluggable ML matcher (bring-your-own-model/
                          # -features) — scaffold, see
                          # docs/ML-Matcher-Integration-Guide.md
  evaluation/               # blocking + rule evaluation harnesses
  api/
    main.py, deps.py, schemas.py, jobs.py   # FastAPI app wiring
    backends/                 # index_backend.py (the pluggable-storage seam),
                          # sql_backend.py (SQLite), parquet_backend.py (local mode)
    ingest/                    # publish.py / publish_local.py (batch),
                          # incremental.py / local_score.py (single/few-record)
    routers/                   # health, runs, records, audit, dashboard
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
- [`Deterministic-Rules-Guide.md`](docs/Deterministic-Rules-Guide.md) — the five
  matching rules, their tiers, and precision/recall evaluation.
- [`FS-Matcher-Production-Guide.md`](docs/FS-Matcher-Production-Guide.md) — the
  Fellegi-Sunter matcher's train/promote/serve/swap lifecycle.
- [`API-Design.md`](docs/API-Design.md) — the FastAPI route contract.
- [`Application-Architecture.md`](docs/Application-Architecture.md) — how the
  backend and `empi-dashboard/` fit together end to end.

See the [repo-root README](../README.md) for how this fits with `empi-dashboard/`.
