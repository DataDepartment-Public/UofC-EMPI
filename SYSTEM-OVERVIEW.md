# eMPI Entity Matching System — Overview & Runbook

One system, two halves of the same `DataDepartment-Public/UofC-EMPI` repo,
developed on separate branches rather than as separate projects:

| Branch | Folder name used below | What it is |
|---|---|---|
| `jason-api` (forked as `lily-api`) | `UofC-EMPI-jason-api` | Python — the entity-resolution pipeline + FastAPI service |
| `dashboard-main` (forked as `lily-dashboard`) | `UofC-EMPI-dashboard-main` | Next.js — the reviewer-facing dashboard (BFF + React UI) |

This document (and the runbook in particular) was written against a local
workspace with both branches checked out side by side as sibling folders,
since that's how the two services find each other over `localhost`:

```bash
git clone -b jason-api      git@github.com:DataDepartment-Public/UofC-EMPI.git UofC-EMPI-jason-api
git clone -b dashboard-main git@github.com:DataDepartment-Public/UofC-EMPI.git UofC-EMPI-dashboard-main
```

(swap in `lily-api` / `lily-dashboard` for this fork's branches). They are not
linked by a git submodule or monorepo tooling — just two checkouts of the
same repo, on different branches, that talk to each other over HTTP on
`localhost`. This document is the connective tissue: what each piece does,
how they fit together, and the exact commands to bring the whole thing up
from a clean checkout.

For deep detail on any one piece, see the docs already inside each branch
(linked throughout). This file only covers what neither of them says on its
own: that they're one system, and how to actually run it end to end.

---

## 1. What this system does

AllianceChicago's eMPI ("enterprise Master Patient Index") problem: patient
records arrive from multiple source systems and the same real person often
has several different `PATID`s with inconsistent formatting (nicknames,
typos, old addresses, missing SSNs, etc.). This system:

1. **Cleans** raw patient records into a normalized form.
2. **Blocks** them into small candidate groups so we don't compare every
   record to every other record (an O(n²) non-starter at real volume).
3. **Applies deterministic matching rules** (e.g. exact SSN + DOB) to decide
   which candidate pairs are the same person.
4. **Clusters** confirmed matches into master patient records.
5. Publishes the result to a **reviewer dashboard**, where a human can browse
   every master record, drill into *why* a match was made, and manually
   merge/unmerge/approve edge cases the rules didn't auto-resolve.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Browser  (localhost:3000)                                          │
│    React UI — Dashboard tab · Dataset tab · Model Explanation page  │
└───────────────▲───────────────────────────────────────────────────────┘
                │ same-origin fetch to /api/*
┌───────────────┴───────────────────────────────────────────────────────┐
│  Next.js (Node.js)  —  UofC-EMPI-dashboard-main/web                  │
│    app/page.tsx, app/dataset/*        — the React pages              │
│    app/api/*/route.ts                 — BFF: proxies to FastAPI,     │
│                                          injects X-Reviewer-Id        │
└───────────────▲───────────────────────────────────────────────────────┘
                │ HTTP (localhost:8000), server-side only —
                │ the browser never calls FastAPI directly
┌───────────────┴───────────────────────────────────────────────────────┐
│  FastAPI (uvicorn)  —  UofC-EMPI-jason-api/src/api                    │
│    /health  /runs  /records  /clusters/{mid}  /dashboard/summary      │
│    /audit  /audit/merge  /audit/unmerge                               │
└───────────────▲───────────────────────────────────────────────────────┘
                │ POST /runs schedules this in the background:
┌───────────────┴───────────────────────────────────────────────────────┐
│  Pipeline (pure Python/pandas)  —  src/pipeline.py                    │
│    clean → block → deterministic rules → cluster                      │
│    writes Parquet + a RunManifest under data/, then "publishes"       │
│    the final entities into SQLite empi.db                             │
└─────────────────────────────────────────────────────────────────────┘
```

**Two storage tiers**, by design (see `UofC-EMPI-jason-api/docs/API-Design.md`):
- `data/*.parquet` + `data/runs/run_<id>.json` — the immutable record of what
  each pipeline run produced. Never edited after the fact.
- `data/empi.db` (SQLite) — the system of record the dashboard reads/writes.
  A "publish" step loads a run's output here; reviewer merge/unmerge actions
  mutate this directly and are never overwritten by a later pipeline run for
  records a human has already touched.

**Why a BFF instead of the browser calling FastAPI directly:** the Next.js
route handlers under `web/app/api/` are the only thing that talks to FastAPI.
This lets the server inject a trusted `X-Reviewer-Id` header server-side (so
the audit log can't be spoofed from the browser) and gives the frontend one
place to validate every response against a zod schema (`web/lib/schemas.ts`),
so a backend contract change surfaces as a build error instead of a silent
runtime bug.

**Note on FastAPI route paths:** the backend's own routes have *no* `/api`
prefix (e.g. `GET /dashboard/summary`, `GET /records`). The `/api` prefix
only exists on the Next.js BFF side (`GET /api/dashboard/summary` from the
browser), which then calls the unprefixed FastAPI route server-side. Don't
be confused by the `docs/API-Design.md` examples, some of which predate this
detail.

### Further reading, by topic

| Topic | Doc |
|---|---|
| Full backend route + SQLite schema contract | `UofC-EMPI-jason-api/docs/API-Design.md` |
| Full frontend/backend system design | `UofC-EMPI-jason-api/docs/Application-Architecture.md` (identical copy in the dashboard repo) |
| Dashboard functional spec (what every screen must do) | `UofC-EMPI-dashboard-main/docs/Dashboard-Guide.md` |
| Interactive mock of the UI (static HTML, mock data) | `UofC-EMPI-dashboard-main/demo/dashboard-demo.html` |
| Field-by-field data cleaning rules | `UofC-EMPI-jason-api/docs/Data-Cleaning-Guide.md` |
| Blocking (candidate-pair generation) design | `UofC-EMPI-jason-api/docs/Blocking-Guide.md` |
| Deterministic matching rules | `UofC-EMPI-jason-api/docs/Deterministic-Rules-Guide.md` |
| Branding (colors/fonts/logo) | `UofC-EMPI-dashboard-main/docs/Alliance-Chicago-Branding.md` |

---

## 3. Running it locally — from a clean checkout

Two things must both be running: the FastAPI backend (port 8000) and the
Next.js frontend (port 3000). Start the backend first.

### 3.1 Backend — Python environment

The system Python on macOS here is 3.9, which **does not work** — the
codebase uses `str | None`-style type hints evaluated at runtime by
pydantic v2, which requires Python 3.10+. Use a newer interpreter:

```bash
cd UofC-EMPI-jason-api

# if python3.13 isn't already on your machine:
#   brew install python@3.13
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

`requirements.txt` lists `postal` (the Python binding for `libpostal`),
which requires the native `libpostal` C library and a multi-GB data
download to build. It's genuinely optional — `src/preprocessing/transformations.py`
wraps the import in a `try/except` and the cleaning stage just leaves
`Address_normalized` as `NaN` if it's unavailable. Skip it unless you
specifically need address normalization:

```bash
grep -v "^postal$" requirements.txt > /tmp/requirements-no-postal.txt
pip install -r /tmp/requirements-no-postal.txt
```

(If you do want real address normalization, `brew install libpostal` first,
then `pip install -r requirements.txt` as-is.)

### 3.2 Backend — get some data into the pipeline

The repo ships no sample raw data. Generate a synthetic patient file (it
deliberately includes messy formatting, accented characters, and planted
duplicate pairs so every matching rule has something to find):

```bash
python scripts/generate_synthetic_raw.py --n 4000 --dup-rate 0.2 --seed 42
# → writes data/raw/MDM_Population.csv
```

(To use a real extract instead, drop a CSV/XLSX with the same 19 columns —
see `docs/Data-Cleaning-Guide.md` — at `data/raw/` and skip this step.)

### 3.3 Backend — start the API

```bash
uvicorn src.api.main:app --port 8000
```

This creates `data/empi.db` on first boot (empty — no entities yet). Verify:

```bash
curl http://localhost:8000/health          # {"status":"ok"}
```

### 3.4 Backend — run the pipeline and publish to the dashboard's DB

`python -m src.pipeline` runs clean→block→rules→cluster and writes Parquet,
but does **not** publish to `empi.db` by itself — only the API's background
job does that (`src/api/jobs.py`). Trigger a run through the API instead of
the bare CLI, so publish happens automatically:

```bash
curl -X POST http://localhost:8000/runs -F "input_path=data/raw/MDM_Population.csv"
# → {"run_id": "...", "status": "queued"}

# poll until status is "succeeded" (a few thousand records finishes in ~2s):
curl http://localhost:8000/runs/<run_id>
```

Sanity check real data is flowing:

```bash
curl http://localhost:8000/dashboard/summary
curl "http://localhost:8000/records?page=1&page_size=3"
```

To reprocess later (e.g. after generating a bigger synthetic file, or a new
raw extract), just `POST /runs` again — publish merges into the existing
`empi.db` rather than wiping it, so prior reviewer merge/unmerge decisions
survive a re-run (see the "reconciliation" section of `API-Design.md`).

### 3.5 Frontend

In a second terminal, with the backend already running:

```bash
cd UofC-EMPI-dashboard-main/web
npm install
npm run dev
```

Open **http://localhost:3000**. The BFF's FastAPI base URL defaults to
`http://localhost:8000`; override it via `EMPI_API_URL` in `web/.env.local`
if the backend runs elsewhere.

### 3.6 Stopping / restarting

```bash
pkill -f "uvicorn src.api.main"
pkill -f "next dev"
```

`empi.db` and everything under `data/` persist across restarts — you only
need to regenerate synthetic data or re-run the pipeline if you want fresh
numbers, not every time you restart the servers.

---

## 4. Verifying it actually works

Beyond `npm run build` / `npx tsc --noEmit` (frontend) and the pytest suite
under `UofC-EMPI-jason-api/tests/` (backend — `pytest` from that repo root),
the real check is exercising the app:

1. **Dashboard tab** (`/`) — 8 KPI cards, a match-status bar chart, model
   info panel, and a performance-over-time line chart should all populate
   with real numbers within a second or two of load (they hit
   `GET /api/dashboard/summary`, which reads `empi.db`).
2. **Dataset tab** (`/dataset`) — search/filter by ID, name, status, merge
   status, birthdate, or SSN last-4. Expand any row: entities with more than
   one confirmed member show "Cluster members"; entities with unresolved
   candidates show "Potential duplicate candidates" with an inline **Merge**
   button.
3. **Merge flow** — click Merge → confirm in the modal → a toast confirms →
   the row's status flips, the Dashboard KPIs update on next poll (15s, or
   immediately via React Query cache invalidation), and a new row appears in
   the **Merge audit log** table at the bottom of the Dataset tab.
4. **Model Explanation** — click any candidate/member name to see the
   field-by-field comparison and the plain-language reason a pair did or
   didn't match, with a back-link to the originating row.

---

## 5. Known gaps / things worth knowing

- **Fixed during initial build-out:** the audit-log API (`GET /audit`) and
  its `useAuditLog` React Query hook already existed but nothing rendered
  them — the Dataset page was missing the audit log table required by the
  Dashboard-Guide spec (and present in the demo). Added
  `web/components/AuditLog.tsx`, wired into `web/app/dataset/page.tsx`.
- **Not fixed — a data-cleaning pipeline quirk, not a dashboard bug:** the
  synthetic generator deliberately injects messy casing and accented
  characters (e.g. `aÑÑa`) to exercise the cleaning stage. A camelCase-split
  heuristic in `src/preprocessing/transformations.py` occasionally
  misfires on the post-transliteration result and inserts a stray space
  (e.g. "aNNa" → "A NNA"). Cosmetic, rare, and confined to synthetic test
  data — call it out if you want it root-caused separately.
- **No probabilistic/ML matching stage yet** — only deterministic rules run
  today (`src/models/deterministic_rules.py`). The Model Explanation page
  intentionally shows the real rule that fired and a genuine field
  comparison rather than a fabricated match probability or SHAP waterfall;
  see the comment at the top of `web/app/dataset/[mid]/explain/page.tsx`.
- **No auth** — the reviewer identity (`reviewer.jclark`) is hardcoded
  server-side in `web/lib/server-api.ts` for the local build. Swapping in
  real SSO/OIDC there is the only change needed to move identity off the
  hardcoded value; the FastAPI contract (`X-Reviewer-Id` header) doesn't
  change.
