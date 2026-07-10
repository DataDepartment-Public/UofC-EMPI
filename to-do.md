# To Do

Ordered by priority. Research paper is last per your note — you'll follow up with venue/scope details.

---

## 1. FastAPI: incremental single/batch record scoring

**Goal:** score one or a handful of new records against the existing population and
persist the result into `empi.db`, without re-running `clean → block → rules →
cluster` over all ~163k records. Confirmed shape: **incremental add, persisted** —
new record(s) get cleaned, blocked against the existing population, run through the
rules (and FS matcher), and written into `empi.db` (create/update entities), same
as a real load would do.

Today `run_stacked_blocking()` (`src/preprocessing/stacked_blocking.py`) always
computes candidate pairs over the *whole* cleaned frame — there's no "new record vs.
existing population" lookup path yet. That's the actual gap to close.

- [ ] **Design the incremental blocking lookup.** Reuse the existing block-key
  functions (`src/preprocessing/blocking.py`, `qgram_blocking.py`) but add a mode
  that, given one or a few new cleaned rows, generates their block keys and looks
  up which *existing* records share a key — instead of recomputing all pairwise
  blocks among existing records. This is the part that actually avoids "redoing
  the whole thing."
  - Needs an index of block key → PATIDs for the current population. Decide
    whether that's built on the fly from the latest `cleaned_*.parquet` each call,
    or persisted/cached (e.g. rebuilt once per full pipeline run, reused for
    incremental calls until the next run). Caching is faster per-call but adds a
    staleness concern if reviewer edits or new incremental adds happen in between.
- [ ] **New module**, e.g. `src/incremental.py`, with a `score_records(records, settings) -> IncrementalScoreResult` entry point:
  1. Clean/normalize input row(s) via the existing `transform_dataframe` path
     (`src/preprocessing/transformations.py`) so incremental records go through
     identical normalization as a batch run.
  2. Candidate lookup against existing population (per the blocking design above).
  3. Run `apply_rules()` (`src/models/deterministic_rules.py`) between the new
     record(s) and only their matched candidates.
  4. **Auto-merge tier** → resolve which existing `mid` this joins (or mint a new
     one via `store.next_mid()`), then `store.upsert_entity` /
     `upsert_entity_member` — this sidesteps `assign_clusters` /
     `build_cluster_assignments`, which operate over a full matches set, so
     merging into an existing cluster needs to be handled directly against
     `empi.db` rather than through those functions.
  5. **Review tier** → insert into `review_candidate` so it surfaces in the
     dashboard's review queue like a normal run would. `store.py` currently only
     has `replace_review_candidates_for_run` (bulk replace scoped to one
     `run_id`) — needs a variant that *appends* without clobbering existing
     review candidates from prior runs/incremental calls.
  6. FS matcher stage (optional, mirrors pipeline Stage 4) — score the new
     record's non-matches with the active model for review-queue evidence,
     if an active model is resolvable.
- [ ] **Persistence/audit trail:** decide whether each incremental call gets its own
  lightweight `run_id` (so it shows up in `GET /runs` history, distinguishable from
  a full batch run) or is untracked. Recommend giving it a `run_id` with some
  `kind: "incremental"` marker on the manifest/record — keeps the audit trail
  consistent with how reviewer merges are already logged.
- [ ] **New route(s):**
  - `POST /records/score` — accepts one record or a list (reuse/extend the
    `RawRecord`-shaped input from `src/api/schemas.py` rather than inventing a new
    input schema).
  - ❓ **Open:** sync response vs. `BackgroundTasks` job like `POST /runs`. A
    single record is probably fast enough to do inline; a "batch" of, say,
    hundreds could still block a request. Suggest: sync for small batches (pick a
    threshold, e.g. ≤50), `BackgroundTasks` + polling (same pattern as `/runs`)
    above that. Confirm the threshold or just always go async for consistency.
- [ ] **Tests:** unit test for the block-key lookup against a known population
  (does it recall the same candidates a full block+rules run would have found for
  that record); integration test hitting `POST /records/score` and asserting the
  new PATID shows up correctly in `empi.db` (`entity`, `entity_member`, and
  `review_candidate` as appropriate) without disturbing existing entities.
- [ ] **Docs:** extend `docs/API-Design.md` with this route (it currently only
  documents `/runs`, `/records`, `/audit`, `/health`) and note the incremental
  blocking behavior in `docs/Blocking-Guide.md`.

---

## 2. Clean up `empi-dashboard/`

Confirmed target layout — flatten `web/` up and adopt Next.js's `src/` directory
convention:

```
empi-dashboard/
  README.md
  package.json
  next.config.ts
  tsconfig.json
  eslint.config.mjs
  postcss.config.mjs
  package-lock.json
  .gitignore
  src/
    app/
    components/
    lib/
  docs/
    API-Design.md
    Alliance-Chicago-Branding.md
    Application-Architecture.md
    Dashboard-Guide.md
```

- [ ] Delete `empi-dashboard/demo/` (`dashboard-demo.html`) — superseded by the
  real Next.js app.
- [ ] Move everything currently under `empi-dashboard/web/` up one level into
  `empi-dashboard/` (git mv to preserve history): `README.md`, `package.json`,
  `package-lock.json`, `next.config.ts`, `tsconfig.json`, `eslint.config.mjs`,
  `postcss.config.mjs`, `.gitignore`.
- [ ] Move `app/`, `components/`, `lib/` into a new `empi-dashboard/src/` (Next.js
  auto-detects `src/app` — no config change needed for the App Router itself).
- [ ] Update path-dependent config for the new `src/` root:
  - `tsconfig.json` path aliases (check `@/*` or similar baseUrl/paths entries).
  - `eslint.config.mjs` if it references `app/`/`components/` paths directly.
  - `next.config.ts` — check for any hardcoded paths.
  - Tailwind/PostCSS content globs in `postcss.config.mjs` if they scan
    `./app/**` etc.
- [ ] Fix internal imports if any use relative paths that assumed the old nesting
  (most should be fine if they're already `@/components/...` style, but verify).
- [ ] Update `README.md` (top-level and `empi-dashboard/`'s) — the current repo
  README's "What's in each folder" section and "Running the whole app locally"
  both reference `empi-dashboard/web/` explicitly (`cd empi-dashboard/web`,
  `[empi-dashboard/web/README.md]`) — these need to become `cd empi-dashboard`.
- [ ] Update `.gitignore` at repo root and dashboard-level if either references
  `empi-dashboard/web/...` paths.
- [ ] Run `npm install && npm run build` (or `next build`) from the new root to
  confirm nothing broke, then `npm run dev` and click through the dashboard.
- [ ] Update `empi-dashboard/docs/*.md` if they reference the old `web/` path
  anywhere (e.g. Dashboard-Guide.md build/run instructions).

---

## 3. Dockerize `empi-service` and `empi-dashboard` (production-ready)

Confirmed scope: production-ready images (multi-stage builds, non-root user,
minimal base images) — not just local-dev convenience wrappers.

- [ ] ❓ **Open: what's the actual deploy target?** (e.g. a specific cloud App
  Service, ECS/Fargate, a plain VM, Azure/AWS/GCP managed container service, or
  still undecided.) This affects base image choice, whether SQLite (`empi.db`) is
  acceptable as-is or needs to move to a networked DB for multi-instance
  deployments, and whether a reverse proxy / TLS termination is handled by the
  platform or needs to be in-image. Defaulting to generic, cloud-agnostic choices
  below until you confirm.

**`empi-service/Dockerfile`**
- [ ] Multi-stage build: a build stage that installs deps with `uv` (per your
  toolchain convention) into a venv, then a slim runtime stage that copies just
  the venv + `src/` — keeps compiled/cache layers out of the final image.
- [ ] Base image: `python:3.11-slim` (repo currently runs 3.13 in `.venv` — confirm
  target Python version) or `python:3.13-slim`.
- [ ] Non-root user (`useradd` + `USER`), read-only filesystem where practical.
- [ ] `data/`, `models/`, `logs/` as declared `VOLUME`s (or bind mounts at deploy
  time) — these are gitignored/runtime-populated, not baked into the image.
  `empi.db` in particular needs to live on a persistent volume, not the
  container's writable layer.
- [ ] `.env` handling: don't bake `.env`/`.env.example` secrets into the image;
  pass config via environment variables at runtime (check `src/config.py`'s
  `Settings` for what's expected).
- [ ] `CMD`/`ENTRYPOINT`: run via `uvicorn src.api.main:app --host 0.0.0.0 --port
  8000` (no `--reload` in production), consider `--workers` count and whether
  `jobs.py`'s in-process run registry (documented as "fine for single-uvicorn-
  process... deploy target") is still valid — multiple workers/replicas would
  break the in-memory job status registry and need a shared store instead. Worth
  flagging explicitly since it's called out as a known limitation in
  `src/api/jobs.py`.
- [ ] Healthcheck hitting `GET /health`.
- [ ] `empi-service/.dockerignore`: exclude `.venv/`, `data/` (except maybe a
  `.gitkeep`), `notebooks/`, `models/experiments/`, `logs/`, `.pytest_cache/`,
  `.ruff_cache/`, `__pycache__/`, `.env`.

**`empi-dashboard/Dockerfile`** (after the folder cleanup above lands, since paths
will change)
- [ ] Multi-stage Next.js build: `deps` stage (`npm ci`), `builder` stage (`npm run
  build`, using Next's `output: "standalone"` if not already set in
  `next.config.ts` — worth adding since it drastically shrinks the runtime image),
  `runner` stage that just copies `.next/standalone` + `.next/static` + `public/`.
- [ ] Base image: `node:20-slim` or `node:22-slim` (check `package.json` engines
  field, if any).
- [ ] Non-root user, `EXPOSE 3000`, `CMD ["node", "server.js"]` (standalone
  output's entry point).
- [ ] Runtime env for the BFF's proxy target (`empi-service` URL) — confirm how
  `lib/server-api.ts` resolves the backend base URL today and make sure it's
  overridable via env var for container networking (service name, not
  `localhost`).
- [ ] `empi-dashboard/.dockerignore`: exclude `node_modules/`, `.next/`, `.git/`.
- [ ] Healthcheck hitting `GET /api/health` (the BFF route).

**Both**
- [ ] `docker-compose.yml` at the repo root wiring both services together
  (even though scope is "production-ready images," a compose file is still the
  practical way to test them together locally before any real deploy) — network
  them so the dashboard reaches `empi-service` by service name, mount a named
  volume for `empi-service/data`.
- [ ] Build and run both standalone (`docker build` + `docker run` per service) and
  confirm parity with the current `uvicorn --reload` / `npm run dev` local setup:
  health checks pass, a full pipeline run completes, the dashboard renders real
  data end-to-end.
- [ ] Decide on image registry / tagging convention if these will actually be
  pushed anywhere (depends on the deploy-target answer above).

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
