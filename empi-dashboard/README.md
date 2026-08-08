# Entity Matching Dashboard — frontend

Reviewer-facing UI for the eMPI entity-resolution pipeline: KPIs, a searchable
registry of resolved patient records with inline unmerge, a candidate-grain
Review Queue for merge/dismiss decisions, per-pair match explanations
(deterministic rule + SHAP waterfall), undoable audit history, and an Admin
tab for live ML threshold tuning. See `docs/Dashboard-Guide.md` and
`docs/Application-Architecture.md` for the functional spec and system design.

Next.js App Router (`src/app/`) + a thin Backend-for-Frontend layer
(`src/app/api/*`) that proxies to the FastAPI service in
`empi-service/src/api/` — the browser never calls FastAPI directly (see
`src/lib/server-api.ts`).

## Running locally

Requires the backend running first:

```bash
# from empi-service/
uvicorn src.api.main:app --port 8000
```

Then, from `empi-dashboard/`:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The BFF's FastAPI base
URL is configured via `EMPI_API_URL` in `.env.local` (defaults to
`http://localhost:8000`).

## Layout

```
src/
  app/
    page.tsx                  # Dashboard tab (KPIs + charts)
    dataset/page.tsx           # Patient Registry tab (table, expand/unmerge, audit log)
    dataset/[mid]/explain/     # Model Explanation sub-page (feature comparison + SHAP)
    review/page.tsx            # Review Queue tab (candidate triage, merge/dismiss)
    admin/page.tsx              # Admin tab (live ML threshold tuning, no auth)
    api/                       # BFF route handlers -> FastAPI
  lib/
    api-client.ts               # browser-side fetch wrapper (zod-validated)
    server-api.ts                # server-only fetch wrapper (BFF -> FastAPI); REVIEWER_ID here
    schemas.ts                    # zod mirrors of src/api/schemas.py
    hooks.ts                        # TanStack Query hooks
  components/                        # Dashboard/Registry/Review/Explanation UI
```

## Checks

```bash
npx tsc --noEmit
npm run lint
npm run build
npm run test          # Vitest + React Testing Library — src/lib/*.test.ts(x), src/components/**/*.test.tsx
```
