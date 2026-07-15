# Entity Matching Dashboard — frontend

Reviewer-facing UI for the eMPI entity-resolution pipeline: KPIs, a searchable
dataset of resolved patient records, inline merge/unmerge, and a per-pair
match explanation. See `docs/Dashboard-Guide.md` and `docs/Application-Architecture.md`
for the functional spec and system design.

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
    dataset/page.tsx           # Dataset tab (table, expand/merge/unmerge)
    dataset/[mid]/explain/     # Model Explanation sub-page
    api/                       # BFF route handlers -> FastAPI
  lib/
    api-client.ts               # browser-side fetch wrapper (zod-validated)
    server-api.ts                # server-only fetch wrapper (BFF -> FastAPI)
    schemas.ts                    # zod mirrors of src/api/schemas.py
    hooks.ts                        # TanStack Query hooks
  components/                        # Dashboard/Dataset/Explanation UI
```

## Checks

```bash
npx tsc --noEmit
npm run lint
npm run build
```
