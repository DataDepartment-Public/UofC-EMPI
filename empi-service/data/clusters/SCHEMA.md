# data/clusters/ — Stage 5 output: cluster assignments `[IMPLEMENTED]`

Full contract: `docs/Data-Contract.md` → "Stage 5 — Clustering."

- **Producer:** `src/models/clustering.py`, the terminal stage of
  `src/pipeline.py`, run once per invocation after Stages 3, 4, and 4.5.
- **Contract:** `contracts.ClusterAssignments`
- **Consumer:** `src/api/ingest/publish.py` — nothing downstream of Stage 5
  reads this file directly except the publish step; the API/dashboard read
  the resolved-output index (`data/empi.db` / `data/local_index/`) instead.
- **File:** `cluster_assignments_<run_id>.parquet`
- **Grain:** one row per **valid** record, including singletons.

**By default, clusters only the deterministic auto-merge edges**
(`data/matches/`). Review-tier confirmations and non-matches are excluded.
The Stage 4/4.5 classifier output is unioned in only if
`settings.fs_feeds_clustering` / `settings.ml_feeds_clustering` is
explicitly turned on (both default `False`) — with both off, clustering
input is byte-identical to `matches` alone.

### Output schema

| Column | Dtype | Notes |
|---|---|---|
| `PATID` | string | Unique; every valid record gets a cluster. |
| `cluster_id` | int64 | Contiguous from 0; deterministic across runs. |
