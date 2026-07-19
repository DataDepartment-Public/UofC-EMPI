# data/runs/ — RunManifest — one JSON per pipeline run `[IMPLEMENTED]`

Full contract: `docs/Data-Contract.md` → "The `RunManifest`."

- **Producer:** `src/pipeline.py`, once per invocation
- **Contract:** `contracts.RunManifest` (pydantic model)
- **Consumers:** `src/api/ingest/publish.py` (batch publish), `fs_matcher/train.py`
  and `ml_matcher/train.py` (input resolution — resolves `cleaned` +
  `candidate_pairs` from here rather than "newest file in a directory", so
  same-run lineage is structurally guaranteed), `scripts/build_eval_workbook.py`
- **File:** `run_<run_id>.json`
- **Why it exists:** threads one `run_id` through every stage's artifact
  name, so a mismatch between e.g. `cleaned` and `candidate_pairs` from two
  different runs is structurally impossible on the orchestrated path.

### Fields

| Field | Type | Notes |
|---|---|---|
| `run_id` | string | UTC timestamp, e.g. `20260617T043941Z` — lexicographically sortable, so "latest" needs no separate version counter. |
| `created_utc` | string | Run start time. |
| `git_sha` | string, optional | Commit the run executed against. |
| `raw_input`, `cleaned`, `candidate_pairs`, `matches`, `non_matches` | `ArtifactRef` | Always populated. |
| `rejects`, `clusters`, `review_evidence` | `ArtifactRef`, optional | Populated by every current run. |
| `matches_model`, `fs_features` | `ArtifactRef`, optional | Populated **only** when an active FS model scored the run. |
| `matches_ml`, `ml_features` | `ArtifactRef`, optional | Populated **only** when an active ML model scored the run (null today — no ML model trained yet). |
| `counts` | `dict[str, int]` | `raw_rows`, `cleaned_rows`, `valid_records`, `candidate_pairs`, `matches`, `non_matches`, `rejects`, `clusters`, `total_clusters`. |

Each `ArtifactRef` carries a project-root-relative `path`, `rows`, and a
`sha256` of the file — so a manifest is also a lightweight integrity check
for every artifact it names.
