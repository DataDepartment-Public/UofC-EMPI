# data/local_index/ — Stage 6, Parquet-backend resolved-output index `[IMPLEMENTED]`

Full contract: `docs/Data-Contract.md` → "Stage 6 — Resolved-output index."

**Only populated when `EMPI_INDEX_BACKEND=parquet`.** With the default
backend (SQLite), this directory stays empty — the same data lives in
`data/empi.db` instead. Both backends implement the exact same interface
(`src/api/backends/index_backend.py::IndexBackend`) and the exact same nine
tables, so everything below applies identically to `empi.db`'s tables.

- **Producers:** `src/api/ingest/publish_local.py` (batch), `src/api/ingest/local_score.py`
  (incremental)
- **Consumer:** `src/api/backends/parquet_backend.py::ParquetIndexBackend` — the same
  FastAPI routes and `empi-dashboard/` UI that read `empi.db` in the default
  configuration.
- **Why it exists:** a fully self-contained local dev/CI/batch deployment
  with zero SQLite dependency. One process-local lock
  (`src/api/deps.py::_PARQUET_BACKEND_LOCK`) serializes requests against it.
- **Unlike Stages 1-5, this is a *mutable* store** — a publish or incremental
  score upserts prior state rather than writing a new immutable, `run_id`-
  stamped file each time.

### The nine tables (one Parquet file each)

| File | Grain | Notes |
|---|---|---|
| `entity.parquet` | one row per resolved entity (singleton or cluster) | `mid`, `origin`, `is_merged`, `confidence`, `match_rule` |
| `entity_member.parquet` | one row per PATID | resolves each PATID to exactly one `mid` |
| `review_candidate.parquet` | one row per unresolved pair awaiting review | includes `fs_match_probability`/`fs_classification_tier` (incremental scoring only) and `ml_match_probability`/`ml_classification_tier` (batch publish, Stage 4.5) when scored |
| `entity_suggestion.parquet` | a reviewer-locked PATID's would-be new grouping | recorded, never auto-applied ("sticky unmerge") |
| `block_key.parquet` | on-disk mirror of the blocking posting lists | supports incremental-scoring point lookups |
| `cleaned_attrs.parquet` | query-by-patid mirror of Stage 1's cleaned output | avoids re-reading the full Stage 1 Parquet per request |
| `record_attrs.parquet` | display fields for the dashboard's `GET /records` | denormalized at publish time |
| `record_raw.parquet` | the "View Raw Data" drawer's un-scrubbed source fields | one JSON blob per PATID |
| `audit_log.parquet` | append-only merge/unmerge/split/dismiss/undo history | also the source of truth for reviewer-locked PATIDs; `prev_mid`/`undo_of`/`undone` carry undo provenance |

Full column-level schema for every table: `Data-Contract.md`'s Stage 6
section (§6a–6e).

**Boolean columns are stored as `int64` (0/1), not `bool`** — deliberately
breaks from Stages 1-5's native `bool` columns, to keep every column
trivially portable between SQLite's dynamic typing and Parquet's columnar
typing without a cast at the boundary.
