# data/ — directory map

**This whole directory (except this file and each subfolder's `SCHEMA.md`)
is gitignored** — regenerable pipeline output + PHI, never committed. Full
source of truth for every schema below: **`docs/Data-Contract.md`**. This
file is a one-page map; each subfolder's own `SCHEMA.md` has the real
column-level detail.

To rebuild everything from scratch:

```bash
python -m src.pipeline --input data/raw/MDM_Population.csv
```

## Inputs (not pipeline output — external/curated)

| Folder | What's in it |
|---|---|
| [`raw/`](raw/SCHEMA.md) | Pipeline input CSVs + eval label sets (`MDM_Population.csv`, `gold_labels.csv`, `synthetic_data.csv`) |
| [`silver_labels/`](silver_labels/SCHEMA.md) | FS matcher training labels — VM-only PHI |

## Pipeline stage output (immutable, `run_id`-stamped Parquet)

| Stage | Folder | Status |
|---|---|---|
| 1 — Clean | [`processed/`](processed/SCHEMA.md) | `[IMPLEMENTED]` |
| 2 — Block | [`blocking/`](blocking/SCHEMA.md) | `[IMPLEMENTED]` |
| 3 — Rules | [`auto_merge/`](auto_merge/SCHEMA.md), [`non_matches/`](non_matches/SCHEMA.md), [`no_match/`](no_match/SCHEMA.md) | `[IMPLEMENTED]` |
| 4 — FS matcher | [`fs_output/`](fs_output/SCHEMA.md) | `[IMPLEMENTED]` |
| 4.5 — ML matcher | [`ml_output/`](ml_output/SCHEMA.md) | `[SCAFFOLD]` — empty until a model is trained; see `docs/ML-Matcher-Integration-Guide.md` |
| 5 — Cluster | [`clusters/`](clusters/SCHEMA.md) | `[IMPLEMENTED]` |
| (manifest) | [`runs/`](runs/SCHEMA.md) | one `run_<run_id>.json` per pipeline invocation |
| (evaluation) | [`evaluations/`](evaluations/SCHEMA.md) | stored end-to-end evaluation reports, keyed by session — see `docs/End-to-End-Evaluation-Guide.md` |

Folder names follow the classification-tier vocabulary in `contracts.py`
(`TIER_AUTO_MERGE`/`TIER_NO_MATCH`) where the folder genuinely holds one
tier's output. `non_matches/`, `fs_output/`, and `ml_output/` are
deliberately **not** tier-named — they hold either a pre-tier pool or a mix
of all three tiers (the tier lives in a column, not the folder). See each
folder's own SCHEMA.md "Naming note" for the specifics.

## Stage 6 — resolved-output index (mutable; what the API/dashboard read)

| Item | Notes |
|---|---|
| `empi.db` | SQLite backend (default, `EMPI_INDEX_BACKEND=sqlite`) — not a folder, a single file at `data/empi.db`. Nine tables; see `docs/Data-Contract.md` Stage 6 or [`local_index/SCHEMA.md`](local_index/SCHEMA.md) (same tables, Parquet form) for the column-level schema. |
| [`local_index/`](local_index/SCHEMA.md) | Parquet backend alternative (`EMPI_INDEX_BACKEND=parquet`) — same nine tables, one Parquet file each. Empty unless that backend is active. |

## Not under `data/`, but related

- `models/fs/` — trained FS matcher artifacts (`fs_model_*.json`, `active.json`). See `docs/FS-Matcher-Production-Guide.md`.
- `models/ml/` — trained ML matcher artifacts, once any exist. See `docs/ML-Matcher-Integration-Guide.md`.

## Keeping this in sync

If you add a new stage output directory, add its `.gitkeep` + `SCHEMA.md`
exceptions to `empi-service/.gitignore` (follow the existing per-directory
pattern) and a row to this file — otherwise a fresh clone won't have the
folder, and the schema doc won't survive a `data/` wipe.
