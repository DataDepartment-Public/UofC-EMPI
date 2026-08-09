# data/blocking/ — Stage 2 output: candidate pairs `[IMPLEMENTED]`

Full contract: `docs/Data-Contract.md` → "Stage 2 — Candidate pairs" and
`docs/Blocking-Guide.md` for the full stacked-blocker design.

- **Producer (production):** `src/preprocessing/stacked_blocking.py::run_stacked_blocking`
  — the 8-block scheme ∪ a typo-tolerant q-gram pass, pruned by graph
  meta-blocking.
- **Contract:** `contracts.CandidatePairs` (`strict=True` — exact, closed schema)
- **Consumer:** deterministic rules (Stage 3)
- **File:** `candidate_pairs_<run_id>.parquet`
- **Grain:** one row per unique candidate pair
- **Splink-compatible:** `PATID_A`/`PATID_B` maps to Splink's `_l`/`_r`.

> ⚠️ **Do not confuse with the standalone `run_blocking.py` CLI.** It runs
> only the 8-block scheme below — no q-gram pass, no meta-blocking prune —
> producing a structurally narrower, algorithmically different candidate pool
> than the orchestrator, even though both can write `candidate_pairs_*.parquet`
> into this same directory. Never substitute one for the other when
> training/serving Stage 4's FS matcher.

### Output schema

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` | string | no | Canonical lower PATID (`PATID_A < PATID_B`). |
| `PATID_B` | string | no | Canonical higher PATID. |
| `source_blocks` | string | no | Pipe-delimited, sorted block IDs, e.g. `"B3\|B5\|B8"`. |
| `n_blocks` | int64 | no | Count of blocks that generated the pair. |

### The 8-block scheme

| Block | Key composition |
|---|---|
| B1 | `SSN_clean` (exact) |
| B3 | DoubleMetaphone(`LastNM_clean`) + full `BirthDT_clean` |
| B4 | `LastNM_clean` + birth year + `FirstNM_clean[:3]` |
| B5 | `Phones_set` intersection (any shared phone) |
| B6 | `Email_clean` (exact) |
| B7 | DoubleMetaphone(`LastNM_clean`) + `ZipCD_clean_base` + birth year |
| B8 | Soundex(`FirstNM_clean`) + Soundex(`LastNM_clean`) + birth year |
| B9 | `LastNM_clean` + `FirstNM_clean` + `last_4_SSN` |

(B2 was removed — subsumed by B3.) A per-key governance cap
(`settings.governance_threshold`, default 500) caps over-large blocks. The
production path additionally unions a q-gram cosine pass and prunes via graph
meta-blocking — see `docs/Blocking-Guide.md`.
