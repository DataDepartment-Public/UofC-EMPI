# Data Contract — eMPI Pipeline

Single source of truth for the **data that moves between pipeline stages**: the
schema, dtype, nullability, file location, and serialization of every artifact
handed from one stage to the next. Where `Data-Cleaning-Guide.md` governs *how
fields are cleaned* and `Deterministic-Rules-Guide.md` governs *which pairs
match*, this document governs *the shape of the data at each boundary* so any
stage can be developed, tested, or replaced against a stable interface.

The authority for each contract is the code, cited inline as `file:symbol`. If
code and this document disagree, the code wins and this document is the bug —
update it in the same change.

**Executable counterpart:** `src/contracts.py` implements these as pandera
`DataFrameModel`s (the bulk frames) and pydantic models (the `Settings` in
`src/config/config.py` and the `RunManifest`). The orchestrator
`src/pipeline.py` and the stage entry points validate every boundary against
them. This document is the human-readable spec; `contracts.py` is the
enforcement — keep them mirrored.

_Last updated: 2026-06-03._

---

## Pipeline overview

```
            data/raw/*.csv|xlsx
                  │  (dtype=str on read — preserves leading zeros)
   ┌──────────────▼───────────────┐
   │ 1. CLEAN / TRANSFORM          │  src/data/transformations.py, clean.py     [IMPLEMENTED]
   └──────────────┬───────────────┘
   data/processed/MDM_Population_cleaned_*.parquet
                  │
   ┌──────────────▼───────────────┐
   │ 2. BLOCKING                   │  src/features/blocking.py, run_blocking.py  [IMPLEMENTED]
   └──────────────┬───────────────┘
   data/blocking/candidate_pairs_*.parquet
                  │
   ┌──────────────▼───────────────┐
   │ 3. DETERMINISTIC RULES        │  src/models/deterministic_rules.py,         [IMPLEMENTED]
   │    (needs cleaned + pairs)    │  run_rules.py
   └───────┬──────────────┬────────┘
           │              │
  data/matches/      data/non_matches/
  matches_*.parquet      non_matches_*.parquet
           │              │
           │   ┌──────────▼───────────────┐
           │   │ 4. MODELING (probabilistic)│  src/models/ (stub)               [PROPOSED]
           │   └──────────┬───────────────┘
           │     model-confirmed edges
           │              │
   ┌───────▼──────────────▼────────┐
   │ 5. CLUSTERING                  │  src/models/clustering.py (stub)            [PROPOSED]
   │   (union of all confirmed edges)│  — note: currently runs inside stage 3
   └──────────────┬─────────────────┘
   data/clusters/cluster_assignments_*.parquet
```

**Entry point.** The canonical way to run stages 1–3 is the orchestrator
`src/pipeline.py` (`python -m src.pipeline`), which runs them in process, threads
one `run_id` through every artifact name, validates each boundary against
`src/contracts.py`, and writes a `RunManifest` to `data/runs/run_<run_id>.json`.
The per-stage CLIs (`clean.py`, `run_blocking.py`, `run_rules.py`) remain for
ad-hoc/incremental work and resolve their inputs by highest version.

Stages 1–3 are implemented. Stages 4–5 are specified here as the **target
contract**; their sections are marked **PROPOSED** and no consumer should assume
those artifacts exist yet.

---

## Global conventions

These apply to **every** artifact in the pipeline.

| Concern | Contract |
|---|---|
| **On-disk format** | Apache Parquet (engine: `pyarrow`). Parquet preserves dtypes natively, so leading zeros on ID-like string fields survive round-trips. |
| **Raw ingest** | Raw CSV/XLSX is read with `dtype=str` (`clean._load`) so `PATID`, `SSN`, `ZipCD`, `last_4_SSN` keep leading zeros before per-field rules run. |
| **Identity key** | `PATID` (string) uniquely identifies a record end-to-end. It is **never transformed** (passthrough from raw). |
| **Pair identity** | A pair is the ordered tuple `(PATID_A, PATID_B)` with `PATID_A < PATID_B` lexicographically. Established once in blocking via `combinations(sorted(patid_list), 2)` and **carried unchanged** through rules and non-matches. Enforced by a `@pa.dataframe_check` on every pair schema. |
| **Null encoding** | Missing string/categorical values are `NaN`; missing dates are `NaT`. `NaN == anything` is `False`, which the rule engine relies on to enforce "both sides present" (`deterministic_rules._agree`). |
| **Versioning (CLIs)** | Per-stage CLI filenames carry `v{N}_{YYYY_MM_DD}`. `N` auto-increments past the highest existing `v{N}` **in that output directory** (date suffix ignored for the bump). |
| **Versioning (orchestrator)** | `src/pipeline.py` names every artifact `*_{run_id}.parquet` and records the exact paths + row counts + SHA-256 in `data/runs/run_<run_id>.json`. |
| **Lineage** | On the orchestrated path, one `run_id` binds the stage outputs, so a mismatch is impossible. On the CLI path, `run_rules` calls `contracts.assert_patid_coverage(pairs, clean)` to fail loudly if the two inputs are from different cleaning runs. |
| **PHI / logging** | No field-level PHI is ever logged. Logs carry aggregate counts only. Audit metadata (`source_blocks`, `match_rule`, `rules_fired`) gives traceability without exposing values. |

### Multi-valued column serialization

Two cleaned columns are collections, not scalars: `Phones_set` and
`full_name_tokens`. Arrow/Parquet has no native `set` type, so the contract is:

- **In memory:** a **sorted `list[str]`** (despite the name, `Phones_set` is a
  list — `transformations.derive_phones_set` returns `sorted({...})`).
- **On disk:** a native Parquet `list<string>` column, written by
  `clean.write_cleaned` (the single cleaned-write path) and read back as a
  NumPy `ndarray`.
- **Consumers MUST tolerate** `ndarray | list | set | str` and normalize to a
  `frozenset` (`blocking._parse_phone_set`, `deterministic_rules._parse_phone_set`).
  The string branch exists only for legacy CSV inputs of the form `"{'a', 'b'}"`.
- The `CleanedRecords` contract enforces this with a `phones_set_is_listlike`
  check, and `tests/integration/test_serialization_roundtrip.py` guards the
  full `write_cleaned → read_parquet → _parse_phone_set` handoff.

---

## Stage 1 — Cleaned dataset  `[IMPLEMENTED]`

- **Producer:** `src/data/transformations.py::transform_dataframe` (via `src/data/clean.py`)
- **Contract:** `contracts.CleanedRecords` (`strict=False` — passthrough columns allowed)
- **Consumers:** blocking (stage 2) and rules (stage 3, for attributes)
- **Location:** `data/processed/MDM_Population_cleaned_*.parquet`
- **Grain:** one row per source record (`PATID`).

For every cleaned source field the producer keeps the original in `<field>_raw`
(never modified) and writes the standardized value to `<field>_clean`. The
columns the contract validates (what downstream depends on):

### Identity & validity

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID` | string | no | Record identity; passthrough, never transformed. |
| `valid_record` | bool | no | `False` if any field-level rule flagged the record, or both names null. **Blocking drops `valid_record == False`** (`blocking._filter_valid_records`). |

### Cleaned attribute columns (consumed downstream)

| Column | Dtype | Nullable | Consumed by |
|---|---|---|---|
| `FirstNM_clean` | string | yes | blocking (B4), rules |
| `LastNM_clean` | string | yes | blocking (B3,B4,B7,B8,B9), rules |
| `BirthDT_clean` | datetime64[ns] / NaT | yes | blocking (B3,B4,B7,B8), rules |
| `SSN_clean` | string (9 digits) | yes | blocking (B1), rules |
| `last_4_SSN` | string (4 digits) | yes | blocking (B9) |
| `Email_clean` | string | yes | blocking (B6), rules |
| `ZipCD_clean_base` | string (5 digits) | yes | blocking (B7) |
| `AddressLine1_clean` | string | yes | rules (NAME_DOB_ADDRESS) |
| `SexAtBirthDSC_clean` | string ∈ {MALE,FEMALE,OTHER} | yes | rules (NAME_DOB_SEX) |
| `Phones_set` | list&lt;string&gt; (see serialization) | yes | blocking (B5), rules (phone agreement) |
| `_dm_LastNM` | string (optional) | yes | blocking (B3), `fs_splink_enhanced_2` phonetic comparison |
| `_dm_FirstNM` | string (optional) | yes | `fs_splink_enhanced_2` phonetic comparison |

`_dm_LastNM` and `_dm_FirstNM` are **optional** in the pandera contract — cleaned parquets written before Phase E2-2 may omit them, in which case blocking recomputes them transparently via its fallback path.

### Other cleaned / derived columns (produced, not yet consumed downstream)

`MiddleNM_clean`, `SuffixNM_clean`, `AddressLine2_clean`, `CityNM_clean`,
`StateCD_clean`, `ZipCD_clean_ext`, `PrimaryPhoneNBR_clean`, `Phone01NBR_clean`,
`Phone02NBR_clean`, `Phone03NBR_clean`, `full_name_tokens` (list&lt;string&gt;),
`full_name_compact` (string), `Address_normalized` (string; `NaN` when
libpostal is unavailable), plus every `<field>_raw`.

---

## Stage 2 — Candidate pairs  `[IMPLEMENTED]`

- **Producer:** `src/features/blocking.py::run_batch_blocking` (via `run_blocking.py`)
- **Contract:** `contracts.CandidatePairs` (`strict=True` — exact, closed schema)
- **Consumer:** deterministic rules (stage 3)
- **Location:** `data/blocking/candidate_pairs_*.parquet`
- **Grain:** one row per unique candidate pair.
- **Splink-compatible:** the `PATID_A`/`PATID_B` layout maps to splink's `_l`/`_r`.

### Output schema

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` | string | no | Canonical lower PATID (`PATID_A < PATID_B`). |
| `PATID_B` | string | no | Canonical higher PATID. |
| `source_blocks` | string | no | Pipe-delimited, sorted block IDs, e.g. `"B3|B5|B8"`. |
| `n_blocks` | int64 | no | Count of blocks that generated the pair. |

### Block definitions (authoritative — supersedes `Deterministic-Rules-Guide.md`)

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

B2 was removed (subsumed by B3). A per-key **governance cap**
(`settings.governance_threshold`, default 500) caps over-large blocks.

---

## Stage 3 — Deterministic matches & non-matches  `[IMPLEMENTED]`

- **Producer:** `src/models/deterministic_rules.py::apply_rules` (via `run_rules.py`)
- **Inputs:** **both** the candidate pairs (stage 2) **and** the cleaned dataset
  (stage 1, for attribute agreement). The CLI guards their correspondence with
  `assert_patid_coverage`; the orchestrator guarantees it in process.
- **Consumers:** clustering (stage 5) consumes matches; modeling (stage 4)
  consumes non-matches.

### 3a — Matches

- **Contract:** `contracts.Matches` (`strict=True`; empty frames skip validation).
- **Location:** `data/matches/matches_*.parquet`
- **Grain:** one row per **confirmed** pair (≥1 rule fired).

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` | string | no | Carried unchanged from blocking. |
| `PATID_B` | string | no | Carried unchanged from blocking. |
| `match_rule` | string | no | Highest-confidence rule that fired (∈ the 6 below). |
| `confidence` | float64 | no | Confidence of `match_rule`, in `[0.970, 1.000]`. |
| `rules_fired` | string | no | Pipe-delimited list of **every** rule that fired. |
| `is_suspicious` | bool | no | `True` if DOB, last name, or (both-present) SSN disagree. |
| `cluster_id` | int64 | no | Connected-component id, stamped by the writer. |
| `source_blocks` | string | no | Passthrough from blocking. |
| `n_blocks` | int64 | no | Passthrough from blocking. |

**Rules** (descending confidence; first to fire wins `match_rule`):

| Rule | Confidence | Agreement predicate |
|---|---|---|
| `EXACT_SSN` | 1.000 | `SSN_clean` |
| `EMAIL_EXACT` | 0.995 | `Email_clean` |
| `NAME_DOB_EMAIL` | 0.990 | First + Last + DOB + Email |
| `NAME_DOB_PHONE` | 0.985 | First + Last + DOB + phone-set intersection |
| `NAME_DOB_SEX` | 0.980 | First + Last + DOB + Sex |
| `NAME_DOB_ADDRESS` | 0.970 | First + Last + DOB + `AddressLine1_clean` |

### 3b — Non-matches

- **Contract:** `contracts.NonMatches` (identical schema to `CandidatePairs`).
- **Location:** `data/non_matches/non_matches_*.parquet`
- **Grain:** the candidate pairs that **no** deterministic rule confirmed — the
  input to the probabilistic/ML stage. Full blocking provenance preserved.
- **Invariant:** `matches ∪ non_matches == candidate_pairs` and
  `matches ∩ non_matches == ∅`, keyed on `(PATID_A, PATID_B)`.

---

## Stage 4 — Probabilistic / ML matching  `[IMPLEMENTED — fs_splink_enhanced; fs_splink_enhanced_2 in development]`

- **Input:** `non_matches` (stage 3b) **+** the cleaned dataset.
- **Outputs:** two artifacts per run — the rich **`contracts.ProbabilisticMatches`** parquet (model-stage native) and the uniform **`contracts.Edges`** projection (for clustering convergence; see below).
- **Splink** fits this slot directly.

### `contracts.ProbabilisticMatches` schema

| Column | Dtype | Required | Notes |
|---|---|---|---|
| `PATID_A` | string | yes | Canonical lower PATID. |
| `PATID_B` | string | yes | Canonical higher PATID. |
| `match_source` | string == `"model"` | yes | Provenance tag. |
| `score` | float64 ∈ [0, 1] | yes | `match_probability` post-`n_blocks` bump. |
| `match_weight` | float64 | yes | log₂(p/(1-p)) at the same bumped p. |
| `classification_tier` | string ∈ {`auto_merge`, `human_review`, `no_match`} | yes | Threshold-derived tier. |
| `veto_reason` | string, nullable | **optional** | Present on `fs_splink_enhanced` output (deterministic-veto layer); absent on `fs_splink_enhanced_2` output (vetoes moved to upstream deterministic stage). Pandera validates both. |
| `source_blocks` | string, nullable | yes | Pipe-joined block names that produced the candidate pair. |
| `n_blocks` | int ≥ 1, nullable | yes | Count of blocks that produced the pair. |

**E2-1 change:** `veto_reason` is now `Optional[Series[str]]` in pandera — producers without a veto layer omit the column entirely. See `tests/unit/test_contracts_probabilistic_optional_veto.py`.

### Uniform edge schema (`contracts.Edges`)  `[PROPOSED — for Stage 5 union]`

| Column | Dtype | Notes |
|---|---|---|
| `PATID_A` | string | Canonical lower PATID. |
| `PATID_B` | string | Canonical higher PATID. |
| `confidence` | float64 | Deterministic rule confidence, or model probability. |
| `match_source` | string ∈ {`deterministic`, `model`} | Provenance of the edge. |
| `evidence` | string | `rules_fired` for deterministic; feature/score summary for model. |

Mid-confidence model edges and `is_suspicious` deterministic edges SHOULD route
to a **review-queue artifact** rather than to automatic clustering.

---

## Stage 5 — Clustering  `[PROPOSED — currently embedded in stage 3]`

> **Current behavior:** connected components are computed *today* inside stage 3
> (`deterministic_rules.assign_clusters`, called from the rules writer) and
> stamped onto matches as `cluster_id` — using **deterministic edges only**.
> Once stage 4 exists, clustering must run **once** over the union of
> deterministic ∪ model edges, or an entity will be split across stages. The
> target is to extract clustering into its own terminal stage.

- **Input:** the concatenation of all confirmed edges (`contracts.Edges`).
- **Algorithm:** union-find / connected components over `(PATID_A, PATID_B)`.
- **Output:** `contracts.ClusterAssignments` — one row per record incl. singletons.

| Column | Dtype | Notes |
|---|---|---|
| `PATID` | string | Unique; every valid record gets a cluster. |
| `cluster_id` | int64 | Contiguous from 0; deterministic across runs. |

Future merge-safety controls (don't bridge a suspicious edge, cap cluster size,
correlation clustering) belong in this stage.

---

## Invariants & open items

**Invariants every stage upholds** (enforced in `src/contracts.py`):

1. **Pair canonical ordering** `PATID_A < PATID_B`, preserved across all pair stages.
2. **`PATID` immutable** across stages; the only cross-stage join key.
3. **Schema validation at every boundary** (read and write).
4. **No PHI in logs** — aggregate counts only.

**Resolved by the contracts layer:**

- *Boundary validation* — `contracts.validate(...)` wired into `clean.py`,
  `run_blocking.py`, `run_rules.py`, and `pipeline.py`; a renamed/missing/
  malformed column now fails loudly at the boundary instead of silently
  NaN-suppressing matches inside the rule engine.
- *Lineage* — orchestrator threads one `run_id` + `RunManifest`; the CLI path is
  guarded by `assert_patid_coverage`.
- *Config* — paths and `governance_threshold` centralized in
  `src/config/config.py::Settings`.

**Remaining (follow-ups):**

- **Duplicated constants & parsers.** `blocking.py` and `deterministic_rules.py`
  still declare their own `COL_*` and `_parse_phone_set`; point them at
  `contracts.py` (names) and a shared `pairs.py` (parser) so there is one copy.
- **`verify_pipeline.py`** still reimplements the cleaned write and has a stale
  "cleaning writes CSV" docstring — switch it to `clean.write_cleaned` /
  `pipeline.run_pipeline`.
- **`Deterministic-Rules-Guide.md`** block table is stale (lists B5 as single
  phone, B8 as initials, a 3,000 cap); the block table above is authoritative.
- **Clustering placement** — extract from stage 3 into stage 5 once modeling exists.
