# Data Contract — eMPI Pipeline

Single source of truth for the **data that moves between pipeline stages**
(stages 1-5) **and the resolved-output store the API layer maintains
downstream of them** (stage 6): the schema, dtype, nullability, file
location, and serialization of every artifact handed from one stage to the
next. Where `Data-Cleaning-Guide.md` governs *how fields are cleaned* and
`Deterministic-Rules-Guide.md` governs *which pairs match*, this document
governs *the shape of the data at each boundary* so any stage can be
developed, tested, or replaced against a stable interface.

The authority for each contract is the code, cited inline as `file:symbol`. If
code and this document disagree, the code wins and this document is the bug —
update it in the same change.

**Executable counterpart:** `src/contracts.py` implements stages 1-5 as
pandera `DataFrameModel`s (the bulk frames) and pydantic models (the
`Settings` in `src/config.py` and the `RunManifest`). The orchestrator
`src/pipeline.py` and the stage entry points validate every boundary against
them. Stage 6 (resolved-output storage) has no pandera contract — it is
row-oriented, not a bulk frame — and is instead enforced by
`src/api/store.py`'s `CREATE TABLE` DDL (SQLite) and
`src/api/parquet_backend.py`'s `_SCHEMAS` (Parquet), which this document's
Stage 6 section mirrors. This document is the human-readable spec; the code
is the enforcement — keep them mirrored.

_Last updated: 2026-07-14._

---

## Pipeline overview

```
            data/raw/*.csv|xlsx
                  │  (dtype=str on read — preserves leading zeros)
   ┌──────────────▼───────────────┐
   │ 1. CLEAN / TRANSFORM          │  src/preprocessing/transformations.py,     [IMPLEMENTED]
   └──────────────┬───────────────┘  src/preprocessing/clean.py
   data/processed/MDM_Population_cleaned_*.parquet
                  │
   ┌──────────────▼───────────────┐
   │ 2. BLOCKING (stacked)          │  src/preprocessing/stacked_blocking.py     [IMPLEMENTED]
   └──────────────┬───────────────┘
   data/blocking/candidate_pairs_*.parquet
                  │
   ┌──────────────▼───────────────┐
   │ 3. DETERMINISTIC RULES        │  src/models/deterministic_rules.py         [IMPLEMENTED]
   │    (needs cleaned + pairs)    │
   └───┬──────────┬──────────┬─────┘
       │          │          │
  data/matches/  data/     data/rejects/
  (auto-merge)   non_matches/ (dropped, audit-only)
       │          │
       │   ┌──────▼────────────────────┐
       │   │ 4. FS MATCHER              │  src/models/fs_matcher/               [IMPLEMENTED]
       │   │ (candidate/feature         │  candidate + feature generator for
       │   │  generator for a           │  a downstream GBT — does NOT feed
       │   │  downstream GBT)           │  clustering
       │   └──┬───────────────────┬─────┘
       │  data/matches_model/  data/FS_output/
       │  (audit, no reader)   (GBT candidates, no reader yet)
       │
   ┌───▼─────────────────────────────┐
   │ 5. CLUSTERING                    │  src/models/clustering.py                [IMPLEMENTED]
   │   (deterministic auto-merge      │  — terminal stage; does NOT union in
   │    edges only)                   │    Stage 4 output (team decision)
   └──────────────┬───────────────────┘
   data/clusters/cluster_assignments_*.parquet
                  │
   ┌──────────────▼───────────────┐
   │ 6. RESOLVED-OUTPUT INDEX       │  src/api/publish.py, incremental.py,      [IMPLEMENTED]
   │    (mutable; API layer)       │  publish_local.py, local_score.py
   └───────────────────────────────┘
   data/empi.db  (or  data/local_index/*.parquet)
```

**Entry point.** The canonical way to run all five stages is the orchestrator
`src/pipeline.py` (`python -m src.pipeline`), which runs them in process,
threads one `run_id` through every artifact name, validates each boundary
against `src/contracts.py`, and writes a `RunManifest` to
`data/runs/run_<run_id>.json`.

**Standalone per-stage CLIs are dev/debug-only, not production.** `clean.py`,
`run_blocking.py`, and `run_rules.py` exist for re-running or inspecting one
stage in isolation during development; they resolve their own inputs by
highest version (see "Naming conventions" below) rather than through a
`RunManifest`, and their output is not part of the production data flow — the
orchestrator never reads it, and neither does the API. `run_blocking.py` in
particular produces a **structurally different, narrower** candidate pool than
the orchestrator (see Stage 2) — never substitute one for the other.

All six stages are implemented and in production, on both of stage 6's
backends (SQLite and Parquet local mode — see its own section).

---

## Global conventions

These apply to **every** artifact in the pipeline.

| Concern | Contract |
|---|---|
| **On-disk format** | Apache Parquet (engine: `pyarrow`). Parquet preserves dtypes natively, so leading zeros on ID-like string fields survive round-trips. |
| **Raw ingest** | Raw CSV/XLSX is read with `dtype=str` (`clean._load`) so `PATID`, `SSN`, `ZipCD`, `last_4_SSN` keep leading zeros before per-field rules run. |
| **Identity key** | `PATID` (string) uniquely identifies a record end-to-end. It is **never transformed** (passthrough from raw). |
| **Pair identity** | A pair is the ordered tuple `(PATID_A, PATID_B)` with `PATID_A < PATID_B` lexicographically. Established once in blocking via `combinations(sorted(patid_list), 2)` and **carried unchanged** through rules, non-matches, and rejects. Enforced by a `@pa.dataframe_check` on every pair schema. |
| **Null encoding** | Missing string/categorical values are `NaN`; missing dates are `NaT`. `NaN == anything` is `False`, which the rule engine relies on to enforce "both sides present" (`deterministic_rules._agree`). |
| **Naming (production)** | `src/pipeline.py` names every artifact `*_<run_id>.parquet`, where `run_id` is a UTC timestamp (`20260617T043941Z`) — lexicographically sortable, so "latest" needs no separate version counter. Every path + row count + SHA-256 is recorded in `data/runs/run_<run_id>.json`. This is the **only production convention**. |
| **Naming (dev/debug CLIs)** | The standalone per-stage CLIs (`clean.py`, `run_blocking.py`, `run_rules.py`) instead write `v{N}_{YYYY_MM_DD}` — `N` auto-incremented past the highest existing `v{N}` **in that output directory** (date suffix ignored for the bump). Not consumed by the orchestrator or the API; see "Naming conventions" below. |
| **Lineage** | On the orchestrated path, one `run_id` binds every stage output, so a mismatch is structurally impossible — this is also why `src/models/fs_matcher/train.py` resolves its inputs from the latest `RunManifest` rather than "newest file in a directory" (see Stage 4). On the standalone-CLI path, `run_rules.py` guards the two inputs it's given with `contracts.assert_patid_coverage`. |
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

- **Producer:** `src/preprocessing/transformations.py::transform_dataframe` (via `src/preprocessing/clean.py`)
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

### Other cleaned / derived columns (produced, not yet consumed downstream)

`MiddleNM_clean`, `SuffixNM_clean`, `AddressLine2_clean`, `CityNM_clean`,
`StateCD_clean`, `ZipCD_clean_ext`, `PrimaryPhoneNBR_clean`, `Phone01NBR_clean`,
`Phone02NBR_clean`, `Phone03NBR_clean`, `full_name_tokens` (list&lt;string&gt;),
`full_name_compact` (string), `Address_normalized` (string; `NaN` when
libpostal is unavailable), plus every `<field>_raw`.

---

## Stage 2 — Candidate pairs  `[IMPLEMENTED]`

- **Producer (production):** `src/preprocessing/stacked_blocking.py::run_stacked_blocking`,
  called from `src/pipeline.py` — the **8-block scheme ∪ a typo-tolerant
  q-gram pass, pruned by graph meta-blocking** (see `docs/Blocking-Guide.md`
  for the full stacked-blocker design).
- **Contract:** `contracts.CandidatePairs` (`strict=True` — exact, closed schema)
- **Consumer:** deterministic rules (stage 3)
- **Location:** `data/blocking/candidate_pairs_*.parquet`
- **Grain:** one row per unique candidate pair.
- **Splink-compatible:** the `PATID_A`/`PATID_B` layout maps to splink's `_l`/`_r`.

> **⚠️ Do not confuse with the standalone `run_blocking.py` CLI.** It runs
> **only** the 8-block scheme below — no q-gram pass, no meta-blocking prune —
> producing a **structurally narrower, algorithmically different** candidate
> pool than the orchestrator, even though both write files matching
> `candidate_pairs_*.parquet` into the same `data/blocking/` directory. Never
> substitute one for the other when training or serving the Stage 4 FS
> matcher — a wrong resolution isn't just "stale data," it's a different
> candidate-generation algorithm than the one the model will serve against in
> production. This is exactly the risk `src/models/fs_matcher/train.py`'s
> `RunManifest`-based input resolution closes structurally (see Stage 4).

### Output schema

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` | string | no | Canonical lower PATID (`PATID_A < PATID_B`). |
| `PATID_B` | string | no | Canonical higher PATID. |
| `source_blocks` | string | no | Pipe-delimited, sorted block IDs, e.g. `"B3|B5|B8"`. |
| `n_blocks` | int64 | no | Count of blocks that generated the pair. |

### Block definitions (the 8-block scheme; authoritative — supersedes `Deterministic-Rules-Guide.md`)

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
(`settings.governance_threshold`, default 500) caps over-large blocks. The
production path additionally unions a q-gram cosine pass and prunes via graph
meta-blocking (ARCS + Cardinality Node Pruning) — see `docs/Blocking-Guide.md`.

---

## Stage 3 — Deterministic matches, non-matches & rejects  `[IMPLEMENTED]`

- **Producer:** `src/models/deterministic_rules.py::apply_rules` +
  `classify_non_matches`, invoked from `src/pipeline.py` (which also owns the
  auto-merge/review **tier split** described below — the tier routing logic
  itself lives in `pipeline.py`, not inside `apply_rules`).
- **Inputs:** **both** the candidate pairs (stage 2) **and** the cleaned dataset
  (stage 1, for attribute agreement). The orchestrator guarantees their
  correspondence in process; the standalone `run_rules.py` CLI guards it with
  `assert_patid_coverage`.
- **Consumers:** clustering (stage 5) consumes `matches`; the Stage 4 FS
  matcher consumes `non_matches`; `rejects` has no downstream reader today
  (terminal audit artifact).

**Rules** (descending confidence; first to fire wins `match_rule`). First/last
name predicates are **fuzzy** (Jaro-Winkler ≥ 0.92 or Damerau-Levenshtein ≤ 1);
all other predicates are exact. Kept in sync with `contracts.RULE_NAMES`.
Every rule fires and records full provenance regardless of tier — only the
routing below differs by tier.

| Rule | Confidence | Tier | Agreement predicate |
|---|---|---|---|
| `SSN_DOB` | 1.000 | auto-merge | `SSN_clean` + `BirthDT_clean` |
| `NAME_DOB_EMAIL` | 0.990 | auto-merge | First + Last + DOB + Email |
| `NAME_DOB_PHONE` | 0.985 | auto-merge | First + Last + DOB + phone-set intersection |
| `NAME_DOB_SEX` | 0.980 | **review** | First + Last + DOB + Sex |
| `NAME_DOB_ADDRESS` | 0.970 | **review** | First + Last + DOB + `AddressLine1_clean` |

### 3a — Matches (auto-merge tier only)

- **Contract:** `contracts.Matches` (`strict=True`; empty frames skip validation).
- **Location:** `data/matches/matches_*.parquet`
- **Grain:** one row per pair confirmed by an **auto-merge-tier** rule
  (`contracts.AUTO_MERGE_RULE_NAMES` = `SSN_DOB`, `NAME_DOB_EMAIL`,
  `NAME_DOB_PHONE`). **`NAME_DOB_SEX` and `NAME_DOB_ADDRESS` never appear
  here** — they're review-tier; a pair they confirm routes to `non_matches` +
  `review_evidence` instead (see 3b).

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` | string | no | Carried unchanged from blocking. |
| `PATID_B` | string | no | Carried unchanged from blocking. |
| `match_rule` | string | no | Highest-confidence **auto-merge-tier** rule that fired (∈ `AUTO_MERGE_RULE_NAMES`). |
| `confidence` | float64 | no | Confidence of `match_rule`, in **`[0.985, 1.000]`** — the auto-merge floor, not the full 5-rule range. |
| `rules_fired` | string | no | Pipe-delimited list of **every** rule that fired (any tier). |
| `is_suspicious` | bool | no | `True` if DOB, last name, or (both-present) SSN disagree. |
| `high_fanout_ssn` | bool | no | `True` if the pair's shared SSN is carried by ≥ `EMPI_SSN_FANOUT_THRESHOLD` patients. |
| `cluster_id` | int64 | no | Connected-component id, stamped by the writer. |
| `source_blocks` | string | no | Passthrough from blocking. |
| `n_blocks` | int64 | no | Passthrough from blocking. |

### 3b — Non-matches (the Stage 4 FS matcher's input — a union of two provenances)

- **Contract:** `contracts.NonMatches` (identical schema to `CandidatePairs`).
- **Location:** `data/non_matches/non_matches_*.parquet`
- **Grain:** `non_matches` is a **union of two distinct provenances**, and
  Stage 4's FS matcher scores both **indistinguishably** — this is easy to
  miss if you only look at the `review_evidence` companion artifact below:
  1. **Review-tier rule-confirmed pairs** — `NAME_DOB_SEX` (~65% adjudicated
     precision) or `NAME_DOB_ADDRESS` (~67%) fired, but the rule's own tier
     means it's routed here rather than auto-merged.
  2. **Genuine rule-undecided pairs** — no rule fired at all, and
     `classify_non_matches` found **fewer than 3** strong-identifier
     contradictions (so not confident enough to reject either).
- **`review_evidence_<run_id>.parquet`** (companion, same directory): the
  full column set for provenance (1) above — `match_rule`, `confidence`,
  `rules_fired`, etc. — that the closed `NonMatches`/`CandidatePairs` schema
  trims. Not part of a strict pandera contract (nothing in the pipeline reads
  it back), but it has exactly one consumer: `src/api/publish.py`, which
  surfaces *why* a review-tier pair was flagged rather than just that it was.

### 3c — Rejects (terminal, audit-only)

- **Contract:** `contracts.Rejects` (subclasses `CandidatePairs`; `strict=True`).
- **Location:** `data/rejects/rejects_*.parquet`
- **Grain:** unconfirmed pairs with **≥ 3** of {full SSN, first, last, DOB}
  strictly disagreeing (calibrated on real run `real_20260620`: 2 conflicts
  still carry ~10% true matches, 3 carry ~0%). **Dropped** from the pipeline —
  nothing downstream reads this artifact back; it exists purely for
  audit/compliance.

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` / `PATID_B` / `source_blocks` / `n_blocks` | (as `CandidatePairs`) | no | Passthrough from blocking. |
| `n_contradictions` | int64 | no | Count of strong-identifier disagreements. |
| `decision` | string | no | Always `"reject"` in this artifact. |
| `reject_rule` | string | no | The reject rule that fired (`contracts.REJECT_RULE_NAMES` = `("STRONG_ID_CONFLICT",)`) — always populated here, since every row that survives the `decision == "reject"` filter already fired it. |

**Invariant:** `matches ⊎ non_matches ⊎ rejects == candidate_pairs` (disjoint
union), keyed on `(PATID_A, PATID_B)`.

---

## Stage 4 — Fellegi-Sunter matcher (candidate/feature generator)  `[IMPLEMENTED]`

- **Producer:** `src/models/fs_matcher/` (`FSMatcher.score` for serving,
  `python -m src.models.fs_matcher.train` for offline training + promotion).
  Invoked from `src/pipeline.py` when an active model is resolvable
  (`registry.resolve_active_model`) and the `non_matches` pool is non-empty;
  otherwise Stage 4 is skipped with a clear log line.
- **Role:** scores the Stage 3 `non_matches` pool and surfaces a **candidate +
  feature set for a downstream GBT** — it does **not** produce edges for
  clustering. Full MLOps lifecycle (train/promote/serve/swap, config knobs,
  deploy-gate): `docs/FS-Matcher-Production-Guide.md`. The trained model
  artifacts themselves (`fs_model_<ts>.json`, `.meta.json`, `active.json`)
  live in `models/fs/` — not under `data/` — gitignored, VM-populated; see
  that guide for the full store layout.
- **Training input resolution:** `train.py` resolves its `cleaned` +
  `candidate_pairs` inputs from the **latest `RunManifest`**
  (`data/runs/run_<run_id>.json`), not by globbing "newest file in a
  directory" — this guarantees same-run lineage *and* guarantees the stacked
  blocker's output is used, never `run_blocking.py`'s narrower pool (see
  Stage 2's warning). Directory-latest resolution is a documented fallback
  only for the rare case no manifest exists yet.

### 4a — ProbabilisticMatches (full audit frame)

- **Contract:** `contracts.ProbabilisticMatches` (`strict=True`).
- **Location:** `data/matches_model/matches_model_<run_id>.parquet`
- **Grain:** every scored `non_matches` pair, all tiers including `no_match`.
- **Status:** terminal — no downstream reader today. Kept for auditability;
  deliberately **not** unioned into clustering.

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` / `PATID_B` | string | no | Canonical pair. |
| `match_source` | string | no | Always `"model"`. |
| `score` | float64 | no | Match probability, `[0.0, 1.0]`. |
| `match_weight` | float64 | no | Log-Bayes-factor match weight. |
| `classification_tier` | string | no | ∈ `{auto_merge, human_review, no_match}` — **informational only**; Stage 4 routes nothing, it labels tiers for audit/the GBT. |
| `veto_reason` | string | yes (optional) | Present only if the producer applies a veto layer; the production matcher omits the column entirely. |
| `source_blocks` / `n_blocks` | string / int64 | yes | Passthrough from blocking, where available. |

### 4b — FSFeatures (the GBT deliverable)

- **Contract:** `contracts.FSFeatures` (`strict=False` — `gamma_<field>` /
  `bf_<field>` feature columns are dynamic extras, checked for presence by
  `validate_fs_features` rather than pandera's closed-schema mode).
- **Location:** `data/FS_output/fs_features_<run_id>.parquet` (pipeline,
  candidate-filtered) and `data/FS_output/fs_features_train_<version>.parquet`
  (train CLI, labeled).
- **Grain:** candidates only — filtered to `match_probability >=
  settings.fs_review_floor` (0.40 default; doubles as the tier boundary *and*
  the candidate cutoff).
- **Status:** terminal today — no in-repo GBT consumer yet; this is the
  contract a future GBT training/serving path will read.

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` / `PATID_B` | string | no | Canonical pair. |
| `match_probability` | float64 | no | Same as `ProbabilisticMatches.score`. |
| `match_weight` | float64 | no | Same as `ProbabilisticMatches.match_weight`. |
| `classification_tier` | string | no | Informational, as above. |
| `label` | float64 (0.0/1.0) | yes | Present (non-null) only on the training feature set; absent/null when scoring. |
| `gamma_<field>` (×7) | int | no | Per-comparison level index (one per Splink comparison: FirstNM, LastNM, BirthDT, SSN, Email, Phones, Address). |
| `bf_<field>` (×N) | float64 | no | Per-comparison Bayes-factor bits, incl. `bf_tf_adj_*` for term-frequency-adjusted fields. |

---

## Stage 5 — Clustering  `[IMPLEMENTED]`

`src/models/clustering.py` is the terminal stage of `src/pipeline.py`, run
once per pipeline invocation after stage 3. **It clusters deterministic
auto-merge matches only** (`AUTO_MERGE_RULES`-tier pairs) — review-tier
confirmations, non-matches, and the Stage 4 FS output are all excluded. Stage
3 still stamps a per-pair `cluster_id` onto `matches` (via
`clustering.assign_clusters`, used internally by
`deterministic_rules.get_match_stats` for audit stats), but the authoritative,
singleton-inclusive assignment is `clustering.build_cluster_assignments`'s
output, written to `data/clusters/`.

**This is a deliberate, current design decision, not a placeholder awaiting
Stage 4:** the team decided the FS matcher is a candidate/feature generator
for a downstream GBT, not an edge source, so its output does **not** union
into clustering — `contracts.Edges` (below) is *not on the roadmap*.

- **Input:** `matches` (deterministic auto-merge edges) + `cleaned` (for the
  full valid-record population, so singletons get a cluster too).
- **Algorithm:** union-find / connected components over `(PATID_A, PATID_B)`.
- **Output:** `contracts.ClusterAssignments` — one row per valid record incl.
  singletons, written to `data/clusters/cluster_assignments_<run_id>.parquet`
  and referenced from the `RunManifest`.

| Column | Dtype | Notes |
|---|---|---|
| `PATID` | string | Unique; every valid record gets a cluster. |
| `cluster_id` | int64 | Contiguous from 0; deterministic across runs. |

Future merge-safety controls (don't bridge a suspicious edge, cap cluster
size, correlation clustering) belong in this stage.

### `contracts.Edges` — not on the current roadmap

A uniform edge schema (`PATID_A`, `PATID_B`, `confidence`, `match_source`,
`evidence`) that would let clustering consume one concatenated
deterministic+probabilistic frame is still defined in `contracts.py` as a
historical placeholder, but **no stage produces it and none is planned to** —
see the design decision above. Don't build against it.

---

## Stage 6 — Resolved-output index (API layer)  `[IMPLEMENTED]`

Unlike stages 1-5 (each an immutable, `run_id`-stamped Parquet file), this is a
**mutable resolved store**: a batch publish or an incremental score upserts
prior state rather than writing a new artifact each time. It is what the
FastAPI service and the `empi-dashboard/` Next.js app actually read and write
— nothing downstream of stage 5 talks to `data/clusters/*.parquet` directly.

- **Producers:**
  `src/api/publish.py::publish_run` — one `RunManifest`'s `clusters` /
  `matches` / `non_matches` / `cleaned` (+ `review_evidence` if present) →
  resolved entities. Batch-only, run once per full pipeline run.
  `src/api/incremental.py::score_records` — one or a few new records scored
  against the *existing* resolved population, no full pipeline re-run. Used
  by `POST /records/score` and the local CLI (`src/api/local_score.py`).
- **Consumers:** `src/api/routers/*` (records, dashboard, audit, runs) and
  `empi-dashboard/`.
- **Two interchangeable backends**, both fully implementing
  `src/api/index_backend.py::IndexBackend` (every table below, both the
  batch-publish and incremental-score paths, and the reviewer audit log):
  - **SQLite** (`src/api/store.py`, `data/empi.db`) — the live,
    multi-request service. Default (`EMPI_INDEX_BACKEND=sqlite`).
  - **Parquet local mode** (`src/api/parquet_backend.py::ParquetIndexBackend`,
    `data/local_index/*.parquet`) — no DB required; a fully self-contained
    local dev/CI/batch/incremental/dashboard deployment with zero SQLite
    dependency. `EMPI_INDEX_BACKEND=parquet`. Batch publish:
    `python -m src.pipeline` then `python -m src.api.publish_local --run-id
    <id>`. Incremental: `python -m src.api.local_score --input record.json`,
    or the same FastAPI service with `EMPI_INDEX_BACKEND=parquet` set.
    One process-local lock (`src/api/deps.py::_PARQUET_BACKEND_LOCK`)
    serializes requests against this backend — it was designed for one-shot
    CLI use (load once, commit once, exit), so a live FastAPI app driving
    concurrent requests against it needs that serialization; SQLite's own
    engine already handles concurrent writers without one.
- **Reconciliation invariant ("sticky unmerge"):** a PATID that has ever
  appeared in `audit_log.patids` is **reviewer-locked** — no later batch
  publish may repoint its `mid`. Its would-be new grouping is written to
  `entity_suggestion` instead, visible but not auto-applied. Only another
  explicit `/audit/*` action can move a locked PATID again.
- **Boolean columns are stored as `int64` (0/1), not `bool`** — the one
  dtype convention stage 6 deliberately breaks from stages 1-5 (where
  `valid_record`/`is_suspicious`/`high_fanout_ssn` are native `bool`), to
  keep every column trivially portable between SQLite's dynamic typing and
  Parquet's columnar typing without a cast at the boundary.

### 6a — Entities & membership

`entity` — one row per resolved entity (singleton or merged cluster).

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `mid` | string | no (key) | `M-{6-digit sequence}` — minted by `next_mid()` (one at a time) or `max_mid_sequence() + 1` (bulk publish loop). |
| `run_id` | string | no | Run (or incremental-score job id) that last touched this entity. |
| `origin` | string ∈ {`deterministic`, `review`, `merge`, `none`} | no | How the entity was formed. |
| `is_merged` | int64 (0/1) | no | `1` iff `>1` unlocked member. |
| `confidence` | float64 | yes | Highest-confidence deterministic pair founding the entity. |
| `match_rule` | string | yes | Rule name for that founding pair; `None` for singletons. |
| `evidence` | string | yes | `rules_fired` (or a manual-merge note) for the founding pair. |
| `updated_utc` | string (ISO-8601) | no | |

Backends: SQLite `entity` table — IMPLEMENTED. Parquet `entity.parquet` —
IMPLEMENTED, both incremental upsert and bulk write from a batch publish
(`ParquetIndexBackend.upsert_entities_bulk`).

`entity_member` — one row per PATID, resolving it to exactly one `mid`.

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `patid` | string | no (key) | An entity's members are every row sharing its `mid`. |
| `mid` | string | no | References `entity.mid`. |
| `is_primary` | int64 (0/1) | no | The lexicographically smallest PATID in the entity, by convention. |
| `added_by` | string | no | `"pipeline"` or a reviewer id. |
| `updated_utc` | string (ISO-8601) | no | |

Backends: SQLite — IMPLEMENTED. Parquet — IMPLEMENTED (same split as
`entity`: incremental upsert and bulk `upsert_entity_members_bulk`).

### 6b — Review & reconciliation

`review_candidate` — an unresolved pair the pipeline routed to human review
(review-tier rule-confirmed, or uncertain-but-not-rejected).

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `id` | int64 | no (key, SQLite only — `AUTOINCREMENT`) | Parquet dedups on the 3-column key below instead of a surrogate id. |
| `patid_a`, `patid_b` | string | no | Canonical pair, `patid_a < patid_b` (same ordering as `contracts.CandidatePairs`). |
| `match_rule` | string | yes | Set only for the review-tier rule-confirmed subset (`NAME_DOB_SEX` / `NAME_DOB_ADDRESS`); `None` for uncertain pairs. |
| `confidence` | float64 | yes | |
| `evidence` | string | yes | `rules_fired`, when `match_rule` is set. |
| `source_blocks` | string | yes | Pipe-delimited block IDs, passthrough from stage 2. |
| `run_id` | string | no | |
| `created_utc` | string (ISO-8601) | no | |
| `fs_match_probability` | float64 | yes | Stage 4 (FS matcher) score, when scored. |
| `fs_classification_tier` | string | yes | Stage 4's tier label. |

Key / replace semantics: a **batch publish replaces wholesale per `run_id`**
(`replace_review_candidates_for_run` — a pair no longer in the latest run's
`non_matches` disappears rather than lingering as a stale suggestion);
**incremental scoring appends**, deduped on `(patid_a, patid_b, run_id)`.

Backends: SQLite — IMPLEMENTED (both semantics). Parquet — IMPLEMENTED
(both semantics: incremental append via `insert_review_candidates`, batch
replace-per-run via `replace_review_candidates_for_run`).

`entity_suggestion` — a reviewer-locked PATID's would-be new grouping,
recorded but not applied (see the sticky-unmerge invariant above).

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `patid` | string | no (key) | |
| `run_id` | string | no | |
| `suggested_mid` | string | no | Synthetic id (`SUGGESTED-{run_id}-{cluster_id}`) — never a real `entity.mid`. |
| `created_utc` | string (ISO-8601) | no | |

Backends: SQLite — IMPLEMENTED. Parquet — IMPLEMENTED (incremental upsert
and bulk write from a batch publish, `upsert_suggestions_bulk`).

### 6c — Persisted lookup indexes (incremental-scoring support)

`block_key` — on-disk mirror of `blocking.BlockingIndex`'s posting lists, so
a single incoming record's candidates are an indexed point lookup instead of
an in-memory rebuild over the whole population.

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `block_id` | string ∈ {B1,B3,B4,B5,B6,B7,B8,B9} | no (key, composite) | Same block definitions as stage 2. |
| `key_value` | string | no (key, composite) | B1 (SSN) / B6 (email) values are SHA-256-hashed (`store.hash_block_key`) before storage — direct identifiers, equality-only use. |
| `patid` | string | no (key, composite) | |

Backends: SQLite — IMPLEMENTED (full rebuild via `replace_block_keys` on
every batch publish; incremental append via `add_block_keys`). Parquet —
IMPLEMENTED (same split — full rebuild on batch publish, incremental append
between).

`cleaned_attrs` — a query-by-patid mirror of the stage 1 `CleanedRecords`
contract, one row per valid PATID, so incremental rule evaluation doesn't
re-read the full ~163k-row stage 1 Parquet per request.

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `patid` | string | no (key) | |
| `first_nm`, `last_nm` | string | yes | |
| `birth_dt` | string (`YYYY-MM-DD`) | yes | Pre-formatted — unlike stage 1's native `datetime64[ns]`. |
| `ssn`, `ssn_last4` | string | yes | |
| `email` | string | yes | |
| `zip_base` | string | yes | |
| `address1` | string | yes | |
| `sex` | string | yes | |
| `phones_json` | string | yes | `json.dumps(sorted(phone_set))` — stage 1's `Phones_set` re-serialized as JSON text (this table is a point-lookup cache, not written via `write_cleaned`, so it doesn't get the native `list<string>` Parquet column stage 1 uses). |
| `run_id` | string | no | |

Backends: SQLite — IMPLEMENTED (full rebuild via `replace_cleaned_attrs`;
incremental upsert via `upsert_cleaned_attrs`). Parquet — IMPLEMENTED (same
split).

### 6d — Display denormalization (dashboard read side)

`record_attrs` — display fields for `GET /records` / `GET /clusters/{mid}`,
denormalized from stage 1's `cleaned` at publish time so the dashboard never
does a per-request read of the full cleaned Parquet.

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `patid` | string | no (key) | |
| `first_name`, `last_name`, `birth_date`, `ssn_last4`, `email`, `zip_code`, `address1`, `sex`, `phone` | string | yes | `birth_date` pre-formatted `YYYY-MM-DD`; `phone` is `PrimaryPhoneNBR_clean`. |
| `run_id` | string | no | |

Backends: SQLite — IMPLEMENTED (`store.upsert_record_attrs_bulk`). Parquet —
IMPLEMENTED (`ParquetIndexBackend.upsert_record_attrs_bulk`). Upsert
semantics by `patid` on both (a batch publish replaces existing rows for the
patids it touches — values can legitimately change run to run for the same
patid — via a batched filter-then-concat, not a per-row loop).

`record_raw` — the "View Raw Data" drawer's un-scrubbed source fields, one
JSON blob per PATID.

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `patid` | string | no (key) | |
| `raw_json` | string | no | `json.dumps` of `publish.RAW_COLUMNS` (every `*_raw` passthrough field from stage 1). |
| `run_id` | string | no | |

Backends: SQLite — IMPLEMENTED. Parquet — IMPLEMENTED
(`ParquetIndexBackend.upsert_record_raw_bulk`), same upsert-by-`patid`
semantics as `record_attrs`.

### 6e — Reviewer audit log

`audit_log` — append-only record of every manual `/audit/merge` or
`/audit/unmerge` action. Also the source of truth for which PATIDs are
reviewer-**locked** (see the sticky-unmerge invariant above).

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `id` | int64 | no (key) | SQLite: native `AUTOINCREMENT`. Parquet: scan existing max `id` + 1 (no autoincrement primitive) — same convention as `mid` minting. |
| `ts_utc` | string (ISO-8601) | no | |
| `user` | string | no | Reviewer id from the trusted `X-Reviewer-Id` header. |
| `action` | string ∈ {`merge`, `unmerge`, `split`} | no | `split` is reserved — not yet emitted by any route. |
| `patids` | string | no | Comma-joined. |
| `mid` | string | no | |
| `prev_state`, `next_state` | string | no | Human-readable state labels (e.g. `"Merged"`, `"Needs review"`). |
| `run_id` | string | yes | |

Backends: SQLite — IMPLEMENTED. Parquet — IMPLEMENTED
(`ParquetIndexBackend.insert_audit_log`/`list_audit_log`).
`ParquetIndexBackend.locked_patids()` reads this table for real (`patids`
column, split on `,`, unioned across rows) — no longer the empty-set stub
local mode started with.

### Status summary

| Table | SQLite | Parquet local mode |
|---|---|---|
| `entity` | IMPLEMENTED | IMPLEMENTED |
| `entity_member` | IMPLEMENTED | IMPLEMENTED |
| `review_candidate` | IMPLEMENTED | IMPLEMENTED |
| `entity_suggestion` | IMPLEMENTED | IMPLEMENTED |
| `block_key` | IMPLEMENTED | IMPLEMENTED |
| `cleaned_attrs` | IMPLEMENTED | IMPLEMENTED |
| `record_attrs` | IMPLEMENTED | IMPLEMENTED |
| `record_raw` | IMPLEMENTED | IMPLEMENTED |
| `audit_log` | IMPLEMENTED | IMPLEMENTED |

All nine tables are fully implemented on both backends as of 2026-07-14 —
the `to-do.md` operationalization plan (Phases 1-3: bulk-publish support,
dashboard read-side parity, and `audit_log`/reviewer-lock parity) is done.
The two backends now cover the same functional surface: batch pipeline runs,
incremental single/few-record scoring, the full dashboard read side, and
reviewer merge/unmerge — see the Parquet-local-mode commands in this stage's
intro bullets above.

---

## The `RunManifest`

Written by `src/pipeline.py` to `data/runs/run_<run_id>.json` (pydantic model,
`contracts.RunManifest`), threading one `run_id` through every artifact so
lineage is structurally guaranteed on the orchestrated path. It exists
specifically to replace fragile "latest version in the directory" resolution —
`src/models/fs_matcher/train.py` relies on this guarantee directly (Stage 4).

| Field | Type | Notes |
|---|---|---|
| `run_id` | string | UTC timestamp, e.g. `20260617T043941Z`. |
| `created_utc` | string | Run start time. |
| `git_sha` | string, optional | Commit the run executed against. |
| `raw_input`, `cleaned`, `candidate_pairs`, `matches`, `non_matches` | `ArtifactRef` | Always populated. |
| `rejects`, `clusters`, `review_evidence` | `ArtifactRef`, optional | Populated by every current run; optional for backward-compat with older manifest shapes. |
| `matches_model`, `fs_features` | `ArtifactRef`, optional | Populated **only** when an active FS model scored the run (null if Stage 4 was skipped). |
| `counts` | `dict[str, int]` | `raw_rows`, `cleaned_rows`, `valid_records`, `candidate_pairs`, `matches`, `non_matches`, `rejects`, `clusters`, `total_clusters`. |

Each `ArtifactRef` carries a project-root-relative `path`, `rows`, and a
`sha256` of the file.

---

## `data/` directory audit — is everything necessary?

One row per subdirectory: who writes it, who reads it back (if anyone), and
whether that's by design.

| Directory | Producer(s) | Consumer(s) | Status |
|---|---|---|---|
| `data/raw/` | (external upload) | Stage 1 (`clean.py`) | Active — pipeline input |
| `data/processed/` | Stage 1 | Stage 2, Stage 3 (attribute join), `fs_matcher/train.py` | Active |
| `data/blocking/` | Stage 2 (stacked, production) / standalone `run_blocking.py` (8-block-only, dev — see warning) | Stage 3, `fs_matcher/train.py` | Active |
| `data/matches/` | Stage 3 (auto-merge tier) | `src/api/publish.py`, `src/evaluation/rule_eval.py` | Active |
| `data/non_matches/` | Stage 3 (`non_matches_*` + `review_evidence_*` companion) | Stage 4 (scores `non_matches`), `src/api/publish.py` (both files) | Active |
| `data/rejects/` | Stage 3 | *(none)* | Terminal — audit only, no reader |
| `data/matches_model/` | Stage 4 (`ProbabilisticMatches`) | *(none)* | Terminal — audit only, deliberately not unioned into clustering |
| `data/FS_output/` | Stage 4 (pipeline candidates + `train.py` labeled training set) | *(none yet)* | Terminal today — the contract a future GBT will read |
| `data/clusters/` | Stage 5 | `src/api/publish.py` | Active |
| `data/runs/` | `src/pipeline.py` (`RunManifest`) | `src/api/publish.py`, `fs_matcher/train.py` (input resolution), `scripts/build_eval_workbook.py` | Active |
| `data/silver_labels/` | *(external, VM-only)* | `fs_matcher/train.py` | VM-only PHI input, gitignored |
| `data/empi.db` | `src/api/publish.py` | `src/api/store.py` + routers | Active — serves the review dashboard |
| `data/local_index/` | `src/api/publish.py` / `publish_local.py` (batch), `src/api/incremental.py` / `local_score.py` (incremental) | `src/api/parquet_backend.py`-backed routes (records/dashboard/audit), `src/api/local_score.py` | Active — full parity with `data/empi.db` (Stage 6) |
| `models/fs/` *(not under `data/`)* | `fs_matcher/train.py` | Stage 4 (`registry.resolve_active_model`) | See `docs/FS-Matcher-Production-Guide.md` for the full model-store layout |

---

## Invariants & open items

**Invariants every stage upholds** (enforced in `src/contracts.py`):

1. **Pair canonical ordering** `PATID_A < PATID_B`, preserved across all pair stages.
2. **`PATID` immutable** across stages; the only cross-stage join key.
3. **Schema validation at every boundary** (read and write) — including
   `rejects`, validated against `contracts.Rejects` in both `pipeline.py` and
   `run_rules.py`.
4. **No PHI in logs** — aggregate counts only.
5. **Sticky unmerge** (stage 6 only) — a PATID that has ever appeared in
   `audit_log.patids` is never repointed to a different `mid` by a later
   batch publish; only another explicit `/audit/*` action can move it.

**Resolved by the contracts layer:**

- *Boundary validation* — `contracts.validate(...)` wired into `clean.py`,
  `run_blocking.py`, `run_rules.py`, and `pipeline.py`; a renamed/missing/
  malformed column now fails loudly at the boundary instead of silently
  NaN-suppressing matches inside the rule engine.
- *Lineage* — orchestrator threads one `run_id` + `RunManifest`; the CLI path is
  guarded by `assert_patid_coverage`; `fs_matcher/train.py` resolves its
  inputs from the manifest rather than directory-globbing.
- *Config* — paths and `governance_threshold` centralized in
  `src/config.py::Settings`.

**Remaining (follow-ups):**

- **Duplicated constants & parsers** (verified still true). `blocking.py` and
  `deterministic_rules.py` still declare their own `COL_*` and
  `_parse_phone_set`; point them at `contracts.py` (names) and a shared
  `pairs.py` (parser) so there is one copy.
- **`verify_pipeline.py`** still reimplements the cleaned write and has a stale
  "cleaning writes CSV" docstring — switch it to `clean.write_cleaned` /
  `pipeline.run_pipeline`.
