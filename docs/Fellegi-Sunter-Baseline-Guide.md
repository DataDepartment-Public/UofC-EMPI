# Fellegi-Sunter Baseline Model — Build Guide

> **Status.** Shipped. Untouched as the head-to-head reference against the enhanced model.
> **Code.** `models/experiments/fs_splink_baseline/`
> **Authority.** This document mirrors the code. If they disagree, the code wins — update the doc.
> **Last updated.** 2026-06-17.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Where this fits in the pipeline](#2-where-this-fits-in-the-pipeline)
3. [What Fellegi-Sunter actually is (primer)](#3-what-fellegi-sunter-actually-is-primer)
4. [Module layout](#4-module-layout)
5. [Data flow — inputs and outputs](#5-data-flow--inputs-and-outputs)
6. [Splink settings — the comparison vector](#6-splink-settings--the-comparison-vector)
7. [Training — `u`-estimation + EM + match-prevalence prior](#7-training--u-estimation--em--match-prevalence-prior)
8. [Prediction and classification](#8-prediction-and-classification)
9. [Outputs and the cross-model evaluation contract](#9-outputs-and-the-cross-model-evaluation-contract)
10. [How to run](#10-how-to-run)
11. [Troubleshooting](#11-troubleshooting)
12. [Tests](#12-tests)
13. [Diagnostics, known limitations, and Phase A → B transition](#13-diagnostics-known-limitations-and-phase-a--b-transition)

---

## 1. TL;DR

The baseline is a **probabilistic record-linkage model** built on Splink 4.x over DuckDB. For each blocked candidate pair `(PATID_A, PATID_B)` it produces a `match_probability ∈ [0, 1]` and one of three tiers: `auto_merge` (≥ 0.90), `human_review` (0.50 ≤ p < 0.90), `no_match` (< 0.50). It is the **first probabilistic stage** in the eMPI pipeline and the reference matcher against which any future model is judged.

**Plain-English summary.** For every pair the blocking stage said *"might be the same patient,"* this model asks *"how strongly does the evidence (matching name, matching DOB, matching SSN, …) support that?"* It learns the strength of each piece of evidence from the data itself (via an algorithm called EM) and combines them into a single probability per pair.

---

## 2. Where this fits in the pipeline

```
raw  →  1. clean  →  2. block  →  3. deterministic rules ─┬─► data/matches/        (auto-confirmed)
                                                           └─► data/non_matches/    (the fuzzy remainder)
                                                                 ↓
                                                        4. Fellegi-Sunter (this doc)
                                                                 ↓
                                                         eval_schema parquet for head-to-head
```

The baseline historically scored **all** ~205k blocked candidate pairs because Stage 3 (deterministic rules) was built *after* the baseline shipped. After Stage 3 landed on `develop`, the baseline kept its full-pool behavior for head-to-head purposes; the enhanced model (see `Fellegi-Sunter-Enhanced-Guide.md`) is the one that runs after the rules stage on the non_matches subset.

---

## 3. What Fellegi-Sunter actually is (primer)

Fellegi-Sunter (1969) is the foundational mathematical framework for probabilistic record linkage. The pieces matter for understanding everything downstream.

### 3.1 Two probabilities per comparison level

For each comparison field (name, DOB, SSN, …) and each agreement level (exact match, fuzzy match, no match), the model carries two probabilities:

- **`m` = P(level | the pair IS a true match)** — how often real duplicates land at this level
- **`u` = P(level | the pair is NOT a match)** — how often two unrelated records land at this level by chance

### 3.2 Weight per level = log₂(m/u)

Each agreement level contributes evidence to the score, measured in **bits**:

> `weight = log₂(m / u)`

- If `m > u`: positive bits → "this agreement is more common in matches than in non-matches" → evidence FOR a match.
- If `m < u`: negative bits → "this agreement is more common in non-matches" → evidence AGAINST. We call this a **weight inversion** and log a warning when it appears; it usually signals an under-determined comparison level.
- If `m = u`: zero bits → no signal.

### 3.3 Total log-odds and final probability

For a pair, sum the per-comparison weights into a `match_weight` (total bits). Combine with the **match-prevalence prior** (P[two random records are duplicates]) to get a `match_probability ∈ [0, 1]`:

> `match_probability = 1 / (1 + 2^-match_weight × (1-π)/π)`, where π is the prior.

### 3.4 Where `m` and `u` come from — EM

EM (Expectation-Maximization) is an iterative algorithm. Given the comparison vector and a "blocking rule" that defines which pairs to look at, EM alternates between (E) assigning each pair a "soft" probability of being a match using the current m/u values, and (M) re-estimating m/u from those soft assignments. It converges to a local optimum. We run **three EM sessions** with different blocking rules (SSN, Email, Soundex-name + birth-year) because no single rule is broad enough on its own.

### 3.5 Where `u` comes from for fields not in the EM blocking rule — random sampling

EM cannot estimate `u` for the field used *as* the blocking rule (because all pairs agree on it by construction). For those fields, Splink estimates `u` by **randomly sampling pairs from the cleaned dataset** and measuring how often they agree at each level. This is the `estimate_u_using_random_sampling` step — and it is the source of an important calibration issue covered in the enhanced model's guide.

---

## 4. Module layout

| File | Purpose |
|---|---|
| `fellegi_sunter_baseline.py` | The whole matcher: settings, linker, training, prediction, classification, diagnostics. |
| `run_synthetic_baseline.py` | Sandbox runner. Uses `models/common/synthetic_data.py` fixtures + a 1e4 `u_max_pairs` budget. Safe to run anywhere. |
| `run_real_baseline.py` | **VM-only.** Reads the highest-versioned cleaned parquet + candidate-pairs parquet, runs with `u_max_pairs=1e6`, writes `models/outputs/fs_splink_baseline__<data-version>.parquet` + a diagnostics JSON. |
| `requirements.txt` | Pinned Splink + DuckDB versions to match the trained-model artifacts. |
| `__init__.py` | Re-exports `run_fs_baseline`, `MODEL_NAME`, `DEFAULT_AUTO_MERGE_THRESHOLD`, `DEFAULT_REVIEW_FLOOR`. |

Public entry point: `run_fs_baseline(candidate_pairs_path, df_clean, ...) -> pd.DataFrame` (`fellegi_sunter_baseline.py:658`).

---

## 5. Data flow — inputs and outputs

### Inputs

1. **Cleaned patient index** (parquet): `data/processed/MDM_Population_cleaned_v<N>_<date>.parquet`. Schema enforced by `src/contracts.py::CleanedRecords`. Required columns:
   - `PATID` (string)
   - `FirstNM_clean`, `LastNM_clean`, `BirthDT_clean`, `SSN_clean`, `last_4_SSN`, `Email_clean`, `ZipCD_clean_base`, `Phones_set`

2. **Candidate pairs** (parquet): `data/blocking/candidate_pairs_v<N>_<date>.parquet` (or `src/features/outputs/blocking/` on the VM — note the dir mismatch). Schema: `CandidatePairs` contract (`PATID_A, PATID_B, source_blocks, n_blocks`).

### Outputs

1. **Standard 5-col eval_schema parquet** at `models/outputs/fs_splink_baseline__<data-version>.parquet`. This is the **cross-model evaluation contract** — every matcher writes the same shape so the validation notebook can do apples-to-apples head-to-head:

| Column | Type | Notes |
|---|---|---|
| `PATID_A` | str | `PATID_A < PATID_B` (canonical order) |
| `PATID_B` | str | |
| `model_name` | str | constant `"fs_splink_baseline"` |
| `score` | float | = `match_probability` |
| `predicted_tier` | str | one of `auto_merge`, `human_review`, `no_match` |

2. **Phase A diagnostics** at `models/artifacts/fs_splink_baseline/diagnostics__<data-version>.json` — non-PHI: trained `m`/`u` per level, per-EM-session probabilities, the match-prevalence prior. Used for calibration only; never logs identifier values.

3. **`full_output=True` rich frame** (optional, in-memory): adds `match_probability`, `match_weight`, `classification_tier`, `source_blocks`, `n_blocks`, and Splink's `gamma_*` per-field agreement-level columns. Used by the validation notebook for §5 / §8 / §9 sampling and figures.

---

## 6. Splink settings — the comparison vector

`build_settings()` (`fellegi_sunter_baseline.py:256`) constructs the Splink `SettingsCreator` object that defines the comparison vector. Below is the **full list of comparisons** with the design rationale for each.

### 6.1 Comparison summary

| Field | Levels (top → bottom) | Notes |
|---|---|---|
| **FirstNM** | exact / JW ≥ 0.92 / JW ≥ 0.85 / else | TF adjustments on; thresholds tightened from default to drop the m<u "kinda similar" band (Plan R1) |
| **LastNM** | exact / JW ≥ 0.9 / JW ≥ 0.8 / JW ≥ 0.7 / else | Splink defaults; TF adjustments on |
| **DOB (`_dob_str`)** | exact / ±1 day / ±1 month / else | Dropped the ±1-year level (m << u) |
| **SSN** (custom 4-level) | null / exact 9-digit / last-4 match / else | Graded: full > partial > none |
| **Email** (custom 4-level) | null / exact full / exact username (different domain) / else | Dropped the JW>0.88-username fuzzy band (m<u) |
| **Phones_array** | array intersect ≥ 2 / array intersect ≥ 1 / else | Splink `ArrayIntersectAtSizes([2, 1])` |
| **ZIP** (custom 4-level) | null / exact 5-digit / first-3-digit match / else | The 3-digit-prefix level was added back in R2 to prevent a negative-evidence cascade |

**Not in the baseline:** there is **no** Address comparison. An Address Custom comparison was tried in Plan R2 and reverted on 2026-06-15 — EM trained large positive weights on Address levels using `u`-estimates from random pairs, but the candidate pool is heavily preselected for households, so Address agreement is far more common in the pool than in random pairs. The enhanced model fixes this by **locking** Address `m`/`u` to candidate-pool-aware priors rather than trusting EM.

### 6.2 A representative snippet — the SSN comparison

```python
cl.CustomComparison(
    output_column_name="SSN",
    comparison_levels=[
        cll.NullLevel(COL_SSN),                                    # at least one side missing
        cll.ExactMatchLevel(COL_SSN),                              # full 9-digit match
        cll.CustomLevel(                                           # last-4 match only
            sql_condition=(
                f"{COL_SSN_LAST4}_l IS NOT NULL "
                f"AND {COL_SSN_LAST4}_r IS NOT NULL "
                f"AND {COL_SSN_LAST4}_l = {COL_SSN_LAST4}_r"
            ),
            label_for_charts="Last 4 digits match",
        ),
        cll.ElseLevel(),                                           # else: different / unknown
    ],
)
```

(See `fellegi_sunter_baseline.py:299-313` for the live version.)

The `_l` / `_r` suffixes are Splink conventions for the left / right record of a pair in SQL. `CustomLevel` lets you write arbitrary DuckDB SQL for any agreement predicate the built-in `ComparisonLevel`s don't cover.

### 6.3 `retain_intermediate_calculation_columns=True` — Phase A

The Splink settings keep both `retain_intermediate_calculation_columns` and `retain_matching_columns` on (`fellegi_sunter_baseline.py:371-372`). This bloats the in-memory frame with `gamma_*`, `tf_*`, `bf_*` columns — **deliberately**, so we can audit calibration. Phase B will flip both off once we trust the model.

---

## 7. Training — `u`-estimation + EM + match-prevalence prior

`train_model(linker, u_max_pairs=1e6, seed=42)` (`fellegi_sunter_baseline.py:412`) is a four-step procedure.

### 7.1 Step 1 — random-sample `u`-estimation

```python
linker.training.estimate_u_using_random_sampling(max_pairs=u_max_pairs, seed=seed)
```

Splink draws `max_pairs` random record pairs from the cleaned dataset (no blocking applied), measures how often each comparison level agrees, and that's the `u` value for that level. Production budget is 1e6 pairs.

### 7.2 Steps 2-4 — three EM sessions with complementary blocking rules

Each EM session fixes one **EM blocking rule** (a constraint that all pairs entering EM must satisfy — narrow enough that "matches" actually outnumber non-matches in that subset) and estimates `m` (and the remaining un-blocked `u`) for the other fields.

```python
# Session 1 — SSN-exact anchor (high precision, ~21% coverage).
linker.training.estimate_parameters_using_expectation_maximisation(
    f"l.{COL_SSN} = r.{COL_SSN} AND l.{COL_SSN} IS NOT NULL"
)
# Session 2 — Email-exact anchor (~32% coverage).
linker.training.estimate_parameters_using_expectation_maximisation(
    f"l.{COL_EMAIL} = r.{COL_EMAIL} AND l.{COL_EMAIL} IS NOT NULL"
)
# Session 3 — Soundex(FN)+Soundex(LN)+BirthYear (broad ~99% coverage).
linker.training.estimate_parameters_using_expectation_maximisation(
    "l._sx_FirstNM = r._sx_FirstNM AND l._sx_LastNM = r._sx_LastNM "
    "AND l._birth_year = r._birth_year"
)
```

The three sessions are complementary by design: Sessions 1 and 2 are precision anchors that produce reliable `m` estimates on the un-blocked fields; Session 3's broad coverage estimates `m` for SSN and Email themselves.

### 7.3 Step 5 — match-prevalence prior (`λ`)

```python
linker.training.estimate_probability_two_random_records_match(
    deterministic_matching_rules=[
        f"l.{COL_SSN} = r.{COL_SSN} AND l.{COL_SSN} IS NOT NULL",
        f"l.{COL_EMAIL} = r.{COL_EMAIL} AND l.{COL_EMAIL} IS NOT NULL",
        "l._dm_LastNM = r._dm_LastNM AND l._dob_str = r._dob_str "
        "AND l._dm_LastNM IS NOT NULL AND l._dob_str IS NOT NULL",
    ],
    recall=0.80,
)
```

Tells Splink: *"These three deterministic rules collectively catch ~80% of true matches. From their hit-rate, estimate the population-level fraction of pairs that are true matches (π) and use it as the Bayesian prior when computing `match_probability`."*

### 7.4 Step 6 — weight-inversion diagnostic

`_warn_on_weight_inversions(linker)` (`fellegi_sunter_baseline.py:482`) walks every comparison level and logs a warning where `m < u`. These warnings are not errors — some are expected (the JW<0.5 mismatch levels in the enhanced model are deliberately `m < u`) — but they're a signal that a level may be under-trained on sparse data.

---

## 8. Prediction and classification

### 8.1 `predict_pairs(linker, candidate_pairs_df)` (`fellegi_sunter_baseline.py:512`)

Calls `linker.inference.predict()`, then post-processes:

1. Maps Splink's `PATID_l` / `PATID_r` columns to canonical `PATID_A` / `PATID_B` (with `A < B` always).
2. Drops "shim passthrough" columns that Splink uses internally.
3. **Re-attaches `source_blocks` and `n_blocks`** from the candidate-pairs frame — these are *pair-level* columns that can't ride through Splink's `additional_columns_to_retain` (which only harvests record-level columns).

### 8.2 `classify_pairs(df_predictions, auto_merge_threshold, review_floor)` (`fellegi_sunter_baseline.py:547`)

Pure post-processing — no retraining. Pure threshold cut:

```python
p = df_predictions["match_probability"]
tier = pd.Series("no_match", index=df_predictions.index, dtype="object")
tier = tier.mask(p >= review_floor, "human_review")
tier = tier.mask(p >= auto_merge_threshold, "auto_merge")
```

Defaults: `DEFAULT_AUTO_MERGE_THRESHOLD = 0.90`, `DEFAULT_REVIEW_FLOOR = 0.50` (`fellegi_sunter_baseline.py:107-108`). Constants live in the module so notebooks, evaluation code, and visualizations all auto-track any retuning.

---

## 9. Outputs and the cross-model evaluation contract

`to_evaluation_schema(df_classified)` (`fellegi_sunter_baseline.py:579`) projects the rich classified frame to the standardized 5-column shape. The contract — `models/common/eval_schema.py::EVAL_SCHEMA_COLUMNS` — is enforced by `validate_evaluation_frame()`:

```python
EVAL_SCHEMA_COLUMNS = ["PATID_A", "PATID_B", "model_name", "score", "predicted_tier"]
VALID_TIERS = {"auto_merge", "human_review", "no_match"}
```

Any new matcher that doesn't conform won't load into the head-to-head notebook §10.

The `run_real_baseline.py` runner also writes a **diagnostics JSON** containing the trained m/u parameters and per-EM-session estimates, used by the validation notebook §7 for non-PHI calibration audits.

---

## 10. How to run

### 10.1 Sandbox (off-VM)

```bash
python models/experiments/fs_splink_baseline/run_synthetic_baseline.py
```

Uses fixtures from `models/common/synthetic_data.py` and `u_max_pairs=1e4` (avoids the single-CPU u-sampling issue — see Troubleshooting). Takes a few seconds.

### 10.2 Production (VM, `empi_env` conda env)

```bash
# From the project root, with empi_env activated:
python models/experiments/fs_splink_baseline/run_real_baseline.py

# Optional overrides:
python models/experiments/fs_splink_baseline/run_real_baseline.py \
    --cleaned-index data/processed/MDM_Population_cleaned_v4_2026_06_11.parquet \
    --candidate-pairs src/features/outputs/blocking/candidate_pairs_v4_2026_06_11.parquet \
    --data-version v4_2026_06_11 \
    --u-max-pairs 1e6 \
    --auto-merge-threshold 0.90 \
    --review-floor 0.50
```

Without overrides, the runner auto-resolves to the **highest-versioned parquet** in each input directory and derives `data_version` from the candidate-pairs filename.

### 10.3 Expected log output (last ~15 lines)

```
INFO Blocking time: 335.17 seconds
INFO Predict time: 1.38 seconds
INFO predict_pairs: scored 205000 pairs
INFO classify_pairs: {'no_match': 198000, 'human_review': 1369, 'auto_merge': 5631}   ← representative numbers
INFO Tier breakdown: {'no_match': 198000, 'human_review': 1369, 'auto_merge': 5631}
INFO Score distribution: min=0.0001  p25=0.001  median=0.005  p75=0.020  max=1.0000
INFO Wrote 205000 scored pairs -> models/outputs/fs_splink_baseline__v4_2026_06_11.parquet
INFO Wrote diagnostics -> models/artifacts/fs_splink_baseline/diagnostics__v4_2026_06_11.json
```

### 10.4 What success looks like

- Tier counts add up to the total pair count.
- The `auto_merge` count should be ~3% of all pairs on the real cohort.
- Some `Weight inversion (m < u)` warnings are expected — mainly on rarely-populated levels (Email "else", SSN "else"). They do not invalidate the run.
- The diagnostics JSON exists and contains m/u values for every comparison.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: u-estimation failed because this host reports a single CPU and u_max_pairs=1e6 (>1e4) triggers Splink's salted DuckDB sampling path` | Splink salts the DuckDB u-sampling join with `cpu_count()` partitions when `max_pairs > 1e4`; on a 1-CPU host that collapses to 1 partition and Splink errors. | Run on a multi-core allocation, OR call with `--u-max-pairs 1e4`. The runner already auto-retries with 1e4 on this exact error. |
| `Could not access the DuckDB connection on DuckDBAPI (_con). Splink internals may have changed` | We use a Splink private attribute (`db_api._con`) to register the candidate_pairs table in the same DuckDB connection the blocking rule queries. Splink's private API changed across a minor version. | Pin the Splink version in `requirements.txt` (already done: `splink>=4.0.16,<5.0.0`). |
| Many "Weight inversion" warnings on Phones / Address levels | The candidate pool is preselected for likely-same-household pairs; EM's random-sample `u` under-counts within-pool agreement prevalence → trained weights are miscalibrated. | This is *exactly* what the enhanced model fixes via locked manual priors. Not a baseline bug — a baseline limitation. |
| All scores near 0 | Comparison vector and/or candidate_pairs table mis-registered in DuckDB. | Check the `_register_candidate_pairs` call ran without error; verify `df_model` and `candidate_pairs_df` use the same `PATID` values. |

---

## 12. Tests

Three layers, all share session-scoped fixtures from `tests/conftest.py`:

- **`tests/unit/test_fellegi_sunter_baseline.py`** — `prepare_model_input`, `build_settings`, `classify_pairs` thresholds, `to_evaluation_schema` projection.
- **`tests/integration/test_fellegi_sunter_baseline.py`** — full pipeline on synthetic data: train → predict → classify → eval_schema parquet round-trip.
- **`tests/regression/test_fellegi_sunter_baseline.py`** — known-pair sanity checks (e.g., a hand-constructed "obvious duplicate" must land in `auto_merge`).

Fixtures train Splink **once per session**:

```python
# tests/conftest.py
FS_U_MAX_PAIRS_SANDBOX = 1e4   # avoid the single-CPU u-sampling issue in tests

@pytest.fixture(scope="session")
def fs_classified(fs_df_clean, fs_candidate_pairs_path):
    return fs.run_fs_baseline(
        fs_candidate_pairs_path, fs_df_clean,
        u_max_pairs=FS_U_MAX_PAIRS_SANDBOX, full_output=True,
    )
```

Write new tests as **additional assertions against existing fixtures** rather than retraining.

---

## 13. Diagnostics, known limitations, and Phase A → B transition

### 13.1 Phase A → B

The baseline is currently in **Phase A** (calibration-friendly): `retain_intermediate_calculation_columns=True`, `retain_matching_columns=True`. This keeps every `gamma_*`, `tf_*`, `bf_*` Splink intermediate column on the output frame for audit. Phase B (production-friendly) flips both off — smaller output, faster predict, no behavioral change. Phase B is gated on "we trust the model" — currently blocked on the calibration findings the manual review surfaced (see the enhanced model's guide).

### 13.2 Known limitations the manual-review exercise surfaced

These motivated the enhanced model. Repeated here for completeness; full discussion in `Fellegi-Sunter-Enhanced-Guide.md`:

1. **One pair with same name + DOB + DIFFERENT SSN was placed in `auto_merge` at score 0.93.** Probability thresholds alone cannot enforce clinical safety; populated-SSN disagreement needs to be a structural veto, not a comparison level.
2. **~45% of borderline (0.40-0.95) pairs were `family-same-household` false positives.** EM-trained `u` for Address and Phones agreement was under-counted because the candidate pool is preselected for household neighbors.
3. **Strict precision in the 0.85-0.95 band was ~9%.** The 0.90 auto-merge threshold is too permissive for a clinical patient-merge.

### 13.3 What the baseline is still good for

- A reference point for "what a vanilla EM-trained FS model looks like on this data."
- The head-to-head baseline in the validation notebook §10.
- An ablation: if a future change makes the *enhanced* model look worse, run both and compare which intervention regressed.

---

## 14. What's next — Phase E3 refactor

This module (`fellegi_sunter_baseline.py`) is scheduled for replacement in **Phase E3-1 / E3-2** of the FS Refactor + Repo Cleanup plan. The plan refactors `fs_splink_baseline/` onto the shared `models/common/fs_base.py` OO scaffold that `fs_splink_enhanced_2/` already consumes: `fellegi_sunter_baseline.py` becomes a thin `FSBaseline(FSModel)` subclass in `fs_baseline.py`, with comparisons extracted to `comparisons.py`. Runners and tests are rewritten against the `FSModel` API in the same phase; no shim layer is introduced. The baseline's head-to-head reference role, threshold constants, and evaluation-schema output are all preserved — only the internal implementation shape changes. See `docs/superpowers/specs/2026-06-23-fs-refactor-design.md` for the full design.

---

## See also

- `docs/Fellegi-Sunter-Enhanced-Guide.md` — the production matcher built on top of this baseline's lessons.
- `docs/Data-Contract.md` — schema contracts at every pipeline boundary.
- `docs/Data-Cleaning-Guide.md` — authoritative cleaning rules (Stage 1).
- `docs/Deterministic-Rules-Guide.md` — Stage 3 deterministic rules.
- `notebooks/fellegi_sunter/fellegi_sunter_validation.ipynb` — manual-validation entry point (VM only).
