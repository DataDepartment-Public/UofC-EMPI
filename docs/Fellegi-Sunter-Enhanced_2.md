# Fellegi-Sunter Enhanced_2 — Model Build Guide

> **Status:** in development on `feature/fs-baseline-splink`. Sections fill incrementally with each phase commit (E2-0 → E2-6).

## TL;DR

`fs_splink_enhanced_2` is the third Fellegi-Sunter (FS) experiment in the EMPI pipeline, succeeding `fs_splink_baseline` and `fs_splink_enhanced`. It is the first FS model in the repo to:

1. **Replace EM with supervised m-estimation** using a labeled synthetic pairs dataset (`data/synthetic/synthetic_train_v3.csv`, 40,000 labeled pairs).
2. **Drop deterministic vetoes from the FS scoring layer entirely** — veto logic moves to the deterministic-rules stage upstream (separate workstream).
3. **Add four new comparison features** (middle name, sex-as-positive, phonetic name agreement, plus a bundle: DOB month-day swap, ZIP base, full-name-compact, email local-part-only).
4. **Be built on a shared object-oriented base** (`models/common/fs_base.py`) that future FS experiments will reuse.

The goal: restore positive recall lost in `fs_splink_enhanced` (the §10 confusion matrix showed **0/3 `same` verdicts** reaching auto_merge or human_review) while preserving its household-FP suppression.

## Why this model exists — the recall-collapse story

The validation-notebook §10 confusion matrix on 42 reviewer-labeled pairs:

```
                Baseline FS                   Combined (rules + enhanced FS)
                no_match  hr  am                no_match  hr  am
different          4      23   4                   30      1   0
unsure             2       6   2                    9      1   0
same               1       1   1                    3      0   0   ← recall collapsed
```

The combined system has excellent **no_match precision (30/31 different verdicts)** but **lost all positive recall**: every `same` verdict ended up in `no_match`, and nothing reached `auto_merge` or `human_review`. Three architectural causes:

1. **EM over-credited Address and Phones agreement** because the blocking candidate pool is preselected for likely household neighbors. The locked manual priors in `manual_priors.py` partially fixed this but were estimated from only 42 reviewer labels.
2. **The veto stack was tuned to catch all household false positives.** Combined with the EM-pessimistic m-priors, true positives could not climb above the 0.95 auto-merge threshold.
3. **Comparisons were under-expressive.** No middle-name signal, no phonetic-name signal, no DOB month-day swap level — the model could not distinguish "real positive with one corrupted field" from "household neighbor".

Supervised m-estimation from a 40,000-pair synthetic dataset (with rich case-type coverage including sibling/twin/spouse/cousin household contamination negatives) directly resolves cause #1. Removing the vetoes resolves cause #2. The four new comparisons resolve cause #3.

## Pipeline position

```
raw → clean → block → deterministic rules ─┬─► data/matches/             (auto-confirmed)
                                            └─► data/non_matches/         (~169k pairs)
                                                  ↓
                              Stage 4: fs_splink_enhanced_2 (this model)
                                                  ↓
                                       data/matches_model_v2/<run_id>.parquet
                                       models/outputs/fs_splink_enhanced_2__<v>.parquet
```

`fs_splink_enhanced_2` runs **alongside** `fs_splink_enhanced` during the development phase, not as a replacement, so the validation notebook §11 can compare baseline / enhanced / enhanced_2 head-to-head on the same 42-label set. Promotion to the production orchestrator (`src/pipeline.py`) happens only after acceptance criteria are met.

## Fellegi-Sunter primer

(See `docs/Fellegi-Sunter-Baseline-Guide.md` "FS primer" section for the full derivation of m/u, log₂(m/u) match weight, and EM. The enhanced_2 model uses the same scoring algebra; only the training procedure changes.)

## Object-oriented architecture (`models/common/fs_base.py`)

Phase E2-1 introduces a shared OO base at `models/common/fs_base.py` that enhanced_2 (and future FS experiments) consume. Five concepts:

### `ComparisonSpec` + `ComparisonRegistry`

A `ComparisonSpec` is a frozen dataclass with `(name, builder, notes)` where `builder` is a zero-arg callable returning the comparison's Splink dict form. Lazy invocation lets the registry live at module-import time without paying Splink's import cost.

`ComparisonRegistry` is an ordered, **immutable-by-copy** collection. Mutation methods (`with_added`, `with_removed`, `with_replaced`) return new registries — the original is never modified, so a subclass's class-level registry constant is safe under concurrent runs and tests. Duplicate names raise at construction. Ordering is preserved: the order specs are declared is the order Splink sees them in `settings["comparisons"]`.

### `TrainingStrategy` (ABC) + `EMTraining` / `SupervisedTraining`

`TrainingStrategy.train(linker, df_clean)` is the single abstract method. Concrete strategies:

- **`EMTraining(em_blocking_rules, prior_rules, recall, u_max_pairs, seed)`** — random u-sampling + one EM session per blocking rule + (optional) `estimate_probability_two_random_records_match`. Used by baseline/enhanced; not refactored onto the base yet.
- **`SupervisedTraining(labels_df, label_col, u_max_pairs, seed, labels_table_name)`** — random u-sampling (still on the **real** cohort, not synthetic) + `register_labels_table` + `estimate_m_from_pairwise_labels`. Used by enhanced_2.

Both strategies call the shared `_estimate_u_with_guard` (converting Splink's single-CPU salting error into an actionable RuntimeError) and `_warn_on_weight_inversions` (m<u sign-flip diagnostic).

### `ClassificationConfig`

```python
@dataclass
class ClassificationConfig:
    auto_merge_threshold: float = 0.95
    review_floor: float = 0.40
    n_blocks_bump_threshold: int = 2
    n_blocks_bump_max_bits: float = 4.0
```

`__post_init__` enforces `0 ≤ review_floor ≤ auto_merge_threshold ≤ 1` and `n_blocks_bump_max_bits ≥ 0`. Enhanced_2's thresholds will be retuned in E2-4; for now the base defaults match enhanced.

### `FSModel` (ABC)

Subclasses configure:
- `model_name: str` — self-identifying tag for the eval-schema output
- `registry: ComparisonRegistry` — assembled from `ComparisonSpec`s
- `classification_config: ClassificationConfig`
- `training: TrainingStrategy` — set in `__init__`
- (optional overrides) `candidate_pairs_table_name`, `unique_id_column`, `eval_schema_columns`, `probabilistic_matches_columns`

Subclasses implement:
- `prepare_model_input(df_clean) -> pd.DataFrame` — project-specific derived-column logic (Splink shim columns, phone arrays, etc.)
- `build_settings() -> dict` — full Splink settings dict assembly from the registry

Base provides:
- `build_linker(df_model, candidate_pairs_df)` — registers candidate_pairs in DuckDB, instantiates Linker
- `train(linker, df_clean)` — delegates to `self.training`
- `predict(linker, candidate_pairs_df)` — runs `linker.inference.predict()`, canonicalizes PATID_A/B, merges source_blocks/n_blocks back on
- `classify(df_predictions)` — applies n_blocks log-odds bump, then thresholds → tier
- `to_evaluation_schema(df_classified)` — 5-col cross-model projection
- `to_probabilistic_matches(df_classified)` — `ProbabilisticMatches` projection (omits `veto_reason` by default; subclass + `probabilistic_matches_columns` override re-adds it)
- `run(candidate_pairs_df, df_clean, full_output)` — train + predict + classify in one call

### Why this shape

The functional baseline + enhanced modules accumulated ~1,000 lines each of Splink boilerplate (linker construction, prediction post-processing, threshold logic, projection helpers) that all three FS experiments share. The OO base lifts that shared mass out, leaving each `fs_splink_*` experiment as ~200 lines of project-specific differences (registry contents, EM rules, custom derived columns). Refactoring baseline + enhanced onto this base is a deferred follow-up.

## Cleaning prerequisites (Phase E2-2)

The phonetic-name comparisons (`LastNM_Phonetic`, `FirstNM_Phonetic`) need Double-Metaphone codes for every cleaned record. As of Phase E2-2, the cleaning stage emits them directly:

| Column | Producer | Notes |
|---|---|---|
| `_dm_LastNM` | `src.data.transformations.transform_dataframe` | Double-Metaphone primary code of `LastNM_clean`. `None` for null/empty input. |
| `_dm_FirstNM` | `src.data.transformations.transform_dataframe` | Double-Metaphone primary code of `FirstNM_clean`. `None` for null/empty input. |

Both are **optional** in the `CleanedRecords` pandera contract so cleaned parquets written before E2-2 still validate. `src/features/blocking.py::_compute_derived_columns` checks for these columns on the input frame and only recomputes them when absent — the regression test `tests/regression/test_dm_codes_persisted.py` pins byte-identical blocking output between the two code paths.

The authoritative `_dm_primary` implementation lives in `src/data/transformations.py`. Blocking imports it from there, so the encoding can never drift between stages.

`fs_splink_enhanced_2` reads these columns directly off the cleaned parquet — no derivation needed inside the FS module. This is the cleanest division: cleaning owns derived columns; matching owns m/u estimation.

## Module layout

```
models/experiments/fs_splink_enhanced_2/
├── __init__.py                       # re-exports FSEnhanced2, run_fs_enhanced_2
├── comparisons.py                    # ComparisonSpec builders + ENHANCED_2_REGISTRY
├── fs_enhanced_2.py                  # class FSEnhanced2(FSModel) + run_fs_enhanced_2
├── run_synthetic_enhanced_2.py       # sandbox runner (used by integration test)
└── (run_real_enhanced_2.py shipped in Phase E2-5)
```

`comparisons.py` declares one `_build_*` function per Splink comparison; each returns the comparison dict directly. The registry is composed at import time via `build_registry(include_address: bool)` so the sandbox (without `CityNM_clean` / `StateCD_clean`) can drop Address + Household_discount cleanly.

`fs_enhanced_2.py` contains a thin `FSEnhanced2(FSModel)` subclass. It sets `model_name`, owns the candidate-pairs blocking-rule SQL and shim columns, implements `prepare_model_input` (calls `_compute_derived_columns` from blocking and adds `Phones_array` + shim columns), and implements `build_settings` (uses Splink's `SettingsCreator` for the base settings dict, then replaces `comparisons` with the registry's `build_all()` output). All Splink-linker boilerplate (train/predict/classify/projections) is inherited from `FSModel`.

## Comparison registry (final, Phase E2-3)

The full registry (`ENHANCED_2_REGISTRY` in `comparisons.py`), in declaration order — this is also the order Splink lists in the trained settings, which the validation notebook §11 will display:

| # | Name | New in E2-3? | Levels |
|---|---|---|---|
| 1 | `FirstNM` | carried over | null / JW≥0.92 / JW≥0.85 / JW<0.5 / else |
| 2 | `LastNM` | new level | null / exact / JW≥0.92 / **full_name_compact exact** (NEW) / JW≥0.88 / JW<0.5 / else |
| 3 | `BirthDT` | new level | null / exact / ±1 day / ±1 month / **month-day swap** (NEW) / else |
| 4 | `SSN` | carried over | null / exact / last-4 match / 5-9 conflict / else |
| 5 | `Email` | carried over | null / exact / local-part match (diff domain) / else |
| 6 | `Phones` | carried over | array intersect ≥2 / ≥1 / else |
| 7 | `ZIP` | carried over | null / exact / 3-prefix / else |
| 8 | `MiddleNM` | **NEW** | null / exact / first-initial match / mismatch |
| 9 | `Sex_positive` | **NEW** | null / exact / OTHER-either / M↔F mismatch / else |
| 10 | `LastNM_Phonetic` | **NEW** | null / DM-equal / mismatch (reads `_dm_LastNM`) |
| 11 | `FirstNM_Phonetic` | **NEW** | null / DM-equal / mismatch (reads `_dm_FirstNM`) |
| 12 | `Household_discount` | carried over | null / household-indicator-without-identity-match / else (gated on `include_address`) |
| 13 | `Address` | carried over | null / exact / same-city-state-zip / else (gated on `include_address`) |

The synthetic sandbox runs with `include_address=False` (records frame lacks `CityNM_clean` / `StateCD_clean`) so comparisons 12 and 13 drop out there.

**Audit-flagged levels** (Phase E2-0 surfaced two — neither dropped):
- `Sex_positive[M↔F]` had 0 positives in the labeled set, so Splink reports m=None for that level after training. This is the *desired* behavior — combined with the high u (~6,700 random-pair occurrences in the audit), the level acts as strong anti-evidence in scoring without ever being trained as a positive case. No manual clamp needed.
- `FirstNM_Phonetic[null]` had only 10 positives — minor; first names are rarely null in synthetic. m for that level may be noisy but the level rarely fires.

| Comparison | New? | Levels | E2-0 audit |
|---|---|---|---|
| **MiddleNM** | NEW | null-either / exact / initial-match / mismatch | ✅ All 4 levels OK (exact: 2,543 pos / 25 neg; mismatch: 665 pos / 694 neg — strong signal both directions) |
| **Sex_positive** | NEW (was a veto) | null-either / exact / OTHER-either / MALE↔FEMALE | ⚠ `MALE↔FEMALE` has **0 positives, 6,702 negatives** — by design (synthetic generator does not produce sex-swap true matches). Will manually prime m≈0.01 (very low) so the strong negative log-weight survives. Other levels OK. |
| **LastNM_Phonetic (DM)** | NEW | phonetic-equal / mismatch | ✅ OK (equal: 11,106 pos / 9,967 neg; mismatch: 4,894 pos / 14,033 neg) |
| **FirstNM_Phonetic (DM)** | NEW | phonetic-equal / mismatch / null-either | ⚠ `null_either` has 10 positives — small sample; first names rarely null. Other levels OK. |
| **DOB month-day swap** | NEW level inside existing `BirthDT` | exact / month-day-swap / same-year / mismatch | ✅ All levels OK; swap is essentially diagnostic (67 pos / 1 neg) |
| **ZipBase** | NEW | null / exact / 3-prefix-match / mismatch | ✅ All levels OK (prefix3_match: 1,013 pos / 7,206 neg) |
| **full_name_compact** | NEW level inside existing `LastNM` | compact-exact / mismatch | ✅ OK (compact_exact: 6,916 pos / 1,844 neg) |
| **Email local-part-only** | NEW level inside existing `Email` | exact / local-only / mismatch / null | ✅ OK; local-only is a perfect positive signal (392 pos / 0 neg) |

The `MALE↔FEMALE` finding is **expected**: a sex-mismatch true positive would be extremely rare in real records, so synthetic doesn't generate any. The model will still penalize MALE↔FEMALE pairs heavily because u is large (6,702 random-pair occurrences) and m is manually clamped low.

## Supervised training procedure (Phase E2-3 + E2-5-fix3)

Implemented in `models/common/fs_base.py::SupervisedTraining.train()`. Two paths exist depending on whether the labels' PATIDs resolve in the production records frame.

### The PATID-resolution constraint (the bug E2-5-fix3 addresses)

`splink.training.estimate_m_from_pairwise_labels` computes m by **looking up each labeled pair in the records table bound to the linker**, computing its comparison vector, and tallying frequencies per level. If labels reference PATIDs not in the records table, the inner-join produces zero rows and Splink silently falls back to **Bayesian floor m values** (typically `1/N` where N is small), giving a model that scores from u alone. The output looks plausible (bimodal distribution from u variance) but is uncalibrated.

`SupervisedTraining` guards against this with a pre-flight check: if the labels' positive PATIDs do not resolve in the records frame the linker is bound to AND no `labels_records_df` is provided, it raises a clear `ValueError` before invoking Splink.

### Path A — single-linker (sandbox / when labels' PATIDs ARE in `df_clean`)

1. **Validate** that every PATID referenced by `label == 1` rows of `labels_df` exists in `df_clean`. Otherwise raise.
2. **Filter labels to positives.** Splink ignores any score column; every registered row is treated as a positive.
3. **Rename to Splink's schema.** Labels table needs `<unique_id_column>_l` / `_r` columns (for `PATID` that's `PATID_l` / `PATID_r`). The fs_base helper handles the `PATID_A`/`PATID_B` → `_l`/`_r` rename. (Splink's docstring says "unique_id_l" but the real names follow the configured `unique_id_column_name` — verified against `splink==4.0.16` in `splink/internals/block_from_labels.py`.)
4. **Register + train.** `linker.table_management.register_table(positives, "synthetic_labels", overwrite=True)` then `linker.training.estimate_m_from_pairwise_labels("synthetic_labels")`.
5. **Random u-estimation on the live linker.** Same `linker.training.estimate_u_using_random_sampling(max_pairs=u_max_pairs, seed=seed)` as enhanced/baseline.
6. **Weight-inversion diagnostic** logs levels where m<u (sign-flipped) — usually a sign that the level had insufficient positive labels.

### Path B — split-training (production: synthetic labels, real cohort)

This is the **typical production case** — the synthetic training labels reference synthetic PATIDs (e.g. `S000098581`) which do not exist in the real cohort. `SupervisedTraining.__init__` accepts a `labels_records_df` parameter: the records frame whose PATIDs the labels DO reference (`data/synthetic/synthetic_blocking_testing.csv` for the standard synthetic-train-v3 label set).

1. **Validate** that the labels' positive PATIDs resolve in `labels_records_df`.
2. **Build an auxiliary "m-training" linker** bound to `labels_records_df` with the same settings as the production linker (registry, candidate-pairs blocking rule, retain flags). Splink requires the candidate-pairs table to exist for binding; an empty `pd.DataFrame` is registered since the m-training procedure never calls predict.
3. **Train m on the auxiliary linker.** Same register-labels + `estimate_m_from_pairwise_labels` as Path A, but resolving against `labels_records_df`.
4. **Estimate u on the live (production) linker.** u must reflect the real-cohort distribution — random sampling from synthetic would mis-calibrate the negative-class baseline.
5. **Copy trained m values** from the auxiliary linker's `_settings_obj.comparisons` to the live linker's, level-by-level, via `_copy_m_probabilities` (writes `_m_probability` on each `ComparisonLevel` — same private-attribute pattern Splink itself uses internally and that `manual_priors.apply_manual_priors` uses in the enhanced module).
6. **Weight-inversion diagnostic** runs on the live linker post-merge.

The two-linker pattern is the standard FS recipe for "supervised m + cohort u" — m carries the positive-class distribution learned from labels, u carries the random-pair distribution learned from the production frame. Cross-pollination at step 5 unifies them.

**Why not just train m on the real cohort directly?** Because we don't have real-data labels at scale. The synthetic generator produces 16,000 deliberately-engineered positives (with rich case_type coverage) — orders of magnitude more than the 42 reviewer labels. Split-training lets us use the rich synthetic signal for m while preserving the real-cohort signal for u.

## Threshold-tuning rationale (Phase E2-4)

`scripts/sweep_enhanced_2_thresholds.py` is the calibration tool. It accepts any scored-pairs frame + labels frame and sweeps the prescribed grid (`auto_merge ∈ {0.85, 0.90, 0.925, 0.95}` × `review_floor ∈ {0.35, 0.40, 0.50, 0.55}`), reporting precision and recall at each combination plus the Pareto-feasible points under the gates.

**Synthetic sweep (10,000 test labels, run locally as the first-pass calibration):**

| auto_merge | review_floor | precision_AM | recall_AM | recall_AM∪HR |
|---|---|---|---|---|
| 0.85 | (any) | 0.754 | 0.972 | 0.979 |
| 0.90 | (any) | 0.777 | 0.969 | 0.979 |
| 0.925 | (any) | 0.793 | 0.966 | 0.979 |
| **0.95** | **0.40** | **0.822** | **0.963** | **0.979** |

(Within each `auto_merge` row, `review_floor` changes the human_review tier size but not the AM precision or AM∪HR recall — the residual 21 positives in no_match score very low across all combinations.)

**Why no synthetic combination meets the ≥95% precision gate:** the synthetic test set is intentionally adversarial — ~85% of its negatives are `NM-HARD-*` or `NM-HH-*` case types (deliberately sharing one or more identifier fields). Real blocked candidate pairs are far less adversarial, so the precision on the 42 reviewer-labeled real pairs is expected to be materially higher than the synthetic 82.2%.

**Decision (within-grid Pareto-best on synthetic):** keep `auto_merge_threshold = 0.95` and `review_floor = 0.40` — matches the enhanced model's defaults. The synthetic sweep validates the current point as the best in-grid choice. The real-data sweep (post-E2-5, on the VM, using the 42 reviewer labels parsed from the validation notebook) will finalize the numbers; `ClassificationConfig` defaults are revisited then.

To re-run the sweep on the VM after a real cohort run:
```bash
# 42 reviewer labels (gold) — final precision + recall calibration
python scripts/sweep_enhanced_2_thresholds.py --mode real \
  --scored models/outputs/fs_splink_enhanced_2__<version>.parquet \
  --labels-csv /path/to/reviewer_labels_42.csv \
  --output data/threshold_sweep_real.csv
```

### Silver-labels validation (VM-only)

`data/silver_labels/` (gitignored, VM-only — labels reference real PATIDs from `MDM_Population.csv`) holds Stage-3 deterministic-rule confirmations as an all-positive held-out validation set (~99% precision per adjudicator). Silver labels are **not** part of the supervised m-training set — they remain held out specifically so we can measure how enhanced_2 ranks pairs the deterministic stage already caught.

**Workflow caveat:** silver-labeled pairs are filtered out of `data/non_matches/<run_id>.parquet` by Stage 3 (that's where they came from). To score them with enhanced_2, the real runner (E2-5) must score against the **full candidate pool** (`data/blocking/candidate_pairs_<run_id>.parquet`), not the non_matches subset. The E2-5 runner exposes a `--score-full-candidate-pool` flag for exactly this use.

```bash
# 1. Score the full candidate pool for evaluation purposes (E2-5)
python -m models.experiments.fs_splink_enhanced_2.run_real_enhanced_2 \
  --score-full-candidate-pool

# 2. Sweep against silver labels (recall-only signal — precision_AM is
#    trivially 1.0 since silver has no negatives; the script warns).
python scripts/sweep_enhanced_2_thresholds.py --mode real \
  --scored models/outputs/fs_splink_enhanced_2__<version>_full_pool.parquet \
  --labels-csv data/silver_labels/silver_labels.csv \
  --output data/threshold_sweep_silver.csv
```

**Interpretation of silver-labels sweep:**
- `recall_AM` tells us *what fraction of deterministic-rule confirmations enhanced_2 would also auto-merge*. A high number is corroborative — enhanced_2 catches what the rules catch. A low number reveals signals the rules use that enhanced_2 is missing (and is a candidate for an additional Stage-4 comparison).
- `recall_AM∪HR` tells us *what fraction reaches at least the human-review tier*. Should approach 100% — if not, enhanced_2 is systematically under-scoring real positives.
- `precision_AM` is uninformative here (no negatives) and the script flags it.

## Prediction & classification (Phase E2-3)

> **⚠ Calibration depends on upstream `classify_non_matches`** — see [Upstream contradiction filter](#upstream-contradiction-filter-classify_non_matches) for the full diagnosis. Stage-4 scoring sees only the `review`-tier pairs from Stage 3's three-way split; pairs with ≥ `DEFAULT_REJECT_MIN_CONTRADICTIONS` contradicting strong identifiers are dropped into `data/rejects/<run_id>.parquet` and never reach the FS scorer. Post-merge real-cohort runs land against this filtered pool. The sandbox and integration-test paths are unaffected because synthetic test pairs don't broadly trigger anti-evidence levels.

Inherited from `FSModel`. Three phases:

### Prediction
`linker.inference.predict().as_pandas_dataframe()` returns the full comparison-vector frame. Post-processing in `FSModel.predict`:
- Canonicalize PATID_A/B from Splink's `PATID_l` / `PATID_r` (`min`/`max` on the pair).
- Merge `source_blocks` + `n_blocks` back on from the original candidate_pairs frame (Splink doesn't pass them through additional_columns_to_retain).

### n_blocks log-odds bump
Identical to enhanced: `+1 bit` per block above `n_blocks_bump_threshold=2`, capped at `n_blocks_bump_max_bits=4.0`. Bit-space math:
```
weight     = log2(p / (1 - p))
weight'    = weight + min(max(n_blocks - threshold, 0), max_bits)
p'         = 1 / (1 + 2^(-weight'))
```
Implemented as `FSModel._apply_n_blocks_bump` (pure pandas/numpy, no Splink).

### Threshold classification
```
p' >= auto_merge_threshold       -> "auto_merge"
review_floor <= p' < auto_merge  -> "human_review"
p' < review_floor                -> "no_match"
```
Both bounds are **inclusive at the floor**. Defaults are `0.95 / 0.40` (enhanced's defaults; retuned in Phase E2-4).

**Veto override is removed.** There is no `veto_reason` column on the classified frame and none in the `ProbabilisticMatches` projection. Vetoes will live in the upstream deterministic-rules stage going forward (separate workstream).

### Sandbox smoke results (Phase E2-3)

Run via `python -m models.experiments.fs_splink_enhanced_2.run_synthetic_enhanced_2` against the synthetic test split (10,000 pairs, 80/20 neg/pos):

```
                      auto_merge  human_review  no_match
label=0 (8000)              416           488      7,096
label=1 (2000)            1,925            32        43
```

- **True positives in auto_merge:** 96.3% (1,925/2,000)
- **True positives in AM∪HR:** 97.9% (1,957/2,000) — recall lifted dramatically vs the enhanced model's `0/3 same → AM∪HR` on the 42-label set
- **False positives in auto_merge:** 5.2% (416/8,000) — the threshold sweep in Phase E2-4 will tune this down

The integration test `tests/integration/test_fs_enhanced_2_sandbox.py` pins the three E2-3 gates: (a) one `gamma_*` column per registered comparison; (b) household-contamination pairs (NM-HH-*) mean score below `review_floor=0.40` with ≥60% in `no_match` tier; (c) NM-COMMON-* bottom decile below 0.05.

## Output contracts

Two artifacts per real-data run:

### `models/outputs/fs_splink_enhanced_2__<v>.parquet` — cross-model eval schema

```
PATID_A | PATID_B | model_name | score | predicted_tier
```

Identical 5-column shape used by `fs_splink_baseline` and `fs_splink_enhanced`. The validation notebook §11 (Phase E2-6) reads all three through this contract for head-to-head. `model_name == "fs_splink_enhanced_2"`. `score` is `match_probability ∈ [0, 1]` post-`n_blocks` bump.

### `data/matches_model_v2/matches_model_<run_id>.parquet` — ProbabilisticMatches

Schema enforced by `src.contracts.ProbabilisticMatches`. Column order from `FSModel.probabilistic_matches_columns`:

```
PATID_A | PATID_B | match_source | score | match_weight | classification_tier | source_blocks | n_blocks
```

**Contract change in E2-1: `veto_reason` is now optional.** Producers that don't apply a veto layer (enhanced_2) omit the column entirely; producers that do (enhanced) include it as `Series[str]` with nullable values. Both pass `validate(df, ProbabilisticMatches)`. The optionality is encoded as `Optional[Series[str]]` in pandera — see `tests/unit/test_contracts_probabilistic_optional_veto.py` for the three cases (absent / present-null / present-populated).

The default `FSModel.to_probabilistic_matches` projection omits `veto_reason`. A subclass that wants it back overrides `probabilistic_matches_columns` to include it and overrides the projection method to populate it from `df_classified[VETO_REASON_COL]`.

## How to run

### Sandbox (any host)

```bash
python -m models.experiments.fs_splink_enhanced_2.run_synthetic_enhanced_2
```

Loads `data/synthetic/synthetic_blocking_testing.csv` (records, deconstructed from the paired CSVs), `data/synthetic/synthetic_train_v3.csv` (40k labels), and `data/synthetic/synthetic_test_v3.csv` (10k labels). Runs with `include_address=False` (the records frame lacks `CityNM_clean` / `StateCD_clean`) and `u_max_pairs=1e4` (stays under Splink's single-CPU salting limit). Prints tier breakdown + confusion matrix; validates `ProbabilisticMatches` round-trip. No on-disk artifacts (sandbox).

Used as the integration-test fixture (`tests/integration/test_fs_enhanced_2_sandbox.py`).

### VM — production scoring (real cohort)

```bash
# Default: score the post-Stage-3 non_matches pool.
python -m models.experiments.fs_splink_enhanced_2.run_real_enhanced_2
```

Inputs (auto-resolved to the highest-versioned file via `models.common.versioning.latest_versioned`):
- `data/processed/MDM_Population_cleaned_v*_*.parquet` — records (real PHI)
- `data/non_matches/non_matches_v*_*.parquet` — Stage-3 output (~169k pairs the rules did not confirm)
- `data/synthetic/synthetic_train_v3.csv` — labels for supervised m-training
- `data/synthetic/synthetic_blocking_testing.csv` — labels-records frame for split-training (m trains here; pass `--labels-records ''` to disable)

Outputs:
- `models/outputs/fs_splink_enhanced_2__<data-version>.parquet` — 5-col eval schema (head-to-head input for the validation notebook §11)
- `data/matches_model_v2/fs_splink_enhanced_2_matches_model__<data-version>.parquet` — ProbabilisticMatches contract (validated, union-ready for Stage 5)
- `models/artifacts/fs_splink_enhanced_2/diagnostics__<data-version>.json` — non-PHI counts + score quantiles

`<data-version>` is auto-parsed from the input filename (e.g. `v4_2026_06_11`) so refreshes don't overwrite earlier runs.

### VM — silver-labels validation (full candidate pool)

```bash
python -m models.experiments.fs_splink_enhanced_2.run_real_enhanced_2 \
    --score-full-candidate-pool
```

Scores `src/features/outputs/blocking/candidate_pairs_v*_*.parquet` (the FULL pre-rules pool, ~205k pairs on the current cohort) instead of `non_matches`. Required because silver-labeled pairs (Stage-3 deterministic confirmations from `data/silver_labels/`) were *removed* from `non_matches` by Stage 3 — they need to be back in the scoring pool to be evaluable. Output filenames get a `_full_pool` suffix:
- `models/outputs/fs_splink_enhanced_2__<v>_full_pool.parquet`
- `data/matches_model_v2/fs_splink_enhanced_2_matches_model__<v>_full_pool.parquet`

### Expected logs

Aggregate-only — never PHI. Example trailing lines:

```
INFO Training FSEnhanced2 (u_max_pairs=1e+06, include_address=True)...
INFO FSModel[fs_splink_enhanced_2]: linker built (N records, M candidate pairs)
INFO SupervisedTraining: registering 16000 positive labels for m-training (columns: PATID_l, PATID_r)
INFO FSModel[fs_splink_enhanced_2]: training complete
INFO FSModel[fs_splink_enhanced_2]: scored M pairs
INFO FSModel[fs_splink_enhanced_2]: classify {'no_match': ..., 'human_review': ..., 'auto_merge': ...}
INFO Wrote N rows -> models/outputs/fs_splink_enhanced_2__<v>.parquet (eval_schema)
INFO Wrote N rows -> data/matches_model_v2/fs_splink_enhanced_2_matches_model__<v>.parquet (ProbabilisticMatches)
INFO Tier breakdown: {...}
INFO Score distribution: min=...  p25=...  median=...  p75=...  max=...
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: u-estimation failed because this host reports a single CPU and u_max_pairs=1e+06 (>1e4) triggers Splink's salted DuckDB sampling path...` | Single-CPU host. | The runner auto-retries with `u_max_pairs=1e4`; you'll see the retry log. To suppress entirely, pass `--u-max-pairs 1e4` explicitly. |
| `FileNotFoundError: No files matching ('non_matches_v*_*.parquet', 'non_matches_*Z.parquet')` (or cleaned / candidate_pairs equivalent) | Auto-resolution checks both naming conventions (standalone CLI `v<N>_<date>` and pipeline orchestrator `<run_id>`) and neither produced a match. | Confirm Stages 1 + 2 (+ 3) have been run on this host. The runner also looks in `data/blocking/` AND `src/features/outputs/blocking/` for candidate pairs. Pass `--cleaned-index` / `--non-matches` / `--candidate-pairs` to pin a specific file. |
| `Labels CSV not found: data/synthetic/synthetic_train_v3.csv` | The synthetic training set isn't on this host. | `git pull` to fetch `data/synthetic/*.csv` (whitelisted in `.gitignore`); confirm both `synthetic_train_v3.csv` and `synthetic_test_v3.csv` are present. |
| `Splink: m probability not trained for X — comparison level was never observed in the training data` | Some comparison level had 0 positive labels in `synthetic_train_v3.csv`. | Expected for `Sex_positive[MALE↔FEMALE]` (synthetic has 0 sex-swap positives by design) and a handful of rare-level / DOB-swap cases. The strong-negative weight still comes from the high u. To eliminate, expand the synthetic generator to cover those levels. |
| `ValueError: SupervisedTraining: N/N labeled PATIDs are absent from the 'df_clean' records frame` | Single-linker training was attempted but the labels' PATIDs don't exist in the production records (the silent-broken case fixed in E2-5-fix3). | Pass `--labels-records data/synthetic/synthetic_blocking_testing.csv` so m-training runs on a separate auxiliary linker. The runner's default already does this; explicit `--labels-records ''` disables it. |
| `Splink: m probability not trained for X` reported on **every** level of **every** comparison (run with no `--labels-records`) | Labels' PATIDs unresolvable in `df_clean` → m at floor values for the whole model. | Same fix as the row above. Always supply `--labels-records` when the labels and the production records come from disjoint PATID spaces. |
| `Splink: Weight inversion (m < u) on comparison X level Y` | m landed below u for that level — informational. | The level still scores correctly. Inspect the diagnostics JSON to see which comparisons + levels are affected; if many, the labels set may need broader coverage. |
| `Splink: Invalid table names provided (only l. and r. are valid): cp.PATID_A, cp.PATID_B` | Splink's static settings validator doesn't know about the runtime-registered `candidate_pairs` table. | Benign warning — by design. The candidate-pairs blocking rule uses an EXISTS subquery DuckDB decorrelates at runtime. Unchanged since the baseline. |
| `ProbabilisticMatches validation fails on n_blocks` | Scoring pool had `n_blocks` as float instead of int. | Pandera contract enforces `int ≥ 1`. Re-run blocking to refresh the candidate-pairs parquet, or coerce `n_blocks = n_blocks.astype("int64")` before passing to the runner. |

## Tests

*(Placeholder — fills in Phase E2-6.)*

## Upstream contradiction filter (`classify_non_matches`)

Stage 3 (`src/models/deterministic_rules.classify_non_matches`, landed on `develop` via PR #11 and merged into this branch in Phase E2-5b) is the team's chosen pattern for keeping anti-evidence pairs out of Stage 4. It performs a three-way split on every candidate pair:

1. **Confirmed** — at least one deterministic rule fires → `data/matches/<run_id>.parquet`
2. **Reject** — ≥ `DEFAULT_REJECT_MIN_CONTRADICTIONS` strong identifiers strictly disagree → `data/rejects/<run_id>.parquet`, dropped from the pipeline
3. **Review** — neither confirmed nor rejected → `data/non_matches/<run_id>.parquet`, flows into Stage 4

Stage 4 (this module) therefore receives only the "review" survivors — ambiguous pairs without enough contradicting evidence to discard, but without enough corroborating evidence to confirm. The pre-merge `data/non_matches/` artifact contained every candidate pair the deterministic rules could not confirm; the post-merge artifact contains only the review-tier subset.

### Why this resolves the E2-5 calibration gap

Phase E2-5 surfaced a structural calibration gap: enhanced_2's supervised m-training cannot learn m values for anti-evidence levels (levels that fire only on negatives). `splink.training.estimate_m_from_pairwise_labels` learns m from positive observations only — levels positives never fire fall back to Splink's Bayesian floor (~0.05). For anti-evidence levels where u is small (random pairs rarely fire them either), this produces an inverted weight: `log2(0.05/0.001) = +5.6 bits of POSITIVE evidence` when the level fires, exactly the opposite of the intent.

On the 2026-06-21 pre-merge VM run against the 169,180-pair `non_matches` pool, this produced 105,528 auto_merges (62% of the pool) with a score median of 0.998 — bimodal at the top because household-contamination FPs were being credited rather than penalised.

Upstream contradiction-filtering addresses the same failure mode at the architecturally correct layer: pairs with contradicting strong identifiers (the population on which anti-evidence inversion does the most damage) are dropped *before* Stage 4 sees them. The anti-evidence comparison levels inside the FS model (`Household_discount`, `Sex_positive[M↔F]`, `SSN[9-digit mismatch]`) still produce inflated scores when they fire on the surviving review-tier pairs, but the population on which that inflation matters is much smaller.

**Design precedent:** see `docs/Deterministic-Rules-Guide.md` — the team's deterministic engine has consistently chosen "filter on contradictions / corroborate by rule" over "veto by single signal." The `EMAIL_EXACT` standalone rule was *removed*, not vetoed, when shared family/clinic inboxes drove ~63–80% adjudicator precision. `classify_non_matches` extends that same philosophy to the non-matches pool.

### Reversibility

If post-merge measurement on the next VM session shows residual over-scoring on review-tier pairs, a downstream **corroboration gate** (a positive-predicate requirement on auto_merge promotion, mirroring the Stage 3 `RULES` registry) remains available as a follow-up phase. It was scoped as Phase E2-5-fix5 and deliberately dropped in E2-5b because `classify_non_matches` addresses the same failure mode upstream. Documentation lives in `docs/superpowers/specs/2026-06-22-develop-integration-design.md`.

### Current status

The pre-merge enhanced_2 artifacts (`models/outputs/fs_splink_enhanced_2__20260617T043941Z.parquet` and `data/matches_model_v2/...`) are **known miscalibrated** for the reason above and should be deleted on the VM before the next run. Post-merge VM runs of `run_real_enhanced_2.py` will score against the contradiction-filtered `data/non_matches/<run_id>.parquet` and are expected to land in a sensible tier distribution. Headline numbers from the first post-merge real-cohort run go in a follow-up doc-only commit once that run completes.

## Known limitations + what's next

1. **Anti-evidence over-scoring is mitigated, not eliminated.** Stage 4 sees only review-tier pairs from `classify_non_matches`, so the household-FP population that inflated raw scores in pre-merge runs is now mostly dropped before scoring. Residual inflation is possible on borderline review pairs where one anti-evidence level fires; revisit if real-cohort measurement shows auto_merge precision below threshold.
2. **42 reviewer labels is a small calibration set.** Phase E2-6 threshold sweep against them may not have enough statistical power to detect small precision/recall differences. Add 30–50 more pairs at the new boundaries before promoting.
3. **Synthetic positives may over-represent corruption.** m values for "All other" / mismatch levels are higher than real-data positives would justify (because synthetic positives are deliberately corrupted at high rates). The downstream effect is weaker negative weights on these levels in real scoring. Mitigation: silver labels (real-PATID positives, ~99% precision) can complement synthetic in a future m-training pass.
4. **OO base is not yet adopted by baseline / enhanced.** Refactoring those onto `models/common/fs_base.py` is deferred to a separate workstream.
5. **No Stage 5 clustering yet.** The `ProbabilisticMatches` artifact is union-ready for a future clustering step that combines deterministic + probabilistic edges.

## See also

- `docs/Fellegi-Sunter-Baseline-Guide.md` — original FS implementation (EM-only)
- `docs/Fellegi-Sunter-Enhanced-Guide.md` — second FS experiment (EM + manual priors + vetoes)
- `docs/Deterministic-Rules-Guide.md` — upstream Stage 3 rules
- `docs/Data-Contract.md` — schema contracts at every stage boundary
- `data/synthetic/coverage_report.csv` — full per-level audit table (Phase E2-0 output)
