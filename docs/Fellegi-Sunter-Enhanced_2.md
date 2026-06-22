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

## Supervised training procedure (Phase E2-3)

Implemented in `models/common/fs_base.py::SupervisedTraining.train()`. Three steps in order:

1. **Random u-estimation on the real cohort.** `linker.training.estimate_u_using_random_sampling(max_pairs=u_max_pairs, seed=seed)`. Random pairs from the production records — *not* synthetic — because synthetic negatives don't capture blocking-induced household co-occurrence in real data.
2. **Filter labels to positives.** Splink's `estimate_m_from_pairwise_labels` treats every row in the labels table as a positive match (it ignores any score column). The labels frame is filtered to `label == 1` rows.
3. **Rename to Splink's labels-table schema.** The labels table needs `<unique_id_column>_l` / `<unique_id_column>_r` columns — for our `PATID` unique-id setting that's `PATID_l` / `PATID_r`. The fs_base helper handles the rename from `PATID_A` / `PATID_B`. (Splink's docstring shows "unique_id_l" but that's placeholder text — the real names follow the configured `unique_id_column_name`. Verified against `splink==4.0.16` in `splink/internals/block_from_labels.py`.)
4. **Register + train.** `linker.table_management.register_table(positives, "synthetic_labels", overwrite=True)` then `linker.training.estimate_m_from_pairwise_labels("synthetic_labels")`.
5. **Weight-inversion diagnostic.** `_warn_on_weight_inversions(linker)` logs any comparison level where m<u (sign-flipped) — typically a sign that a level had insufficient positive labels.

No EM session is run. No manual priors are applied. Splink computes m by looking up each labeled positive pair in the records table, computing the full comparison vector, and tallying frequencies per level — the canonical Fellegi-Sunter supervised estimator.

## Prediction & classification (Phase E2-3)

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

*(Placeholder — fills in Phase E2-5.)*

```bash
# Sandbox smoke (any host)
python models/experiments/fs_splink_enhanced_2/run_synthetic_enhanced_2.py

# Real cohort (VM)
python models/experiments/fs_splink_enhanced_2/run_real_enhanced_2.py
```

## Troubleshooting

*(Placeholder — fills in Phase E2-5.)*

## Tests

*(Placeholder — fills in Phase E2-6.)*

## Known limitations + what's next

*(Placeholder — fills in Phase E2-6.)*

## See also

- `docs/Fellegi-Sunter-Baseline-Guide.md` — original FS implementation (EM-only)
- `docs/Fellegi-Sunter-Enhanced-Guide.md` — second FS experiment (EM + manual priors + vetoes)
- `docs/Deterministic-Rules-Guide.md` — upstream Stage 3 rules
- `docs/Data-Contract.md` — schema contracts at every stage boundary
- `data/synthetic/coverage_report.csv` — full per-level audit table (Phase E2-0 output)
