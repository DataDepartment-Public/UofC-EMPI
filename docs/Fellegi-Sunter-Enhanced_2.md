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

*(Placeholder — fills in Phase E2-1.)*

The shared base provides:
- `FSModel` (ABC): owns Splink-linker boilerplate, n_blocks bump, classify, output projections.
- `TrainingStrategy` (ABC) with concrete subclasses `EMTraining` (used by baseline / enhanced) and `SupervisedTraining` (used by enhanced_2).
- `ComparisonSpec` + `ComparisonRegistry`: declarative, ordered, mutable-by-copy.
- `ClassificationConfig`: thresholds + n_blocks-bump constants in one dataclass.

`fs_splink_enhanced_2/` is a thin composition over this base — see "Module layout" once Phase E2-3 ships.

## Module layout

*(Placeholder — fills in Phase E2-3.)*

## Comparison registry

*(Placeholder — fills in Phase E2-3 with the audit-informed final registry. The registry below lists the **proposed** new comparisons plus their E2-0 audit verdicts.)*

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

## Supervised training procedure

*(Placeholder — fills in Phase E2-3.)*

Outline:
1. Load `data/synthetic/synthetic_train_v3.csv` (40k pairs with `label`, `case_type`, `_l` / `_r` paired columns).
2. Build the Splink linker with `ENHANCED_2_REGISTRY` comparisons.
3. `estimate_u_using_random_sampling` on the real cleaned cohort (unchanged from baseline / enhanced).
4. `estimate_m_from_pairwise_labels` on the synthetic pairs table.
5. No EM session. No manual priors except the `MALE↔FEMALE` clamp noted above.

## Prediction & classification

*(Placeholder — fills in Phase E2-3.)*

Same scoring algebra as baseline / enhanced. The `n_blocks` log-odds bump (+1 bit per block above 2, capped +4) is retained. The veto-override branch is **removed** — there are no vetoes in this module.

## Output contracts

*(Placeholder — fills in Phase E2-1 / E2-5.)*

- **`models/outputs/fs_splink_enhanced_2__<v>.parquet`**: 5-col cross-model eval schema (PATID_A, PATID_B, model_name, score, predicted_tier). Identical contract to baseline and enhanced; the validation notebook §11 reads all three via this contract.
- **`data/matches_model_v2/matches_model_<run_id>.parquet`**: `ProbabilisticMatches` projection. `veto_reason` becomes **optional** in `src/contracts.py` (E2-1 contract change) so this module's projection can omit it.

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
