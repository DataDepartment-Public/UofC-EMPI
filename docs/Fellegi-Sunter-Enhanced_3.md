# Fellegi-Sunter Enhanced_3 — Model Build Guide

> **Status:** in development on `feature/fs-baseline-splink`. Sections fill incrementally with each phase commit (E3-1 → E3-6). The headline match-weight table and test metrics are filled after the first VM run.

## TL;DR

`fs_splink_enhanced_3` is the fourth Fellegi-Sunter (FS) experiment in the EMPI pipeline, after `fs_splink_baseline`, `fs_splink_enhanced`, and `fs_splink_enhanced_2`. It is a deliberate **step toward simplicity and interpretability**:

1. **Seven two-level comparisons** — FirstNM, LastNM, BirthDT, SSN, Email, Phones, Address — each a single *Exact match vs All-other* distinction (preceded by a standard Splink null no-evidence level). No JW bands, no ±1-day DOB, no SSN last-4, no household anti-evidence. The point is a match-weight table a teammate can read off **one Bayes factor per field**.
2. **m trained supervised from the real-cohort silver labels** (`data/silver_labels/silver_labels_v1_2026_06_21.csv`), not synthetic — via `estimate_m_from_pairwise_labels` on a train split.
3. **lambda seeded from the pipeline's deterministic rules** via `estimate_probability_two_random_records_match`; **u from random sampling** on the cohort.
4. **Evaluated on a held-out test split** of the same silver labels (which carry both positive and negative labels) — a real confusion matrix, precision, and recall, plus Splink's label-based accuracy and threshold-selection tools.
5. Reuses the shared OO base (`models/common/fs_base.py`) — no new training machinery beyond two small backward-compatible hooks.

## Why this model exists

The earlier experiments traded interpretability for FP suppression: `fs_splink_enhanced` added vetoes + manual priors + JW bands + a household composite; `fs_splink_enhanced_2` added supervised m from *synthetic* labels plus four more comparisons (13 total). Each step added moving parts that are hard to explain to clinical/governance stakeholders and hard to audit field-by-field.

enhanced_3 asks a different question: **how well does the simplest defensible FS model do when its m-probabilities come from real, high-precision labels?** A model whose every weight is a single exact-match Bayes factor is one a non-specialist can read directly off `match_weights_chart`. The two new inputs that make this viable:

- **Real silver labels with both classes.** Earlier supervised training used synthetic positives; the silver labels are Stage-3 deterministic confirmations (~99% adjudicator precision) and — per the dataset author — include negatives, so they support both supervised m-training *and* a genuine held-out precision/recall evaluation.
- **A deterministic-rule lambda prior.** Seeding the overall match prevalence from the pipeline's own high-precision rules anchors the prior without hand-tuning.

## Pipeline position

```
raw → clean → block → deterministic rules ─┬─► data/matches/        (auto-confirmed)
                                            └─► data/non_matches/    (review pool)
                                                  ↓
                            Stage 4: fs_splink_enhanced_3 (this model)
                                                  ↓
                       data/matches_model_v2/fs_splink_enhanced_3_*.parquet
                       models/outputs/fs_splink_enhanced_3__<v>.parquet
```

For **evaluation** (validation notebook §12) enhanced_3 scores the **full candidate pool directly** — *all Alliance data through the FS model, bypassing the deterministic-rules stage* — because the silver labels are Stage-3 confirmations that Stage 3 removes from `non_matches`. Scoring the full pool keeps the silver-labeled pairs present so test metrics can be computed. enhanced_3 runs **alongside** the other FS models during development, not as a replacement.

## Fellegi-Sunter primer

See `docs/Fellegi-Sunter-Baseline-Guide.md` "FS primer" for the m/u, log₂(m/u) match weight, and the estimation algebra. enhanced_3 uses the same scoring math; only the comparison structure (minimal) and the training inputs (real silver labels + deterministic-rule prior) differ.

## Module layout — `models/experiments/fs_splink_enhanced_3/`

| File | Responsibility |
|---|---|
| `comparisons.py` | The 7 two-level builders, `ENHANCED_3_REGISTRY` + `build_registry(include_address=)`, and `PRIOR_RULES` (deterministic rules translated to Splink SQL) + `DEFAULT_PRIOR_RECALL`. |
| `fs_enhanced_3.py` | `FSEnhanced3(FSModel)` — composes the registry + `SupervisedTraining(prior_rules=PRIOR_RULES)` + `ClassificationConfig()` defaults. `prepare_model_input` reuses blocking's `_compute_derived_columns` + `Phones_array` shim; `build_settings` assembles the dedupe-only settings. Public `run_fs_enhanced_3(...)`. |
| `run_synthetic_enhanced_3.py` | Sandbox smoke test on synthetic data (stdout only; `u_max_pairs=1e4`). |
| `run_real_enhanced_3.py` | **VM-only** runner: loads silver labels, stratified train/test split, trains, scores the production pool, evaluates the held-out test split, writes eval-schema + ProbabilisticMatches parquet + diagnostics JSON (metrics + trained m/u). |

### Shared-base hooks added for enhanced_3 (`models/common/fs_base.py`)

Both backward-compatible (default off; existing baseline / enhanced / enhanced_2 callers unaffected):

- **`SupervisedTraining(prior_rules=…, prior_recall=…)`** — when `prior_rules` is set, the strategy calls `estimate_probability_two_random_records_match` on the live linker **before** m/u estimation, seeding lambda from the deterministic rules.
- **`FSModel.run(return_linker=True)`** — returns `(result, linker)` so the validation notebook can feed the trained Splink linker to `linker.visualisations.*` / `linker.evaluation.*` charts.

## Comparisons

Each of the seven comparisons has exactly three levels: a **null** no-evidence level (missing field on either side → zero weight), an **exact-match** level, and an **all-other** catch-all. Term-frequency adjustments are enabled on the high-cardinality identity fields (FirstNM, LastNM, Email) so a shared common name counts for less than a shared rare one — and so the notebook's `tf_adjustment_chart` is populated.

| Comparison | Column | Exact-match level | TF? |
|---|---|---|---|
| FirstNM | `FirstNM_clean` | string equality | ✅ |
| LastNM | `LastNM_clean` | string equality | ✅ |
| BirthDT | `_dob_str` (YYYY-MM-DD) | string equality | — |
| SSN | `SSN_clean` | full 9-digit equality | — |
| Email | `Email_clean` | string equality | ✅ |
| Phones | `Phones_array` | shared phone (`list_intersect ≥ 1`) | — |
| Address | `AddressLine1_clean` | street-line equality | — |

> **Why a null level if the model is "two-level"?** The null level is a *no-evidence* level — it contributes zero weight, so a record with a missing field is neither rewarded nor penalised. Only *populated-and-equal* vs *populated-and-different* carries evidence. That is the two-level distinction the design intends; the null level is standard Splink practice, not a third outcome.

> **Phones is multi-valued**, so its "exact match" notion is *sharing at least one phone number* (array intersection ≥ 1); an empty/null list on either side is the null level.

### Derived match-weight table (m / u / log₂ Bayes factor)

Filled from the first VM run via validation-notebook **§12.4** (saved to `notebooks/fellegi_sunter/figures/enhanced_3_match_weights__<VERSION_TAG>.csv`). Each row is one comparison level.

| Comparison | Level | m | u | log₂ BF |
|---|---|---|---|---|
| FirstNM | Exact match | _TBD_ | _TBD_ | _TBD_ |
| FirstNM | All other | _TBD_ | _TBD_ | _TBD_ |
| LastNM | Exact match | _TBD_ | _TBD_ | _TBD_ |
| LastNM | All other | _TBD_ | _TBD_ | _TBD_ |
| BirthDT | Exact match | _TBD_ | _TBD_ | _TBD_ |
| BirthDT | All other | _TBD_ | _TBD_ | _TBD_ |
| SSN | Exact match | _TBD_ | _TBD_ | _TBD_ |
| SSN | All other | _TBD_ | _TBD_ | _TBD_ |
| Email | Exact match | _TBD_ | _TBD_ | _TBD_ |
| Email | All other | _TBD_ | _TBD_ | _TBD_ |
| Phones | Shared phone | _TBD_ | _TBD_ | _TBD_ |
| Phones | All other | _TBD_ | _TBD_ | _TBD_ |
| Address | Exact match | _TBD_ | _TBD_ | _TBD_ |
| Address | All other | _TBD_ | _TBD_ | _TBD_ |

## Training procedure

Owned by `SupervisedTraining`, in this order:

```
a. estimate_probability_two_random_records_match(PRIOR_RULES, recall=0.9)   → lambda
b. estimate_m_from_pairwise_labels(silver-label TRAIN split, positives)     → m
c. estimate_u_using_random_sampling(max_pairs=1e6 VM / 1e4 sandbox)         → u
```

- **`PRIOR_RULES`** translate `src/models/deterministic_rules.py::RULES` into Splink `l.<col> = r.<col>` SQL: SSN+DOB; NAME+DOB+EMAIL; NAME+DOB+PHONE; NAME+DOB+SEX; NAME+DOB+ADDRESS. `DEFAULT_PRIOR_RECALL = 0.9` (the rules are high-recall over true matches in the candidate pool). These seed lambda only — they do **not** select training pairs.
- **m** is estimated from the silver-label *positives* in the train split. Silver-label PATIDs are real-cohort IDs present in the cleaned parquet, so training is **single-linker** (no split-training / `labels_records_df` needed).
- **u** comes from random pairs — it does **not** use the labels. Note: exact-match levels for rare identifiers (SSN, DOB, Email, Phones) are essentially never observed in random sampling, so Splink leaves their u at the settings default and logs "u not trained for … Exact match". This is expected for a high-discrimination exact field and is documented under **Known limitations**.

## Classification

Reuses the base `FSModel.classify()`: the n_blocks log-odds bump (≥2 blocks, capped at 4 bits) then threshold binning with `ClassificationConfig` defaults **`auto_merge = 0.95`, `review_floor = 0.40`**:

- `score < 0.40` → `no_match`
- `0.40 ≤ score < 0.95` → `human_review`
- `score ≥ 0.95` → `auto_merge`

> Held-out test metrics in §12 / the runner are computed on **un-bumped** scores (the test pairs carry no blocking provenance), so they reflect the raw model score.

## Outputs

| Artifact | Path | Contract |
|---|---|---|
| Cross-model eval schema | `models/outputs/fs_splink_enhanced_3__<v>[_full_pool].parquet` | 5-col `PATID_A \| PATID_B \| model_name \| score \| predicted_tier` |
| Probabilistic matches | `data/matches_model_v2/fs_splink_enhanced_3_matches_model__<v>[_full_pool].parquet` | `ProbabilisticMatches` (validated; `veto_reason` omitted) |
| Diagnostics (non-PHI) | `models/artifacts/fs_splink_enhanced_3/diagnostics__<v>[_full_pool].json` | tier counts, score quantiles, split sizes, **test metrics** (precision/recall/F1 + confusion), trained m/u settings |

## Test-split evaluation results

Filled after the first VM run (`run_real_enhanced_3.py --score-full-candidate-pool`; diagnostics JSON `test_metrics`):

| Metric (auto_merge as positive) | Value |
|---|---|
| Precision | _TBD_ |
| Recall | _TBD_ |
| F1 | _TBD_ |
| Confusion (tier × label) | _TBD_ |

Sandbox reference (synthetic data, illustrative only): positive auto_merge recall ≈ 78%, precision ≈ 94%.

## How to run

```bash
# Sandbox smoke test (synthetic data, off-VM OK).
python -m models.experiments.fs_splink_enhanced_3.run_synthetic_enhanced_3

# VM: train on silver labels + score the full candidate pool + write metrics.
python -m models.experiments.fs_splink_enhanced_3.run_real_enhanced_3 \
    --score-full-candidate-pool

# VM: regenerate validation-notebook §12, then run the notebook.
python scripts/_inject_section_12.py
```

## Tests

- `tests/unit/test_fs_enhanced_3_comparisons.py` — registry shape, 3-level structure, TF placement, prior-rule translation (21 tests).
- `tests/unit/test_fs_base_enhanced_3_hooks.py` — `prior_rules` lambda seeding + `return_linker` (8 tests).
- `tests/unit/test_fs_enhanced_3_model.py` — `FSEnhanced3` wiring (7 tests).
- `tests/unit/test_fs_enhanced_3_real_runner.py` — PHI-free runner helpers: split, canonicalize, metrics, label validation (7 tests).

The full train→score round-trip is exercised by `run_synthetic_enhanced_3.py` (not a pytest test — it trains a real Splink model).

## Known limitations

- **u untrained for rare exact levels.** SSN / DOB / Email / Phones exact-match levels are ~never seen in random u-sampling, so they keep Splink's default u. This under-states their discriminating power slightly. A future refinement could set literature-informed u for these levels (mirroring `manual_priors.py` in the enhanced module).
- **Deliberately under-expressive.** No typo tolerance (JW), no near-DOB, no SSN last-4, no household anti-evidence. enhanced_3 will under-recall positives with a single corrupted identity field and cannot discount same-household false positives the way enhanced_2 does. That is the explicit interpretability trade-off — compare head-to-head in the validation notebook before promoting.
- **Test metrics use un-bumped scores** (see Classification).

## See also

- `docs/Fellegi-Sunter-Enhanced_2.md` — the richer supervised predecessor.
- `docs/Fellegi-Sunter-Enhanced-Guide.md` — the EM + vetoes + manual-priors model.
- `docs/Deterministic-Rules-Guide.md` — the `RULES` that `PRIOR_RULES` mirror.
- `models/common/fs_base.py` — the shared OO base (`FSModel`, `SupervisedTraining`, `ClassificationConfig`).
