# Fellegi-Sunter Enhanced_3 — Model Build Guide

> **Status:** built on `feature/fs-baseline-splink` (E3-1 → E3-6); first VM run scored + evaluated on data version **`v6_2026_06_25`**. Match-weight table, test metrics, and the "Results & insights" + "Proposed updates for enhanced_4" sections below reflect that run.

## TL;DR

`fs_splink_enhanced_3` is the fourth Fellegi-Sunter (FS) experiment in the EMPI pipeline, after `fs_splink_baseline`, `fs_splink_enhanced`, and `fs_splink_enhanced_2`. It is a deliberate **step toward simplicity and interpretability**:

1. **Seven two-level comparisons** — FirstNM, LastNM, BirthDT, SSN, Email, Phones, Address — each a single *Exact match vs All-other* distinction (preceded by a standard Splink null no-evidence level). No JW bands, no ±1-day DOB, no SSN last-4, no household anti-evidence. The point is a match-weight table a teammate can read off **one Bayes factor per field**.
2. **m trained supervised from the real-cohort silver labels** (`data/silver_labels/silver_labels_v1_2026_06_21.csv`), not synthetic — via `estimate_m_from_pairwise_labels` on a train split.
3. **lambda seeded from the pipeline's deterministic rules** via `estimate_probability_two_random_records_match`; **u from random sampling** on the cohort.
4. **Evaluated on a held-out test split** of the same silver labels (which carry both positive and negative labels) — a real confusion matrix, precision, and recall, plus Splink's label-based accuracy and threshold-selection tools.

> **Silver-labels schema:** `PATID_A, PATID_B, silver_label`, where `silver_label` is a boolean (`True` = match, `False` = non-match). The runner's `--label-col` defaults to `silver_label` and coerces `True`/`False` to `1`/`0`.
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

From the first VM run (`v6_2026_06_25`), via validation-notebook **§12.4** (saved to `notebooks/fellegi_sunter/figures/enhanced_3_match_weights__v6_2026_06_25.csv`). Each row is one comparison level; the match weight is log₂(m/u).

| Comparison | Level | m | u | log₂ BF |
|---|---|---|---|---|
| FirstNM | Exact match | 0.883 | 0.00119 | **+9.54** |
| FirstNM | All other | 0.117 | 0.999 | −3.10 |
| LastNM | Exact match | 0.807 | 0.00092 | **+9.78** |
| LastNM | All other | 0.193 | 0.999 | −2.37 |
| BirthDT | Exact match | 0.998 | 0.000045 | **+14.43** |
| BirthDT | All other | 0.002 | 1.000 | **−8.93** |
| SSN | Exact match | 0.938 | _untrained¹_ | _(default)_ |
| SSN | All other | 0.062 | 1.000 | −4.02 |
| Email | Exact match | 0.579 | _untrained¹_ | _(default)_ |
| Email | All other | 0.421 | 1.000 | −1.25 |
| Phones | Shared phone | 0.551 | 0.0000155 | **+15.12** |
| Phones | All other | 0.449 | 1.000 | −1.16 |
| Address | Exact match | 0.206 | 0.0000173 | **+13.54** |
| Address | All other | 0.794 | 1.000 | −0.33 |

¹ The SSN and Email **exact** levels never co-occur in random u-sampling (two random records essentially never share a full SSN or full email), so Splink leaves their u untrained and the exact weight falls back to a default rather than a data-grounded value. DOB / Phones / Address exact levels did get a (tiny) sampled u. See **Results & insights** and **Known limitations**.

> **How to read this table** — two patterns drive everything downstream. (1) *Agreement is decisive*: every exact level is strongly positive (+9 to +15 bits). (2) *Disagreement is mostly cheap*: the "All other" weights are small for every field **except BirthDT** (−8.93), because among true matches a disagreeing field is common (e.g. m=0.79 for Address "All other" → 79% of true-match pairs differ on AddressLine1, so a differing address barely moves the score). That asymmetry is the precision story below.

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

From the first VM run (`v6_2026_06_25`), validation-notebook §12.1. Held-out test split: **10,213 positives / 30,748 negatives**.

Confusion matrix (rows = silver label, cols = predicted tier):

| silver label | no_match | human_review | auto_merge |
|---|---|---|---|
| different (0) | 27,529 | 437 | 2,782 |
| same (1) | 70 | 464 | 9,679 |

| Metric | auto_merge as positive | auto_merge ∪ human_review as positive |
|---|---|---|
| Precision | **0.777** (9,679 / 12,461) | 0.759 (10,143 / 13,362) |
| Recall | **0.948** (9,679 / 10,213) | 0.993 (10,143 / 10,213) |
| F1 | **0.854** | 0.860 |

Headline: **recall is strong (94.8% at auto_merge, 99.3% if human_review counts), but auto_merge precision is only 77.7%** — 2,782 truly-different pairs auto-merge. The full analysis is in **Results & insights** below.

Sandbox reference (synthetic data, illustrative only): positive auto_merge recall ≈ 78%, precision ≈ 94%.

## Results & insights (v6_2026_06_25)

All figures referenced below were produced by validation-notebook §12 and saved to
`notebooks/fellegi_sunter/figures/<name>__v6_2026_06_25.{png,csv}`.

1. **High recall, precision-limited.** The model recovers almost every true match (94.8% to auto_merge, 99.3% to auto_merge ∪ human_review) but auto-merges 2,782 truly-different pairs — **77.7% auto_merge precision**. For an MPI where an auto-merge silently fuses two patients' records, that false-merge rate is the gating problem, not recall. *(confusion PNG)*

2. **Under-penalizing disagreement is the root cause.** The match-weight table shows every "All other" (disagreement) level is weak except BirthDT. This is not a bug in training — it is what the silver-label positives say: a disagreeing field is *common* among true matches, so m(All-other) is high and log₂(m/u) ≈ 0:
   - Address All-other **−0.33** (m=0.794 → 79% of true matches differ on AddressLine1)
   - Phones All-other **−1.16** (m=0.449), Email All-other **−1.25** (m=0.421)
   - FirstNM **−3.10**, LastNM **−2.37**, SSN **−4.02**
   - only **BirthDT All-other −8.93** (m=0.002) actually bites.
   So a conflicting SSN, email, address, or phone barely lowers a score. *(match-weights CSV/PNG)*

3. **DOB + one name alone clears auto-merge.** Exact BirthDT (+14.4) + exact FirstNM or LastNM (+9.5) is ≈ +24 bits — enough to overcome the strongly negative prior and land at probability ≈ 1.0 **with no corroborating identifier**. Because the other disagreement penalties are tiny (finding 2), nothing pulls such a pair back. This is the false-positive mechanism: different people who share a birthday and a common name (and twins / siblings). *(waterfall PNG shows the mirror case — a DOB *mismatch* of −8.93 correctly sinks a same-surname pair to p≈0.00006.)*

4. **The false positives are high-confidence, so threshold-tuning will not rescue precision.** The 2,782 false auto-merges sit at score ≥ 0.95 by definition; the human_review band (0.40–0.95) holds only 437 negatives. Raising the auto-merge threshold would shed recall faster than it sheds false merges. **The fix must be structural — stronger disagreement weights — not a threshold move.** *(score-histogram PNG; confirm precisely against the threshold-selection tool once exported.)*

5. **Address agreement is over-weighted relative to how rare it is.** Exact AddressLine1 contributes **+13.5 bits** (near-conclusive) but only 21% of true matches actually share it (m=0.206). A near-conclusive reward on a *household-shared* signal inflates scores for same-address / different-person pairs — exactly the failure enhanced_2's `Household_discount` composite was built to catch. *(match-weights CSV)*

6. **SSN and Email exact `u` are untrained.** Two random records never share a full SSN or email in u-sampling, so those exact levels fall back to a default u rather than a data-grounded one (DOB/Phones/Address got a tiny sampled u). Their exact weights are therefore not calibrated from this cohort — a latent risk if the default diverges from reality. *(match-weights CSV footnote ¹; m_u_parameters / parameter_estimates PNGs)*

7. **Scores are bimodal / over-confident.** The full pool splits 138,214 no_match / 4,678 human_review / 61,913 auto_merge — only **2.3%** lands in the review band. Two-level comparisons admit few intermediate score combinations, so pairs are pushed to the extremes and clerical review gets a near-empty queue. *(score-histogram PNG)*

8. **What this says vs enhanced_2.** enhanced_3's deliberate simplicity is what made findings 2/3/5 legible — they were always implicit, but a one-Bayes-factor-per-field model surfaces them cleanly. enhanced_2 already carries the machinery that targets them (multi-level mismatch bands like JW<0.5 and SSN 9-digit conflict, the `Household_discount` composite, manual priors). The takeaway is not "enhanced_2 is better" but "enhanced_3 isolated *which* of enhanced_2's extra parts are load-bearing for precision."

## Proposed updates for enhanced_4 (prioritized by expected precision impact)

- **P1 — Turn "All other" into explicit *conflict* vs *missing* levels.** The single biggest lever (findings 2, 3). Split each disagreement level so a *populated-and-conflicting* field is penalized hard while a *missing* field stays neutral: SSN full-9 conflict, Email different-domain/local conflict, BirthDT year-gap, name JW<0.5. The enhanced_2 builders (`models/experiments/fs_splink_enhanced_2/comparisons.py`) already implement these — port them.
- **P2 — Stop DOB+name-only auto-merges; add a household-discount composite.** Require at least one corroborating strong identifier (SSN / email / phone) before a DOB+name pair can auto-merge, and penalize "shared address but identity conflict" (findings 3, 5). Reuse enhanced_2's `Household_discount`.
- **P3 — Ground the SSN & Email exact `u`.** Set literature- or cohort-informed u for the untrained exact levels (finding 6), mirroring `models/experiments/fs_splink_enhanced/manual_priors.py` (`fix_u_probability`).
- **P4 — Re-tune thresholds *with* the threshold-selection tool, expecting limited gain.** Because the FPs are high-confidence (finding 4), document precision/recall across candidate thresholds rather than nudging 0.95 — and treat structural fixes (P1/P2) as the real precision lever.
- **P5 — Add near-match levels to repopulate the review band.** Name JW bands, ±1-day / month-day-swap DOB, SSN last-4 (findings 1, 7) — recovers the 534 positives currently missed and gives clerical review a meaningful, non-empty queue.

Recommended sequencing: **P1 → P2** first (directly attack the false-merge rate), then **P3 / P5** (calibration + recall/queue), with **P4** as a measurement step throughout.

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

- **Auto_merge precision ceiling ≈ 78% (measured, v6).** The dominant limitation: 2,782 truly-different test pairs auto-merge because disagreement levels under-penalize (see **Results & insights** findings 2–4). This is structural, not a threshold artifact — addressed by enhanced_4 P1/P2.
- **`u` untrained for the SSN & Email exact levels.** Confirmed in the v6 match-weights CSV: those two exact levels never co-occur in random u-sampling, so they keep Splink's default u (DOB/Phones/Address did get a tiny sampled u). Their exact weights are not cohort-calibrated. Fix: enhanced_4 P3 (manual/literature u, mirroring `manual_priors.py`).
- **Deliberately under-expressive.** No typo tolerance (JW), no near-DOB, no SSN last-4, no household anti-evidence. enhanced_3 under-recalls positives with a single corrupted identity field (534 missed in the v6 test split) and cannot discount same-household false positives the way enhanced_2 does. That is the explicit interpretability trade-off — and the v6 results quantify exactly which of those omissions cost the most (enhanced_4 P1/P2/P5).
- **Over-confident score distribution.** Only 2.3% of the full pool lands in the 0.40–0.95 review band, so clerical review gets a near-empty queue (finding 7).
- **Test metrics use un-bumped scores** (see Classification).

## See also

- `docs/Fellegi-Sunter-Enhanced_2.md` — the richer supervised predecessor.
- `docs/Fellegi-Sunter-Enhanced-Guide.md` — the EM + vetoes + manual-priors model.
- `docs/Deterministic-Rules-Guide.md` — the `RULES` that `PRIOR_RULES` mirror.
- `models/common/fs_base.py` — the shared OO base (`FSModel`, `SupervisedTraining`, `ClassificationConfig`).
