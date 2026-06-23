# Fellegi-Sunter Enhanced Model — Build Guide

> **Status.** Shipped. Stage 4 of the production pipeline.
> **Code.** `models/experiments/fs_splink_enhanced/`
> **Authority.** This document mirrors the code. If they disagree, the code wins — update the doc.
> **Last updated.** 2026-06-17.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Why the enhanced model exists — the manual-review story](#2-why-the-enhanced-model-exists--the-manual-review-story)
3. [Where this fits in the pipeline](#3-where-this-fits-in-the-pipeline)
4. [What Fellegi-Sunter actually is (primer)](#4-what-fellegi-sunter-actually-is-primer)
5. [Module layout](#5-module-layout)
6. [Data flow — inputs and outputs](#6-data-flow--inputs-and-outputs)
7. [What changed from the baseline (summary table)](#7-what-changed-from-the-baseline-summary-table)
8. [The Splink settings — comparison vector + locked priors](#8-the-splink-settings--comparison-vector--locked-priors)
9. [Deterministic vetoes — the safety net](#9-deterministic-vetoes--the-safety-net)
10. [Manual priors — candidate-pool-aware calibration](#10-manual-priors--candidate-pool-aware-calibration)
11. [Training](#11-training)
12. [Prediction and classification — vetoes + n_blocks bump + thresholds](#12-prediction-and-classification--vetoes--n_blocks-bump--thresholds)
13. [Output contracts — eval_schema and ProbabilisticMatches](#13-output-contracts--eval_schema-and-probabilisticmatches)
14. [How to run](#14-how-to-run)
15. [Troubleshooting](#15-troubleshooting)
16. [Tests](#16-tests)
17. [Diagnostics, known limitations, and what's next](#17-diagnostics-known-limitations-and-whats-next)

---

## 1. TL;DR

The enhanced model is the **production probabilistic matcher** for the eMPI pipeline. It runs as **Stage 4**, scoring the ~169k post-rules `non_matches` pool. It takes the baseline Fellegi-Sunter model and adds four targeted improvements, every one of which is a direct response to a specific failure surfaced in a 42-pair manual review:

1. **Deterministic vetoes** (clinical safety net): four hard-no-match rules — `ssn_conflict`, `dob_year_gap`, `gender_conflict` (strict MALE↔FEMALE only), `dob_and_ssn4_conflict` — applied after FS scoring, before classification.
2. **Candidate-pool-aware manual priors** (calibration fix): m/u values for Address, Phones, and key mismatch levels are **locked** to hand-set priors rather than EM-trained, because EM's random-sample `u` under-counts within-pool agreement on these fields.
3. **A `Household_discount` comparison** (negative-interaction signal): captures "shared household indicator AND clearly different person" — the pattern that dominated the borderline FPs.
4. **n_blocks score bump** (cross-block evidence): adds up to +4 bits of weight when a pair was found via multiple blocking strategies.

**Tighter thresholds:** `auto_merge = 0.95` (up from 0.90), `review_floor = 0.40` (down from 0.50). Trades some recall for precision (clinical priority) and routes more borderline pairs to human review.

**Plain-English summary.** The baseline was well-engineered, but EM couldn't see that our candidate pool is biased toward household neighbors — so it overweighted "shared address" and "shared phone" as evidence of being the same person. The enhanced model fixes that calibration, adds a safety net so SSN conflicts can't slip through, and tightens thresholds so the model is conservative where patient safety demands it.

---

## 2. Why the enhanced model exists — the manual-review story

This is the *why* before any of the *what*. Skip if you read the baseline guide.

### 2.1 The problem: no ground truth

We have **no labels for `MDM_Population`**. There is no answer key telling us which pairs are real duplicates. That means:

- We cannot compute precision/recall on the full population
- We cannot use ROC curves or AUC
- We cannot pick thresholds from a held-out set
- **The only way to know if the baseline is working is to inspect its decisions and judge them by hand**

### 2.2 The sampling strategy

In `notebooks/fellegi_sunter/fellegi_sunter_validation.ipynb` two targeted sampling passes drew **42 pairs** for manual review, deliberately placed on the **decision boundaries** where model failures actually matter:

| Section | Sample | Purpose |
|---|---|---|
| §9.1 | 20 pairs randomly drawn from the `human_review` tier | Characterize what failure modes land in the middle band — pairs adjudicators would see in production |
| §9.2 | 12 pairs each from `[0.45, 0.55)` (around `review_floor`) and `[0.85, 0.95)` (around `auto_merge`) | Probe whether thresholds are placed correctly |

For each pair: rendered the side-by-side identifier table + Splink `gamma_*` per-field agreement levels, so the reviewer could see *exactly which fields the model weighted heavily* and judge each decision.

### 2.3 What the review revealed — three failure modes

**Failure #1 — Critical: a pair at score 0.93 with same name + DOB but DIFFERENT SSN landed in `auto_merge`.** In a clinical patient-merge context this is a never-event. Probability thresholds cannot enforce clinical safety; populated-SSN disagreement must be **structurally exclusionary**, not just one comparison level among many. → motivated the **vetoes** layer.

**Failure #2 — Systemic: 19 of 42 borderline pairs (~45%) were `family-same-household` false positives.** Different people sharing an address, phone, and last name. The baseline scored these high because the EM-trained weights for Address-exact and Phones-intersect were strongly positive (+18 bits in the worst case). **Root cause:** EM estimates `u` (= P[agreement | not a match]) by randomly sampling pairs from the whole dataset. Our candidate pool **isn't random** — it's preselected by blocking for likely household members, so shared address/phone is far more common in the pool than in random pairs. EM mistook this for "these fields are strong match evidence." → motivated the **manual priors** + **`Household_discount`** comparison.

**Failure #3 — Precision shortfall: ~9% strict precision in the 0.85-0.95 band.** Of 11 pairs labeled `same` in that range, only 1 was unambiguously a real match. Healthcare patient-merge needs ≥99% strict precision. → motivated the **threshold tightening to 0.95** and the **n_blocks bump** (rewards cross-block independent evidence).

### 2.4 Why hand-review was the right methodology

Worth defending explicitly:

1. **Without labels, there's no benchmark to run.** EM converges to a local optimum that may or may not match reality. Inspection is the only check.
2. **42 boundary pairs beat 500 random pairs.** Random tells you what the *population* looks like; boundary sampling tells you whether the *model* is working.
3. **The labels are reusable.** They now serve as the head-to-head test set in validation notebook §10, so the enhanced model can be evaluated against the same 42 pairs.
4. **It surfaces *systematic* failures.** Three distinct failure modes from 42 pairs is a much stronger signal than three errors buried in 1,000 random samples.

---

## 3. Where this fits in the pipeline

```
raw  →  1. clean  →  2. block  →  3. deterministic rules ─┬─► data/matches/        (auto-confirmed; ~35k, ~100% precision)
                                                           └─► data/non_matches/    (~169k fuzzy remainder)
                                                                 ↓
                                                        4. Fellegi-Sunter ENHANCED (this doc)
                                                                 ↓
                                                         data/matches_model/        (ProbabilisticMatches contract)
```

The enhanced model **only scores pairs the deterministic rules could not confirm**. The rules already handle the easy ~35k cases at near-100% precision; refocusing the probabilistic model on the fuzzy remainder lets every change in the enhanced model concentrate on the actual hard cases.

A future Stage 5 (clustering) is *proposed* — see `docs/Data-Contract.md` for the `Edges` and `ClusterAssignments` schemas. The ProbabilisticMatches contract is union-ready with the deterministic Matches contract for that step.

---

## 4. What Fellegi-Sunter actually is (primer)

The same primer as the baseline guide — copied here so this doc is fully standalone.

### 4.1 Two probabilities per comparison level

For each comparison field (name, DOB, SSN, address, phone, …) and each agreement level (exact / fuzzy / mismatch / null), the model carries two probabilities:

- **`m` = P(level | the pair IS a true match)** — how often real duplicates land at this level
- **`u` = P(level | the pair is NOT a match)** — how often two unrelated records land at this level by chance

### 4.2 Weight per level = log₂(m / u)

Each level contributes evidence to the total score in **bits**:

- `m > u`: **positive** bits → "this agreement is more common in matches than in non-matches" → evidence FOR.
- `m < u`: **negative** bits → evidence AGAINST. Deliberate in the enhanced model for explicit-mismatch levels (FirstNM JW<0.5, SSN full mismatch, Household_discount).

### 4.3 Total log-odds + match-prevalence prior → `match_probability`

Sum per-level weights into a `match_weight` (total bits). Combine with a population-level prior π (P[two random records are duplicates]):

> `match_probability = 1 / (1 + 2^-match_weight × (1-π)/π)`

### 4.4 Where m and u come from

- **EM (Expectation-Maximization)** trains `m` on the fields *not* used as the EM blocking rule, plus `u` on those fields.
- **Random sampling** estimates `u` on the fields *that ARE* used as the EM blocking rule (because EM can't see them). This is the step that miscalibrates — see §10.

The enhanced model **locks** m/u via Splink's `fix_m_probability` / `fix_u_probability` settings on the levels where random-sample `u` is wrong.

---

## 5. Module layout

| File | Purpose |
|---|---|
| `fs_enhanced.py` | `FSEnhanced(FSModel)` subclass: `prepare_model_input`, `build_settings` (calls `apply_manual_priors`), `classify` override (calls `super().classify()` then `apply_vetoes`), and the `run_fs_enhanced` public entry point. Splink boilerplate inherited from `models/common/fs_base.py::FSModel`. |
| `comparisons.py` | One `_build_*` function per Splink comparison; `ENHANCED_REGISTRY` composed at import time via `ComparisonRegistry`. |
| `deterministic_vetoes.py` | The 4-veto safety layer. `apply_vetoes(df_predictions, df_clean) -> df` annotates each pair with a `veto_reason: str \| None`. |
| `manual_priors.py` | Locked m/u values for Address, Phones, name-mismatch levels, SSN mismatch, Household_discount. `apply_manual_priors(settings_dict)` mutates the Splink settings dict in place. |
| `run_synthetic_enhanced.py` | Sandbox runner. Uses `models/common/synthetic_data.py` fixtures + `u_max_pairs=1e4`. Safe to run anywhere. The synthetic fixture lacks some address columns; the module degrades gracefully. |
| `run_real_enhanced.py` | **VM-only.** Reads the highest-versioned cleaned parquet + candidate-pairs parquet, runs with `u_max_pairs=1e6`, writes both the 5-col eval_schema parquet and the ProbabilisticMatches parquet. |
| `requirements.txt` | Pinned Splink + DuckDB versions to match the trained-model artifacts. |
| `__init__.py` | Re-exports `run_fs_enhanced`, `MODEL_NAME`, threshold constants, `to_probabilistic_matches`. |

Public entry point: `run_fs_enhanced(candidate_pairs_path, df_clean, full_output=False, return_diagnostics=False, u_max_pairs=1e6)` in `fs_enhanced.py`.

---

## 6. Data flow — inputs and outputs

### Inputs

1. **Cleaned patient index** (parquet): `data/processed/MDM_Population_cleaned_*.parquet`. Schema enforced by `src/contracts.py::CleanedRecords`. Required columns include the same baseline set **plus** `AddressLine1_clean`, `CityNM_clean`, `StateCD_clean`, `SexAtBirthDSC_clean` (for the Address comparison and the gender veto).

2. **Non-matches parquet** (Stage 3 output): `data/non_matches/non_matches_<run_id>.parquet`. Schema: `NonMatches` contract — identical to `CandidatePairs` but distinguished by provenance (these are the pairs the deterministic rules did not confirm).

### Outputs

The enhanced runner writes **two artifacts** per run, each for a different downstream consumer:

1. **5-col eval_schema parquet** at `models/outputs/fs_splink_enhanced__<data-version>.parquet`. Same shape as the baseline: `PATID_A, PATID_B, model_name, score, predicted_tier`. Consumed by the validation notebook §10 for head-to-head against the baseline.

2. **ProbabilisticMatches parquet** at `data/matches_model/matches_model_<run_id>.parquet` (`src/contracts.py::ProbabilisticMatches`):

| Column | Type | Notes |
|---|---|---|
| `PATID_A`, `PATID_B` | str | canonical-ordered |
| `match_source` | str | always `"model"` |
| `score` | float ∈ [0, 1] | post-bump `match_probability` |
| `match_weight` | float | log-odds in bits, post-bump |
| `classification_tier` | str | `auto_merge` / `human_review` / `no_match` |
| `veto_reason` | str \| null | name of the first veto that fired, or null |
| `source_blocks` | str \| null | pipe-delimited list from blocking |
| `n_blocks` | int \| null | number of blocks that emitted this pair |

> **Contract change (Phase E2-1, 2026-06-21):** `veto_reason` is now **optional** in `ProbabilisticMatches` so the `fs_splink_enhanced_2` matcher (which has no veto layer — vetoes moved to the upstream deterministic-rules stage) can omit it. `fs_splink_enhanced` continues to emit the column unchanged. Pandera validates frames with or without the column present.

This is the union-ready artifact for a future Stage 5 (clustering) — combine with Stage 3's `Matches` frame via the `Edges` projection (`confidence=score`, `evidence=veto_reason`).

3. **Diagnostics JSON** (non-PHI) at `models/artifacts/fs_splink_enhanced/diagnostics__<data-version>.json`. Trained m/u, per-EM-session estimates, match-prevalence prior. Used by the validation notebook for calibration audits.

---

## 7. What changed from the baseline (summary table)

| Concern | Baseline | Enhanced | Section |
|---|---|---|---|
| **auto_merge threshold** | 0.90 | **0.95** | §12 |
| **review_floor** | 0.50 | **0.40** | §12 |
| **Pipeline position** | scores all candidate pairs | scores only Stage-3 non_matches | §3 |
| **Deterministic vetoes** | none | 4 rules: SSN, DOB-gap, gender, DOB+SSN4 | §9 |
| **Address comparison** | absent (reverted in baseline R2 due to miscalibration) | **present**, with locked m/u | §8 / §10 |
| **Phones m/u** | EM-trained, severely overweighted ≥1 intersect | **locked** to candidate-pool-aware priors | §10 |
| **Name JW<0.5 mismatch levels** | absent (typos and totally-different lumped together) | **explicit** levels with locked anti-evidence m/u | §8 |
| **SSN full-mismatch level** | absent in the score (just an "else") | **explicit** level with -2.3 bit m/u | §8 |
| **Household_discount comparison** | absent | **new composite** signal | §8 |
| **n_blocks score bump** | not applied | **+1 bit per extra block, capped at +4 bits** | §12 |
| **Output artifacts** | one (5-col eval_schema) | two (eval_schema + ProbabilisticMatches) | §13 |

Everything else — the cleaned-data contract, the blocking output, the m/u-from-EM mechanism, the candidate-pairs DuckDB registration, the FirstNM JW [0.92, 0.85] thresholds, the DOB ±1d/±1m levels, the email username level, the ZIP 3-prefix level, the `retain_intermediate_calculation_columns=True` Phase A setting — is identical to the baseline.

---

## 8. The Splink settings — comparison vector + locked priors

`build_settings(include_address=True) -> dict` at `fellegi_sunter_enhanced.py:311`. Key differences from the baseline:

### 8.1 Returns a dict (not a `SettingsCreator`)

```python
creator = SettingsCreator(link_type="dedupe_only", ..., comparisons=[...])
settings_dict = creator.get_settings("duckdb").as_dict()
```

This matters because Splink's `ComparisonLevel` constructors **do not accept `m_probability` / `u_probability` kwargs**. The override mechanism is to build the settings dict and mutate `comparison_levels[i]["m_probability"]` directly, then set `fix_m_probability = True` / `fix_u_probability = True` so EM treats those values as locked rather than re-estimating them. That's what `apply_manual_priors(settings_dict)` does — it's called at the end of `build_settings()` to lock the priors before the dict is handed to the Linker.

### 8.2 New: name JW<0.5 mismatch levels

Inserted before the trailing `ElseLevel` of the default `NameComparison` via the `_insert_level_before_else()` helper:

```python
_insert_level_before_else(
    settings_dict,
    output_column_name=COL_FIRST_NM,
    new_level={
        "sql_condition": (
            f"jaro_winkler_similarity({COL_FIRST_NM}_l, {COL_FIRST_NM}_r) < 0.5 "
            f"AND {COL_FIRST_NM}_l IS NOT NULL AND {COL_FIRST_NM}_r IS NOT NULL"
        ),
        "label_for_charts": f"Jaro-Winkler distance of {COL_FIRST_NM} < 0.5",
    },
)
```

**Why:** the baseline's `else` bucket lumped typos (`Jon` vs `John`) with totally-different names (`Jon` vs `Maria`) and gave them the same weight. Splitting them lets the model assign **strong anti-evidence** (m=0.005 / u=0.30 ≈ −3 bits for FirstNM, ≈ −5 bits for LastNM) to genuinely different names without punishing typos.

### 8.3 New: SSN full-mismatch level (defense in depth)

Same `_insert_level_before_else()` mechanism. Even though the `ssn_conflict` veto already rejects SSN-different pairs, the level ensures the FS score itself reflects the conflict in any diagnostic mode that bypasses the veto layer.

### 8.4 New: `Household_discount` composite comparison

A whole new comparison appended to the settings dict:

```python
{
    "output_column_name": "Household_discount",
    "comparison_levels": [
        { "sql_condition": "<one side null>", "is_null_level": True, ... },
        {
            "sql_condition": (
                "("
                "  (AddressLine1_clean_l = AddressLine1_clean_r AND ...IS NOT NULL)"
                "  OR (len(list_intersect(Phones_array_l, Phones_array_r)) >= 1)"
                ") "
                "AND jaro_winkler_similarity(FirstNM_clean_l, FirstNM_clean_r) < 0.7 "
                "AND _dob_str_l != _dob_str_r"
            ),
            "label_for_charts": "Household indicator without identity match",
        },
        { "sql_condition": "ELSE", "label_for_charts": "All other comparisons" },
    ],
}
```

Fires when **(shared address OR shared phone) AND clearly different first name AND different DOB**. This is the negative-interaction signal the per-field comparisons cannot express — locked to m=0.05 / u=0.45 ≈ **−3.2 bits anti-evidence**. The single biggest expected mover for the family-same-household class.

### 8.5 New: Address comparison (4-level)

Appended after Household_discount:

| Level | Definition |
|---|---|
| `null` | Either side missing |
| `Exact match on AddressLine1_clean` | Full street-line agreement |
| `Same City + State + Zip` | Geographic agreement when street differs |
| `else` | Different |

Locked m/u from `manual_priors.ADDRESS_MU` (see §10).

### 8.6 What stayed the same

DOB exact/±1d/±1m, the SSN graded null/exact/last-4/else cascade, the Email null/exact/exact-username/else, Phones `ArrayIntersectAtSizes([2, 1])` (level *structure* unchanged but **m/u locked**), ZIP null/exact/3-prefix/else, the FirstNM `[0.92, 0.85]` JW thresholds, the LastNM defaults, the `retain_intermediate_calculation_columns=True` Phase A setting, and the candidate-pairs DuckDB registration.

---

## 9. Deterministic vetoes — the safety net

`models/experiments/fs_splink_enhanced/deterministic_vetoes.py`. `apply_vetoes(df_predictions, df_clean)` is called between `predict_pairs()` and `classify_pairs()` and annotates each scored pair with a `veto_reason: str | None`.

### 9.1 The four rules and their precedence

When multiple rules would fire on the same pair, the **first** in this order wins:

| # | Rule | Trigger |
|---|---|---|
| 1 | `ssn_conflict` | Both `SSN_clean` populated, full 9-digit mismatch |
| 2 | `dob_year_gap` | Both `BirthDT_clean` populated, `\|year_A − year_B\| ≥ 5` |
| 3 | `gender_conflict` | Both `SexAtBirthDSC_clean` populated, **strictly MALE↔FEMALE only** (MALE↔OTHER and FEMALE↔OTHER do *not* veto, to preserve trans/nonbinary records) |
| 4 | `dob_and_ssn4_conflict` | DOB exact mismatch AND last-4 SSN mismatch (both populated) |

Order is by **clinical evidence strength**: a populated-SSN disagreement is the single strongest "different person" signal we have.

### 9.2 What the veto does at classification time

In `classify_pairs()` (`fellegi_sunter_enhanced.py:815-817`):

```python
if VETO_REASON_COL in out.columns:
    vetoed = out[VETO_REASON_COL].notna()
    tier = tier.mask(vetoed, "no_match")
```

A non-null `veto_reason` forces `classification_tier = "no_match"` **regardless of probabilistic score**. No score, however high, can override a veto. The `veto_reason` is preserved in the rich frame and in the `ProbabilisticMatches` artifact (stripped from the 5-col legacy eval-schema).

### 9.3 Graceful schema degradation

Each veto checks its own required columns independently. If a required column is missing (e.g., the synthetic fixture lacks `SexAtBirthDSC_clean`), that one veto degrades to all-False with a warning, and the others still run. This is the "graceful-skip" pattern at `deterministic_vetoes.py:110-125`.

### 9.4 PHI discipline

`apply_vetoes` logs **aggregate counts only**: total vetoed pairs and per-rule firings *before* precedence resolution. **No identifier values ever leave this module's logs.**

```
INFO apply_vetoes: 109918/169180 pairs vetoed (per-rule firings before precedence:
     {'ssn_conflict': 42218, 'dob_year_gap': 88433, 'gender_conflict': 29111,
      'dob_and_ssn4_conflict': 41957})
```

---

## 10. Manual priors — candidate-pool-aware calibration

`models/experiments/fs_splink_enhanced/manual_priors.py`. The whole reason the enhanced model exists from a calibration perspective.

### 10.1 The miscalibration story (in detail)

EM estimates `u` (= P[level | not a match]) for fields not in the EM blocking rule by **drawing random pairs** from the cleaned dataset. The implicit assumption is that "random pairs from the dataset" is a fair stand-in for "non-matching pairs in our candidate pool."

For our cohort this assumption is false. Blocking deliberately **preselects pairs that look like they might be matches** — including pairs that share an address or a phone number because they live in the same household. So in the candidate pool, shared address and shared phone are **common**, even among non-matches. But random-pair sampling drew unrelated patients from across the dataset, who almost never share addresses or phones, and concluded "agreement is very rare among non-matches" → very low `u` → very high `log₂(m/u)` weight.

**Net result in the baseline:** Phones intersect ≥1 carried a +18 bit positive weight. A pair with the same first name + last name + shared phone + different DOB would land high in `human_review` or even `auto_merge`. That's exactly the family-same-household FP class the manual review flagged.

### 10.2 The fix: lock m/u to hand-set values

`apply_manual_priors(settings_dict)` walks the settings dict and, for each target comparison level whose `label_for_charts` matches an entry in the priors map, sets:

```python
level["m_probability"]      = m
level["u_probability"]      = u
level["fix_m_probability"]  = True   # EM holds these fixed across iterations
level["fix_u_probability"]  = True
```

Comparisons covered: `Address`, `Phones_array`, `FirstNM_clean` (JW<0.5 level only), `LastNM_clean` (JW<0.5 level only), `SSN` (full-mismatch level only), `Household_discount` (all levels).

### 10.3 The prior values and why they look the way they do

**Address** — `manual_priors.ADDRESS_MU`:

| Level | m̂ | û | log₂(m/u) bits | Direction |
|---|---|---|---|---|
| `Exact match on AddressLine1_clean` | 0.67 | 0.55 | **+0.28** | Near-zero (advantage washed out by pool preselection) |
| `Same City + State + Zip` | 0.30 | 0.30 | **0.00** | Zero (regional agreement alone is uninformative) |
| `All other comparisons` | 0.03 | 0.15 | **−2.32** | **Anti-evidence** (address truly disagreeing in a preselected pool is a strong negative signal) |

**Phones_array** — `manual_priors.PHONES_MU`:

| Level | m̂ | û | log₂(m/u) bits |
|---|---|---|---|
| `Array intersection size >= 2` | 0.40 | 0.05 | **+3.0** (rare even within households — strong positive) |
| `Array intersection size >= 1` | 0.50 | 0.55 | **~0** (was +18 in baseline; this is the household-contamination fix) |
| `All other comparisons` | 0.10 | 0.40 | **−2.0** |

**Name JW<0.5 mismatch** — `FIRSTNM_JW_LT_05_MU`, `LASTNM_JW_LT_05_MU`:

- FirstNM JW<0.5: `(0.005, 0.30)` → ~−6 bits
- LastNM JW<0.5: `(0.005, 0.15)` → ~−5 bits (last names cluster more, so stronger signal)

**SSN full-mismatch** — `SSN_FULL_MISMATCH_MU`: `(0.01, 0.05)` → ~−2.3 bits.

**Household_discount** — `HOUSEHOLD_DISCOUNT_MU`: `(0.05, 0.45)` → ~−3.2 bits.

### 10.4 Refreshing the priors

After each new round of labeled review pairs, **refresh these values**. They were initially set from the 42-pair manual review described in §2. The contract of `apply_manual_priors` is stable; the dict values are not.

---

## 11. Training

`train_model(linker, u_max_pairs=1e6, seed=42)` at `fellegi_sunter_enhanced.py:620` is **identical to the baseline's training procedure**: random-sample `u`-estimation, three complementary EM sessions (SSN-anchor, Email-anchor, Soundex+BirthYear broad), match-prevalence prior (recall=0.80), and a weight-inversion diagnostic.

The only difference is **what EM is allowed to touch**: levels with `fix_m_probability=True` / `fix_u_probability=True` are held fixed across iterations. That's where the manual priors are protected from being overwritten by EM.

The "weight inversion" warnings on locked levels (e.g., `Address all other comparisons`, `Household_discount fired`, `SSN full mismatch`, name JW<0.5) are **expected and intentional** — these levels are deliberately `m < u` because they carry anti-evidence. Don't be alarmed by them in the log output.

---

## 12. Prediction and classification — vetoes + n_blocks bump + thresholds

`classify_pairs()` at `fellegi_sunter_enhanced.py:755` is where the enhanced model diverges most from the baseline. Three things happen in order:

### 12.1 Step 1 — n_blocks score bump

If a pair was found by ≥3 blocking strategies, its `match_weight` is bumped in log-odds (bit) space:

```python
bump_bits = (out["n_blocks"] - n_blocks_bump_threshold).clip(
    lower=0, upper=n_blocks_bump_max_bits
)
if bool((bump_bits > 0).any()):
    p_safe = out["match_probability"].clip(lower=1e-12, upper=1 - 1e-12)
    weight = np.log2(p_safe / (1.0 - p_safe))
    weight_bumped = weight + bump_bits
    out["match_probability"] = 1.0 / (1.0 + np.power(2.0, -weight_bumped))
```

Defaults: `n_blocks_bump_threshold = 2`, `n_blocks_bump_max_bits = 4.0`. A pair found in 3 blocks gets +1 bit; 4 blocks gets +2 bits; 6+ blocks caps at +4 bits.

**Why:** different blocking strategies look at different evidence (SSN, phonetic name + DOB, phone intersect, etc.). A pair found by multiple blocks has **independent corroborating evidence** the FS model wouldn't otherwise see. The cap prevents runaway boosting.

### 12.2 Step 2 — threshold tier assignment

Same as the baseline mechanism, but new defaults:

```python
DEFAULT_AUTO_MERGE_THRESHOLD = 0.95
DEFAULT_REVIEW_FLOOR = 0.40
```

- `match_probability >= 0.95` → `auto_merge`
- `0.40 <= match_probability < 0.95` → `human_review`
- `match_probability < 0.40` → `no_match`

The 0.95 floor reflects the clinical priority on minimizing false positives. The 0.40 floor (down from 0.50) is deliberately permissive — we'd rather route a borderline pair to human review than silently drop it.

### 12.3 Step 3 — veto override

Any pair with a non-null `veto_reason` is forced to `no_match`, regardless of probabilistic score. No score, however high, can override a veto.

### 12.4 What `run_fs_enhanced` actually orchestrates

`run_fs_enhanced(candidate_pairs_path, df_clean, ...)` at `fellegi_sunter_enhanced.py:963` is the public entry point. It:

1. Loads candidate pairs (parquet → DataFrame)
2. Calls `prepare_model_input(df_clean)` to build the model-input frame with derived columns (DOB-string, phonetic codes, phones array, etc.)
3. Calls `build_linker(df_model, candidate_pairs_df)` — which calls `build_settings()`, which calls `apply_manual_priors()`
4. Calls `train_model(linker, u_max_pairs=...)`
5. Calls `predict_pairs(linker, candidate_pairs_df)` — re-attaches `source_blocks` / `n_blocks` from the candidate pairs frame
6. Calls `apply_vetoes(predictions, df_clean)` — annotates with `veto_reason`
7. Calls `classify_pairs(predictions_with_vetoes)` — n_blocks bump → thresholds → veto override
8. Projects to the eval-schema via `to_evaluation_schema()`, or returns the rich frame when `full_output=True`

---

## 13. Output contracts — eval_schema and ProbabilisticMatches

Two projection functions, two output artifacts, two consumers.

### 13.1 `to_evaluation_schema()` — for the head-to-head notebook

Same 5-col contract as the baseline (`models/common/eval_schema.py::EVAL_SCHEMA_COLUMNS`):

```
PATID_A | PATID_B | model_name | score | predicted_tier
```

`model_name` is constant `"fs_splink_enhanced"`. Written to `models/outputs/fs_splink_enhanced__<data-version>.parquet`. Consumed by validation notebook §10 for apples-to-apples confusion matrices and tier-shift sankeys against the baseline.

### 13.2 `to_probabilistic_matches()` — for Stage 5 clustering

Validates against `src/contracts.py::ProbabilisticMatches`. Written to `data/matches_model/matches_model_<run_id>.parquet` by the orchestrator (`src/pipeline.py`), or `data/matches_model/fs_splink_enhanced_matches_model__<data-version>.parquet` by the standalone runner. Preserves enough richness to (a) explain *why* each pair landed where it did (via `veto_reason`, `match_weight`, `source_blocks`, `n_blocks`) and (b) feed a future clustering step. The projection is at `fellegi_sunter_enhanced.py:876`.

---

## 14. How to run

### 14.1 Sandbox (off-VM)

```bash
python models/experiments/fs_splink_enhanced/run_synthetic_enhanced.py
```

Uses fixtures from `models/common/synthetic_data.py` and `u_max_pairs=1e4`. The synthetic fixture lacks `AddressLine1_clean` and `SexAtBirthDSC_clean`, so the Address comparison + the gender veto are degraded gracefully. The other three vetoes (`ssn_conflict`, `dob_year_gap`, `dob_and_ssn4_conflict`) and the locked Phones priors still run.

### 14.2 Production (VM, `empi_env` conda env, end-to-end pipeline)

The preferred entry point — runs all four stages with one in-memory cleaned frame and a single `run_id`:

```bash
python -m src.pipeline
```

Writes `data/matches_model/matches_model_<run_id>.parquet` (ProbabilisticMatches contract).

### 14.3 Production (VM, standalone enhanced runner)

When you have on-disk Stage-2 candidate-pairs and want to re-run only the enhanced FS:

```bash
python models/experiments/fs_splink_enhanced/run_real_enhanced.py

# Or override paths / version tag:
python models/experiments/fs_splink_enhanced/run_real_enhanced.py \
    --cleaned-index data/processed/MDM_Population_cleaned_v4_2026_06_11.parquet \
    --candidate-pairs src/features/outputs/blocking/candidate_pairs_v4_2026_06_11.parquet \
    --data-version v4_2026_06_11 \
    --u-max-pairs 1e6 \
    --auto-merge-threshold 0.95 \
    --review-floor 0.40
```

Writes both the 5-col eval_schema parquet (for the notebook) and the ProbabilisticMatches parquet (for Stage 5).

### 14.4 Expected log output (last ~15 lines, real cohort)

```
INFO apply_manual_priors: locked 3/3 levels on comparison Address
INFO apply_manual_priors: locked 3/3 levels on comparison Phones_array
INFO apply_manual_priors: locked 1/1 levels on comparison FirstNM_clean
INFO apply_manual_priors: locked 1/1 levels on comparison LastNM_clean
INFO apply_manual_priors: locked 1/1 levels on comparison SSN
INFO apply_manual_priors: locked 1/1 levels on comparison Household_discount
INFO train_model: training complete (u + 3 EM sessions + prevalence prior)
INFO predict_pairs: scored 169180 pairs
INFO apply_vetoes: 109918/169180 pairs vetoed (per-rule firings before precedence:
     {'ssn_conflict': 42218, 'dob_year_gap': 88433, 'gender_conflict': 29111,
      'dob_and_ssn4_conflict': 41957})
INFO classify_pairs: n_blocks bump applied to 19078/169180 pairs (max bump 4.0 bits)
INFO classify_pairs: {'no_match': 141157, 'auto_merge': 26920, 'human_review': 1103}
INFO [4/5] MODEL — tier breakdown: {'no_match': 141157, 'auto_merge': 26920,
     'human_review': 1103} -> data/matches_model/matches_model_<run_id>.parquet
```

### 14.5 What success looks like

- All six `apply_manual_priors: locked …` lines fire (3/3 Address, 3/3 Phones_array, 1/1 each on the others).
- Tier counts add up to the input count.
- The `auto_merge` count is realistic for the cohort (~15-17% of scored non_matches on the current data).
- The ProbabilisticMatches parquet validates against the contract on write — any schema drift fails loudly via pandera.

---

## 15. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: u-estimation failed because this host reports a single CPU` | Splink salts `cpu_count()` partitions when `max_pairs > 1e4`. | `--u-max-pairs 1e4`, or run on a multi-core allocation. The runner auto-retries. |
| Log warning `apply_manual_priors: <comparison> present but no level labels matched the prior dict` | Comparison-level `label_for_charts` was renamed somewhere without updating `manual_priors.py`. | Restore the label match. **This is critical** — a silent unlocked level means EM trains the calibration-bug `u` on that field. |
| Many `Weight inversion (m < u)` warnings on locked levels | **Expected.** Anti-evidence levels (FirstNM JW<0.5, SSN full mismatch, Household_discount, Address-different) are deliberately `m < u`. | Ignore for locked levels. Investigate only for un-locked levels. |
| `apply_vetoes: skipping <rule> — df_clean missing columns […]` | Synthetic-fixture path. The veto degrades to all-False; the others still run. | No action needed in the synthetic sandbox. On the VM, this indicates a cleaned-parquet schema regression — escalate. |
| Splink `_con` AttributeError | Splink private API drifted across a minor version. | Already pinned in `requirements.txt`. Don't relax the bound without re-validating `_register_candidate_pairs`. |

---

## 16. Tests

Three layers:

- **`tests/unit/test_fs_enhanced_e4_levels.py`** — unit tests for the FirstNM/LastNM JW<0.5 levels, SSN 5-9 mismatch level, Household_discount comparison structure.
- **`tests/unit/test_fs_enhanced_e5_thresholds_bump.py`** — boundary tests for `classify_pairs`: tier cuts at 0.40 and 0.95, n_blocks bump math (one-bit, capped, weight-in-sync), `to_probabilistic_matches` projection, veto override precedence.
- **`tests/unit/test_fs_enhanced_manual_priors.py`** — `apply_manual_priors` is idempotent, mutates the right comparisons, no-ops gracefully when target comparisons are absent.
- **`tests/regression/test_fs_enhanced_vetoes.py`** — known-pair sanity checks: a hand-constructed `ssn_conflict` pair lands in `no_match` regardless of its match_probability.

Tests build small in-memory fixtures inline rather than retraining a full Splink model.

---

## 17. Diagnostics, known limitations, and what's next

### 17.1 Headline numbers (real cohort, run `20260617T043941Z`)

- 169,180 non-match pairs scored
- 109,918 pairs (65%) vetoed before classification
- 19,078 pairs received an n_blocks bump (max +4 bits)
- **Final tiers: 141,157 no_match / 1,103 human_review / 26,920 auto_merge**

The 26,920 auto_merge figure (~16% of scored non_matches) is **higher than expected** — the validation notebook §10 head-to-head against the 42 labeled pairs is the gate for whether to ship as-is or iterate on priors.

### 17.2 Known limitations

- **The 42 labels are limited.** Three failure modes from 42 pairs is a strong signal, but the prior values are only as good as the sample. The plan calls for re-labeling 30-50 more pairs from the enhanced model's borderline regions if the acceptance criteria don't hold.
- **Address normalization is "raw `AddressLine1_clean`."** We deliberately did not depend on libpostal-derived `Address_normalized` because libpostal isn't always installed. Address-exact agreement is brittle to whitespace and punctuation variation — the §9 manual review flagged at least one pair where the same address was written with different spacing.
- **The gender veto is strict M↔F only.** This is deliberate (preserves trans/nonbinary records) but means M↔OTHER and F↔OTHER pairs are *not* vetoed even if they should clearly be.
- **EM still trains everything outside the locked levels.** If the candidate pool changes shape (e.g., blocking rules are retuned), the unlocked m/u values will need to be re-verified.

### 17.3 What's next

- **Validation §10 head-to-head** must produce ≥95% strict precision in combined-system `auto_merge` and ≥66% recall on `same` verdicts to AM ∪ HR before promotion.
- **Stage 5 (clustering)** — union of deterministic + probabilistic edges into final `cluster_id`s. The ProbabilisticMatches contract is union-ready with the Matches contract via the proposed `Edges` projection (`docs/Data-Contract.md`).
- **Phase B audit-flag flip** — once we trust the model, set `retain_intermediate_calculation_columns=False` / `retain_matching_columns=False` for a smaller, faster output.
- **Active-learning loop** — a separate experiment, not blocking on this model. The §10 §10.3 per-veto sample is the seed for that work.

---

## 18. Object-oriented architecture

`FSEnhanced` is a thin subclass of `FSModel` in `models/common/fs_base.py` (shipped in Phase E3-3). The per-module entry surface lives in `fs_enhanced.py` (subclass + `run_fs_enhanced`) and `comparisons.py` (`ENHANCED_REGISTRY`).

**Veto + manual-prior wiring:** `FSEnhanced.classify()` calls `super().classify()` (n_blocks bump → threshold tiers) and then passes the result through `apply_vetoes(df_classified, self._df_clean)`, where `self._df_clean` is stashed on `self` during `prepare_model_input()`. `apply_manual_priors(settings_dict)` is called at the end of `FSEnhanced.build_settings()`, locking the candidate-pool-aware m/u values before the settings dict is handed to the Linker. No base-class hook points were added; the overrides are fully self-contained in `FSEnhanced`. This preserves the original functional-module behavior exactly (see commit `f6e286e` for the design rationale).

For the full OO design narrative (ABC contracts, `ComparisonRegistry`, `TrainingStrategy` hierarchy), see the "Object-oriented architecture" section in `docs/Fellegi-Sunter-Enhanced_2.md`.

---

## See also

- `docs/Fellegi-Sunter-Baseline-Guide.md` — the reference model this one is measured against.
- `docs/Data-Contract.md` — the schemas at every pipeline boundary (the human-readable counterpart to `src/contracts.py`).
- `docs/Data-Cleaning-Guide.md` — authoritative Stage 1 cleaning rules.
- `docs/Deterministic-Rules-Guide.md` — Stage 3 deterministic rules (the upstream of this stage's input pool).
- `notebooks/fellegi_sunter/fellegi_sunter_validation.ipynb` — the manual-validation entry point. §9 is the 42-pair review that motivated this whole rebuild; §10 is the head-to-head against the baseline.
