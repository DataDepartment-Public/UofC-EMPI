# Fellegi-Sunter Enhanced_4 — Model Build Guide

> **Status:** built on `feature/fs-baseline-splink` (E4-1 → E4-7); **not yet run on the VM**. Match-weight table and results sections are placeholders pending the first real-cohort run.

## TL;DR

`fs_splink_enhanced_4` is the fifth Fellegi-Sunter (FS) experiment in the EMPI pipeline, after `fs_splink_baseline`, `fs_splink_enhanced`, `fs_splink_enhanced_2`, and `fs_splink_enhanced_3`. It is the **structural precision fix** for enhanced_3's precision ceiling (~78% auto_merge), realized by implementing every proposal in enhanced_3's "Proposed updates for enhanced_4" section (P1–P5):

1. **Ten multi-level comparisons with explicit conflict-vs-missing mismatch levels** — each identity field now carries a hard-conflict level (populated-and-disagreeing) separate from a null/missing level (neutral, no evidence). The single uninformative "All other" level that diluted enhanced_3's disagreement penalties is replaced with specific conflict levels that contribute real negative weight.
2. **A deterministic corroboration gate in `classify()`** — after threshold-based tier assignment, an `auto_merge` pair is demoted to `human_review` unless it carries corroborating evidence beyond DOB + name: a person-unique signal (SSN or Email agreement) or both household signals (phone AND address). This closes the twin/cohabitant false-positive hole that accounted for enhanced_3's precision floor.
3. **Grounded SSN / Email exact u** — the two exact levels that were left untrained in enhanced_3 (random pairs never co-occur on a full SSN or email, so Splink left their u at a default) are now set to `1/n_distinct` from the real cohort before training and pinned via `fix_u_probability=True`.
4. **m stays purely supervised** from the real-cohort silver labels — no manual m-locking (unlike `fs_splink_enhanced.manual_priors`). Because the silver labels are deterministic confirmations, they almost never populate the hard-conflict levels, so those penalties are floor-driven: the corroboration gate — not the comparison weights — is the primary precision lever.
5. Reuses the shared OO base (`models/common/fs_base.py`) — no new training hooks needed beyond those added for enhanced_3.

> **Key metric to watch on the first VM run:** does auto_merge precision rise materially above enhanced_3's ~78% at comparable recall, and how many true matches does the gate demote (the recall cost)?

## Why this model exists

Enhanced_3's first VM run (`v6_2026_06_25`) delivered recall ≈ 95% but auto_merge precision ≈ 78%. The post-mortem (see `docs/Fellegi-Sunter-Enhanced_3.md` "Results & insights") established three structural causes:

1. **Under-penalized disagreement.** The single "All other" level for each field is weak because true matches commonly disagree on address, phone, and even email — so m(All-other) is high and log₂(m/u) ≈ 0 for most fields. Only BirthDT disagreement contributed real negative weight (−8.93 bits).
2. **DOB + one name alone clears auto-merge.** Exact BirthDT (+14.4 bits) plus exact FirstNM or LastNM (+9.5 bits) sums to ≈ +24 bits — enough to auto-merge with no corroborating identifier. Same-birthday sibling or twin pairs plus common-name pairs are the FP mechanism.
3. **The false positives are high-confidence (≥0.95), so threshold tuning cannot rescue precision.** Raising the auto-merge threshold sheds recall faster than it sheds false merges.

Enhanced_4 is the structural fix: explicit conflict levels penalize populated-but-disagreeing fields; the gate blocks DOB+name-only auto-merges unless a person-unique or dual-household signal corroborates.

## Pipeline position

```
raw → clean → block → deterministic rules ─┬─► data/matches/        (auto-confirmed)
                                            └─► data/non_matches/    (review pool)
                                                  ↓
                            Stage 4: fs_splink_enhanced_4 (this model)
                                                  ↓
                       data/matches_model_v2/fs_splink_enhanced_4_*.parquet
                       models/outputs/fs_splink_enhanced_4__<v>.parquet
```

For **evaluation** (and to keep the silver-labeled pairs in the scoring pool) the runner supports `--score-full-candidate-pool`, which bypasses the post-rules non_matches filter — identical to enhanced_3's evaluation approach. The corroboration gate fires inside `classify()` on both the production pool and the held-out test split, so test metrics reflect the full system.

## Fellegi-Sunter primer

See `docs/Fellegi-Sunter-Baseline-Guide.md` "FS primer" for the m/u, log₂(m/u) match weight, and the estimation algebra. Enhanced_4 uses the same scoring math; the changes are in the comparison structure (multi-level conflict-vs-missing) and the post-classification gate.

## Module layout — `models/experiments/fs_splink_enhanced_4/`

| File | Responsibility |
|---|---|
| `comparisons.py` | The 10 comparison builders, `ENHANCED_4_REGISTRY` + `build_registry(include_address=)`, and `PRIOR_RULES` / `DEFAULT_PRIOR_RECALL` (copied verbatim from enhanced_3 for identical lambda seeding). |
| `fs_enhanced_4.py` | `FSEnhanced4(FSModel)` subclass — composes the registry + `SupervisedTraining(prior_rules=PRIOR_RULES)` + `ClassificationConfig()` defaults. `prepare_model_input` stashes `df_clean` on `self`; `build_settings` calls `_ground_untrained_u`; `classify` override applies the corroboration gate after the base-class tier assignment. Public entry: `run_fs_enhanced_4(...)`. `MODEL_NAME = "fs_splink_enhanced_4"`. |
| `corroboration_gate.py` | `apply_corroboration_gate(df_classified, df_clean)` — demotes uncorroborated `auto_merge` pairs to `human_review`, annotating the `corroboration` column. |
| `run_synthetic_enhanced_4.py` | Sandbox smoke test on synthetic data (stdout only; `u_max_pairs=1e4`). Confirms end-to-end flow, `ProbabilisticMatches` validation, and gate firing. |
| `run_real_enhanced_4.py` | **VM-only** runner: loads silver labels, stratified train/test split, trains, scores the production pool, evaluates the held-out test split (gate applied), writes eval-schema + ProbabilisticMatches parquet + diagnostics JSON. |

### Shared-base hooks reused from enhanced_3 (`models/common/fs_base.py`)

No new hooks were added to `fs_base.py` for enhanced_4. The two hooks enhanced_3 introduced are reused unchanged:

- **`SupervisedTraining(prior_rules=…, prior_recall=…)`** — seeds lambda via `estimate_probability_two_random_records_match` before m/u.
- **`FSModel.run(return_linker=True)`** — returns `(result, linker)` for the validation notebook's Splink charts.

## Comparisons

Enhanced_4 defines **10 comparisons** assembled by `build_registry(include_address=)`. The base 7 run in the sandbox; the address-gated 3 require `CityNM_clean` / `StateCD_clean` (available in production, absent in the synthetic sandbox).

### Design principle: null = no evidence, conflict = hard penalty

For every field, the null level is a *no-evidence* level (missing on either side → zero weight). An *exact* or near-match level contributes positive weight. A *hard-conflict* level (both fields populated but clearly disagree) contributes real negative weight. The single "All other" catch-all that blurred this distinction in enhanced_3 is replaced with named conflict levels wherever the field warrants it.

### Comparison levels

| Comparison | Gate | Levels (first-match wins) |
|---|---|---|
| FirstNM | base | null / exact(TF) / JW≥0.92 / JW≥0.88 / **JW<0.5 hard-conflict** / else |
| LastNM | base | null / exact(TF) / full_name_compact / JW≥0.95 / JW≥0.88 / DM-phonetic / **JW<0.5 hard-conflict** / else |
| BirthDT | base | null / exact / month-day-swap / DamerauLevenshtein≤1 / ±1 day / ±1 month / ±1 year / else |
| SSN | base | null / exact-9 / last-4 match / **full-9 mismatch** / else |
| Email | base | null / exact(TF) / same-username-diff-domain / JW≥0.88 / else |
| Phones | base | null+empty / ≥2 shared / ≥1 shared / else |
| ZIP | base | null / 5-digit exact / 3-digit prefix / else |
| State | address | null / exact / else |
| Household_discount | address | null / household-indicator-without-identity / else |
| Address | address | null / exact street / Same City+State+Zip / else |

**Term-frequency adjustments** are enabled on FirstNM, LastNM, and Email (exact levels), so a shared common name counts for less than a shared rare one.

**Notable additions vs enhanced_3:**

- `full_name_compact` level on LastNM catches hyphenation/spacing variance (e.g. `MARTINEZ-CASTILLO ↔ MARTINEZCASTILLO`) above the JW bands.
- DM-phonetic level on LastNM catches same-sound/different-spelling after JW bands.
- Month-day-swap level on BirthDT catches transposed MM/DD entry (inserted right after `exact`, before the ±1-year band — otherwise the `≤1 year` band, matching every same-year pair, would absorb the transposition). Splink's `DateOfBirthComparison` also auto-inserts a `DamerauLevenshtein≤1` level after it.
- SSN full-9 mismatch and name JW<0.5 are the explicit conflict levels that replace the weak "All other" in enhanced_3.
- `Household_discount` composite anti-evidence: fires when the pair shares an address or phone but name/DOB disagree — targets same-household/different-person FPs. Requires address columns (address-gated).
- ZIP and State comparisons add geographic near-match evidence below the Address level.

### Why conflict-vs-missing matters

In enhanced_3, an SSN or email disagreement fell into "All other" with weight ≈ 0 because m(All-other) was high (true matches often disagree on these fields). In enhanced_4, populated-and-conflicting SSN falls into "full-9 mismatch" with a weight that approaches −∞ as m approaches 0. Because the silver-label positives (which are deterministic confirmations that passed Stage 3's exact-SSN and name+DOB+email rules) almost never populate this conflict level, m for the conflict level is near zero and its Bayes factor is strongly negative.

> **Floor-driven conflict weights caveat.** Because m is supervised from silver-label *positives* that almost never exhibit SSN/email/name conflict (they are deterministic confirmations), the conflict level m-values are floor-driven rather than information-theoretically estimated. The Bayes factor may appear very large but is statistically fragile for rare conflict levels. This is acceptable in practice — the corroboration gate is the primary precision lever, not the conflict weights — but should be noted when interpreting the match-weight table.

### Derived match-weight table (m / u / log₂ Bayes factor)

_TBD — pending first VM run (`run_real_enhanced_4.py --score-full-candidate-pool`)_

## Training procedure

Identical to enhanced_3 — owned by `SupervisedTraining`, in this order:

```
a. estimate_probability_two_random_records_match(PRIOR_RULES, recall=0.9)   → lambda
b. estimate_m_from_pairwise_labels(silver-label TRAIN split, positives)     → m
c. estimate_u_using_random_sampling(max_pairs=1e6 VM / 1e4 sandbox)         → u
```

**`PRIOR_RULES`** are copied verbatim from enhanced_3: SQL translations of `src/models/deterministic_rules.py::RULES` (SSN+DOB; NAME+DOB+EMAIL; NAME+DOB+PHONE; NAME+DOB+SEX; NAME+DOB+ADDRESS). `DEFAULT_PRIOR_RECALL = 0.9`.

**m** is estimated from the silver-label positives in the train split. Silver-label PATIDs are real-cohort IDs present in the cleaned parquet — single-linker training (no `labels_records_df` needed).

**u** comes from random pairs. The SSN and Email exact levels are grounded before this step (see below) via `fix_u_probability=True` so random sampling does not overwrite the grounded values. After training, `audit_untrained_u(linker)` reports any remaining non-null levels that still have no u (logged as a warning; stored in diagnostics).

## u-Grounding for SSN and Email exact levels

In enhanced_3, the SSN and Email *exact* levels were left with untrained u values because two random records essentially never share a full SSN or email — Splink's random-sampling step never observes those co-occurrences and leaves u at its default.

Enhanced_4 grounds these before training in `_ground_untrained_u(settings)`:

- For each target comparison (SSN, Email), u is set to `1 / n_distinct(col)` computed from `df_clean`, the probability two random records drawn uniformly collide on that identifier.
- `fix_u_probability = True` is set on the level, so the random-sampling step honors the pinned value rather than overwriting it.
- If the column is unavailable on `df_clean` or has zero distinct values, a hardcoded fallback is used: **SSN → 1e-6**, **Email → 5e-4**.

Only u is grounded; m stays purely supervised. This mirrors the `fix_u_probability` mechanism in `fs_splink_enhanced/manual_priors.py` without locking m.

## Corroboration gate

The gate is the primary precision lever in enhanced_4. It runs inside `FSEnhanced4.classify()` after the base-class n_blocks bump and threshold-based tier assignment, implemented in `corroboration_gate.py`.

### Tiered rule

Identifiers are split into two classes by how uniquely they name a *person*:

| Class | Signals | Rationale |
|---|---|---|
| Person-unique | SSN exact agreement, Email exact agreement | One live human per value; one signal is sufficient |
| Household | Phone set intersection ≥1, AddressLine1 exact agreement | Shared among cohabitants; need both together |

For each `auto_merge` pair:

```
ssn_agree      = both SSN_clean populated AND equal
email_agree    = both Email_clean populated AND equal
phone_share    = both Phones_set non-empty AND |set_A ∩ set_B| ≥ 1
address_agree  = both AddressLine1_clean populated AND equal

person_unique   = ssn_agree OR email_agree
household_count = int(phone_share) + int(address_agree)   # 0, 1, or 2

keep  = person_unique OR (household_count ≥ 2)
demote = auto_merge AND NOT keep
```

Demoted pairs are moved from `auto_merge` → `human_review`. They are never moved to `no_match`, and pairs already in `human_review` / `no_match` are never promoted. The `corroboration` column records `"demoted_no_corroboration"` on demoted pairs and `None` elsewhere.

### Why household signals only count in pairs

Twins and cohabitants legitimately share a phone number and a mailing address. A pair with matching DOB + one name + one shared household signal is exactly the twin / same-household false-positive that enhanced_3 leaked at high confidence. Requiring *both* household signals together (phone AND address) — or one truly person-unique signal — closes that hole without penalizing a genuine duplicate that also happens to agree on SSN or email.

### Relationship to the FS score and Household_discount

The gate is deterministic and orthogonal to the probabilistic score. The `Household_discount` comparison already *down-weights* shared-household evidence inside the FS score; the gate is the hard backstop for twin/cohabitant survivors that the score alone cannot remove. The gate does not interact with `Household_discount` — they target the same FP class from different angles.

### Degraded-signal handling

If any corroborating column is absent from `df_clean`, that signal degrades to all-False (with a warning), so `household_count` caps at 1 and the household keep-branch becomes unreachable — pairs must satisfy the person-unique branch to stay `auto_merge`. If *all* corroborating columns are absent, the gate skips entirely and returns `corroboration=None` everywhere (fail-open only when no signal is available at all; the synthetic sandbox exercises this path on records without address columns).

### Gate on test metrics

The corroboration gate fires on the held-out test split inside `_score_pairs_with_trained_model` → `model.classify()`. Test metrics (precision/recall/confusion) therefore reflect the full enhanced_4 system — FS score + gate — not just the score alone. The key number to watch is how many true-match positives are demoted (the recall cost of the gate) vs how many true-negative auto_merges are blocked (the precision gain).

## Classification

`FSEnhanced4.classify()` applies in order:

1. **n_blocks log-odds bump** (base class) — ≥2 blocks, capped at 4 bits.
2. **Threshold binning** with `ClassificationConfig` defaults **`auto_merge = 0.95`, `review_floor = 0.40`**:
   - `score < 0.40` → `no_match`
   - `0.40 ≤ score < 0.95` → `human_review`
   - `score ≥ 0.95` → `auto_merge`
3. **Corroboration gate** — demotes uncorroborated `auto_merge` pairs to `human_review`.

Test metrics use un-bumped scores (the test pairs carry no blocking provenance), mirroring enhanced_3.

## Outputs

| Artifact | Path | Contract |
|---|---|---|
| Cross-model eval schema | `models/outputs/fs_splink_enhanced_4__<v>[_full_pool].parquet` | 5-col `PATID_A \| PATID_B \| model_name \| score \| predicted_tier` |
| Probabilistic matches | `data/matches_model_v2/fs_splink_enhanced_4_matches_model__<v>[_full_pool].parquet` | `ProbabilisticMatches` (validated; `veto_reason` omitted; `corroboration` column is internal, dropped before projection) |
| Diagnostics (non-PHI) | `models/artifacts/fs_splink_enhanced_4/diagnostics__<v>[_full_pool].json` | tier counts, score quantiles, split sizes, trained m/u settings, `grounded_u` (SSN/Email exact u values used), `gate_demotions_full_pool`, `untrained_u_after_training`, **test metrics** (precision/recall/F1 + confusion) |

## Test-split evaluation results

_TBD — pending first VM run (`run_real_enhanced_4.py --score-full-candidate-pool`)_

The key metric to watch is whether auto_merge precision rises materially above enhanced_3's ~78% at comparable recall, and how many true matches the corroboration gate demotes (the recall cost). The diagnostics JSON records `gate_demotions_full_pool` for production scoring and the test-split confusion matrix for held-out evaluation; both reflect the full system including the gate.

## Results & insights

_TBD — pending first VM run (`run_real_enhanced_4.py --score-full-candidate-pool`)_

> Sub-note: the key metric to watch is whether auto_merge precision rises materially above enhanced_3's ~78% at comparable recall, and how many true matches the gate demotes (the recall cost). If precision rises but recall drops substantially, the next iteration should consider relaxing the gate (e.g. allowing a single strong household signal rather than two) or adding more near-match levels (P5 in enhanced_3's roadmap) to repopulate the gate's keep-branch for genuine duplicates that lack SSN/email.

## How to run

```bash
# Sandbox smoke test (synthetic data, off-VM OK).
python -m models.experiments.fs_splink_enhanced_4.run_synthetic_enhanced_4

# VM: train on silver labels + score the full candidate pool + write metrics.
python -m models.experiments.fs_splink_enhanced_4.run_real_enhanced_4 \
    --score-full-candidate-pool

# VM: build the evaluation notebook and run it.
python scripts/_build_enhanced_4_evaluation_nb.py
# then open and run notebooks/fellegi_sunter/fellegi_sunter_4_evaluation.ipynb on the VM
```

The real runner accepts the same CLI flags as `run_real_enhanced_3.py`: `--cleaned-index`, `--candidate-pairs`, `--silver-labels`, `--label-col` (default `silver_label`), `--test-size` (default 0.2), `--split-seed` (default 42), `--data-version`, `--u-max-pairs`.

## Tests

- `tests/unit/test_fs_enhanced_4_real_runner.py` — PHI-free runner helpers: split, canonicalize, metrics, label validation, u-grounding, gate-demotion diagnostics.
- `tests/unit/test_fs_enhanced_4_gate.py` — `apply_corroboration_gate`: person-unique keep, dual-household keep, single-household demotion, no-column degraded-signal, empty-frame edge case.
- `tests/unit/test_fs_enhanced_4_u_grounding.py` — `_ground_untrained_u`: cohort-derived 1/n_distinct path, fallback when column absent, `fix_u_probability=True` enforcement, `audit_untrained_u` reporting.

30 tests total. The full train→score→gate round-trip is exercised by `run_synthetic_enhanced_4.py` (not a pytest test — it trains a real Splink model).

## Known limitations

- **Floor-driven conflict penalties (m supervised from positives).** The silver labels are high-precision deterministic confirmations. True matches that passed Stage 3's exact-SSN and name+DOB rules almost never exhibit SSN/name conflict — so m for the conflict levels is near-zero by construction, not by evidence. The conflict Bayes factors are large but statistically fragile. The corroboration gate is the real precision lever; treat conflict weights as reinforcement, not primary signal.
- **The gate trades auto_merge recall for precision.** True matches with only DOB + name agreement and no corroborating identifier (no shared SSN, email, phone, or address) are demoted to `human_review` rather than auto-merged. For a patient record that genuinely changed address, phone, and email since a prior registration, the gate correctly routes to clerical review — but recall at auto_merge is lower than enhanced_3's 95%.
- **Residual twin/cohabitant hole.** A twin pair that additionally shares an email or SSN (vanishingly rare in practice) would satisfy the person-unique branch and stay `auto_merge`. Equally, two household members who share both phone and address *and* happen to have the same birthday and similar name would satisfy the dual-household branch. The `Household_discount` comparison fires on the shared-household / name-disagree pattern inside the FS score, but it does not block the gate. These edge cases are accepted residuals; a future iteration could add a DOB+household veto.
- **SSN and Email exact u are 1/n_distinct approximations.** This is better than Splink's default but is a collision-probability estimate, not a learned value. If the real cohort has a population of records that share SSNs (e.g. data-entry placeholders like 999-99-9999), 1/n_distinct underestimates u. Cleaning removes known placeholder SSNs; any residual duplicates would bias u slightly high (making the SSN exact Bayes factor slightly smaller than it should be).
- **`u` untrained for other rare levels after grounding.** `audit_untrained_u()` reports any non-null levels that still have no u after training. New comparison levels added in future iterations should check this output after the first VM run.
- **Test metrics use un-bumped scores** (see Classification). The n_blocks bump affects production scoring but not the test-split metrics.

## See also

- `docs/Fellegi-Sunter-Enhanced_3.md` — the predecessor model and the P1–P5 proposals this model realizes.
- `docs/Data-Contract.md` — `ProbabilisticMatches` schema (`veto_reason` optional; `corroboration` is internal and dropped before projection).
- `docs/Deterministic-Rules-Guide.md` — the `RULES` that `PRIOR_RULES` mirror for lambda seeding.
- `models/common/fs_base.py` — the shared OO base (`FSModel`, `SupervisedTraining`, `ClassificationConfig`).
