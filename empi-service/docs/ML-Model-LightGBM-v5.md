# The ML Model — LightGBM v5 (confident match vs. everything else)

This doc explains **which** machine-learning model the eMPI pipeline serves at
Stage 4.5, **what** it decides, **how** it was built, and **how** it is wired
into the pipeline. It is the model-specific companion to
`docs/ML-Matcher-Integration-Guide.md` (the generic pluggable-matcher contract)
and `docs/Nonmatch-Gate-Guide.md` (the gate that feeds it).

**v5 replaced the earlier class-1-is-*ambiguous* generation of this model
(v3/v4), whose serving code and doc have been deleted.** The training notebooks
for those versions are kept under `notebooks/ml_model/confident_match/` as
research history; their exported artifacts no longer load (§9).

---

## 1. What the model is

A **LightGBM gradient-boosted binary classifier** that, given a candidate
patient-record pair, predicts whether that pair is a **confident match**.

- Source notebook:
  `notebooks/ml_model/confident_match/pair_classifier_lightgbm_confident_match_v5.ipynb`.
- Native target: class **1 = confident match**, class **0 = everything else**
  (ambiguous ∪ confident non-match).
- Trained on the **whole** candidate pool — no population filter.

### Why the label points this way

This is the *only* substantive change from v4, and it is a serving-correctness
change, not a modelling one. v3 and v4 trained class 1 = *ambiguous*, so every
consumer had to invert:

| | v3 / v4 (retired) | **v5** |
|---|---|---|
| model's class `1` | ambiguous (v4: ∪ non-match) | **confident match** |
| served `match_probability` | `1 − predict_proba[:, 1]` | `predict_proba[:, 1]` |
| serve adapter | `MatchProbabilityAdapter` — swapped the probability columns | `DirectMatchAdapter` — pass-through |
| SHAP contributions | **negated**, base value included | passed through unchanged |

The SHAP negation was the sharpest edge in the serving path: get it backwards
and the reviewer sees a waterfall that reads plausibly and is exactly wrong
(see `docs/Explanations-Guide.md` §sign convention). v5 removes the inversion
rather than guarding it.

**The decision boundary is unchanged.** v4's `P(class 1) ≥ 0.30` and v5's
`P(confident match) ≥ 0.70` are the same cut, so `ml_auto_merge_threshold`
stays at **0.70** and the held-out metrics are directly comparable across
versions.

---

## 2. The population it was trained on, and why

v5 trains on the **whole** gold-labelled candidate pool (`data/gold_labels/…`,
VM-only PHI) — matches, ambiguous pairs, and confident non-matches alike, the
latter two folded into class 0. This is inherited from v4 and is a real
improvement over v3, which dropped the ≈142k confident non-matches and
therefore *could not identify a true non-match at all*.

At serve time the model nevertheless scores only the Stage-4.25 gate's
survivors (§3), so its training population is a superset of its serving
population. That asymmetry is deliberate and safe in this direction: seeing the
easy negatives during training calibrates the boundary, while the gate removes
them before serving so the model spends its capacity where it matters. It does
mean the honest precision estimate is the **serving-population** one (notebook
§8.5, `test_metrics.metrics_auto_merge_serving_population` in the sidecar), not
the full-pool one — the full pool is padded with easy negatives the model will
never actually be asked about.

---

## 3. How it's used in the pipeline (Stage 4.5)

```
3. deterministic rules ──► non-matches pool (uncertain pairs)
        │
        ▼
4. FS matcher  ── audit-only: scores + features, routes nothing
        │
        ▼
4.25 non-match GATE ── drop P(plausible) < gate_threshold
        │                 → only the plausible survivors continue
        ▼
4.5 ML matcher (THIS MODEL)
        │  score = P(confident match)      ← native, no inversion
        │  score ≥ 0.70 → auto_merge (confident match)
        │  score <  0.70 → human_review (ambiguous)
        ▼
6. clustering  (ML auto_merge edges union in — ml_feeds_clustering=True)
```

Two design decisions make this coherent:

1. **A dedicated model is the non-match gate.** Stage 4.25
   (`src/models/nonmatch_gate/`, see `docs/Nonmatch-Gate-Guide.md`) discards
   confident non-matches first — everything scoring below
   `P(plausible) = 0.30` — and passes the survivors on. The gate shares this
   model's feature builder (§4), so both stages see identical features. (The FS
   matcher held this role before and remains the fallback when no gate model is
   active; with neither, the ML matcher scores the full non-matches pool with a
   warning. v5 tolerates that degraded path better than v3 did, since it *has*
   seen non-matches — but the gate is still the supported configuration.)

2. **The ML model runs as a 2-tier classifier.** With non-matches gated out
   upstream, the ML matcher never needs a `no_match` tier. It is configured with
   no floor at all, so every scored pair lands in exactly one of two tiers:
   `auto_merge` (confident match) or `human_review` (ambiguous). The
   effective three pipeline tiers are produced across stages:
   **rules-reject + gate-discard = `no_match`; ML = `auto_merge` / `human_review`.**

**Score direction.** High score = confident match = `auto_merge`; low score =
ambiguous = `human_review`. Unlike v3/v4, nothing inverts anywhere along the
path — the number the model emits is the number recorded.

**Clustering.** `ml_feeds_clustering` is `True` — this model's `auto_merge` tier
forms real merge edges. Its precision at the chosen threshold is therefore a
production safety property, and `ml_auto_merge_threshold` is a safety dial, not
a cosmetic one. Notebook §8.3 tabulates precision and wrong-merge counts by
threshold for exactly this decision.

---

## 4. The 12 features (unchanged since v3)

Built by `FeatureBuilderV5` (`src/models/ml_matcher/lightgbm_v5.py`). The
feature set did not change in v4 or v5, and the **non-match gate imports this
same class**, so there is exactly one implementation to keep correct. All
features are computed from the two records' cleaned attributes (no FS features
needed). Missing values → `NaN`, handled natively by LightGBM.

| Feature | Type | Definition |
|---|---|---|
| `sim_jw_first` / `sim_jw_last` / `sim_jw_middle` | num | Jaro-Winkler on the name |
| `sound_first` / `sound_last` | cat | Metaphone code match (sound-alike): same / different / missing |
| `sim_lev_email` | num | normalized Levenshtein on email |
| `sim_lev_address1` | num | normalized Levenshtein on address line 1 |
| `addr_token_jaccard` | num | token-set Jaccard of the address |
| `cmp_street_num` | cat | first numeric token of the address, matched exactly |
| `ssn_digit_frac` | num | fraction of position-wise matching SSN digits |
| `sim_dob` | num | normalized Levenshtein on the `YYYYMMDD` birth-date string |
| `sim_phones` | num | best (max) Jaro-Winkler across all cross pairs of the two phone sets |

The 3 categoricals use the fixed category order `['missing', 'same',
'different']` (must match training). `MiddleNM_clean` is not a
contract-guaranteed cleaned column, so the builder tolerates its absence (→
`sim_jw_middle = NaN`).

LightGBM matches features **positionally**, so column order matters:
`DirectMatchAdapter._align` reorders inputs to the fitted model's
`feature_name_`, and the notebook's export cell asserts that its own
`FEATURE_COLS` equals the module's before dumping the artifact.

LightGBM config (notebook §7, unchanged from v3/v4): `binary` objective, 1000
estimators with early stopping, `learning_rate=0.05`, `num_leaves=15`,
`class_weight='balanced'` (class 1 is the minority here), `random_state=42`.

---

## 5. How it's wired in (the serve-only path)

The model is **served, not retrained in-pipeline** (`MLMatcher.train()` and the
`train` CLI persist/promote path remain stubs; retraining is done in the
notebook). Two things in `src/models/ml_matcher/lightgbm_v5.py` do the work:

- **`FeatureBuilderV5`** — the BYOF feature builder (§4). Returns `PATID_A`,
  `PATID_B` + the 12 features. The pipeline passes it into the Stage-4.5
  `MLMatcher` construction (`src/pipeline.py`), and
  `src/models/nonmatch_gate/gate.py` imports it as its own default.
- **`DirectMatchAdapter`** — the BYOM wrapper. A pass-through: it does *not*
  swap probability columns and does *not* negate contributions, because the
  inner model's class 1 is already the served positive class. It still earns
  its place by realigning input columns to the training order and by being the
  stable pickled type for the artifact.

Loading & registry (`src/models/ml_matcher/registry.py`):

- Artifact: `models/ml/ml_model_confident_match_v5_<ts>.pkl` — a `joblib` dump
  of the `DirectMatchAdapter` (which pickles the fitted LightGBM inside).
- `load_model_artifact()` = `joblib.load`. Discovered via the `ml_model_*.pkl`
  glob — the descriptive middle of the filename is free-form — and the active
  one is named by `models/ml/active.json`.
- The `.meta.json` sidecar carries provenance + held-out
  `test_metrics.metrics_auto_merge` (precision/recall), read by the deploy gate.
  v5 sidecars additionally carry `model_version`, `serve_adapter`, and
  `metrics_auto_merge_serving_population`.

Data available to the feature builder at serve time is the gated pool
(`PATID_A`, `PATID_B`, `source_blocks`, `n_blocks`) joined to the cleaned
records (`FirstNM_clean`, `LastNM_clean`, `MiddleNM_clean`, `BirthDT_clean`,
`SSN_clean`, `Email_clean`, `AddressLine1_clean`, `Phones_set`, …).

---

## 6. Configuration

Set in `empi-service/.env` (see `.env.example`); all overridable via
`EMPI_`-prefixed env vars. **No setting changes when moving from v4 to v5** —
the boundary is identical, only its statement changed.

| Setting | Value | Meaning |
|---|---|---|
| `ml_auto_merge_threshold` | **0.70** | `score ≥ this` → `auto_merge` (the notebook operating point). |
| `ml_feeds_clustering` | `True` | The `auto_merge` tier forms real merge edges. |
| `gate_threshold` | `0.30` | The **non-match gate** boundary — pairs scoring below `P(plausible) = this` are discarded before the ML model (`EMPI_GATE_THRESHOLD`). |
| `fs_review_floor` | `0.40` | The FS candidate cutoff; the gate boundary **only** on the FS fallback path (`EMPI_FS_REVIEW_FLOOR`). |

---

## 7. (Re)exporting the model from the notebook

The notebook's §11 produces and promotes the artifact:

1. **§11.0** archives any pre-v5 `ml_model_*.pkl` (+ sidecars + `active.json`)
   into `models/ml/legacy_pre_v5/`, out of the registry's glob. Do not skip
   this — see §9.
2. **§11.1** wraps the fitted `model` in `DirectMatchAdapter`, `joblib.dump`s it
   to `models/ml/ml_model_<ts>.pkl`, writes the `.meta.json` sidecar (held-out
   `metrics_auto_merge` at `score ≥ 0.70`, plus the serving-population metrics),
   and calls `registry.promote(...)` to write `active.json`.
3. **§11.2** reloads the artifact exactly as the pipeline does and asserts that
   the served score equals the notebook's `proba_test` and that the SHAP
   contributions reconstruct it. This is the check that catches an inverted
   adapter before it reaches a reviewer.

Run the notebook top-to-bottom (it needs `model`, `proba_test`, `y_test`,
`idx_test`), then run the pipeline — Stage 4.5 will resolve and serve the
promoted model. Confirm with the `[6/7] MODEL(ML) — tiers {auto_merge,
human_review}` log line (no `no_match` key) and
`data/ml_output/ml_features_<run>.parquet`.

---

## 8. Outputs

Per run, Stage 4.5 writes to `data/ml_output/`:

- `matches_ml_<run_id>.parquet` — the uniform 5-col `ClassificationResults`
  (`PATID_A, PATID_B, model_name, score, predicted_tier`) audit frame.
- `ml_features_<run_id>.parquet` — the `MLFeatures` candidate parquet: the pair
  keys + `match_probability` + `classification_tier` + the 12 feature columns,
  for every survivor.
- `ml_explanations_<run_id>.parquet` — per-pair SHAP contributions
  (`PairExplanations`), in the served score's frame so positive pushes toward
  `auto_merge`. Served by `GET /explanations/ml_matcher/{a}/{b}`; see
  `docs/Explanations-Guide.md`.

---

## 9. Pre-v5 artifacts no longer load — by design

A pre-v5 `.pkl` pickled a `MatchProbabilityAdapter` from
`src.models.ml_matcher.lightgbm_v3`. **That module has been deleted**, so
`joblib.load` now raises `ModuleNotFoundError` on those artifacts rather than
deserializing them.

That is the desired behavior, not a regression to work around. While both
generations were importable the failure mode was silent: the two adapters
satisfy the same interface, both loaded without complaint, and no score was out
of range either way — so a legacy artifact resolved by
`resolve_active_model`'s newest-by-mtime fallback (which kicks in whenever
`active.json` is missing or malformed) would have been served with v5
semantics: every score inverted, every merge decision backwards. A crash at load
is strictly better than that.

Notebook §11.0 still moves any pre-v5 artifacts to `models/ml/legacy_pre_v5/`,
now purely so the store is tidy and nobody hits a confusing `ModuleNotFoundError`
in the middle of a pipeline run.

---

## 10. Known limitations / next steps

- **Training population ⊃ serving population.** Trained on the whole pool,
  served on gate survivors. Read `metrics_auto_merge_serving_population` (§8.5)
  as the production-precision estimate; the full-pool number is inflated by easy
  negatives.
- **Fit on gold.** The end-to-end evaluation headline is only leakage-free at
  `python scripts/evaluate_all.py --holdout strict`; see
  `docs/End-to-End-Evaluation-Guide.md`.
- **Serve-only.** Reproducible retraining via `MLMatcher.train()` + the `train`
  CLI is a documented extension point (see the integration guide), not yet
  implemented — §11 of the notebook is the promotion path.
- **Random stratified split.** Kept for head-to-head comparability with v3/v4; a
  PATID-grouped split would give a stricter generalization estimate.
