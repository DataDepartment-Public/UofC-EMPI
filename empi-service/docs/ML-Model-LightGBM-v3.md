# The ML Model — LightGBM v3 (match vs. ambiguous)

This doc explains **which** machine-learning model the eMPI pipeline serves at
Stage 4.5, **what** it decides, **how** it was built, and **how** it is wired
into the pipeline. It is the model-specific companion to
`docs/ML-Matcher-Integration-Guide.md` (the generic pluggable-matcher contract)
and `docs/Nonmatch-Gate-Guide.md` (the gate that feeds it).

---

## 1. What the model is

A **LightGBM gradient-boosted binary classifier** that, given a *plausible*
patient-record pair, predicts whether that pair is a **confident match** or an
**ambiguous / hard case** that a human should review.

- Source notebook: `notebooks/ml_model/pair_classifier_lightgbm_ambiguous_v3.ipynb`.
- Native target: class **1 = ambiguous**, class **0 = confident match**
  (positive = "route to a human reviewer").
- Held-out performance (notebook §8, full test set): **ROC AUC ≈ 0.9955**,
  **PR AUC ≈ 0.9936**. At the operating point (below), the ambiguous class runs
  ~0.91 precision / ~0.99 recall.

It is deliberately a **2-class** model, not 3-class: it never has to recognise a
"different patient" (true non-match). That job is done upstream — see §3.

---

## 2. The population it was trained on, and why

The model was trained on **gold-labelled** pairs
(`data/gold_labels/…`, VM-only PHI) filtered to the **plausible pool**: pairs
whose gold label is either a match *or* flagged ambiguous. Confident non-matches
(≈142k of the ≈205k labelled pairs) were **dropped** before training (notebook
§3). So the model only ever learned the boundary *within* plausible pairs:
"confident match" vs "ambiguous".

This is the single most important fact about the model: **it cannot identify true
non-matches**, because it never saw any. Feeding it non-matches would produce
misleadingly confident scores. The pipeline is built so it never has to (§3).

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
        │  score = P(confident match) = 1 − P(ambiguous)
        │  score ≥ 0.70 → auto_merge (confident match)
        │  score <  0.70 → human_review (ambiguous)
        ▼
6. clustering  (ML auto_merge edges union in — ml_feeds_clustering=True by default)
```

Two design decisions make this coherent:

1. **A dedicated model is the non-match gate.** Because the ML model can't
   judge true non-matches, Stage 4.25 (`src/models/nonmatch_gate/`, see
   `docs/Nonmatch-Gate-Guide.md`) discards them first — it drops everything
   scoring below `P(plausible) = 0.30` and passes only the plausible survivors
   to the ML model, which therefore sees roughly the population it was trained
   on. The gate shares this model's `V3FeatureBuilder`, so both stages see
   identical features. (The FS matcher held this role before and remains the
   fallback when no gate model is active; with neither, the ML matcher scores
   the full non-matches pool with a warning.)

2. **The ML model runs as a 2-tier classifier.** With non-matches gated out
   upstream, the ML matcher never needs a `no_match` tier. It is configured with
   `ml_review_floor = 0.0`, so every scored pair lands in exactly one of two
   tiers: `auto_merge` (confident match) or `human_review` (ambiguous). The
   effective three pipeline tiers are produced across stages:
   **rules-reject + gate-discard = `no_match`; ML = `auto_merge` / `human_review`.**

**Score direction.** The notebook model's class 1 is *ambiguous*, but the
pipeline maps a **high** score to `auto_merge`. So at serve time the score is
inverted to `P(confident match) = 1 − P(ambiguous)` (§4, the adapter). A high
score = confident match = auto_merge; a low score = ambiguous = human_review.

**Operating point.** `auto_merge` at `score ≥ 0.70` mirrors the notebook exactly:
it flags ambiguous at `P(ambiguous) ≥ 0.30`, i.e. confident match at
`1 − P(ambiguous) ≥ 0.70`.

**Clustering.** `ml_feeds_clustering` is `True` — this model's `auto_merge` tier forms merge edges. The gate makes
the ML `auto_merge` tier more defensible, but it should be validated against true
non-matches before being unioned into clustering.

---

## 4. The 12 features (v3 feature engineering)

Built by `V3FeatureBuilder` (`src/models/ml_matcher/lightgbm_v3.py`), ported
verbatim from notebook §5. All are computed from the two records' cleaned
attributes (no FS features needed). Missing values → `NaN`, handled natively by
LightGBM.

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
'different']` (must match training). `MiddleNM_clean` is not a contract-guaranteed
cleaned column, so `V3FeatureBuilder` tolerates its absence (→ `sim_jw_middle =
NaN`).

LightGBM config (notebook §7): `binary` objective, 1000 estimators (early
stopping ~403), `learning_rate=0.05`, `num_leaves=15`, `class_weight='balanced'`,
`random_state=42`.

---

## 5. How it's wired in (the serve-only path)

The model is **served, not retrained in-pipeline** (`MLMatcher.train()` and the
`train` CLI persist/promote path remain stubs; retraining is done in the
notebook). Two small classes in `src/models/ml_matcher/lightgbm_v3.py` do the
work:

- **`V3FeatureBuilder`** — the BYOF feature builder (§4). Returns `PATID_A`,
  `PATID_B` + the 12 features. The pipeline passes it into the Stage-4.5
  `MLMatcher` construction (`src/pipeline.py`).
- **`MatchProbabilityAdapter`** — the BYOM wrapper. Wraps the fitted
  `LGBMClassifier` and swaps its probability columns so
  `predict_proba(X)[:, 1] = P(confident match)`. The pipeline reads column 1 as
  `match_probability`, so this is what makes "high score → auto_merge" correct.

Loading & registry (`src/models/ml_matcher/registry.py`):

- Artifact: `models/ml/ml_model_<ts>.pkl` — a `joblib` dump of the
  `MatchProbabilityAdapter` (which pickles the fitted LightGBM inside).
- `load_model_artifact()` = `joblib.load`. Discovered via the `ml_model_*.pkl`
  glob; the active one is named by `models/ml/active.json`.
- `ml_model_<ts>.meta.json` sidecar carries provenance + held-out
  `test_metrics.metrics_auto_merge` (precision/recall), read by the deploy gate.

Data available to the feature builder at serve time is the FS-gated pool
(`PATID_A`, `PATID_B`, `source_blocks`, `n_blocks`) joined to the cleaned
records (`FirstNM_clean`, `LastNM_clean`, `MiddleNM_clean`, `BirthDT_clean`,
`SSN_clean`, `Email_clean`, `AddressLine1_clean`, `Phones_set`, …).

---

## 6. Configuration

Set in `empi-service/.env` (see `.env.example`); all overridable via
`EMPI_`-prefixed env vars.

| Setting | Value | Meaning |
|---|---|---|
| `ml_auto_merge_threshold` | **0.70** | `score ≥ this` → `auto_merge` (the notebook operating point). |
| `ml_review_floor` | **0.0** | Removes the `no_match` tier → 2-tier classifier. Also the candidate-parquet cutoff (0.0 keeps every survivor). |
| `ml_feeds_clustering` | `False` | Audit-only; keep off until validated against true non-matches. |
| `gate_threshold` | `0.30` | The **non-match gate** boundary — pairs scoring below `P(plausible) = this` are discarded before the ML model (`EMPI_GATE_THRESHOLD`). |
| `fs_review_floor` | `0.40` | The FS candidate cutoff; the gate boundary **only** on the FS fallback path (`EMPI_FS_REVIEW_FLOOR`). |

---

## 7. (Re)exporting the model from the notebook

The notebook's final **export cell** (§11) produces and promotes the artifact:

1. Wraps the fitted `model` in `MatchProbabilityAdapter` and `joblib.dump`s it to
   `models/ml/ml_model_<ts>.pkl`.
2. Computes held-out `metrics_auto_merge` at `score ≥ 0.70` and writes the
   `.meta.json` sidecar.
3. Calls `registry.promote(...)` to write `active.json`.

Run the notebook top-to-bottom (it needs `model`, `proba_test`, `y_test`,
`idx_test`), then run the pipeline — Stage 4.5 will resolve and serve the
promoted model. Confirm with the `[6/7] MODEL(ML) — tiers {auto_merge, human_review}`
log line (no `no_match` key) and `data/ml_output/ml_features_<run>.parquet`.

---

## 8. Outputs

Per run, Stage 4.5 writes to `data/ml_output/`:

- `matches_ml_<run_id>.parquet` — the uniform 5-col `ClassificationResults`
  (`PATID_A, PATID_B, model_name, score, predicted_tier`) audit frame.
- `ml_features_<run_id>.parquet` — the `MLFeatures` candidate parquet: the pair
  keys + `match_probability` + `classification_tier` + the 12 feature columns,
  for every survivor.
- `ml_explanations_<run_id>.parquet` — per-pair SHAP contributions
  (`PairExplanations`), sign-normalized to the **served** score so positive
  pushes toward `auto_merge`. Served by `GET /explanations/ml_matcher/{a}/{b}`;
  see `docs/Explanations-Guide.md`.

---

## 9. Known limitations / next steps

- **No true-non-match class.** The model relies entirely on the Stage-4.25
  gate to remove non-matches. With no gate at all (no gate model *and* no FS
  model), its scores on the raw pool are not trustworthy — hence the fallback
  warning.
- **Training population ≠ serving population exactly.** It trained on
  gold-plausible pairs; at serve time it scores the *gate*-plausible pool.
  These overlap heavily but are not identical — the gate misses ~0.1% of true
  plausible pairs at its operating point.
- **Serve-only.** Reproducible retraining via `MLMatcher.train()` + the `train`
  CLI is a documented extension point (see the integration guide), not yet
  implemented.
- To make the model safe for auto-merge into clustering, evaluate its
  `auto_merge` tier against true non-matches, then consider
  `ml_feeds_clustering=True`.
