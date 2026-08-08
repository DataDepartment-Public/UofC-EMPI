# data/ml_output/ — Stage 4.5 output: ML matcher audit frame + candidates `[SCAFFOLD]`

Full contract: `docs/Data-Contract.md` → "Stage 4.5 — ML matcher" (§4.5a, §4.5b).

Merged folder — holds both artifacts the ML matcher produces per run.
Previously split across two directories (`data/matches_ml/` for the audit
frame, `data/ML_output/` for the candidate deliverable); consolidated since
both are the ML matcher's output as a whole and always written together.
Empty until a model is trained; see `docs/ML-Matcher-Integration-Guide.md`.

- **Producer:** `src/models/ml_matcher/` (`MLMatcher.score` for serving,
  `python -m src.models.ml_matcher.train` — a CLI skeleton whose training
  call is a deliberate stub). Invoked from `src/pipeline.py` when an active
  model is resolvable and the `non_matches` pool is non-empty; otherwise
  Stage 4.5 is skipped with a log line — structurally identical gating to
  Stage 4.
- **Role:** structurally identical to Stage 4 — implements
  `src.models.base.PairClassifier`, scores the same `non_matches` pool, and
  emits the same shape of audit + candidate artifacts. Differs in being
  bring-your-own-model (`src.models.ml_matcher.base.MLModel`) and
  bring-your-own-features (`FeatureBuilder`, optionally enriched with Stage
  4's `FSFeatures`).
- **Trained model artifacts** live separately in `models/ml/` — not under
  `data/`.
- **Naming note:** not tier-named (same reasoning as `data/fs_output/`) —
  contains all three tiers mixed; the tier lives in each row's
  `predicted_tier`/`classification_tier` column.

## 4.5a — matches_ml (full audit frame, ClassificationResults-shaped)

- **Contract:** `contracts.ClassificationResults` (the same shared 5-column
  shape every classifier stage emits: `PATID_A`, `PATID_B`, `model_name`,
  `score`, `predicted_tier`)
- **File:** `matches_ml_<run_id>.parquet`
- **Grain:** every scored `non_matches` pair, all tiers
- **Status:** audit frame; feeds Stage 5's optional edge union when
  `settings.ml_feeds_clustering` is on

## 4.5b — MLFeatures (the candidate deliverable)

- **Contract:** `contracts.MLFeatures` (`strict=False` — BYOF means feature
  column names/count are the implementer's choice; only `PATID_A`/`PATID_B`
  are validated by name, via `validate_ml_features`)
- **File:** `ml_features_<run_id>.parquet`
- **Grain:** every scored pair, no floor — a candidate-inclusion floor used
  to filter this file, but it's gone: keeping every pair is what makes the
  parquet a complete record of the stage's scores, which an offline
  threshold sweep needs (see `MLMatcher.to_ml_features`'s docstring)

## 4.5c — ml_explanations (per-pair SHAP)

- **Contract:** `contracts.PairExplanations` (`validate_pair_explanations`)
- **File:** `ml_explanations_<run_id>.parquet`
- **Grain:** every pair the ML matcher scored
- **Note:** contributions are sign-normalized to the **served** score
  (`P(confident match)`), not the underlying model's `P(ambiguous)` — positive
  always pushes toward `auto_merge`.
- **Consumer:** `GET /explanations/ml_matcher/{a}/{b}`. See
  `docs/Explanations-Guide.md`.

Full detail: `docs/Data-Contract.md`.
