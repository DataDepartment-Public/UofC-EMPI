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
- **Grain:** candidates only — filtered to `match_probability >=
  settings.ml_review_floor` (0.40 default)

Full detail: `docs/Data-Contract.md`.
