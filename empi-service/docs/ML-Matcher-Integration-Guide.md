# ML Matcher — Integration Guide

> **Status:** SPEC, not yet built. Nothing under `src/models/ml_matcher/`
> exists in the codebase today. This document is the contract to build your
> model against — Jason will wire the pipeline side (registry, pipeline
> stage, schema validation) around whatever you produce here. Treat this as
> a handoff spec, not a description of shipped code.

## 1. Where your model fits

```
raw ─► clean ─► block ─► deterministic rules (Stage 3)
                              │
                              ▼
                    Fellegi-Sunter matcher (Stage 4, audit-only)
                              │
                              ▼
                    ┌─────────────────────┐
                    │   YOUR MODEL HERE    │   ← new stage, between FS and clustering
                    │  (src/models/ml_matcher/) │
                    └─────────────────────┘
                              │
                              ▼
                    clustering (Stage 5, terminal)
```

The deterministic rules stage already splits every candidate pair into three
buckets: pairs it's confident enough about get **auto-merged** directly into
clustering; everything else falls into a `non_matches` pool. The FS matcher
(Splink, Stage 4) scores that pool and produces a candidate/feature file, but
its output is **audit-only today** — it does not feed clustering. Your model
scores that same `non_matches` pool (optionally enriched with FS's features)
and is likewise a scored classifier, not an automatic merger — whether its
`auto_merge`-tier output eventually feeds clustering is a config toggle
Jason will add on the pipeline side once your model is validated, not
something your code needs to worry about.

**Your job is exactly this:** given a pool of candidate patient-record pairs,
produce a calibrated match probability and a three-way tier for each pair.

## 2. What already exists today that you can use as input

### `candidate_pairs` — the pair pool

Every pair you'll score. Real columns (see `src/contracts.py`,
`CandidatePairs`):

| Column | Type | Notes |
|---|---|---|
| `PATID_A` | str | canonical smaller PATID |
| `PATID_B` | str | canonical larger PATID |
| `source_blocks` | str | which blocking strategy(s) surfaced this pair |
| `n_blocks` | int | how many blocking strategies agreed |

### `df_clean` — the cleaned record attributes, keyed by `PATID`

Join this onto both sides of a pair (see how `deterministic_rules.py` and
`fs_matcher/matcher.py` both do this — `_materialize_pairs` /
`prepare_model_input` are worth reading as reference). Relevant columns:
`FirstNM_clean`, `LastNM_clean`, `BirthDT_clean`, `SSN_clean`, `last_4_SSN`,
`Email_clean`, `ZipCD_clean_base`, `AddressLine1_clean`,
`SexAtBirthDSC_clean`, `Phones_set` (a set of cleaned phone numbers),
`valid_record`.

### `FSFeatures` — the readiest-made feature set (already produced every run)

The FS matcher already writes this parquet per pipeline run
(`data/FS_output/fs_features_<run_id>.parquet`), and a **labeled** version for
training via `python -m src.models.fs_matcher.train` (writes a labeled GBT
training feature file — this is very likely your best starting point for a
first model, since it reuses the same labeled pairs FS itself trained on).
Base columns (`src/contracts.py`, `FSFeatures`):

| Column | Type | Notes |
|---|---|---|
| `PATID_A`, `PATID_B` | str | pair key |
| `match_probability` | float | FS's own Fellegi-Sunter score |
| `match_weight` | float | FS's score in log-odds (bits) |
| `classification_tier` | str | FS's own tier — informational, not a decision |
| `label` | float, optional | 0/1, present only on the training file |

Plus, dynamically, **one `gamma_<field>` and one `bf_<field>` column per
compared field** (name, dob, ssn, email, phone, address, sex — see
`src/models/fs_matcher/comparisons.py` for the exact field list). `gamma_*` is
Splink's discrete agreement-level index for that field (0 = strong
disagreement, higher = closer agreement, exact levels are comparison-specific
— e.g. exact match vs. Jaro-Winkler-close vs. no match); `bf_*` is that
level's Bayes-factor contribution in bits. Together these are effectively
**pre-engineered per-field similarity features** — you likely don't need to
write your own string-distance functions for name/address/etc. from scratch;
a gradient-boosted tree over `gamma_*`/`bf_*` alone is a very reasonable
first model, and Splink's own docs explain how those levels are derived if
you want the details.

You are **not required** to use `FSFeatures` at all — you can build features
directly off `candidate_pairs` + `df_clean` (e.g. your own name-similarity
metric, a phone/email network feature, whatever you want). Both paths are
first-class; see the interface below.

## 3. The interface your code needs to satisfy

Two small pieces, both plain Python — no framework lock-in:

```python
from typing import Protocol
import pandas as pd

class FeatureBuilder(Protocol):
    def build_features(
        self,
        candidate_pairs: pd.DataFrame,
        df_clean: pd.DataFrame,
        fs_features: pd.DataFrame | None = None,   # None if you don't want it
    ) -> pd.DataFrame:
        """Return a frame keyed by PATID_A/PATID_B with your feature columns."""
        ...

class MLModel(Protocol):        # plain scikit-learn duck typing
    def fit(self, X, y) -> "MLModel": ...
    def predict_proba(self, X): ...   # shape (n, 2), column 1 = P(match)
```

`MLModel` is intentionally just scikit-learn's estimator shape. A
`GradientBoostingClassifier`, `XGBClassifier`, `LGBMClassifier`, or
`CatBoostClassifier` already satisfies this with **zero adapter code** — just
hand the fitted estimator over. If you want to use PyTorch/TensorFlow/a
neural net, write a ~10-line wrapper class exposing `fit`/`predict_proba` in
front of it; the pipeline only ever calls those two methods.

Deliverables from you, concretely:
1. A `FeatureBuilder` implementation (a class or even just a function with
   that signature).
2. A trained model object satisfying `MLModel` — however you serialize it
   (pickle/joblib/ONNX — your call, flag your preference so the loader on
   the pipeline side matches) is fine, just document the format.
3. The threshold values you'd recommend for your model (see §4) based on
   your held-out evaluation.

## 4. What your scored output needs to look like

The pipeline expects a uniform 5-column shape from every classifier stage
(rules, FS, and yours) — `PATID_A, PATID_B, model_name, score, predicted_tier`
— where `predicted_tier ∈ {"auto_merge", "human_review", "no_match"}`. Use
the exact same threshold pattern FS already uses
(`src/models/fs_matcher/base.py`, `FSModel.classify()`):

```python
def classify(scored: pd.DataFrame, auto_merge_threshold: float, review_floor: float) -> pd.DataFrame:
    p = scored["match_probability"]
    tier = pd.Series("no_match", index=scored.index)
    tier = tier.mask(p >= review_floor, "human_review")
    tier = tier.mask(p >= auto_merge_threshold, "auto_merge")
    return scored.assign(classification_tier=tier)
```

Report your recommended `auto_merge_threshold` / `review_floor` from your
held-out evaluation (precision/recall at each threshold) — these become
config values on the pipeline side (mirrors `fs_auto_merge_threshold` /
`fs_review_floor` in `src/config.py` today), not something baked into your
model code.

## 5. Training data / labels

Don't start from zero — this project already has labeled pairs:
- The FS matcher trains on **silver labels** (`SupervisedTraining` in
  `src/models/fs_matcher/base.py`) — ask Jason for the current silver-label
  file location and the labeled `FSFeatures` training parquet
  (`fs-train`'s output), which is likely your fastest path to a first model.
- `scripts/eval_against_labels.py` and the eval harness under
  `src/evaluation/` show the project's existing precision/recall @ methodology
  — match that convention (report metrics the same way FS's `.meta.json`
  sidecars do) so your model's numbers are directly comparable to FS's and
  to the deterministic rules'.
- Gold labels are a work in progress (per project notes) — silver +
  synthetic is what's available today; treat silver-label precision numbers
  as a proxy, not ground truth, same caveat that applies to the rules and FS
  evaluations already in this repo.

## 6. What Jason is building around this (so you don't duplicate it)

You don't need to build any of the following — it's pipeline-side scaffolding:
- **Model registry / promotion** — an active-model pointer + deploy-gate
  pattern mirroring `src/models/fs_matcher/registry.py` (a retrained model
  only promotes if its held-out metrics don't regress past a margin).
- **Pipeline wiring** — a new stage between FS and clustering that resolves
  the active model, skips cleanly if none is active (same pattern FS uses
  today — the pipeline runs fine with no ML model plugged in), and persists
  your output as a per-run parquet.
- **The configurable clustering-feed toggle** — whether your `auto_merge`
  tier actually unions into clustering's edges (vs. staying audit-only, like
  FS is today) is a settings flag, off by default until your model is
  validated enough to trust for auto-merge.
- **Schema validation** (pandera contracts) on your output shape.

## 7. Open decisions to settle together before you start

1. **Model serialization format** — pickle, joblib, or ONNX? Whichever you
   pick determines how the pipeline-side loader is written.
2. **Feature source** — do you want to build primarily off FS's `gamma_*`/
   `bf_*` columns (fast path, reuses Splink's per-field similarity work), or
   do you have features in mind that need direct access to `df_clean` (e.g.
   something involving `Phones_set` network structure, or a text-embedding
   similarity on names/addresses not captured by Splink's comparisons)?
   Either is fine — just confirms which of `candidate_pairs`/`df_clean`/
   `fs_features` your `FeatureBuilder` actually needs.
3. **Held-out evaluation protocol** — same temporal/random split convention
   as the rest of this project's eval work, so your numbers are apples-to-
   apples with the FS matcher's `.meta.json` metrics.
