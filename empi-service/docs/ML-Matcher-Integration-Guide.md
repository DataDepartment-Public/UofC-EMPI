# ML Matcher — Integration Guide

> **Status:** Scaffold built and tested; your two extension points are still
> stubs. Everything under `src/models/ml_matcher/` exists today — the
> `MLMatcher` class, the model registry (active-pointer + deploy-gate,
> mirroring the FS matcher's), the `train` CLI, pipeline wiring as Stage 4.5
> (with schema validation), and a full unit test suite
> (`tests/unit/models/ml_matcher/`). Run `pytest tests/unit/models/ml_matcher/
> -v` first to confirm the scaffold works in your environment.
>
> What's actually left, concretely — everything else in this doc is context
> for these four:
> 1. A `FeatureBuilder` implementation (§3).
> 2. A trained `MLModel`-compatible estimator (§3).
> 3. `MLMatcher.train()` — currently a stub that raises `NotImplementedError`
>    (`src/models/ml_matcher/matcher.py`).
> 4. `registry.load_model_artifact()` — currently a stub; the one function
>    that deserializes whatever format you save your model in
>    (`src/models/ml_matcher/registry.py`). This is where your answer to
>    §8.1 (serialization format) actually gets wired in.

## 1. Where your model fits

```
[1/7] clean ─► [2/7] block ─► [3/7] deterministic rules
                                        │
                                        ▼
                          [4/7] Fellegi-Sunter matcher (audit-only)
                                        │
                                        ▼
                          [5/7] non-match gate — drops confident non-matches
                                        │
                                        ▼
                          ┌─────────────────────┐
                          │   YOUR MODEL HERE    │   [6/7] — between the gate and clustering
                          │  (src/models/ml_matcher/) │
                          └─────────────────────┘
                                        │
                                        ▼
                          [7/7] clustering (terminal)
```

(`[n/7]` matches the tags you'll actually see in pipeline log lines —
`MODEL(FS)` is `[4/7]`, `GATE` is `[5/7]`, `MODEL(ML)` is `[6/7]`.)

The deterministic rules stage already splits every candidate pair into three
buckets: pairs it's confident enough about get **auto-merged** directly into
clustering; everything else falls into a `non_matches` pool. The **non-match
gate** (Stage 4.25, `src/models/nonmatch_gate/` — see
`docs/Nonmatch-Gate-Guide.md`) scores that pool with `P(plausible)` and
discards everything below `settings.gate_threshold` as a confident non-match,
so only the *plausible* survivors reach your model. Your model scores that
subset (optionally enriched with FS's features) and is a scored classifier, not
an automatic merger — whether its `auto_merge`-tier output feeds clustering is a
config toggle (`settings.ml_feeds_clustering`, **on** by default — see §4).

> The FS matcher used to be that gate (`_fs_plausible_pool` in
> `src/pipeline.py`, keeping pairs with `match_probability >=
> fs_review_floor`). It still runs as the **fallback** when no gate model is
> active, or when `EMPI_GATE_SUPERSEDES_FS=false`; otherwise Stage 4 is
> audit-only.

Because the non-matches are gated out upstream, the served LightGBM v5 model
(`docs/ML-Model-LightGBM-v5.md`) runs as a **2-tier** classifier
(`ml_review_floor = 0.0`): confident match → `auto_merge`, ambiguous →
`human_review`; it does not emit `no_match`. A model
that *does* need to distinguish non-matches can still emit all three tiers by
setting a non-zero `ml_review_floor` — the `classify()` machinery supports it.

**Your job is exactly this:** given a pool of candidate patient-record pairs,
produce a calibrated match probability and a tier for each pair.

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
(`data/fs_output/fs_features_<run_id>.parquet`), and a **labeled** version for
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

## 3. Where you plug in — the class hierarchy

Three layers, outer to inner. You only ever touch the innermost one.

```
src.models.base.PairClassifier          (Protocol — the pipeline's contract)
        │  model_name: str
        │  run(candidate_pairs, df_clean, **kwargs) -> ClassificationResults
        │
        ▼
src.models.ml_matcher.matcher.MLMatcher  (ALREADY BUILT — satisfies PairClassifier)
        │  score() / classify() / predict() / run() / to_ml_features() — all
        │  implemented and tested. train() is a stub — see §3.3.
        │  Wraps:
        ├── self.feature_builder: FeatureBuilder   ← YOU implement this
        └── self.model:            MLModel          ← YOU provide this (fitted)
```

`PairClassifier` (`src/models/base.py`) is the shared contract every scoring
stage satisfies — deterministic rules, the FS matcher, and `MLMatcher`. It's a
`typing.Protocol`, not an ABC (structural typing — `MLMatcher` satisfies it by
having the right shape, no explicit inheritance), and its `run()` is what
`src/pipeline.py` actually calls. **You never touch this layer** — `MLMatcher`
already satisfies it, fully implemented, tested
(`tests/unit/models/ml_matcher/test_ml_matcher_matcher.py`).

Your job is the two Protocols `MLMatcher` wraps (`src/models/ml_matcher/base.py`):

```python
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

You pass both into the already-built matcher: `MLMatcher(model=your_model,
feature_builder=YourFeatureBuilder())`. Until you do, it uses
`NotImplementedFeatureBuilder` (raises immediately with a message pointing
back here) and a `None` model (raises on `predict`/`run`) — that's the
scaffold's deliberate default, not a bug if you hit it before wiring yours up.

### 3.1 See it work end to end first

`tests/unit/models/ml_matcher/test_ml_matcher_matcher.py` has a complete toy
`FeatureBuilder`/`MLModel` pair (`_FakeFeatureBuilder`, `_FakeModel`) and
exercises the exact call shape — `run()` in, `ClassificationResults` out.
Reading that file (or having your coding agent read it) is the fastest way to
see the real usage pattern before writing anything. Run it directly:

```bash
pytest tests/unit/models/ml_matcher/ -v
```

### 3.2 What "deliverables" actually means, concretely

1. A `FeatureBuilder` implementation (a class or even just a function with
   that signature).
2. A trained model object satisfying `MLModel` — however you serialize it
   (pickle/joblib/ONNX — your call) is fine, just document the format and
   implement `registry.load_model_artifact()` to match (§8.1).
3. `MLMatcher.train()` filled in — see §3.3.
4. The threshold values you'd recommend for your model (see §4) based on
   your held-out evaluation.

### 3.3 Filling in `MLMatcher.train()`

The method signature already exists (`src/models/ml_matcher/matcher.py`) —
it's a stub that raises `NotImplementedError` until you implement it:

```python
def train(
    self,
    candidate_pairs: pd.DataFrame,
    df_clean: pd.DataFrame,
    labels: pd.DataFrame,
    fs_features: pd.DataFrame | None = None,
) -> None:
    """Fit self.model in place."""
```

A real implementation calls `self.build_features(...)`, derives `X`/`y` from
`labels` (the label file's format — silver-label CSV, whatever — is your
choice), and calls `self.model.fit(X, y)`. `src/models/ml_matcher/train.py`
is the CLI that calls this — it already resolves `candidate_pairs`/`df_clean`
from the latest pipeline run's manifest and handles `--promote`; it stops
today exactly at `model.train(...)` with the `NotImplementedError` above.

## 4. What your scored output looks like (already implemented — nothing to write here)

The pipeline expects a uniform 5-column shape from every classifier stage
(rules, FS, and yours) — `PATID_A, PATID_B, model_name, score, predicted_tier`
— where `predicted_tier ∈ {"auto_merge", "human_review", "no_match"}`. You
don't write this — `MLMatcher.classify()` already does it, with the exact
same inclusive-boundary threshold pattern `FSModel.classify()` uses
(`src/models/fs_matcher/base.py`), tested in
`test_ml_matcher_matcher.py::test_classify_thresholds_inclusive`.

All you provide are the threshold *values* — `ClassificationConfig
(auto_merge_threshold, review_floor)` — based on your held-out evaluation
(precision/recall at each threshold). These are already real, defaulted
settings in `src/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `ml_auto_merge_threshold` | `0.95` | score ≥ this → `auto_merge` tier |
| `ml_review_floor` | `0.40` | score ≥ this → `human_review` tier; also the cutoff for landing in the `MLFeatures` candidate parquet at all |
| `ml_deploy_gate_margin` | `0.02` | a retrained model may only promote if held-out precision/recall are within this margin of the currently-active model's |
| `ml_feeds_clustering` | `False` | when `True`, your `auto_merge`-tier pairs union into clustering edges alongside the deterministic rules' |
| `ml_model_dir` | `models/ml/` | where trained artifacts + `active.json` live |
| `ml_active_model` | `None` | explicit override; when unset, the registry resolves `active.json` (or the newest artifact) |

Report your recommended values for the first two; Jason (or you) update the
defaults once your held-out numbers are in.

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

## 6. What's already built around this (so you don't duplicate it)

None of the following is left to build — it's already in the repo, tested:
- **Model registry / promotion** (`src/models/ml_matcher/registry.py`) — an
  active-model pointer + deploy-gate, mirroring
  `src/models/fs_matcher/registry.py` exactly: `resolve_active_model()`,
  `passes_deploy_gate()`, `promote()`. A retrained model only promotes if its
  held-out `auto_merge`-tier precision/recall don't regress past
  `ml_deploy_gate_margin`. The one piece *not* filled in is
  `load_model_artifact()` — deserializing whatever format you save your
  model in (§3, §8.1).
- **Pipeline wiring** (`src/pipeline.py`, Stage 4.5) — resolves the active
  model, skips cleanly if none is active or `non_matches` is empty (same
  pattern FS uses — the pipeline runs fine with no ML model plugged in), and
  persists your output as a per-run parquet in `data/ml_output/` (the audit
  frame and the candidate feature file both land there).
- **The configurable clustering-feed toggle** (`settings.ml_feeds_clustering`,
  default `False`) — whether your `auto_merge` tier actually unions into
  clustering's edges, or stays audit-only like FS is today.
- **Schema validation** (pandera contracts `MLFeatures` /
  `ClassificationResults` in `src/contracts.py`) on your output shape —
  `MLMatcher.run()`/`to_ml_features()` already produce validated frames.

## 7. Verifying your integration

Three checkpoints, cheapest first:

1. **Scaffold sanity** — `pytest tests/unit/models/ml_matcher/ -v`. Should
   pass with zero changes on your part; confirms your environment can import
   everything before you write real code.
2. **Your code in isolation** — write unit tests for your `FeatureBuilder`
   and trained model the same way `test_ml_matcher_matcher.py` tests the
   scaffold (swap `_FakeFeatureBuilder`/`_FakeModel` for real ones).
3. **End to end in the real pipeline** — once `MLMatcher.train()` and
   `load_model_artifact()` are implemented:
   ```bash
   python -m src.models.ml_matcher.train --promote   # trains, writes ml_model_<ts>.json + .meta.json, promotes
   python -m src.pipeline --input data/raw/MDM_Population.csv
   ```
   Watch for `[6/7] MODEL(ML) — scoring N plausible pairs with ML model ...`
   in the log (not the `skipped` variant) — that confirms the registry
   resolved your promoted model and Stage 4.5 actually ran. Check
   `data/ml_output/ml_features_<run_id>.parquet` for your output.

## 8. Open decisions to settle together before you start

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
