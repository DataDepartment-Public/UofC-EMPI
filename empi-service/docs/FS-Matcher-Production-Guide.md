# FS Matcher — Production Guide

The **Fellegi-Sunter (FS) matcher** is Stage 4 of the eMPI pipeline. This guide
explains what it does, how to train / deploy / swap it, and the contract of the
feature file it produces for the downstream model. It is written for a reviewer
who needs to operate and maintain the matcher — no Splink internals required.

---

## 1. What it is and where it sits

The pipeline resolves patient records in five stages:

```
raw → 1. clean → 2. blocking → 3. deterministic rules ─┬─► matches (auto-merge)
                                                        └─► non-matches (uncertain)
                                                              │
                                                              ▼
                                            4. FS matcher  (this component)
                                              • scores the non-matches
                                              • surfaces likely-match CANDIDATES
                                              • emits per-pair FEATURES
                                              │
                                              ▼  (feeds the downstream GBT, NOT clustering)
                                       5. clustering → cluster_assignments
                                          (deterministic auto-merge edges only)
```

The deterministic rules (Stage 3) confidently **auto-merge** the easy pairs and
send everything uncertain to the **non-matches** pool. The FS matcher's job is to
comb that no-match pool and **surface additional candidate pairs** — pairs that
look like the same patient — together with a rich set of **features**, for a
downstream **Gradient-Boosted-Tree (GBT)** to make the final call.

**Important:** the FS matcher does **not** merge anything and does **not** affect
clustering. Clustering continues to group only the deterministic auto-merge
edges. The FS matcher is a *candidate + feature generator* that hands off to the
GBT. This keeps the automatic-merge behavior unchanged while giving the GBT a
strong, interpretable signal.

**Model:** 7 two-level comparisons (First name, Last name, Date of birth, SSN,
Email, Phone, Address), each contributing one interpretable weight. The model is
trained from **silver labels** (high-precision deterministic confirmations).
Held-out performance is roughly **precision ≈ 0.78 / recall ≈ 0.95**. The model
structure is deliberately **frozen**; the GBT downstream is the mechanism for
lifting precision further.

---

## 2. MLOps lifecycle: train → promote → serve → swap

The matcher is **trained offline** and **served pre-trained** — the pipeline
never retrains at run time and never needs the PHI labels to score.

```
  ┌─ TRAIN (offline, on the VM, has the PHI silver labels) ──────────────┐
  │  python -m src.models.fs_matcher.train                               │
  │    → models/fs/fs_model_<ts>.json        (the trained model)         │
  │    → models/fs/fs_model_<ts>.meta.json   (metrics + provenance)      │
  │    → data/FS_output/fs_features_train_*.parquet  (GBT training data) │
  └──────────────────────────────────────────────────────────────────────┘
                         │  --promote  (passes the deploy-gate?)
                         ▼
  ┌─ ACTIVATE ───────────────────────────────────────────────────────────┐
  │  models/fs/active.json  → points at the model to serve               │
  └──────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─ SERVE (every pipeline run — no labels, no training) ────────────────┐
  │  python -m src.pipeline                                               │
  │    Stage 4 loads the active model, scores non-matches, writes:       │
  │      data/matches_model/matches_model_<run>.parquet  (audit)         │
  │      data/FS_output/fs_features_<run>.parquet        (GBT candidates)│
  └──────────────────────────────────────────────────────────────────────┘

  SWAP a model:  point EMPI_FS_ACTIVE_MODEL at a different fs_model_*.json,
                 or re-run train with --promote.
```

---

## 3. Runbook

All commands run from the `empi-service/` directory on the VM (conda env
`empi_env`; use `python -m pip …` for installs).

### 3.1 Train a model

```bash
python -m src.models.fs_matcher.train
```

Resolves its `cleaned`/`candidate_pairs` inputs from the **latest `RunManifest`**
in `data/runs/` (guaranteeing same-run lineage and that the orchestrator's
stacked blocker produced them, not the narrower standalone `run_blocking.py`
CLI — see `docs/Data-Contract.md` Stage 2's warning), falling back to
directory-latest resolution only if no manifest exists yet. Also resolves the
silver labels at `data/silver_labels/silver_labels_v1_2026_06_21.csv`. It does
an 80/20 stratified split (train the model / measure held-out precision &
recall), writes the model + a `.meta.json` with the metrics, and writes the
**labeled GBT training feature file**. It does **not** activate the model.

Useful flags: `--run-id ID` (pin a specific run's manifest instead of the
latest), `--silver-labels PATH`, `--cleaned-index PATH` / `--candidate-pairs
PATH` (explicit overrides, skip manifest resolution for that input),
`--test-size 0.2`, `--split-seed 42`, `--data-version TAG`.

### 3.2 Train **and** deploy

```bash
python -m src.models.fs_matcher.train --promote
```

After training, the **deploy-gate** compares the new model's held-out
precision/recall against the currently-active model. If they are within the
allowed margin, `active.json` is repointed to the new model; otherwise promotion
is **refused** (exit code 1) and the active model is left untouched. Override a
refusal only deliberately with `--force-promote`.

### 3.3 Serve (run the pipeline)

```bash
python -m src.pipeline
```

Stage 4 loads the active model and scores the non-matches pool. If **no** model
is active yet, Stage 4 is skipped with a clear log line and the rest of the
pipeline runs normally (see *Bootstrapping* below).

### 3.4 Swap the served model without retraining

```bash
EMPI_FS_ACTIVE_MODEL=/abs/path/to/models/fs/fs_model_<ts>.json python -m src.pipeline
```

Or repoint `active.json` by promoting a different model. The explicit
`EMPI_FS_ACTIVE_MODEL` override wins over `active.json`.

---

## 4. Configuration (cutoffs & paths)

Every setting is overridable via an `EMPI_`-prefixed environment variable (see
`src/config.py`).

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `fs_review_floor` | `EMPI_FS_REVIEW_FLOOR` | `0.40` | **The candidate cutoff.** Pairs scoring at/above this are the candidates written to the GBT feature file. Also the boundary of the `human_review` tier. |
| `fs_auto_merge_threshold` | `EMPI_FS_AUTO_MERGE_THRESHOLD` | `0.95` | Labels the `auto_merge` tier. **Informational only** — the FS matcher merges nothing; this just tags high-confidence pairs. |
| `fs_deploy_gate_margin` | `EMPI_FS_DEPLOY_GATE_MARGIN` | `0.02` | How much held-out precision/recall a retrain may drop before promotion is refused. |
| `fs_active_model` | `EMPI_FS_ACTIVE_MODEL` | `None` | Explicit model file to serve (overrides `active.json`). |
| `fs_model_dir` | `EMPI_FS_MODEL_DIR` | `models/fs` | Model store (trained JSON + meta + `active.json`). |
| `fs_output_dir` | `EMPI_FS_OUTPUT_DIR` | `data/FS_output` | Where the GBT feature files are written. |

> **One knob to widen/narrow the candidate net:** lower `EMPI_FS_REVIEW_FLOOR`
> to surface more (lower-confidence) candidates to the GBT; raise it to surface
> fewer, higher-confidence ones. Because the GBT only ever sees pairs above this
> floor, the floor sets the **maximum recall** the GBT can achieve — set it with
> that trade-off in mind.

---

## 5. The GBT feature file (`FSFeatures`)

Written to `data/FS_output/fs_features_<run_id>.parquet` on each pipeline run
(candidates only) and to `data/FS_output/fs_features_train_<version>.parquet` by
the train CLI (the labeled training set). One row per candidate record pair:

| Column | Type | Description |
|---|---|---|
| `PATID_A`, `PATID_B` | str | The pair (canonical order `PATID_A < PATID_B`). |
| `match_probability` | float | FS match probability in `[0, 1]`. |
| `match_weight` | float | FS total evidence in **bits** (log₂ odds). |
| `classification_tier` | str | `auto_merge` / `human_review` (informational). |
| `gamma_<field>` | int | For each of the 7 comparisons, **which level** matched (e.g. 2 = exact, 0 = null/missing). |
| `bf_<field>` | float | For each comparison, its **Bayes factor** — how much that field shifted the odds. |
| `label` | float | `1`/`0` on the **training** file; absent/null when scoring. |

The `gamma_*`/`bf_*` columns are the key hand-off: they let the GBT learn
**field interactions** the additive FS score cannot (e.g. "SSN matches but DOB
conflicts"). The GBT trains on the labeled rows of the training file and, at
inference, consumes the per-run candidate file.

> The exact `gamma_`/`bf_` column names follow each field's output name — e.g.
> `gamma_SSN_clean`, `bf_SSN_clean`, `gamma_FirstNM_clean`, `bf_Phones`. The
> high-cardinality name/email fields also carry a `bf_tf_adj_<field>`
> term-frequency adjustment column.

### Audit artifact

Alongside the feature file, the pipeline writes
`data/matches_model/matches_model_<run_id>.parquet` (`ProbabilisticMatches`): the
**full** scored non-matches set (including pairs below the candidate floor) with
tiers and scores. It is a read-only audit/review record — nothing downstream
consumes it, and it is **not** unioned into clustering.

---

## 6. Retraining when performance degrades

Entity resolution drifts as the data changes. When precision/recall on new data
degrade:

1. Refresh the **silver labels** (or add newly adjudicated pairs).
2. Retrain: `python -m src.models.fs_matcher.train`. Inspect the new model's
   `.meta.json` (`test_metrics.metrics_auto_merge`).
3. Deploy behind the gate: `… --train … --promote`. The gate blocks a model that
   is materially worse than the active one, so a bad retrain cannot silently ship.
4. If a deploy ever regresses in production, roll back by repointing
   `EMPI_FS_ACTIVE_MODEL` (or `active.json`) at the previous `fs_model_<ts>.json`
   — every trained model is retained in `models/fs/`.

The model's **comparison structure is frozen**; retraining re-estimates the
weights from labels, it does not change the fields or levels. A structural change
is a deliberate, separately-validated model revision.

---

## 7. Bootstrapping (first deployment)

The model store (`models/fs/`) is **not** in version control — models are trained
on the VM where the PHI data lives. A fresh checkout therefore has **no active
model**, and Stage 4 is skipped (the pipeline still runs clean → block → rules →
cluster). To bring the matcher online the first time:

```bash
python -m src.models.fs_matcher.train --promote     # trains + activates the first model
python -m src.pipeline                              # Stage 4 now scores
```

From then on, every pipeline run serves the active model until you promote a new
one.

---

## 8. Where the code lives

```
src/models/fs_matcher/
  matcher.py       FSMatcher — model + serving (load/score) + feature projection
  comparisons.py   the frozen 7 comparisons + the prior rules that seed the model
  base.py          shared FS model machinery (training, classification, projections)
  train.py         the fs-train CLI (train → persist → deploy-gate)
  registry.py      model store: active-model resolution + deploy-gate
  versioning.py    input-file resolution helper
src/pipeline.py    Stage 4 integration (load active model, emit the two artifacts)
src/config.py      the fs_* settings above
src/contracts.py   FSFeatures + ProbabilisticMatches schemas
tests/unit/test_fs_matcher_*.py, test_config_fs.py   the test suite
```
