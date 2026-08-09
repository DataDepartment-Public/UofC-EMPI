# Non-Match Gate — the served confident-non-match filter (Stage 4.25)

The gate is the pipeline's **confident-non-match filter**: it decides which of the
deterministic rules' `non_matches` are plausible enough to reach the ML matcher, and
discards the rest for the remainder of the run.

This job used to belong to the **FS matcher's `no_match` tier**. It now belongs to a
LightGBM classifier trained specifically for it —
`notebooks/ml_model/confident_nonmatch/pair_classifier_lightgbm_nonmatch_gate_v1.ipynb`.
FS still runs (Stage 4) but is **audit-only**: it writes its `ProbabilisticMatches` and
`FSFeatures` artifacts and routes nothing.

Related docs: `docs/ML-Model-LightGBM-v5.md` (the Stage-4.5 model the gate feeds),
`docs/ML-Matcher-Integration-Guide.md` (the Stage-4.5 extension contract),
`docs/FS-Matcher-Production-Guide.md` (the gate this replaced).

---

## 1. What it decides

| | |
|---|---|
| **Population** | the whole candidate pool — every pair the deterministic rules did not confirm or reject |
| **Score** | `predict_proba(X)[:, 1]` = **`P(plausible)`** = `P(match ∪ ambiguous)` |
| **Pass** (`score >= gate_threshold`) | plausible — a real match or a genuine hard case → continues to Stage 4.5, tiered `human_review` in the audit frame |
| **Drop** (`score < gate_threshold`) | confident non-match → discarded |

Unlike the ML matcher's model, the gate's `predict_proba` needs **no adapter** — class 1 is
already the "keep it" class, so the pickle is the raw fitted `LGBMClassifier`.

The gate emits only two of the three shared tiers (`human_review` / `no_match`). It makes
no merge decision — that is Stage 4.5's job — and never feeds clustering.

### The asymmetry that sets the threshold

A dropped pair is **unrecoverable**: nothing downstream reconsiders it. A wrongly-kept pair
merely costs a bit of Stage-4.5 compute and, at worst, lands in a reviewer's queue. So the
operating point favors **plausible recall**, not precision or accuracy.

`gate_threshold = 0.30` is the notebook's §8.3 operating point. On the held-out test set:

| threshold | kept fraction | plausible recall | plausible miss rate | kept precision |
|---|---|---|---|---|
| 0.10 | 0.319 | 0.9994 | 0.0006 | 0.958 |
| **0.30** | **0.317** | **0.9992** | **0.0008** | **0.963** |
| 0.50 | 0.316 | 0.9981 | 0.0019 | 0.966 |
| 0.70 | 0.308 | 0.9882 | 0.0118 | 0.981 |

Held-out ROC AUC 0.9997, PR AUC 0.9994. Note how flat *kept fraction* is across the range —
raising the threshold buys very little pool reduction and costs recall quickly.

## 2. Features

The gate reuses the ML matcher's builder verbatim —
`src.models.ml_matcher.lightgbm_v5.FeatureBuilderV5`, the same 12 features (3 categorical +
9 numeric) documented in `docs/ML-Model-LightGBM-v5.md` §4 — the same class object, not a
copy. That is deliberate: the gate and the matcher see **identical inputs**, so their
decisions are directly comparable and there is one feature implementation to keep correct,
not two.

Only the training target and population differ between the two models.

## 3. Where it sits

```
3. deterministic rules ──► non_matches
                              │
4. FS matcher ────────────────┤  AUDIT-ONLY (scores, writes artifacts, routes nothing)
                              │
4.25 NON-MATCH GATE ──────────┤  drop P(plausible) < gate_threshold
                              ▼
4.5 ML matcher                   confident match (auto_merge) vs ambiguous (human_review)
                              ▼
5. clustering (terminal)         deterministic auto-merge edges only
```

`src/pipeline.py` logs the stage as `[5/7] GATE`.

**Fallbacks**, in order:

1. An active gate model + `gate_supersedes_fs` on (the default) → the gate runs.
2. No active gate model (or `gate_supersedes_fs=false`) **and** FS ran → the **legacy FS
   gate** (`_fs_plausible_pool`) filters the pool, logged as a fallback.
3. Neither → the full `non_matches` pool reaches Stage 4.5 **ungated**, with a `WARNING`.
   Stage 4.5 has no true-non-match class, so its output is unreliable in this state.

An empty `non_matches` pool skips the stage entirely.

A **corrupt or unloadable** gate artifact fails the run loudly rather than falling back —
silently running ungated would pollute every downstream artifact.

## 4. Configuration

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `gate_model_dir` | `EMPI_GATE_MODEL_DIR` | `models/nonmatch_gate` | Artifact store + `active.json`. |
| `gate_active_model` | `EMPI_GATE_ACTIVE_MODEL` | `None` | Explicit artifact override; bypasses `active.json`. |
| `gate_output_dir` | `EMPI_GATE_OUTPUT_DIR` | `data/gate_output` | Per-run audit frame. |
| `gate_threshold` | `EMPI_GATE_THRESHOLD` | `0.30` | **The gate boundary.** At/above → pass. |
| `gate_deploy_gate_margin` | `EMPI_GATE_DEPLOY_GATE_MARGIN` | `0.02` | Promotion guard (see §6). |
| `gate_supersedes_fs` | `EMPI_GATE_SUPERSEDES_FS` | `true` | `false` restores the legacy FS gate. |

## 5. Output

Per run, `data/gate_output/gate_results_<run_id>.parquet` — a `ClassificationResults` frame
(`PATID_A`, `PATID_B`, `model_name="nonmatch_gate"`, `score`, `predicted_tier`) covering
**every** scored pair, dropped ones included. It is referenced from the `RunManifest` as
`gate_results`, and the manifest's `counts` carry `gate_plausible` / `gate_dropped`.

The dropped pairs live only in this frame — they are not written to `data/no_match/`, which
holds the deterministic rules' rejects. Read `gate_results` when you need to know why a pair
vanished between the rules and the matcher.

`score` is also threaded into the resolved-output index at publish time, as
`review_candidate.gate_score` (`src/api/ingest/publish.py`). It is the **only** number a
gate-dropped row carries — no rule fired and Stage 4.5 never scored it — so without it the
Review Queue's Auto-rejected section is unrankable. It is a separate axis from
`ml_match_probability`, not a fallback for it: a pair the gate passed at 70% plausible can
score near zero on the matcher, and reading the second as "probably not the same person" is
exactly the confusion the two-axis band scale in `empi-dashboard/src/lib/pair-signal.ts`
exists to prevent. `GET /review-queue` filters it via `gate_score_min` / `gate_score_max`.
Rows published before this column existed read NULL until the run is re-published.

Alongside it, `data/gate_output/gate_explanations_<run_id>.parquet` carries the per-pair SHAP
contributions behind each of those verdicts — including the drops, which is what keeps "why was
this pair dropped?" answerable. Served by `GET /explanations/nonmatch_gate/{a}/{b}`; see
`docs/Explanations-Guide.md`.

## 6. Lifecycle — train, export, promote

Training happens in the notebook (there is no `train` CLI for the gate). Its §11 cell:

1. `joblib.dump`s the fitted model to `models/nonmatch_gate/nonmatch_gate_<ts>.pkl`,
2. writes the `nonmatch_gate_<ts>.meta.json` sidecar — provenance, `feature_cols`,
   `gate_threshold`, and `test_metrics.gate_at_threshold`,
3. calls `registry.promote(...)`, which repoints `active.json` **only if the deploy gate
   passes**.

**The deploy gate** compares the new sidecar's `plausible_precision` / `plausible_recall`
against the currently-active model's and refuses a promotion that regresses either by more
than `gate_deploy_gate_margin`. Recall is the metric that matters — see §1. Pass
`force=True` to override deliberately; the pointer records `"forced": true`.

Resolution order at serve time: `gate_active_model` override → `active.json` → newest
`nonmatch_gate_*.pkl` by mtime → `None`.

## 7. Comparing the two gates

Because FS still scores the full `non_matches` pool, one run produces both gates' verdicts
on the same pairs — join `data/gate_output/gate_results_<run_id>.parquet` against
`data/fs_output/matches_model_<run_id>.parquet` on `(PATID_A, PATID_B)`. To put FS back in
the gate seat for a comparison run:

```bash
EMPI_GATE_SUPERSEDES_FS=false python -m src.pipeline --input data/raw/MDM_Population.csv
```

If FS is retired later, drop Stage 4 and `_fs_plausible_pool` together; nothing else in the
pipeline depends on FS output.

## 8. Not yet on the gate

`src/api/ingest/incremental.py` (`POST /records/score`) still annotates persisted review
candidates with FS scores (`fs_match_probability` / `fs_classification_tier`). That path
does no gating, so it was left alone in this change — migrating it means changing the
`review_candidate` columns in both index backends and the dashboard that reads them.

## HIPAA

The gate logs aggregate tier counts only — never field values, block keys, or identifiers.
The `.meta.json` sidecars carry aggregate metrics and provenance only. Preserve both when
adding logging here.
