# Per-Pair SHAP Explanations

Every pair the **non-match gate** (Stage 4.25) and the **ML matcher** (Stage 4.5)
score gets a per-feature SHAP contribution vector, persisted next to the score and
served read-only as a plot-ready waterfall payload.

**Front-end readers: §2 is the contract. You do not need the rest of this document.**

Related: `docs/Nonmatch-Gate-Guide.md`, `docs/ML-Model-LightGBM-v3.md`,
`docs/Data-Contract.md` (the artifact schema), `docs/API-Design.md`.

---

## 1. The endpoint

```
GET /explanations/{model_name}/{patid_a}/{patid_b}[?run_id=<run_id>]
```

| Parameter | Notes |
|---|---|
| `model_name` | `nonmatch_gate` or `ml_matcher`. Anything else → 404. |
| `patid_a` / `patid_b` | Either order — the route canonicalizes to `PATID_A < PATID_B` for you. |
| `run_id` | Optional. **Pass the entity's `run_id`** so the explanation provably matches the score displayed beside it. Omitted, it resolves to the most recent run that produced explanations for that model. |

### 404 is often a normal answer, not an error

Show "no model explanation for this pair", not an error state:

| Case | Why |
|---|---|
| Pair was auto-merged or rejected by the **deterministic rules** | Neither model ever scored it. Its explanation is rule provenance — `match_rule` / `rules_fired`, already on the review candidate. |
| Pair asked for under `ml_matcher` but the **gate dropped it** | It never reached Stage 4.5. Ask `nonmatch_gate` instead — that's where the decision was made. |
| Run has no explanation artifact | Explanations disabled for that run, or the stage was skipped. Scores are still valid. |

**Two explanation kinds exist in this system, and the UI needs both**: SHAP waterfalls
for model decisions, rule provenance for deterministic ones. The rule one is the more
trustworthy of the two — don't let a missing waterfall imply a pair is unexplained.

## 2. The payload

```jsonc
{
  "model": "nonmatch_gate",
  "run_id": "20260728T171925Z",
  "model_file": "nonmatch_gate_20260721T160717Z.pkl",
  "patid_a": "00001",
  "patid_b": "00002",

  "decision": { "score": 0.0474, "tier": "no_match", "threshold": 0.30 },

  "base_value":   -0.500,       // where the waterfall starts
  "final_margin": -3.000,       // where it ends; sigmoid(final_margin) == score
  "units": "log_odds",
  "top_n": 8,                   // suggestion, not a limit — all features are returned
  "axis": { "min": -3.4, "max": -0.1 },

  "features": [                 // ordered by |shap| descending
    {
      "name": "sim_dob",
      "label": "Date of birth similarity",
      "value": 0.25,            // number, string (categoricals), or null (missing)
      "display_value": "0.250", // preformatted; null when the feature is missing
      "shap": -1.5,
      "start": -0.5,            // ← draw a rect from `start`
      "end":   -2.0,            // ← to `end`
      "direction": "negative",
      "cumulative_prob": 0.119
    }
  ]
}
```

### Drawing it

```
for each feature in features:
    draw a rect from `start` to `end`, colored by `direction`
    label it with `label` and `display_value`
```

That's the whole algorithm. Every position is precomputed: `features[0].start ==
base_value`, each bar's `start` equals the previous bar's `end`, and the last bar's
`end` equals `final_margin`. You never compute an offset, and you never need to know
what a base value or a log-odd is.

### Rules for reading it correctly

- **`direction` is already normalized.** Positive always pushes toward the model's
  positive decision — *plausible* for the gate, *confident match* for the matcher.
  Never infer direction from the raw sign of anything else; the matcher's underlying
  model is inverted and the API un-inverts it for you.
- **Bars are log-odds** (`units`). They do **not** sum to the probability. If you want
  a probability axis, use `cumulative_prob` per step — do not add `shap` values as
  probabilities.
- **`value: null`** means the feature was *missing for this pair*, which is different
  from zero and is genuinely what the model branched on. Render it as "not available",
  not as `0`.
- **`threshold`** is the decision boundary in probability space; `decision.score` is
  comparable to it directly.
- Say "**contributed to**", not "caused". SHAP allocates credit; it is not a
  counterfactual. Fixing a DOB typo will not move the score by exactly that bar's width.

### Two known display traps

**Correlated features split credit.** `sim_lev_address1`, `addr_token_jaccard`, and
`cmp_street_num` are three views of the same address; `sim_jw_last` and `sound_last`
are two views of the last name. SHAP divides the signal among them, so any single
address bar understates how much address evidence mattered. Consider grouping them
visually, or at least don't invite the conclusion that "address barely matters".

**With only 12 features, most bars are small.** `top_n` (default 8) is the suggested
cut; collapsing the remainder into one "N other features" bar reads far better than
twelve bars where four are invisible.

## 3. What the numbers are

Exact **TreeSHAP**, computed by LightGBM's own `predict_proba(X, pred_contrib=True)`.

The `shap` package is deliberately **not** a dependency of `empi-service`: for
LightGBM models `shap.TreeExplainer` delegates to that same implementation, and the
two agree to `atol=1e-8` (verified). Importing `shap` would pull `numba`/`llvmlite`
into the pipeline and API image for identical output. Use `shap` in the notebooks,
where its global plots (beeswarm, dependence) are the actual value.

### Why they're precomputed, not computed on demand

An explanation must explain **the decision that was recorded**. Computing on request
would resolve "the currently active model" and rebuild features from "the current
cleaned mirror" — both of which drift as models are promoted and data republished. A
reviewer could then see `score = 0.04, dropped` beside a chart derived from a model
that never scored that pair. That is exactly the lineage mismatch the `RunManifest`
exists to prevent, so contributions are produced by the same call that produces the
score and persisted beside it.

The endpoint therefore loads **no model**, recomputes **nothing**, and reads the
immutable run artifact whose sha256 the manifest records.

## 4. The artifact

Written per run, per model:

| Model | Path |
|---|---|
| Non-match gate | `data/gate_output/gate_explanations_<run_id>.parquet` |
| ML matcher | `data/ml_output/ml_explanations_<run_id>.parquet` |

Contract: `contracts.PairExplanations` (`validate_pair_explanations`). Columns:

| Column | Notes |
|---|---|
| `PATID_A` / `PATID_B` | Canonical pair (`A < B`); the frame is **sorted** by this so the endpoint's single-pair read is a row-group pushdown, not a scan. |
| `model_name` | `nonmatch_gate` / `ml_matcher`. |
| `score`, `predicted_tier` | The recorded decision, so the artifact stands alone. |
| `base_value` | The model's expected raw output (log-odds). |
| `model_file` | Which serving artifact produced these contributions. |
| `shap_<feature>` × N | float32 contribution, sign-normalized. |
| `feat_<feature>` × N | The feature value that was actually scored. |

It is deliberately **self-contained** — carrying the feature values as well as the
contributions is what lets the endpoint serve a complete payload without rebuilding
features, which is the entire point (§3).

**Coverage is every scored pair, including the ones the gate dropped.** Those drops are
unrecoverable and `gate_results` + `gate_explanations` are their only record, so "why
was this pair dropped?" stays answerable. Cost is ~20 µs and ~75 bytes per pair per
model: about 4 s and 15 MB per 200k pairs.

## 5. Configuration

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `explanations_enabled` | `EMPI_EXPLANATIONS_ENABLED` | `true` | Emit the artifacts. Off → the endpoint 404s for that run. |
| `explanation_top_n` | `EMPI_EXPLANATION_TOP_N` | `8` | The `top_n` hint in the payload. |

## 6. Extending

A BYOM model that exposes neither `contributions(X)` nor LightGBM's `pred_contrib`
simply produces no explanations: the stage logs it and the run completes normally. A
missing explanation must never fail a run whose scores are fine.

To make a new model explainable, give it either shape. If its positive class is not
the pipeline's positive decision, negate the contributions **and** the base value in
that wrapper — see `MatchProbabilityAdapter.contributions`, and the regression test
`test_adapter_negates_so_positive_means_confident_match`. Getting this backwards
produces a waterfall that reads perfectly plausibly and is exactly wrong.

## HIPAA

Contributions derive from similarity scores, never from field values, and nothing in
this path is logged. The stored `feat_*` values are similarity metrics (e.g. `0.83`,
`"different"`), not identifiers.
