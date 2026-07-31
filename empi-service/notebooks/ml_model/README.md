# `ml_model/` notebooks

Research/training notebooks for the pluggable ML matcher (Stage 4.5), grouped by
what the model's target is.

```
ml_model/
├── confident_match/     # output usable as P(confident match) — the served ml_matcher family
│   ├── pair_classifier_lightgbm_v1.ipynb          # base match vs. not-match (simple features)
│   ├── pair_classifier_lightgbm_ambiguous_v1.ipynb
│   ├── pair_classifier_lightgbm_ambiguous_v2.ipynb
│   ├── pair_classifier_lightgbm_ambiguous_v3.ipynb # 12-feature v3 (class 1 = ambiguous)
│   ├── pair_classifier_lightgbm_ambiguous_v4.ipynb # v3 features, non-matches folded into class 1
│   └── pair_classifier_lightgbm_confident_match_v5.ipynb  # ★ THE SERVED MODEL
├── confident_nonmatch/  # non-match gate — output is P(plausible) = P(match ∪ ambiguous)
│   └── pair_classifier_lightgbm_nonmatch_gate_v1.ipynb
└── three_class/         # single 3-way head: confident match / non-match / ambiguous
    └── pair_classifier_lightgbm_3class_v1.ipynb
```

## Which one is in production

Only two: **`confident_match_v5`** (Stage 4.5, the ML matcher) and
**`nonmatch_gate_v1`** (Stage 4.25, the gate). See
`docs/ML-Model-LightGBM-v5.md` and `docs/Nonmatch-Gate-Guide.md`.

The rest are research history. v1–v4 are kept because they are the record of how
v5's target was arrived at — v4 in particular is the baseline v5's metrics are
compared against (same split, same features, flipped label).

> ⚠️ **v1–v4 score in the opposite direction from v5.** Their class 1 is
> *ambiguous*, so `predict_proba[:, 1]` there is `P(ambiguous)`, whereas v5's is
> `P(confident match)`. Their serving code (`lightgbm_v3.py`, its
> `MatchProbabilityAdapter`, and the standalone inference notebooks) has been
> **deleted**, and any artifact they exported no longer loads — `joblib.load`
> raises `ModuleNotFoundError`. That is intentional: the two generations look
> identical at load time and produce in-range scores either way, so a silent
> inversion was the alternative. If you re-run one of these notebooks, treat its
> model as notebook-local and do not promote it.
