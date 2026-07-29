# `ml_model/` notebooks

Research/training notebooks for the pluggable ML matcher (Stage 4.5), grouped by
what the model's target is.

```
ml_model/
├── confident_match/     # output usable as P(confident match) — the served ml_matcher family
│   ├── pair_classifier_lightgbm_v1.ipynb          # base match vs. not-match (simple features)
│   ├── pair_classifier_lightgbm_ambiguous_v1.ipynb
│   ├── pair_classifier_lightgbm_ambiguous_v2.ipynb
│   ├── pair_classifier_lightgbm_ambiguous_v3.ipynb # 12-feature v3 — the served model
│   └── pair_classifier_lightgbm_ambiguous_v4.ipynb # v3 target, but non-matches folded into class 1
├── confident_nonmatch/  # non-match gate — output is P(plausible) = P(match ∪ ambiguous)
│   └── pair_classifier_lightgbm_nonmatch_gate_v1.ipynb
├── three_class/         # single 3-way head: confident match / non-match / ambiguous
│   └── pair_classifier_lightgbm_3class_v1.ipynb
└── inference/           # run a saved .pkl on a labeled test set (no training)
    ├── inference_confident_match.ipynb     # score a confident-match model
    └── inference_confident_nonmatch.ipynb  # score a non-match-gate model
```

## Inference notebooks

Both load a `.pkl` you choose and evaluate it on a **pairs** test set
(`data/synthetic_data/synthetic_test_v3.csv` by default — `label`: 1 = match,
0 = not-match; there is **no ambiguous column**, so evaluation collapses to
match vs. not-match).

They reuse the production **`V3FeatureBuilder`** (the 12 v3 features), so they
serve the **v3-feature family only**: `ambiguous_v3`, `ambiguous_v4`,
`nonmatch_gate_v1`, and the exported `ml_matcher` artifact. Models trained on a
different feature set (base `pair_classifier_lightgbm_v1`, `ambiguous_v1/v2`,
`three_class`) are out of scope for these notebooks.
