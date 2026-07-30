"""ML non-match gate (pipeline Stage 4.25).

Decides which of the deterministic rules' `non_matches` are **plausible**
(a real match or a genuine hard case) and which are **confident non-matches**
to be discarded before the ML matcher runs — the job the FS matcher used to
do in the pipeline.

The served model is the LightGBM `nonmatch_gate` classifier trained in
`notebooks/ml_model/confident_nonmatch/pair_classifier_lightgbm_nonmatch_gate_v1.ipynb`
over the same 12 features as the Stage-4.5 ML matcher (it reuses
`src.models.ml_matcher.lightgbm_v3.V3FeatureBuilder`), so `predict_proba[:, 1]`
is `P(plausible)` directly — no adapter. See `docs/Nonmatch-Gate-Guide.md`.
"""

from src.models.nonmatch_gate.gate import GateResult, MODEL_NAME, NonMatchGate

__all__ = ["NonMatchGate", "GateResult", "MODEL_NAME"]
