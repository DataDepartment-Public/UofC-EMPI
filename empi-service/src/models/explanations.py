"""Per-pair SHAP explanations, shared by the non-match gate (Stage 4.25) and
the ML matcher (Stage 4.5).

WHY THIS IS COMPUTED AT SCORE TIME, NOT ON DEMAND
--------------------------------------------------
An explanation has to explain **the decision that was recorded**, not the
decision the currently-active model would make now. Recomputing on demand
would resolve "the active model" and rebuild features from "the current
cleaned mirror" — both of which drift as models are promoted and data is
republished, so a reviewer could see `score=0.04, dropped` beside a chart
derived from a model that never scored the pair. That is the same lineage
mismatch `RunManifest` exists to prevent, so contributions are produced by
the same call that produces the score and persisted next to it.

WHAT THE NUMBERS ARE
--------------------
Exact TreeSHAP, via LightGBM's own `pred_contrib=True`. Verified numerically
identical (atol 1e-8) to `shap.TreeExplainer` — which for LightGBM models
simply delegates to the same C++ implementation. The `shap` package is
therefore NOT a dependency here: it would pull numba/llvmlite into the
pipeline and API image for identical output. Use `shap` in the notebooks,
where its global plots (beeswarm, dependence) are the actual value.

Contributions are in **log-odds**: they sum to the model's raw margin, not to
`predict_proba`. `build_payload` carries both — raw contributions for the bar
geometry, and a `cumulative_prob` per step for a probability-labelled axis.

SIGN CONVENTION
---------------
Normalized so that **positive always pushes toward the model's positive
decision** — plausible for the gate, confident-match for the ML matcher. This
module never adjusts signs itself; the model wrapper is responsible for handing
them over already normalized, so no caller has to know how a given model was
trained.

Both models served today train their class 1 *as* the served positive decision
— the gate's is `plausible`, the ML matcher's (LightGBM v5,
`lightgbm_v5.DirectMatchAdapter`) is `confident match` — so contributions pass
through untouched. A future model whose positive class is the *opposite* of the
served decision must negate its contributions **and** its base value inside its
own wrapper. Skipping that yields a waterfall that reads plausibly and is
exactly wrong; it was the sharpest edge in the previous (class 1 = *ambiguous*)
generation of this model, and removing the inversion is why v5 exists.

PHI / HIPAA: contributions are derived from similarity scores, never from
field values, and nothing here is logged.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Column prefixes in the persisted explanation frame.
SHAP_PREFIX = "shap_"
FEAT_PREFIX = "feat_"
BASE_VALUE_COL = "base_value"

#: Reviewer-facing feature names for the v3 feature set both served models
#: share. Owned here so they label identically and the dashboard never
#: hardcodes a feature name; an unlisted feature falls back to `feature_label`'s
#: readable form rather than failing.
FEATURE_LABELS: dict[str, str] = {
    "sim_jw_first": "First name similarity",
    "sim_jw_last": "Last name similarity",
    "sim_jw_middle": "Middle name similarity",
    "sound_first": "First name sounds alike",
    "sound_last": "Last name sounds alike",
    "sim_dob": "Date of birth similarity",
    "ssn_digit_frac": "SSN digits matching",
    "sim_lev_email": "Email similarity",
    "sim_lev_address1": "Street address similarity",
    "addr_token_jaccard": "Address word overlap",
    "cmp_street_num": "Street number",
    "sim_phones": "Phone number similarity",
}

#: Features whose stored value is a category label rather than a number.
CATEGORICAL_VALUE_FEATURES = frozenset({"sound_first", "sound_last", "cmp_street_num"})


def feature_label(name: str) -> str:
    """Display label for a feature, falling back to a readable form of the
    raw column name so an unlabelled new feature still renders sanely."""
    return FEATURE_LABELS.get(name, name.replace("_", " ").capitalize())


# ── Computing contributions ──────────────────────────────────────────────────
def supports_contributions(model: Any) -> bool:
    """Whether `model` can produce SHAP contributions.

    True for anything exposing an explicit `contributions(X)` (the ML
    matcher's adapter) or a LightGBM-shaped `predict_proba(X,
    pred_contrib=True)`. BYOM means a plugged-in model may support neither —
    the stage then skips explanations rather than failing the run.
    """
    if hasattr(model, "contributions"):
        return True
    return hasattr(model, "booster_") or hasattr(model, "predict_proba")


def compute_contributions(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    """Exact TreeSHAP contributions for `X`, or None if `model` can't produce
    them.

    Returns an `(n_rows, n_features + 1)` array whose last column is the base
    value (the model's expected raw output), matching LightGBM's
    `pred_contrib` layout. Signs are already normalized by the model wrapper.
    """
    if model is None:
        return None
    if hasattr(model, "contributions"):
        contrib = model.contributions(X)
        return None if contrib is None else np.asarray(contrib)
    try:
        contrib = model.predict_proba(X, pred_contrib=True)
    except TypeError:
        # Not a LightGBM-shaped estimator — BYOM without contribution support.
        return None
    contrib = np.asarray(contrib)
    if contrib.ndim != 2 or contrib.shape[1] != X.shape[1] + 1:
        # Multiclass LightGBM returns (n, (n_features+1) * n_classes); the
        # pipeline's models are binary, so anything else is unexpected.
        logger.warning(
            "Unexpected pred_contrib shape %s for %d features; skipping "
            "explanations", contrib.shape, X.shape[1],
        )
        return None
    return contrib


# ── The persisted frame ──────────────────────────────────────────────────────
def build_explanation_frame(
    pairs: pd.DataFrame,
    features: pd.DataFrame,
    contributions: np.ndarray,
    *,
    model_name: str,
    scores: np.ndarray | pd.Series,
    tiers: np.ndarray | pd.Series,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble the per-pair explanation artifact.

    Deliberately **self-contained** — it carries the score, the tier, the base
    value, every contribution AND every feature value, so the endpoint serves
    a complete payload from one file and never rebuilds features (which would
    reintroduce the drift this whole design avoids).

    Sorted by pair key so Parquet row-group statistics make the endpoint's
    single-pair filtered read a cheap predicate pushdown rather than a scan.
    """
    # Model-agnostic by construction: the roster is whatever the caller
    # scored, never a hardcoded feature list. Keeps this module usable by any
    # future model without importing its feature builder (and without the
    # import cycle that would create).
    cols = feature_cols if feature_cols is not None else list(features.columns)
    out = pd.DataFrame(
        {
            "PATID_A": pairs["PATID_A"].to_numpy(),
            "PATID_B": pairs["PATID_B"].to_numpy(),
            "model_name": model_name,
            "score": np.asarray(scores, dtype="float64"),
            "predicted_tier": np.asarray(tiers, dtype=object).astype(str),
            BASE_VALUE_COL: contributions[:, -1].astype("float64"),
        }
    )
    for i, col in enumerate(cols):
        out[f"{SHAP_PREFIX}{col}"] = contributions[:, i].astype("float32")
    for col in cols:
        values = features[col]
        # Categoricals are stored as their label; numerics stay float (NaN is
        # meaningful — it is what the model actually branched on).
        out[f"{FEAT_PREFIX}{col}"] = (
            values.astype(object).where(values.notna(), None)
            if col in CATEGORICAL_VALUE_FEATURES
            else values.astype("float32")
        )
    return out.sort_values(["PATID_A", "PATID_B"]).reset_index(drop=True)


def explanation_feature_names(frame: pd.DataFrame) -> list[str]:
    """Feature names present in a persisted explanation frame, in column
    order (which is the model's training order)."""
    return [c[len(SHAP_PREFIX):] for c in frame.columns if c.startswith(SHAP_PREFIX)]


# ── The serving payload ──────────────────────────────────────────────────────
def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _display_value(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if name in CATEGORICAL_VALUE_FEATURES:
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return f"{float(value):.3f}"


def build_payload(
    row: pd.Series,
    *,
    threshold: float | None = None,
    run_id: str | None = None,
    model_file: str | None = None,
    top_n: int = 8,
    feature_names: list[str] | None = None,
) -> dict:
    """Turn one persisted explanation row into a plot-ready waterfall payload.

    Every bar arrives with its `start`/`end` already computed, so the UI draws
    a rectangle per row and does no arithmetic — it needs no notion of a base
    value, log-odds, or SHAP at all. Features are ordered by descending
    |contribution|, which is the order a waterfall reads best.
    """
    names = feature_names or explanation_feature_names(row.to_frame().T)
    base = float(row[BASE_VALUE_COL])

    ranked = sorted(
        names, key=lambda n: abs(float(row.get(f"{SHAP_PREFIX}{n}", 0.0) or 0.0)), reverse=True,
    )

    features: list[dict] = []
    cursor = base
    for name in ranked:
        shap_value = float(row.get(f"{SHAP_PREFIX}{name}", 0.0) or 0.0)
        start, end = cursor, cursor + shap_value
        cursor = end
        raw_value = row.get(f"{FEAT_PREFIX}{name}")
        if raw_value is not None and name not in CATEGORICAL_VALUE_FEATURES:
            try:
                raw_value = None if pd.isna(raw_value) else float(raw_value)
            except (TypeError, ValueError):
                raw_value = None
        features.append(
            {
                "name": name,
                "label": feature_label(name),
                "value": raw_value,
                "display_value": _display_value(name, row.get(f"{FEAT_PREFIX}{name}")),
                "shap": shap_value,
                "start": start,
                "end": end,
                "direction": "positive" if shap_value >= 0 else "negative",
                "cumulative_prob": _sigmoid(end),
            }
        )

    margin = cursor
    bounds = [base, margin, *[f["end"] for f in features]]
    pad = max(0.05, 0.05 * (max(bounds) - min(bounds)))
    return {
        "model": str(row["model_name"]),
        "run_id": run_id,
        "model_file": model_file,
        "patid_a": str(row["PATID_A"]),
        "patid_b": str(row["PATID_B"]),
        "decision": {
            "score": float(row["score"]),
            "tier": str(row["predicted_tier"]),
            "threshold": threshold,
        },
        "base_value": base,
        "final_margin": margin,
        "units": "log_odds",
        "top_n": min(top_n, len(features)),
        "axis": {"min": min(bounds) - pad, "max": max(bounds) + pad},
        "features": features,
    }


__all__ = [
    "BASE_VALUE_COL",
    "CATEGORICAL_VALUE_FEATURES",
    "FEATURE_LABELS",
    "FEAT_PREFIX",
    "SHAP_PREFIX",
    "build_explanation_frame",
    "build_payload",
    "compute_contributions",
    "explanation_feature_names",
    "feature_label",
    "supports_contributions",
]
