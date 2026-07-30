"""NonMatchGate — the pipeline's confident-non-match gate (Stage 4.25).

Scores every pair in the rules' `non_matches` pool with `P(plausible)` and
splits the pool in two:

* **pass** (`P(plausible) >= threshold`) — a real match or a genuine hard
  case; flows on to the Stage-4.5 ML matcher, which decides confident match
  vs ambiguous.
* **drop** (`P(plausible) < threshold`) — a confident non-match; discarded
  for the rest of the run.

This is the job the FS matcher used to do via its `no_match` tier. The gate
model is trained on the **whole** candidate pool (target `1` = match ∪
ambiguous) rather than FS's probabilistic match score, and reuses the ML
matcher's `V3FeatureBuilder` — the two models see identical features, so the
gate's decision and the matcher's are directly comparable.

The dropped pairs are the costly error (unrecoverable downstream), so the
operating point favors plausible **recall**: `settings.gate_threshold` is
0.30, where the notebook measures 0.999 held-out plausible recall.

Emits the uniform 5-col `ClassificationResults` shape shared with every
classifier stage, using two tiers: `human_review` for pairs that pass (the
gate makes no merge decision — that is the ML matcher's job) and `no_match`
for the pairs it drops. It is not a `PairClassifier` merger and never feeds
clustering.

PHI / HIPAA: logs aggregate counts only, mirroring the FS and ML matchers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.contracts import TIER_HUMAN_REVIEW, TIER_NO_MATCH
from src.models.ml_matcher.base import FeatureBuilder
from src.models.ml_matcher.lightgbm_v3 import V3FeatureBuilder

logger = logging.getLogger(__name__)

MODEL_NAME = "nonmatch_gate"
DEFAULT_THRESHOLD = 0.30


class NonMatchGate:
    """Confident-non-match gate over candidate pairs.

    Parameters
    ----------
    model :
        Fitted classifier whose ``predict_proba(X)[:, 1]`` is
        ``P(plausible)`` — the raw LightGBM classifier the notebook exports
        (no probability-swapping adapter, unlike the ML matcher's).
    feature_builder :
        Defaults to `V3FeatureBuilder` — the same 12 features the gate model
        was trained on.
    threshold :
        Pairs scoring at/above this pass the gate.
    """

    model_name = MODEL_NAME

    def __init__(
        self,
        model: object,
        feature_builder: FeatureBuilder | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.model = model
        self.feature_builder = feature_builder or V3FeatureBuilder()
        self.threshold = float(threshold)

    # ── Scoring ───────────────────────────────────────────────────────────────
    def build_features(
        self, candidate_pairs: pd.DataFrame, df_clean: pd.DataFrame,
    ) -> pd.DataFrame:
        return self.feature_builder.build_features(candidate_pairs, df_clean)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score a feature frame. Returns `PATID_A`/`PATID_B` +
        `plausible_probability`."""
        if self.model is None:
            raise RuntimeError(
                "NonMatchGate.predict: no model attached — load one first."
            )
        X = features.drop(columns=["PATID_A", "PATID_B"], errors="ignore")
        proba = np.asarray(self.model.predict_proba(self._align(X)))
        score = proba[:, 1] if proba.ndim == 2 else proba
        out = features[["PATID_A", "PATID_B"]].copy()
        out["plausible_probability"] = score
        return out

    def classify(self, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """Split scores into pass (`human_review`) / drop (`no_match`)."""
        p = df_predictions["plausible_probability"]
        out = df_predictions.copy()
        out["classification_tier"] = np.where(
            p >= self.threshold, TIER_HUMAN_REVIEW, TIER_NO_MATCH
        )
        counts = {k: int(v) for k, v in out["classification_tier"].value_counts().items()}
        logger.info(
            "NonMatchGate[%s]: threshold=%.2f tiers %s",
            self.model_name, self.threshold, counts,
        )
        return out

    def score(
        self, candidate_pairs: pd.DataFrame, df_clean: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build features, score, and classify in one call. Returns
        `PATID_A`/`PATID_B`/`plausible_probability`/`classification_tier`."""
        return self.classify(self.predict(self.build_features(candidate_pairs, df_clean)))

    # ── Projections ───────────────────────────────────────────────────────────
    def to_evaluation_schema(self, df_classified: pd.DataFrame) -> pd.DataFrame:
        """Project a `score()` output to the shared 5-col
        `ClassificationResults` shape (`src.contracts`)."""
        required = {"PATID_A", "PATID_B", "plausible_probability", "classification_tier"}
        missing = required - set(df_classified.columns)
        if missing:
            raise ValueError(
                f"to_evaluation_schema is missing columns {sorted(missing)}; "
                "expected the output of score()/classify()."
            )
        return pd.DataFrame(
            {
                "PATID_A": df_classified["PATID_A"].to_numpy(),
                "PATID_B": df_classified["PATID_B"].to_numpy(),
                "model_name": self.model_name,
                "score": df_classified["plausible_probability"].to_numpy(),
                "predicted_tier": df_classified["classification_tier"].to_numpy(),
            }
        )

    def apply(
        self, candidate_pairs: pd.DataFrame, df_clean: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Gate `candidate_pairs`, the pipeline's one call.

        Returns ``(surviving_pairs, evaluation_frame)`` where
        ``surviving_pairs`` is the subset of `candidate_pairs` that passed —
        the input frame's own rows, so passthrough columns (`source_blocks`/
        `n_blocks`) are preserved — and ``evaluation_frame`` is the full
        `ClassificationResults` audit frame covering every scored pair.
        """
        classified = self.score(candidate_pairs, df_clean)
        evaluation = self.to_evaluation_schema(classified)
        passed = classified.loc[
            classified["classification_tier"] == TIER_HUMAN_REVIEW, ["PATID_A", "PATID_B"]
        ]
        survivors = candidate_pairs.merge(
            passed, on=["PATID_A", "PATID_B"], how="inner"
        ).reset_index(drop=True)
        return survivors, evaluation

    # ── internals ─────────────────────────────────────────────────────────────
    def _align(self, X):
        """Reorder columns to the fitted model's training feature order when
        known, so column ordering can't silently corrupt predictions."""
        names = getattr(self.model, "feature_name_", None)
        if names is not None and isinstance(X, pd.DataFrame) and set(names).issubset(X.columns):
            return X[list(names)]
        return X


__all__ = ["NonMatchGate", "MODEL_NAME", "DEFAULT_THRESHOLD"]
