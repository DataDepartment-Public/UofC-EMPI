"""MLMatcher — the pluggable ML matcher (pipeline Stage 4.5).

Bring-your-own-model / bring-your-own-features candidate + feature generator,
structurally parallel to the FS matcher (`src/models/fs_matcher/`): loads a
pre-trained model artifact and scores the rules' `non_matches` pool without
retraining. Its output is audit-only by default — see
`settings.ml_feeds_clustering`.

SCAFFOLDING STATUS: the pluggable surface (`FeatureBuilder`/`MLModel`
Protocols in `base.py`, threshold classification, the shared 5-col output
projection satisfying `src.models.base.PairClassifier`) is real and usable
today. `train()` is a deliberate stub — feature preprocessing and model
fitting are left to whoever plugs in a real model (see
`docs/ML-Matcher-Integration-Guide.md`).

PHI / HIPAA: logs aggregate counts only, mirroring the FS matcher.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.contracts import TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH
from src.models.ml_matcher.base import (
    ClassificationConfig,
    FeatureBuilder,
    MLModel,
    NotImplementedFeatureBuilder,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "ml_matcher"


class MLMatcher:
    """Satisfies `src.models.base.PairClassifier` structurally — no
    inheritance needed (Protocol conformance is by shape)."""

    model_name = MODEL_NAME

    def __init__(
        self,
        model: MLModel | None = None,
        feature_builder: FeatureBuilder | None = None,
        classification_config: ClassificationConfig | None = None,
    ):
        """
        Parameters
        ----------
        model : optional
            A fitted (or, before training, unfitted) scikit-learn-compatible
            estimator (`fit`/`predict_proba`). `None` until loaded or
            trained — `predict()` raises if called with no model attached.
        feature_builder : optional
            Defaults to `NotImplementedFeatureBuilder` (raises on use) — pass
            a real `FeatureBuilder` implementation to make this usable.
        classification_config : optional
            Thresholds. Defaults to `ClassificationConfig()` (0.95 / 0.40);
            the pipeline passes settings-derived thresholds
            (`ml_auto_merge_threshold` / `ml_review_floor`).
        """
        self.model = model
        self.feature_builder = feature_builder or NotImplementedFeatureBuilder()
        self.classification_config = classification_config or ClassificationConfig()

    # ── Feature building (BYOF) ────────────────────────────────────────────────
    def build_features(
        self,
        candidate_pairs: pd.DataFrame,
        df_clean: pd.DataFrame,
        fs_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return self.feature_builder.build_features(candidate_pairs, df_clean, fs_features)

    # ── Scoring (BYOM) ──────────────────────────────────────────────────────────
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score a feature frame (`PATID_A`/`PATID_B` + feature columns) with
        the attached model. Returns `PATID_A`/`PATID_B` + `match_probability`."""
        if self.model is None:
            raise RuntimeError(
                "MLMatcher.predict: no model attached — load or train one first."
            )
        X = features.drop(columns=["PATID_A", "PATID_B"], errors="ignore")
        proba = np.asarray(self.model.predict_proba(X))
        score = proba[:, 1] if proba.ndim == 2 else proba
        out = features[["PATID_A", "PATID_B"]].copy()
        out["match_probability"] = score
        return out

    def classify(self, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """Threshold `match_probability` into `classification_tier` — the
        same cutoffs `FSModel.classify()` uses, minus the n_blocks bump
        (BYOF means the pipeline can't assume `n_blocks` survived feature
        building)."""
        cfg = self.classification_config
        p = df_predictions["match_probability"]
        tier = pd.Series(TIER_NO_MATCH, index=df_predictions.index, dtype="object")
        tier = tier.mask(p >= cfg.review_floor, TIER_HUMAN_REVIEW)
        tier = tier.mask(p >= cfg.auto_merge_threshold, TIER_AUTO_MERGE)
        out = df_predictions.copy()
        out["classification_tier"] = tier
        counts = {k: int(v) for k, v in out["classification_tier"].value_counts().items()}
        logger.info("MLMatcher[%s]: classify %s", self.model_name, counts)
        return out

    # ── Score (build_features + predict + classify in one call) ────────────────
    def score(
        self,
        candidate_pairs: pd.DataFrame,
        df_clean: pd.DataFrame,
        fs_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build features, score, and classify in one call. Returns the rich
        classified frame (`PATID_A`/`PATID_B`/`match_probability`/
        `classification_tier` + every feature column `build_features()`
        produced) — mirrors `FSMatcher.score()`. Project with
        `to_evaluation_schema()` for the shared 5-col `PairClassifier` shape,
        or `to_ml_features()` for the candidate parquet. Calling `score()`
        once and projecting twice avoids re-running feature building/predict
        for each projection.
        """
        features = self.build_features(candidate_pairs, df_clean, fs_features)
        return self.classify(self.predict(features))

    def to_evaluation_schema(self, df_classified: pd.DataFrame) -> pd.DataFrame:
        """Project a `score()`/`classify()` output to the shared 5-col
        `ClassificationResults` shape (`src.models.base.PairClassifier`)."""
        required = {"PATID_A", "PATID_B", "match_probability", "classification_tier"}
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
                "score": df_classified["match_probability"].to_numpy(),
                "predicted_tier": df_classified["classification_tier"].to_numpy(),
            }
        )

    # ── PairClassifier ──────────────────────────────────────────────────────────
    def run(
        self,
        candidate_pairs: pd.DataFrame,
        df_clean: pd.DataFrame,
        fs_features: pd.DataFrame | None = None,
        **kwargs: object,
    ) -> pd.DataFrame:
        """Convenience: `score()` + `to_evaluation_schema()` in one call.
        Returns a frame satisfying `src.contracts.ClassificationResults`."""
        return self.to_evaluation_schema(self.score(candidate_pairs, df_clean, fs_features))

    # ── Candidate feature projection (mirrors FSMatcher.to_fs_features) ────────
    def to_ml_features(
        self, df_classified: pd.DataFrame, *, candidates_only: bool = True,
    ) -> pd.DataFrame:
        """Project a `classify()`-output frame to the `MLFeatures` candidate
        parquet: `PATID_A`/`PATID_B` + `match_probability` +
        `classification_tier` + every BYOF feature column `build_features()`
        produced. When `candidates_only` (the pipeline default), keeps only
        rows at/above the review floor.
        """
        required = {"PATID_A", "PATID_B", "match_probability", "classification_tier"}
        missing = required - set(df_classified.columns)
        if missing:
            raise ValueError(
                f"to_ml_features is missing columns {sorted(missing)}; expected "
                "the output of classify()."
            )
        out = df_classified.reset_index(drop=True).copy()
        if candidates_only:
            floor = self.classification_config.review_floor
            before = len(out)
            out = out[out["match_probability"] >= floor].reset_index(drop=True)
            logger.info(
                "to_ml_features: kept %d/%d candidate pairs (match_probability >= %.2f)",
                len(out), before, floor,
            )
        return out

    # ── Training (STUB) ──────────────────────────────────────────────────────────
    def train(
        self,
        candidate_pairs: pd.DataFrame,
        df_clean: pd.DataFrame,
        labels: pd.DataFrame,
        fs_features: pd.DataFrame | None = None,
    ) -> None:
        """Fit `self.model` in place. STUB — scaffolding only.

        A real implementation builds features (`self.build_features`),
        derives X/y from `labels` (format is the implementer's choice), and
        calls `self.model.fit(X, y)`. See
        `docs/ML-Matcher-Integration-Guide.md`.
        """
        raise NotImplementedError(
            "MLMatcher.train is a scaffold stub — no training implementation "
            "is plugged in yet. See docs/ML-Matcher-Integration-Guide.md."
        )


__all__ = ["MLMatcher", "MODEL_NAME"]
