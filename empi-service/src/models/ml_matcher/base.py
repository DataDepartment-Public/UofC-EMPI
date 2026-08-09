"""Pluggable contracts for the ML matcher (`src/models/ml_matcher/`) — the
bring-your-own-model / bring-your-own-features extension points.

Two Protocols define the plug-in surface:

* `FeatureBuilder` — turns candidate pairs (+ cleaned records, + optionally
  the FS matcher's `FSFeatures`) into a feature frame. `NotImplementedFeatureBuilder`
  is the default — a real implementation is left to whoever plugs in a model
  (see `docs/ML-Matcher-Integration-Guide.md`).
* `MLModel` — scikit-learn duck typing (`fit`/`predict_proba`). Any sklearn,
  XGBoost, LightGBM, or CatBoost estimator already satisfies this with zero
  adapter code; a PyTorch/TensorFlow model needs a small wrapper exposing the
  same two methods.

`MLClassificationConfig` is the stage's threshold config. It carries a single
threshold, deliberately: Stage 4.5 is a **2-tier** classifier. The Stage-4.25
gate owns the discard decision and records every drop it makes in
`data/gate_output/`, so a second stage able to discard would mean a pair could
vanish in either of two places with only one of them audited. The FS matcher
keeps its own 3-tier `fs_matcher.base.ClassificationConfig` (it predates the
gate and still serves as the fallback gate); the two are not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

__all__ = [
    "MLClassificationConfig",
    "FeatureBuilder",
    "MLModel",
    "NotImplementedFeatureBuilder",
]


@dataclass(frozen=True)
class MLClassificationConfig:
    """The ML matcher's single decision threshold.

    `auto_merge_threshold` splits every scored pair into exactly two tiers:
    at/above it, `auto_merge`; below it, `human_review`. There is no floor and
    no `no_match` tier — see the module docstring for why that is structural
    rather than configured.
    """

    auto_merge_threshold: float = 0.70

    def __post_init__(self) -> None:
        if not 0.0 <= self.auto_merge_threshold <= 1.0:
            raise ValueError(
                f"MLClassificationConfig: auto_merge_threshold must be in "
                f"[0, 1], got {self.auto_merge_threshold}"
            )


@runtime_checkable
class FeatureBuilder(Protocol):
    """BYOF extension point."""

    def build_features(
        self,
        candidate_pairs: pd.DataFrame,
        df_clean: pd.DataFrame,
        fs_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return a frame keyed by `PATID_A`/`PATID_B` with feature columns.

        `fs_features` (the FS matcher's `gamma_*`/`bf_*` columns) is an
        optional enrichment join, not a pool filter — a builder that ignores
        it derives features straight from `candidate_pairs`/`df_clean`
        instead. Both paths are first-class.
        """
        ...


@runtime_checkable
class MLModel(Protocol):
    """BYOM extension point — scikit-learn duck typing."""

    def fit(self, X, y) -> "MLModel": ...

    def predict_proba(self, X): ...


class NotImplementedFeatureBuilder:
    """Default `FeatureBuilder` — scaffolding only.

    Ships with no default feature logic; raises the moment it's used. Pass a
    real `FeatureBuilder` implementation to `MLMatcher(feature_builder=...)`
    to make the matcher usable. See `docs/ML-Matcher-Integration-Guide.md`.
    """

    def build_features(
        self,
        candidate_pairs: pd.DataFrame,
        df_clean: pd.DataFrame,
        fs_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "ml_matcher ships with no default feature builder. Implement "
            "FeatureBuilder.build_features (custom features off "
            "candidate_pairs/df_clean, or joining fs_features) and pass it "
            "to MLMatcher(feature_builder=...). See "
            "docs/ML-Matcher-Integration-Guide.md."
        )
