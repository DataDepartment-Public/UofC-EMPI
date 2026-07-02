"""Production Fellegi-Sunter matcher (pipeline Stage 4).

A candidate + feature generator for the downstream Gradient-Boosted-Tree: loads
a pre-trained Splink model artifact, scores the deterministic rules' `non_matches`
pool, and emits the GBT feature parquet (candidates >= review_floor) plus a full
`ProbabilisticMatches` audit frame. Its output does NOT feed clustering.

Lifted, self-contained under `src/` (imports nothing under `models/`). Train with
`python -m src.models.fs_matcher.train`; serve via the pipeline.
"""

from src.models.fs_matcher.matcher import (
    FSMatcher,
    MODEL_NAME,
    classification_config_from_settings,
)

__all__ = ["FSMatcher", "MODEL_NAME", "classification_config_from_settings"]
