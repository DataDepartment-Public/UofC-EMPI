"""Pluggable ML matcher (pipeline Stage 4.5).

Bring-your-own-model / bring-your-own-features candidate + feature generator,
structurally parallel to the FS matcher (`src/models/fs_matcher/`). The
pluggable interface (`FeatureBuilder`/`MLModel` Protocols in `base.py`,
`MLMatcher` in `matcher.py`) is real; training (`MLMatcher.train`) is a
deliberate stub — see `docs/ML-Matcher-Integration-Guide.md` for the
extension contract.

Train with `python -m src.models.ml_matcher.train` once a real training
implementation is plugged in; serve via the pipeline.
"""

from src.models.ml_matcher.matcher import MLMatcher, MODEL_NAME

__all__ = ["MLMatcher", "MODEL_NAME"]
