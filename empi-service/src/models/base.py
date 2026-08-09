"""Shared classifier interface for the eMPI pair-classification stages.

Deterministic rules, the Fellegi-Sunter matcher, and the ML matcher all do
the same conceptual job: take candidate pairs in, classify each into
`auto_merge` / `human_review` / `no_match` (see `src.contracts.
CLASSIFICATION_TIERS`). `PairClassifier` formalizes that shape.

Deliberately a `typing.Protocol`, not an ABC: `src/models/fs_matcher/` is
self-contained by design (`base.py`/`matcher.py` import nothing under
`src.models.*`), and an ABC would force `FSMatcher` to subclass across that
boundary. A Protocol lets any classifier satisfy the interface structurally,
with zero new imports required of existing implementations.

This is a leaf module — it depends only on `src.contracts`, so importing it
from `deterministic_rules.py`, `fs_matcher/`, or `ml_matcher/` never creates
an import cycle.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from src.contracts import TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH

__all__ = [
    "PairClassifier",
    "to_edges",
    "TIER_AUTO_MERGE",
    "TIER_HUMAN_REVIEW",
    "TIER_NO_MATCH",
]


@runtime_checkable
class PairClassifier(Protocol):
    """The contract every classifier stage exposes to `src/pipeline.py`.

    Structural (Protocol), not nominal — a class satisfies this by shape
    alone; it does not need to inherit from anything here.
    """

    model_name: str

    def run(
        self,
        candidate_pairs: pd.DataFrame,
        df_clean: pd.DataFrame,
        **kwargs: object,
    ) -> pd.DataFrame:
        """Classify every row of `candidate_pairs`.

        Returns a frame satisfying `src.contracts.ClassificationResults`
        (`PATID_A, PATID_B, model_name, score, predicted_tier`).
        `**kwargs` exists so implementations like `MLMatcher` can accept
        extra inputs (e.g. `fs_features=`) without breaking the Protocol
        shape for implementations that don't need them.
        """
        ...


def to_edges(results: pd.DataFrame, *, match_source: str) -> pd.DataFrame:
    """Project a `ClassificationResults` frame's `auto_merge`-tier rows to
    the `src.contracts.Edges` shape, for optional union into clustering.

    Non-`auto_merge` rows (human_review/no_match) are dropped — clustering
    only ever consumes auto-merge edges, regardless of source.
    """
    auto = results[results["predicted_tier"] == TIER_AUTO_MERGE]
    return pd.DataFrame(
        {
            "PATID_A": auto["PATID_A"].to_numpy(),
            "PATID_B": auto["PATID_B"].to_numpy(),
            "confidence": auto["score"].to_numpy(),
            "match_source": match_source,
            "evidence": auto["model_name"].to_numpy(),
        }
    ).reset_index(drop=True)
