"""Cross-experiment FS utilities — public surface of `models/common/`.

Re-exports the shared OO base (`fs_base.py`) and the two versioning helpers
(`versioning.py`). Synthetic-data fixtures stay importable from
`models.common.synthetic_data` directly; `eval_schema` was deleted in
Phase E3-0 — its constants are inlined in each runner that needs them.
"""

from models.common.fs_base import (
    ClassificationConfig,
    ComparisonRegistry,
    ComparisonSpec,
    EMTraining,
    FSModel,
    SupervisedTraining,
    TrainingStrategy,
)
from models.common.versioning import latest_versioned, version_tag_from_filename

__all__ = [
    "ClassificationConfig",
    "ComparisonRegistry",
    "ComparisonSpec",
    "EMTraining",
    "FSModel",
    "SupervisedTraining",
    "TrainingStrategy",
    "latest_versioned",
    "version_tag_from_filename",
]
