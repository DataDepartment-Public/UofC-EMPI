"""Cross-experiment FS utilities.

Re-exports the shared object-oriented FS base introduced in Phase E2-1.
The functional modules `eval_schema`, `versioning`, `synthetic_data` remain
importable directly from their own files; nothing is re-exported from them
here to keep the namespace narrow.
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

__all__ = [
    "ClassificationConfig",
    "ComparisonRegistry",
    "ComparisonSpec",
    "EMTraining",
    "FSModel",
    "SupervisedTraining",
    "TrainingStrategy",
]
