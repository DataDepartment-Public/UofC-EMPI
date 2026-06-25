"""tests/unit/test_fs_enhanced_3_model.py

Lightweight wiring tests for FSEnhanced3 — no Splink training (the full
train→score round-trip lives in run_synthetic_enhanced_3.py). These confirm the
model is composed correctly over the shared base: registry, thresholds, and the
SupervisedTraining strategy with the lambda prior wired in.
"""

from __future__ import annotations

import pandas as pd

from models.common.fs_base import SupervisedTraining
from models.experiments.fs_splink_enhanced_3.comparisons import (
    DEFAULT_PRIOR_RECALL,
    PRIOR_RULES,
)
from models.experiments.fs_splink_enhanced_3.fs_enhanced_3 import (
    MODEL_NAME,
    FSEnhanced3,
)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {"PATID_A": ["a", "c"], "PATID_B": ["b", "d"], "label": [1, 0]}
    )


def test_model_name():
    assert MODEL_NAME == "fs_splink_enhanced_3"
    assert FSEnhanced3(labels_df=_labels()).model_name == "fs_splink_enhanced_3"


def test_registry_wired_with_address():
    m = FSEnhanced3(labels_df=_labels(), include_address=True)
    assert len(m.registry) == 7
    assert "Address" in m.registry


def test_registry_wired_without_address():
    m = FSEnhanced3(labels_df=_labels(), include_address=False)
    assert len(m.registry) == 6
    assert "Address" not in m.registry


def test_default_thresholds():
    cfg = FSEnhanced3(labels_df=_labels()).classification_config
    assert cfg.auto_merge_threshold == 0.95
    assert cfg.review_floor == 0.40


def test_training_is_supervised_with_prior():
    m = FSEnhanced3(labels_df=_labels())
    assert isinstance(m.training, SupervisedTraining)
    # Lambda prior is wired by default from the module's PRIOR_RULES.
    assert m.training.prior_rules == PRIOR_RULES
    assert m.training.prior_recall == DEFAULT_PRIOR_RECALL


def test_prior_rules_overridable():
    m = FSEnhanced3(labels_df=_labels(), prior_rules=[])
    assert m.training.prior_rules is None  # empty list disables the prior


def test_build_settings_has_seven_comparisons():
    settings = FSEnhanced3(labels_df=_labels()).build_settings()
    assert len(settings["comparisons"]) == 7
    assert settings["unique_id_column_name"] == "PATID"
