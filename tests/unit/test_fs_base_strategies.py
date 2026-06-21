"""Unit tests for TrainingStrategy concrete classes — signature + ctor invariants.

Splink itself isn't touched here; `train(linker, df_clean)` calls are
exercised against real Splink in tests/integration/ and tests/regression/.
"""

from __future__ import annotations

import pandas as pd
import pytest

from models.common.fs_base import (
    EMTraining,
    SupervisedTraining,
    TrainingStrategy,
)


def test_em_training_is_a_training_strategy():
    em = EMTraining(em_blocking_rules=["l.x = r.x"])
    assert isinstance(em, TrainingStrategy)


def test_em_training_stores_rules_and_defaults():
    em = EMTraining(
        em_blocking_rules=["l.a = r.a", "l.b = r.b"],
        prior_rules=["l.c = r.c"],
        recall=0.5,
        u_max_pairs=2e5,
        seed=7,
    )
    assert em.em_blocking_rules == ["l.a = r.a", "l.b = r.b"]
    assert em.prior_rules == ["l.c = r.c"]
    assert em.recall == 0.5
    assert em.u_max_pairs == 2e5
    assert em.seed == 7


def test_em_training_none_prior_rules_kept_as_none():
    em = EMTraining(em_blocking_rules=["l.x = r.x"])
    assert em.prior_rules is None


def test_supervised_training_is_a_training_strategy():
    labels = pd.DataFrame({"label": [0, 1]})
    st = SupervisedTraining(labels, label_col="label")
    assert isinstance(st, TrainingStrategy)


def test_supervised_training_raises_when_label_col_missing():
    labels = pd.DataFrame({"y": [0, 1]})
    with pytest.raises(ValueError, match="label column"):
        SupervisedTraining(labels, label_col="label")


def test_supervised_training_stores_defaults():
    labels = pd.DataFrame({"label": [0, 1]})
    st = SupervisedTraining(labels)
    assert st.label_col == "label"
    assert st.u_max_pairs == 1e6
    assert st.seed == 42
    assert st.labels_table_name == "synthetic_labels"


def test_training_strategy_is_abstract():
    with pytest.raises(TypeError):
        TrainingStrategy()  # type: ignore[abstract]
