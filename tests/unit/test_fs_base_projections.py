"""Unit tests for FSModel.to_evaluation_schema + to_probabilistic_matches."""

from __future__ import annotations

import pandas as pd
import pytest

from models.common.fs_base import (
    ClassificationConfig,
    ComparisonRegistry,
    ComparisonSpec,
    FSModel,
    SupervisedTraining,
)


class _DummyFS(FSModel):
    model_name = "dummy_fs"
    registry = ComparisonRegistry([ComparisonSpec("X", lambda: {})])
    classification_config = ClassificationConfig()

    def __init__(self):
        self.training = SupervisedTraining(
            pd.DataFrame({"label": [0, 1]}), label_col="label",
        )

    def prepare_model_input(self, df_clean):
        return df_clean

    def build_settings(self):
        return {"comparisons": self.registry.build_all()}


# ─── to_evaluation_schema ─────────────────────────────────────────────────────
def test_to_evaluation_schema_columns_and_values():
    model = _DummyFS()
    rich = pd.DataFrame({
        "PATID_A": ["P0001", "P0003"],
        "PATID_B": ["P0002", "P0004"],
        "match_probability": [0.97, 0.20],
        "classification_tier": ["auto_merge", "no_match"],
        "match_weight": [5.0, -2.0],  # should be dropped
    })
    out = model.to_evaluation_schema(rich)
    assert list(out.columns) == ["PATID_A", "PATID_B", "model_name", "score", "predicted_tier"]
    assert (out["model_name"] == "dummy_fs").all()
    assert out["score"].tolist() == [0.97, 0.20]
    assert out["predicted_tier"].tolist() == ["auto_merge", "no_match"]


def test_to_evaluation_schema_raises_on_missing_required():
    model = _DummyFS()
    bad = pd.DataFrame({"PATID_A": ["P1"], "PATID_B": ["P2"], "match_probability": [0.5]})
    with pytest.raises(ValueError, match="missing"):
        model.to_evaluation_schema(bad)


# ─── to_probabilistic_matches ─────────────────────────────────────────────────
def test_to_probabilistic_matches_columns_and_values():
    model = _DummyFS()
    rich = pd.DataFrame({
        "PATID_A": ["P0001", "P0003"],
        "PATID_B": ["P0002", "P0004"],
        "match_probability": [0.97, 0.20],
        "match_weight": [5.0, -2.0],
        "classification_tier": ["auto_merge", "no_match"],
        "source_blocks": ["B1|B3", "B5"],
        "n_blocks": [2, 1],
    })
    out = model.to_probabilistic_matches(rich)
    expected_cols = [
        "PATID_A", "PATID_B", "match_source", "score", "match_weight",
        "classification_tier", "source_blocks", "n_blocks",
    ]
    assert list(out.columns) == expected_cols
    assert (out["match_source"] == "model").all()
    assert out["score"].tolist() == [0.97, 0.20]
    assert out["source_blocks"].tolist() == ["B1|B3", "B5"]


def test_to_probabilistic_matches_handles_missing_optional_columns():
    model = _DummyFS()
    rich = pd.DataFrame({
        "PATID_A": ["P0001"],
        "PATID_B": ["P0002"],
        "match_probability": [0.5],
        "match_weight": [0.0],
        "classification_tier": ["human_review"],
    })
    out = model.to_probabilistic_matches(rich)
    assert out["source_blocks"].iloc[0] is None
    assert pd.isna(out["n_blocks"].iloc[0])


def test_to_probabilistic_matches_raises_on_missing_required():
    model = _DummyFS()
    bad = pd.DataFrame({
        "PATID_A": ["P1"], "PATID_B": ["P2"],
        "match_probability": [0.5], "classification_tier": ["human_review"],
        # match_weight missing
    })
    with pytest.raises(ValueError, match="missing"):
        model.to_probabilistic_matches(bad)


def test_to_probabilistic_matches_default_projection_omits_veto_reason():
    """enhanced_2 has no veto layer; the default projection must not invent one."""
    model = _DummyFS()
    rich = pd.DataFrame({
        "PATID_A": ["P1"],
        "PATID_B": ["P2"],
        "match_probability": [0.7],
        "match_weight": [1.0],
        "classification_tier": ["human_review"],
    })
    out = model.to_probabilistic_matches(rich)
    assert "veto_reason" not in out.columns
