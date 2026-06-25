"""tests/unit/test_fs_base_enhanced_3_hooks.py

Unit tests for the two backward-compatible fs_base additions made for
enhanced_3:

  1. SupervisedTraining(prior_rules=...) seeds the match-prevalence prior via
     estimate_probability_two_random_records_match before m/u estimation.
  2. FSModel.run(return_linker=True) returns a (result, linker) tuple.

Both are mock-based — no Splink training is performed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from models.common.fs_base import (
    ClassificationConfig,
    ComparisonRegistry,
    FSModel,
    SupervisedTraining,
)


def _labels_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"PATID_A": ["a"], "PATID_B": ["b"], "label": [1]}
    )


def _fake_linker() -> MagicMock:
    linker = MagicMock()
    # _warn_on_weight_inversions iterates this; keep it an empty list so the
    # diagnostic is a no-op instead of choking on a MagicMock.
    linker._settings_obj._parameters_as_detailed_records = []
    return linker


# ── 1. prior_rules seeds lambda ──────────────────────────────────────────────
def test_prior_rules_seed_lambda_before_m_training():
    linker = _fake_linker()
    st = SupervisedTraining(
        labels_df=_labels_df(),
        prior_rules=["l.SSN_clean = r.SSN_clean AND l.SSN_clean IS NOT NULL"],
        prior_recall=0.85,
    )
    st.train(linker, df_clean=None, model=None)

    linker.training.estimate_probability_two_random_records_match.assert_called_once_with(
        deterministic_matching_rules=[
            "l.SSN_clean = r.SSN_clean AND l.SSN_clean IS NOT NULL"
        ],
        recall=0.85,
    )
    linker.training.estimate_m_from_pairwise_labels.assert_called_once()
    linker.training.estimate_u_using_random_sampling.assert_called_once()


def test_no_prior_rules_skips_lambda_seed():
    linker = _fake_linker()
    st = SupervisedTraining(labels_df=_labels_df())  # prior_rules default None
    st.train(linker, df_clean=None, model=None)

    linker.training.estimate_probability_two_random_records_match.assert_not_called()
    linker.training.estimate_m_from_pairwise_labels.assert_called_once()


# ── 2. return_linker flag ────────────────────────────────────────────────────
class _FakeModel(FSModel):
    model_name = "fake"
    registry = ComparisonRegistry([])
    classification_config = ClassificationConfig()

    def prepare_model_input(self, df_clean):
        return df_clean

    def build_settings(self):
        return {}

    def build_linker(self, df_model, candidate_pairs_df, db_api=None):
        return "LINKER_SENTINEL"

    def train(self, linker, df_clean=None):
        return linker

    def predict(self, linker, candidate_pairs_df=None):
        return pd.DataFrame(
            {
                "PATID_A": ["a"],
                "PATID_B": ["b"],
                "match_probability": [0.97],
                "match_weight": [5.0],
            }
        )


def _empty_inputs():
    cp = pd.DataFrame({"PATID_A": [], "PATID_B": []})
    clean = pd.DataFrame({"PATID": []})
    return cp, clean


def test_run_default_returns_single_frame():
    cp, clean = _empty_inputs()
    out = _FakeModel().run(cp, clean)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(FSModel.eval_schema_columns)


def test_run_return_linker_returns_tuple():
    cp, clean = _empty_inputs()
    result, linker = _FakeModel().run(cp, clean, return_linker=True)
    assert isinstance(result, pd.DataFrame)
    assert linker == "LINKER_SENTINEL"


def test_run_return_linker_with_full_output():
    cp, clean = _empty_inputs()
    result, linker = _FakeModel().run(cp, clean, full_output=True, return_linker=True)
    # full_output → rich classified frame, which carries classification_tier.
    assert "classification_tier" in result.columns
    assert linker == "LINKER_SENTINEL"
