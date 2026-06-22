"""Integration test for the split-training path (Phase E2-5-fix3).

Simulates the real-data scenario: labels reference one PATID space (synthetic),
the production records frame contains a disjoint PATID space (mock real cohort).
Without split-training this would silently produce floor-m output; with the
fix, m is trained on the synthetic frame and then transferred to a linker
bound to the mock real cohort for u-estimation and scoring.

Three assertions:

1. **Validation guard fires** when labels reference unresolvable PATIDs and no
   `labels_records_df` is provided.

2. **m_probability is populated** on the live linker's comparison levels
   after split-training completes (vs the floor `None` state of an
   un-trained linker).

3. **End-to-end scoring works** on the mock production candidate pool: the
   model can predict, classify, and project to the ProbabilisticMatches
   contract without errors.

The "mock real cohort" is built by re-stamping synthetic records with a
disjoint PATID prefix (`R0000...` vs `S0000...`), which gives us a frame
that LOOKS like the real cohort to Splink (disjoint unique-ids) without
introducing PHI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.experiments.fs_splink_enhanced_2.fs_enhanced_2 import FSEnhanced2
from src.contracts import ProbabilisticMatches, validate

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTH_DIR = _PROJECT_ROOT / "data" / "synthetic"


# ── Fixtures ─────────────────────────────────────────────────────────────────
def _deconstruct(df: pd.DataFrame) -> pd.DataFrame:
    left_cols = ["PATID_A"] + [c for c in df.columns if c.endswith("_l")]
    right_cols = ["PATID_B"] + [c for c in df.columns if c.endswith("_r")]
    left = df[left_cols].rename(columns=lambda c: c[:-2])
    right = df[right_cols].rename(columns=lambda c: c[:-2])
    return pd.concat([left, right], ignore_index=True).drop_duplicates(subset=["PATID"])


@pytest.fixture(scope="module")
def synthetic_records() -> pd.DataFrame:
    train = pd.read_csv(SYNTH_DIR / "synthetic_train_v3.csv", dtype=str)
    df = _deconstruct(train).reset_index(drop=True)
    df["BirthDT_clean"] = pd.to_datetime(df["BirthDT_clean"], errors="coerce")
    df["valid_record"] = True
    return df


@pytest.fixture(scope="module")
def train_labels() -> pd.DataFrame:
    df = pd.read_csv(SYNTH_DIR / "synthetic_train_v3.csv", dtype=str)
    df["label"] = df["label"].astype(int)
    return df


@pytest.fixture(scope="module")
def mock_real_cohort(synthetic_records: pd.DataFrame) -> pd.DataFrame:
    """Synthetic records with PATIDs re-stamped to a disjoint namespace,
    simulating a 'real cohort' with unique-ids the labels don't reference."""
    df = synthetic_records.copy()
    df["PATID"] = "R" + df["PATID"].str.slice(1)  # S000... -> R000...
    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def mock_candidate_pairs(mock_real_cohort: pd.DataFrame) -> pd.DataFrame:
    """A few random pairs from the mock cohort for the scoring smoke check."""
    sample = mock_real_cohort["PATID"].head(20).tolist()
    pairs = []
    for i in range(0, len(sample) - 1, 2):
        a, b = sorted([sample[i], sample[i + 1]])
        pairs.append({"PATID_A": a, "PATID_B": b, "source_blocks": "smoke", "n_blocks": 1})
    return pd.DataFrame(pairs)


# ── Tests ────────────────────────────────────────────────────────────────────
def test_guard_raises_when_labels_unresolvable_and_no_split_records(
    train_labels: pd.DataFrame, mock_real_cohort: pd.DataFrame,
    mock_candidate_pairs: pd.DataFrame,
):
    """Without labels_records_df, labels' PATIDs must resolve in df_clean —
    otherwise SupervisedTraining raises a clear ValueError before training."""
    model = FSEnhanced2(
        labels_df=train_labels,
        include_address=False,
        u_max_pairs=1e4,
        labels_records_df=None,  # split-training disabled
    )
    df_model = model.prepare_model_input(mock_real_cohort)
    linker = model.build_linker(df_model, mock_candidate_pairs)
    with pytest.raises(ValueError, match="labeled PATIDs are absent"):
        model.train(linker, mock_real_cohort)


def test_split_training_populates_m_on_live_linker(
    train_labels: pd.DataFrame,
    mock_real_cohort: pd.DataFrame,
    mock_candidate_pairs: pd.DataFrame,
    synthetic_records: pd.DataFrame,
):
    """With labels_records_df provided, m is trained on the auxiliary linker
    bound to the synthetic records, then copied to the live linker. After
    training, the live linker's settings_obj has non-default m_probability
    on at least one comparison level (proxy: m_probability is set, not None)."""
    model = FSEnhanced2(
        labels_df=train_labels,
        include_address=False,
        u_max_pairs=1e4,
        labels_records_df=synthetic_records,
    )
    df_model = model.prepare_model_input(mock_real_cohort)
    linker = model.build_linker(df_model, mock_candidate_pairs)
    model.train(linker, mock_real_cohort)

    # At least one comparison should have a populated m on a non-null level.
    n_populated = 0
    for comp in linker._settings_obj.comparisons:
        for level in comp.comparison_levels:
            if getattr(level, "is_null_level", False):
                continue
            if level.m_probability is not None:
                n_populated += 1
    assert n_populated > 0, (
        "Split-training should populate m on at least one non-null level."
    )


def test_split_training_end_to_end_scoring(
    train_labels: pd.DataFrame,
    mock_real_cohort: pd.DataFrame,
    mock_candidate_pairs: pd.DataFrame,
    synthetic_records: pd.DataFrame,
):
    """Full pipeline: train + predict + classify + project. ProbabilisticMatches
    round-trip must validate."""
    model = FSEnhanced2(
        labels_df=train_labels,
        include_address=False,
        u_max_pairs=1e4,
        labels_records_df=synthetic_records,
    )
    df_model = model.prepare_model_input(mock_real_cohort)
    linker = model.build_linker(df_model, mock_candidate_pairs)
    model.train(linker, mock_real_cohort)
    predictions = model.predict(linker, mock_candidate_pairs)
    assert len(predictions) > 0
    classified = model.classify(predictions)
    prob_matches = model.to_probabilistic_matches(classified)
    validated = validate(prob_matches, ProbabilisticMatches)
    assert "veto_reason" not in validated.columns
