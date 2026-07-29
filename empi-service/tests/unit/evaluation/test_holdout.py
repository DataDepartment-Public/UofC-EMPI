"""Guards on the reconstruction of the training notebooks' test folds.

If either notebook changes its `RANDOM_STATE`, split ratios, stratification
target, or the row ordering it loads gold in, `src/evaluation/holdout.py` goes
silently wrong — it would still return *a* fold, just not the one the model was
actually held out on, and every "leakage-safe" number downstream would be
quietly contaminated. These tests are the tripwire.

`test_real_gold_*` are the strongest guards but need the VM's PHI label file;
they skip cleanly on a dev machine. The rest run everywhere against a
gold-shaped fixture.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.holdout import (
    GOLD_AMBIGUOUS_COL,
    GOLD_LABEL_COL,
    RANDOM_STATE,
    TEST_SIZE,
    gold_test_folds,
    load_gold_labels,
)

# The real file, if this is running where the data lives.
_GOLD = (
    Path(__file__).resolve().parents[3]
    / "data" / "gold_labels" / "final_gold_labels_v1_2026_07_05.csv"
)

# Counts the notebooks printed for the served gate and matcher — see
# `pair_classifier_lightgbm_{nonmatch_gate_v1,ambiguous_v3}.ipynb` cell 5.
GOLD_ROWS = 204_805
GOLD_PLAUSIBLE = 62_610


def _fixture_gold(n: int = 1_000, seed: int = 0) -> pd.DataFrame:
    """Gold-shaped labels with a plausible-ish class balance."""
    rng = np.random.default_rng(seed)
    label = rng.random(n) < 0.17
    ambiguous = rng.random(n) < 0.14
    return pd.DataFrame({
        "PATID_A": [f"{i:06d}" for i in range(n)],
        "PATID_B": [f"{i + n:06d}" for i in range(n)],
        "ambiguous_pair": ambiguous,
        "final_gold_label": label,
    })


# ── constants ────────────────────────────────────────────────────────────────
def test_split_constants_match_the_notebooks():
    assert RANDOM_STATE == 42
    assert TEST_SIZE == 0.20


# ── mechanics ────────────────────────────────────────────────────────────────
def test_fold_sizes_are_twenty_percent_of_each_population():
    gold = _fixture_gold()
    folds = gold_test_folds(gold)
    n_plausible = int((gold[GOLD_LABEL_COL] | gold[GOLD_AMBIGUOUS_COL]).sum())

    # sklearn rounds the test split UP (`ceil`), which is exact for the real
    # gold sizes but not for an arbitrary fixture.
    assert len(folds.gate) == math.ceil(len(gold) * TEST_SIZE)
    assert len(folds.matcher) == math.ceil(n_plausible * TEST_SIZE)
    assert len(folds.matcher_population) == n_plausible


def test_folds_are_deterministic():
    gold = _fixture_gold()
    assert gold_test_folds(gold).gate == gold_test_folds(gold).gate


def test_row_order_changes_the_fold():
    """Positional indices make row order load-bearing — a reordered gold file
    is a different split, which is exactly why `load_gold_labels` must not sort."""
    gold = _fixture_gold()
    shuffled = gold.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert gold_test_folds(gold).gate != gold_test_folds(shuffled).gate


def test_strict_holdout_is_a_subset_of_the_gate_fold():
    folds = gold_test_folds(_fixture_gold())
    assert folds.strict <= folds.gate


def test_strict_holdout_excludes_every_matcher_training_pair():
    """The whole point: no pair the matcher was fit on survives into `strict`."""
    folds = gold_test_folds(_fixture_gold())
    matcher_train = folds.matcher_population - folds.matcher
    assert not (folds.strict & matcher_train)


def test_strict_holdout_keeps_pairs_the_matcher_never_saw():
    """Confident non-matches were dropped by the matcher's `keep_mask`, so a
    gate-held-out pair outside its population is safe and must be retained."""
    folds = gold_test_folds(_fixture_gold())
    never_seen = folds.gate - folds.matcher_population
    assert never_seen and never_seen <= folds.strict


def test_pair_keys_are_canonicalized():
    gold = _fixture_gold(50)
    gold.loc[0, ["PATID_A", "PATID_B"]] = ["zzz", "aaa"]
    folds = gold_test_folds(gold)
    assert all(a <= b for a, b in folds.gate)


# ── loading ──────────────────────────────────────────────────────────────────
def test_load_gold_coerces_string_booleans(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text(
        "PATID_A,PATID_B,ambiguous_pair,final_gold_label\n"
        "001,002,True,False\n002,003,0,1\n003,004,,TRUE\n"
    )
    gold = load_gold_labels(path)
    assert gold[GOLD_LABEL_COL].tolist() == [False, True, True]
    assert gold[GOLD_AMBIGUOUS_COL].tolist() == [True, False, False]


def test_load_gold_preserves_leading_zeros(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text(
        "PATID_A,PATID_B,ambiguous_pair,final_gold_label\n007,0042,False,True\n"
    )
    gold = load_gold_labels(path)
    assert gold["PATID_A"].iloc[0] == "007"
    assert gold["PATID_B"].iloc[0] == "0042"


# ── the real file (VM only) ──────────────────────────────────────────────────
@pytest.mark.skipif(not _GOLD.exists(), reason="gold labels are VM-only PHI")
def test_real_gold_population_matches_the_notebooks():
    gold = load_gold_labels(_GOLD)
    plausible = int((gold[GOLD_LABEL_COL] | gold[GOLD_AMBIGUOUS_COL]).sum())
    assert len(gold) == GOLD_ROWS
    assert plausible == GOLD_PLAUSIBLE


@pytest.mark.skipif(not _GOLD.exists(), reason="gold labels are VM-only PHI")
def test_real_gold_fold_sizes_match_the_notebooks():
    folds = gold_test_folds(load_gold_labels(_GOLD))
    assert len(folds.gate) == math.ceil(GOLD_ROWS * TEST_SIZE) == 40_961
    assert len(folds.matcher) == math.ceil(GOLD_PLAUSIBLE * TEST_SIZE) == 12_522
    # Enough leakage-safe pairs left to make the end-to-end number meaningful.
    assert len(folds.strict) > 30_000
