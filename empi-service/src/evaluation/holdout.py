"""Reconstruct the training notebooks' held-out test folds.

**Why this exists.** The Stage-4.25 gate and the served Stage-4.5 v3 matcher
were both trained on `data/gold_labels/final_gold_labels_v1_2026_07_05.csv` —
the same file we want to score the pipeline against. Their notebooks take a
plain random stratified 60/20/20 split, so ~80% of the gold pairs are *training
data* for those two stages. Scoring the end-to-end pipeline on all 204,805 gold
pairs therefore reports a number that is substantially memorized, and it will
look better than production ever will.

Neither notebook persists its split (only the resulting metrics reach the
`.meta.json`), but the split is fully deterministic and depends on nothing but
the gold label columns and the row order of the CSV: `np.arange(n)` split with
`RANDOM_STATE = 42`, stratified on the target. So it can be reproduced here
exactly, without features and without loading a model, and used to restrict the
evaluation to pairs neither model ever saw.

The two folds are **not the same pairs**: the gate splits the whole 204,805-row
file stratified on `plausible`, while the matcher first drops the confident
non-matches (`keep_mask`) and splits the remaining ~62,610 stratified on
`ambiguous`. `TestFolds.strict` is the intersection-safe choice — pairs held out
by the gate *and* (held out by the matcher or never in the matcher's
population). Use `strict` for the headline end-to-end number.

If either notebook's `RANDOM_STATE`, split ratios, stratification target, or row
ordering ever change, this module silently drifts out of sync — the guard is
`tests/unit/evaluation/test_holdout.py`, which pins the fold sizes and class
balances the notebooks printed.

PHI / HIPAA: returns pair keys to the caller (needed to filter), logs counts only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.cluster_eval import PairKey, pair_keys

logger = logging.getLogger(__name__)

__all__ = ["RANDOM_STATE", "TestFolds", "gold_test_folds", "load_gold_labels"]

#: Must equal `RANDOM_STATE` in both training notebooks (§0 imports cell).
RANDOM_STATE = 42
#: `train_test_split(..., test_size=0.20)` — the notebooks' outer split.
TEST_SIZE = 0.20

GOLD_LABEL_COL = "final_gold_label"
GOLD_AMBIGUOUS_COL = "ambiguous_pair"


@dataclass
class TestFolds:
    """Pair keys each model held out, plus the leakage-safe intersection."""

    gate: set[PairKey]
    matcher: set[PairKey]
    matcher_population: set[PairKey]

    @property
    def strict(self) -> set[PairKey]:
        """Pairs no served model was fit on.

        A pair is safe when the gate held it out AND the matcher either held it
        out too or never had it in its population (the confident non-matches
        `keep_mask` dropped — those the matcher genuinely never saw).
        """
        return {k for k in self.gate if k not in self.matcher_population or k in self.matcher}

    def as_dict(self) -> dict:
        return {
            "gate_test_pairs": len(self.gate),
            "matcher_test_pairs": len(self.matcher),
            "matcher_population_pairs": len(self.matcher_population),
            "strict_holdout_pairs": len(self.strict),
        }


def _to_bool(s: pd.Series) -> pd.Series:
    """Boolean coercion copied verbatim from the notebooks' load cell.

    Reproducing this exactly matters: it decides the stratification target, and
    a different NaN policy would shuffle every row into a different fold.
    """
    if s.dtype == bool:
        return s
    mapped = s.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    )
    # The notebooks write `.fillna(False).astype(bool)`; `.where` is the same
    # result without pandas' object-downcasting FutureWarning.
    return mapped.where(mapped.notna(), False).astype(bool)


def load_gold_labels(path) -> pd.DataFrame:
    """Read the gold CSV the way every notebook reads it — row order intact.

    `dtype=str` on the PATID columns is the leading-zeros invariant; row order
    is load-bearing because the fold indices are positional.
    """
    gold = pd.read_csv(path, dtype={"PATID_A": str, "PATID_B": str})
    for col in (GOLD_LABEL_COL, GOLD_AMBIGUOUS_COL):
        if col in gold.columns:
            gold[col] = _to_bool(gold[col])
    return gold


def _test_keys(frame: pd.DataFrame, y: pd.Series) -> set[PairKey]:
    idx = np.arange(len(frame))
    _, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return set(pair_keys(frame.iloc[idx_test]))


def gold_test_folds(gold: pd.DataFrame) -> TestFolds:
    """Reproduce both notebooks' 20% test folds from the gold frame alone."""
    gold = gold.reset_index(drop=True)

    # Gate (confident_nonmatch/…_nonmatch_gate_v1): whole file, target = plausible.
    y_gate = (gold[GOLD_LABEL_COL] | gold[GOLD_AMBIGUOUS_COL]).astype(int)
    gate = _test_keys(gold, y_gate)

    # Matcher (confident_match/…_ambiguous_v3): keep_mask first, then split on
    # `ambiguous`. `reset_index(drop=True)` mirrors the notebook so the
    # positional indices line up.
    plausible = gold[y_gate.astype(bool)].reset_index(drop=True)
    y_matcher = plausible[GOLD_AMBIGUOUS_COL].astype(int)
    matcher = _test_keys(plausible, y_matcher)

    folds = TestFolds(
        gate=gate, matcher=matcher, matcher_population=set(pair_keys(plausible))
    )
    logger.info(
        "Reconstructed notebook test folds — gate=%d, matcher=%d, strict=%d",
        len(folds.gate), len(folds.matcher), len(folds.strict),
    )
    return folds
