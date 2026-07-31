"""Guards on the shared evaluation holdout (`src/evaluation/holdout.py`).

One function defines the split and every site calls it — both training
notebooks and both evaluation CLIs. The properties that matter:

* **order-invariance** — the previous positional `train_test_split` meant a
  re-saved gold CSV silently changed which pairs were held out, invalidating
  already-trained models with nothing raised;
* **stability under growth** — appending labels must not re-partition existing
  pairs, or every retrain silently invalidates the last evaluation;
* **stratification** — the hash is label-independent, so it must take ~the same
  share of every class without being told the classes exist;
* **provenance** — a model fit under a different spec must be reported, because
  for it the "leakage-safe" restriction is a fiction.

`test_real_gold_*` needs the VM's PHI label file and skips cleanly elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.holdout import (
    GOLD_AMBIGUOUS_COL,
    GOLD_LABEL_COL,
    HOLDOUT_FRACTION,
    holdout_bucket,
    holdout_keys,
    holdout_mask,
    holdout_spec,
    is_holdout,
    load_gold_labels,
    verify_model_provenance,
)

_GOLD = (
    Path(__file__).resolve().parents[3]
    / "data" / "gold_labels" / "final_gold_labels_v1_2026_07_05.csv"
)
GOLD_ROWS = 204_805
GOLD_PLAUSIBLE = 62_610


def _fixture_gold(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "PATID_A": [f"{i:06d}" for i in range(n)],
        "PATID_B": [f"{i + n:06d}" for i in range(n)],
        "ambiguous_pair": rng.random(n) < 0.14,
        "final_gold_label": rng.random(n) < 0.17,
    })


# ── the split's definition ───────────────────────────────────────────────────
def test_bucket_is_deterministic_across_calls():
    """hashlib, not the builtin `hash()` — which is salted per process and would
    give a different split on every run."""
    assert holdout_bucket("000123", "004567") == holdout_bucket("000123", "004567")


def test_bucket_is_canonical_in_pair_order():
    assert holdout_bucket("aaa", "bbb") == holdout_bucket("bbb", "aaa")


def test_bucket_is_uniform_enough():
    gold = _fixture_gold()
    buckets = np.array([holdout_bucket(a, b) for a, b in zip(gold.PATID_A, gold.PATID_B)])
    assert 0.0 <= buckets.min() and buckets.max() < 1.0
    assert abs(buckets.mean() - 0.5) < 0.02


def test_holdout_is_the_configured_fraction():
    mask = holdout_mask(_fixture_gold())
    assert abs(mask.mean() - HOLDOUT_FRACTION) < 0.01


def test_membership_is_invariant_to_row_order():
    """THE property positional splitting lacked: re-saving gold in a different
    order must not move a single pair."""
    gold = _fixture_gold()
    shuffled = gold.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert holdout_keys(gold) == holdout_keys(shuffled)


def test_membership_is_stable_when_rows_are_added():
    """Appending new gold labels must not re-partition the existing ones."""
    gold = _fixture_gold(1_000)
    grown = pd.concat([gold, _fixture_gold(1_500).iloc[1_000:]], ignore_index=True)
    before, after = holdout_keys(gold), holdout_keys(grown)
    assert before <= after


def test_membership_is_independent_of_the_label():
    """Stratification is free only because the hash ignores the label. If a
    label correction moved a pair, the previous evaluation would be invalid."""
    gold = _fixture_gold(2_000)
    flipped = gold.assign(final_gold_label=~gold.final_gold_label,
                          ambiguous_pair=~gold.ambiguous_pair)
    assert holdout_keys(gold) == holdout_keys(flipped)


def test_every_class_is_sampled_at_the_same_rate():
    gold = _fixture_gold()
    mask = holdout_mask(gold)
    strata = np.where(gold.ambiguous_pair, "ambiguous",
                      np.where(gold.final_gold_label, "match", "non-match"))
    for cls in ("non-match", "ambiguous", "match"):
        in_cls = strata == cls
        assert in_cls.sum() > 0
        assert abs(mask[in_cls].mean() - HOLDOUT_FRACTION) < 0.03


def test_holdout_keys_are_canonicalized():
    gold = _fixture_gold(500)
    gold.loc[0, ["PATID_A", "PATID_B"]] = ["zzz", "aaa"]
    assert all(a <= b for a, b in holdout_keys(gold))


def test_mask_and_keys_agree():
    gold = _fixture_gold(2_000)
    mask = holdout_mask(gold)
    assert len(holdout_keys(gold)) == int(mask.sum())
    assert all(is_holdout(a, b) for a, b in holdout_keys(gold))


# ── provenance ───────────────────────────────────────────────────────────────
def test_provenance_passes_when_both_models_record_the_current_spec():
    spec = holdout_spec()
    metas = {"ml_matcher": {"holdout_spec": spec}, "nonmatch_gate": {"holdout_spec": spec}}
    assert verify_model_provenance(metas) == []


def test_provenance_flags_a_model_trained_before_the_shared_holdout():
    problems = verify_model_provenance({"ml_matcher": {"notes": "old"}})
    assert len(problems) == 1 and "re-run its notebook" in problems[0]


def test_provenance_flags_a_different_salt_or_fraction():
    """Changing either redefines which pairs are held out, so every model
    trained under the old spec is no longer held out on them."""
    stale = {**holdout_spec(), "salt": "something-else"}
    problems = verify_model_provenance({"nonmatch_gate": {"holdout_spec": stale}})
    assert len(problems) == 1 and "DIFFERENT holdout spec" in problems[0]


def test_provenance_flags_a_missing_sidecar():
    problems = verify_model_provenance({"ml_matcher": None})
    assert len(problems) == 1 and "no meta sidecar" in problems[0]


# ── gold loading ─────────────────────────────────────────────────────────────
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
    assert len(gold) == GOLD_ROWS
    assert int((gold[GOLD_LABEL_COL] | gold[GOLD_AMBIGUOUS_COL]).sum()) == GOLD_PLAUSIBLE


@pytest.mark.skipif(not _GOLD.exists(), reason="gold labels are VM-only PHI")
def test_real_holdout_is_about_a_fifth_of_the_file():
    """~41k — not the ~8k the two independently-drawn folds used to intersect to."""
    gold = load_gold_labels(_GOLD)
    n = len(holdout_keys(gold))
    assert abs(n / GOLD_ROWS - HOLDOUT_FRACTION) < 0.01
    assert 39_000 < n < 43_000
