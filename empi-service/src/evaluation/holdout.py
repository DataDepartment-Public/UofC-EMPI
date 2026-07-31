"""The project's single evaluation holdout — one function, called by everyone.

**Why this exists.** The Stage-4.25 gate and the Stage-4.5 ML matcher are both
trained on `data/gold_labels/final_gold_labels_v1_2026_07_05.csv` — the same
file we want to score the pipeline against. Roughly 80% of the gold pairs are
training data for those stages, so scoring the end-to-end pipeline on all
204,805 pairs reports a number that is substantially memorized.

Both training notebooks and both evaluation CLIs call `holdout_mask` /
`holdout_keys` here. There is exactly one definition of the split, so the two
models are held out on **the same ~41k pairs**, and an evaluation restricted to
them is leakage-free.

HOW THE SPLIT IS DEFINED
------------------------
A pair is in the holdout iff `sha256(PATID_A|PATID_B|salt)`, read as a fraction
in [0, 1), falls below `HOLDOUT_FRACTION`. Three properties follow, and each one
is load-bearing:

* **Order-invariant.** Membership depends only on the pair's own identity, not
  on its row position. `train_test_split` is positional, so re-saving the gold
  CSV in a different row order silently changes which pairs are held out —
  invalidating every already-trained model with nothing raised. That cannot
  happen here.
* **Stable under growth.** Adding or removing gold rows does not move any
  existing pair. Appending 10k labels re-partitions nothing.
* **Stratified for free.** The hash is independent of the label, so it takes
  ~20% of *every* class without being told the classes exist. Measured on the
  real marginals: non-match 19.98%, ambiguous 20.18%, match 19.71%.

`hashlib` — not the builtin `hash()`, which is salted per process and would give
a different split on every run.

WHAT WENT WRONG BEFORE
----------------------
Each notebook used to call `train_test_split` itself, and this module
*reconstructed* those splits by re-deriving each notebook's seed, ratio,
stratification target and row order. Two failures came out of that duplication,
and the design here answers both:

1. **It drifted silently.** When the matcher's notebook changed its
   stratification target, this module kept returning *a* fold — just not the one
   the model was actually held out on. Nothing raised; every "leakage-safe"
   number downstream was simply wrong. One function called by every site cannot
   disagree with itself.

2. **The safe set was tiny.** The two notebooks drew *different* 20% folds, so
   "pairs neither model saw" was their intersection: ~8k of 204,805, about 4% —
   and four fifths of each model's own test set was the other model's training
   data. One shared split makes it the full ~41k, with ~5x the positive labels
   (~12.5k vs ~2.5k).

PROVENANCE
----------
Changing `HOLDOUT_SALT` or `HOLDOUT_FRACTION` redefines the split and
invalidates every model already trained against it. Each notebook records
`holdout_spec()` in its model `.meta.json`; `verify_model_provenance` compares
that against the current spec so a model fit under a different definition is
reported rather than silently trusted. That check is the tripwire the previous
design lacked.

PHI / HIPAA: hashes PATIDs in memory, returns pair keys to the caller (needed to
filter), and logs counts only.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from src.evaluation.cluster_eval import PairKey, canonical_key, pair_keys

logger = logging.getLogger(__name__)

__all__ = [
    "GOLD_AMBIGUOUS_COL",
    "GOLD_LABEL_COL",
    "HOLDOUT_FRACTION",
    "HOLDOUT_SALT",
    "HOLDOUT_VERSION",
    "holdout_bucket",
    "holdout_keys",
    "holdout_mask",
    "holdout_spec",
    "is_holdout",
    "load_gold_labels",
    "verify_model_provenance",
]

GOLD_LABEL_COL = "final_gold_label"
GOLD_AMBIGUOUS_COL = "ambiguous_pair"

#: The split's definition. **Changing either value redefines which pairs are
#: held out and invalidates every model already trained against it** — both
#: notebooks must be re-run. `verify_model_provenance` catches the mismatch.
HOLDOUT_SALT = "empi-holdout-v1"
HOLDOUT_FRACTION = 0.20
#: Bumped whenever the *mechanism* changes (not the salt/fraction), so an old
#: model's recorded spec can be told apart from a merely re-salted one.
HOLDOUT_VERSION = 1


def holdout_bucket(patid_a: str, patid_b: str) -> float:
    """Deterministic uniform value in [0, 1) for a pair, independent of row
    order and of the pair's label. Canonicalized so (a, b) and (b, a) agree."""
    a, b = canonical_key(str(patid_a), str(patid_b))
    digest = hashlib.sha256(f"{a}|{b}|{HOLDOUT_SALT}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def is_holdout(patid_a: str, patid_b: str) -> bool:
    """Whether one pair belongs to the evaluation holdout."""
    return holdout_bucket(patid_a, patid_b) < HOLDOUT_FRACTION


def holdout_mask(
    pairs: pd.DataFrame, a_col: str = "PATID_A", b_col: str = "PATID_B"
) -> np.ndarray:
    """Boolean mask over `pairs` rows — True where the pair is held out.

    This is what the training notebooks call: `~mask` is their train+val
    population, `mask` is the test fold they must never fit on.
    """
    mask = np.fromiter(
        (is_holdout(a, b) for a, b in zip(pairs[a_col], pairs[b_col])),
        dtype=bool,
        count=len(pairs),
    )
    logger.info(
        "Holdout: %d/%d pairs (%.2f%%)", int(mask.sum()), len(mask),
        100.0 * mask.mean() if len(mask) else 0.0,
    )
    return mask


def holdout_keys(
    pairs: pd.DataFrame, a_col: str = "PATID_A", b_col: str = "PATID_B"
) -> set[PairKey]:
    """The held-out pairs of `pairs`, as canonicalized keys.

    This is what the evaluation CLIs call, to restrict scoring to pairs no
    served model was fit on.
    """
    mask = holdout_mask(pairs, a_col, b_col)
    return set(pair_keys(pairs.loc[mask], a_col, b_col))


def holdout_spec() -> dict:
    """The split's definition, for a model's `.meta.json`.

    Recorded at training time and compared at evaluation time, so a model fit
    under a different definition of "held out" is reported rather than trusted.
    """
    return {
        "method": "sha256_bucket",
        "salt": HOLDOUT_SALT,
        "fraction": HOLDOUT_FRACTION,
        "version": HOLDOUT_VERSION,
    }


def _to_bool(s: pd.Series) -> pd.Series:
    """Boolean coercion copied verbatim from the notebooks' load cell, so a
    label frame means the same thing here as it does during training."""
    if s.dtype == bool:
        return s
    mapped = s.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    )
    return mapped.where(mapped.notna(), False).astype(bool)


def load_gold_labels(path) -> pd.DataFrame:
    """Read the gold CSV the way every notebook reads it.

    `dtype=str` on the PATID columns is the leading-zeros invariant. Row order
    no longer matters to the split, but it is still not sorted here — the
    notebooks join features positionally against this frame.
    """
    gold = pd.read_csv(path, dtype={"PATID_A": str, "PATID_B": str})
    for col in (GOLD_LABEL_COL, GOLD_AMBIGUOUS_COL):
        if col in gold.columns:
            gold[col] = _to_bool(gold[col])
    return gold


def verify_model_provenance(model_metas: dict[str, dict | None]) -> list[str]:
    """Check that each served model was trained under the current split spec.

    `model_metas` maps a model name to its `.meta.json` contents (None when the
    model or its sidecar is absent). Returns human-readable problems — empty
    when every model records the current spec.

    A mismatch means the leakage-safe restriction is a fiction for that model:
    it was fit under a different definition of "held out", so the holdout may be
    its training data. Callers should surface this, not swallow it.
    """
    current = holdout_spec()
    problems: list[str] = []
    for name, meta in model_metas.items():
        if meta is None:
            problems.append(f"{name}: no meta sidecar — cannot verify its training split")
            continue
        recorded = meta.get("holdout_spec")
        if recorded is None:
            problems.append(
                f"{name}: trained before the shared holdout existed "
                f"(no holdout_spec in its meta) — re-run its notebook"
            )
        elif recorded != current:
            problems.append(
                f"{name}: trained under a DIFFERENT holdout spec "
                f"({recorded} vs current {current}) — re-run its notebook"
            )
    return problems
