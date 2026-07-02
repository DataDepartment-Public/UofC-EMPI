"""Stacked blocker: 8-block ∪ q-gram → meta-blocking prune.

The production blocking path. It composes the three blocking pieces:

    run_batch_blocking         (src/preprocessing/blocking.py)      8-block scheme
      ∪ run_qgram_blocking     (src/preprocessing/qgram_blocking.py) typo-tolerant
      → prune_candidate_pairs  (src/preprocessing/meta_blocking.py)  CNP/ARCS prune

The q-gram pass adds the last-name-typo pairs every name+DOB block splits apart;
the union recovers them, and meta-blocking then prunes the combined set back down
(the q-gram pass and the union both over-generate). On run `real_20260620` the
stack emits ~109k pairs — about 46.7% fewer than the 8-block scheme alone — while
retaining 99.96% of silver-True pairs and 99.5% of the 8-block recall residual.

Tunable via `settings` (`qgram_*`, `cnp_*`); see `docs/Blocking-Guide.md`. Online
single-record inference still uses the 8-block index (`run_inference_blocking`) —
the q-gram pass needs the full corpus to fit its TF-IDF vocabulary, so it is a
batch-only stage.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.preprocessing.blocking import run_batch_blocking
from src.preprocessing.meta_blocking import prune_candidate_pairs
from src.preprocessing.qgram_blocking import run_qgram_blocking

logger = logging.getLogger(__name__)

_PAIR_COLUMNS = ["PATID_A", "PATID_B", "source_blocks", "n_blocks"]


def union_candidate_pairs(*frames: pd.DataFrame) -> pd.DataFrame:
    """Merge candidate-pair frames on `(PATID_A, PATID_B)`.

    `source_blocks` is the sorted union of each input's pipe-delimited block
    tokens; `n_blocks` is the number of distinct tokens (how many independent
    signals surfaced the pair). Inputs must already use canonical order
    (`PATID_A < PATID_B`); that ordering is preserved.
    """
    present = [f for f in frames if f is not None and not f.empty]
    if not present:
        return pd.DataFrame(columns=_PAIR_COLUMNS)

    combined = pd.concat(
        [f[_PAIR_COLUMNS] for f in present], ignore_index=True
    )
    exploded = combined.assign(
        _tok=combined["source_blocks"].str.split("|")
    ).explode("_tok")
    exploded = exploded[exploded["_tok"].astype(bool)]
    exploded = exploded.drop_duplicates(["PATID_A", "PATID_B", "_tok"])
    exploded = exploded.sort_values(["PATID_A", "PATID_B", "_tok"])

    grouped = exploded.groupby(["PATID_A", "PATID_B"], sort=False)["_tok"]
    out = grouped.agg("|".join).reset_index(name="source_blocks")
    out["n_blocks"] = grouped.size().to_numpy()
    return out[_PAIR_COLUMNS]


def run_stacked_blocking(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Run the full stacked blocker and return the pruned candidate pairs.

    8-block ∪ q-gram → CNP/ARCS prune, all parameters sourced from `settings`.
    Output is the standard candidate-pair schema (`CandidatePairs` contract).
    """
    eight_block = run_batch_blocking(df_clean)
    qgram = run_qgram_blocking(df_clean)
    union = union_candidate_pairs(eight_block, qgram)

    n_qgram_only = len(union) - len(eight_block) if not union.empty else 0
    logger.info(
        "Stacked blocker: 8-block=%d ∪ q-gram=%d → union=%d (+%d q-gram-only)",
        len(eight_block), len(qgram), len(union), max(n_qgram_only, 0),
    )

    pruned = prune_candidate_pairs(union)
    logger.info(
        "Stacked blocker: %d candidate pairs after CNP prune (%.1f%% vs 8-block).",
        len(pruned),
        100 * (len(pruned) / len(eight_block) - 1) if len(eight_block) else 0,
    )
    return pruned
