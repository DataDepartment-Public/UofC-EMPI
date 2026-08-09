"""Graph meta-blocking (ARCS weighting + Cardinality Node Pruning) for the eMPI
stacked blocker.

Blocking over-generates candidate pairs by design. Meta-blocking treats the
candidate set as a graph (records = nodes, candidate pairs = edges), weights each
edge by how *discriminating* the blocks that produced it are, and prunes the
weak edges — cutting the comparison load the downstream rules / probabilistic
stage must carry, with negligible recall loss.

Two pieces:

* **ARCS weight** (Aggregate Reciprocal Comparisons Scheme): an edge's weight is
  the sum over the blocks that produced it of ``1 / (pairs that block emitted)``.
  Edges from small, specific blocks (a shared SSN) score high; edges from huge,
  coarse blocks (a common Soundex + birth year) score low. The typo-tolerant
  q-gram pass carries no block-cardinality signal, so its pairs get a flat
  configured weight (`settings.cnp_qgram_only_weight`).

* **Cardinality Node Pruning (CNP)**: keep an edge if it ranks in the top-k
  ARCS-weighted edges of **either** endpoint (reciprocal-OR). This bounds each
  record's comparisons to ~k of its strongest links.

Validated on run `real_20260620`: CNP top-10 over (8-block ∪ q-gram) cuts the
candidate set ~46.7% vs 8-block alone while retaining 99.96% of silver-True pairs.
See `stacked_blocking.run_stacked_blocking` and `docs/Blocking-Guide.md`.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from src.config import settings
from src.preprocessing.qgram_blocking import QGRAM_SOURCE_LABEL

logger = logging.getLogger(__name__)

Pair = tuple[str, str]


def arcs_weights(
    candidate_pairs: pd.DataFrame,
    qgram_only_label: str = QGRAM_SOURCE_LABEL,
    qgram_only_weight: float | None = None,
) -> dict[Pair, float]:
    """ARCS edge weight per candidate pair.

    Weight = sum over the pair's *block* sources of ``1 / comparisons-in-block``.
    Pairs whose only source is the q-gram pass (no real block) get the flat
    `qgram_only_weight` (default `settings.cnp_qgram_only_weight`), mirroring the
    research stacking recipe.
    """
    if qgram_only_weight is None:
        qgram_only_weight = settings.cnp_qgram_only_weight

    comps: dict[str, int] = defaultdict(int)
    for source in candidate_pairs["source_blocks"]:
        for blk in str(source).split("|"):
            if blk and blk != qgram_only_label:
                comps[blk] += 1

    weights: dict[Pair, float] = {}
    for a, b, source in zip(
        candidate_pairs["PATID_A"],
        candidate_pairs["PATID_B"],
        candidate_pairs["source_blocks"],
    ):
        real = [
            blk for blk in str(source).split("|")
            if blk and blk != qgram_only_label
        ]
        weights[(a, b)] = (
            sum(1.0 / comps[blk] for blk in real) if real else qgram_only_weight
        )
    return weights


def cnp_prune(weights: dict[Pair, float], k: int) -> set[Pair]:
    """Cardinality Node Pruning: keep an edge if it is in the top-k weighted edges
    of *either* endpoint (reciprocal-OR). Ties break deterministically by pair."""
    incident: dict[str, list[tuple[float, Pair]]] = defaultdict(list)
    for (a, b), wt in weights.items():
        incident[a].append((wt, (a, b)))
        incident[b].append((wt, (a, b)))

    keep: set[Pair] = set()
    for edges in incident.values():
        edges.sort(reverse=True)
        for _, pair in edges[:k]:
            keep.add(pair)
    return keep


def prune_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    k: int | None = None,
    qgram_only_weight: float | None = None,
) -> pd.DataFrame:
    """ARCS-weight then CNP-prune a candidate-pair frame.

    Returns the subset of `candidate_pairs` (all columns preserved) whose
    `(PATID_A, PATID_B)` survives CNP top-`k` pruning.
    """
    k = settings.cnp_top_k if k is None else k
    if candidate_pairs.empty:
        return candidate_pairs.reset_index(drop=True).copy()

    weights = arcs_weights(candidate_pairs, qgram_only_weight=qgram_only_weight)
    keep = cnp_prune(weights, k)
    mask = [
        (a, b) in keep
        for a, b in zip(candidate_pairs["PATID_A"], candidate_pairs["PATID_B"])
    ]
    pruned = candidate_pairs[pd.Series(mask, index=candidate_pairs.index)]
    logger.info(
        "Meta-blocking (CNP top-%d): %d -> %d candidate pairs (%.1f%%).",
        k, len(candidate_pairs), len(pruned),
        100 * (len(pruned) / len(candidate_pairs) - 1) if len(candidate_pairs) else 0,
    )
    return pruned.reset_index(drop=True)
