"""Unit tests for graph meta-blocking (src/preprocessing/meta_blocking.py)."""

import pandas as pd

from src.preprocessing.meta_blocking import (
    arcs_weights,
    cnp_prune,
    prune_candidate_pairs,
)


def _pairs(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"PATID_A": a, "PATID_B": b, "source_blocks": s, "n_blocks": len(s.split("|"))}
         for a, b, s in rows]
    )


class TestArcsWeights:
    def test_rarer_block_weighs_more(self):
        # Block B1 emits one pair (weight 1.0); B5 emits two (weight 0.5 each).
        df = _pairs([("A", "B", "B1"), ("C", "D", "B5"), ("E", "F", "B5")])
        w = arcs_weights(df)
        assert w[("A", "B")] == 1.0
        assert w[("C", "D")] == 0.5

    def test_multiple_blocks_sum(self):
        # A pair from two blocks sums their reciprocal-comparison weights.
        df = _pairs([("A", "B", "B1|B5"), ("C", "D", "B5")])
        w = arcs_weights(df)
        # B1 emits 1 pair (1.0), B5 emits 2 pairs (0.5) → 1.5 for the A-B pair.
        assert w[("A", "B")] == 1.5

    def test_qgram_only_gets_flat_weight(self):
        df = _pairs([("A", "B", "QGRAM"), ("C", "D", "B5"), ("E", "F", "B5")])
        w = arcs_weights(df, qgram_only_weight=0.5)
        assert w[("A", "B")] == 0.5  # flat, not derived from a block cardinality

    def test_qgram_token_ignored_in_block_cardinality(self):
        # A pair from B3 + QGRAM is weighted by B3 only; QGRAM does not count as a
        # block in the cardinality denominator.
        df = _pairs([("A", "B", "B3|QGRAM"), ("C", "D", "B3")])
        w = arcs_weights(df, qgram_only_weight=0.5)
        assert w[("A", "B")] == 0.5  # B3 emits 2 pairs → 1/2


class TestCnpPrune:
    def test_keeps_top_k_per_node(self):
        # Node A has three incident edges of descending weight; top-1 keeps only
        # the heaviest, but the lighter edges may survive via their OTHER endpoint.
        weights = {("A", "B"): 3.0, ("A", "C"): 2.0, ("A", "D"): 1.0}
        keep = cnp_prune(weights, k=1)
        # A keeps its top-1 (A-B). B/C/D each have only one edge, so top-1 keeps it.
        assert ("A", "B") in keep
        assert keep == {("A", "B"), ("A", "C"), ("A", "D")}

    def test_prunes_weak_edge_rejected_by_both_endpoints(self):
        # The A-D edge is each endpoint's weakest; with k=1 and a competing
        # stronger edge on both A and D, it is pruned.
        weights = {
            ("A", "B"): 3.0,   # A's strongest
            ("A", "D"): 0.5,   # weak on both ends
            ("D", "E"): 3.0,   # D's strongest
        }
        keep = cnp_prune(weights, k=1)
        assert ("A", "D") not in keep
        assert ("A", "B") in keep and ("D", "E") in keep

    def test_deterministic_on_ties(self):
        weights = {("A", "B"): 1.0, ("A", "C"): 1.0}
        assert cnp_prune(weights, k=1) == cnp_prune(weights, k=1)


class TestPruneCandidatePairs:
    def test_preserves_columns_and_subsets(self):
        df = _pairs([("A", "B", "B1"), ("A", "D", "B5"), ("D", "E", "B1")])
        out = prune_candidate_pairs(df, k=10)
        assert list(out.columns) == ["PATID_A", "PATID_B", "source_blocks", "n_blocks"]
        assert len(out) <= len(df)

    def test_empty_frame(self):
        empty = pd.DataFrame(columns=["PATID_A", "PATID_B", "source_blocks", "n_blocks"])
        assert prune_candidate_pairs(empty, k=10).empty

    def test_large_k_keeps_everything(self):
        df = _pairs([("A", "B", "B1"), ("C", "D", "B5"), ("E", "F", "B5")])
        out = prune_candidate_pairs(df, k=1000)
        assert len(out) == len(df)
