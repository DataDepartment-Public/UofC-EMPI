"""Unit tests for the FS gate that filters the ML input pool
(`src.pipeline._fs_plausible_pool`).

FS acts as the non-match gate: pairs it ranks `no_match` are dropped, and only
the plausible survivors reach the ML matcher (Stage 4.5). Passthrough columns
must survive because the result comes from `non_matches`, not the FS frame.
"""

from __future__ import annotations

import pandas as pd

from src.contracts import TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH
from src.pipeline import _fs_plausible_pool


def _non_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PATID_A": ["A", "A", "A", "B"],
            "PATID_B": ["B", "C", "D", "C"],
            "source_blocks": ["B1", "B3", "B4", "B5"],
            "n_blocks": [1, 2, 1, 3],
        }
    )


def _eval_fs(tiers: list[str]) -> pd.DataFrame:
    nm = _non_matches()
    return pd.DataFrame(
        {
            "PATID_A": nm["PATID_A"],
            "PATID_B": nm["PATID_B"],
            "model_name": "fs_matcher",
            "score": [0.99, 0.55, 0.10, 0.02],
            "predicted_tier": tiers,
        }
    )


def test_drops_only_no_match_pairs():
    eval_fs = _eval_fs([TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH, TIER_NO_MATCH])
    out = _fs_plausible_pool(_non_matches(), eval_fs)
    # A-B (auto_merge) and A-C (human_review) survive; A-D and B-C (no_match) drop.
    assert len(out) == 2
    assert set(zip(out["PATID_A"], out["PATID_B"])) == {("A", "B"), ("A", "C")}


def test_preserves_passthrough_columns():
    eval_fs = _eval_fs([TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH, TIER_NO_MATCH])
    out = _fs_plausible_pool(_non_matches(), eval_fs)
    assert list(out.columns) == ["PATID_A", "PATID_B", "source_blocks", "n_blocks"]
    ab = out[(out.PATID_A == "A") & (out.PATID_B == "B")].iloc[0]
    assert ab["source_blocks"] == "B1" and ab["n_blocks"] == 1


def test_all_no_match_yields_empty_pool():
    eval_fs = _eval_fs([TIER_NO_MATCH] * 4)
    out = _fs_plausible_pool(_non_matches(), eval_fs)
    assert out.empty
    assert list(out.columns) == ["PATID_A", "PATID_B", "source_blocks", "n_blocks"]


def test_none_dropped_when_all_plausible():
    eval_fs = _eval_fs([TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_HUMAN_REVIEW, TIER_AUTO_MERGE])
    out = _fs_plausible_pool(_non_matches(), eval_fs)
    assert len(out) == 4
