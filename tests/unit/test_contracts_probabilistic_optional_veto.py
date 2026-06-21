"""Phase E2-1 contract change: `veto_reason` is now optional in ProbabilisticMatches.

This test pins both behaviors so the enhanced (with veto) and enhanced_2
(without veto) outputs both validate against the same schema.
"""

from __future__ import annotations

import pandas as pd

from src.contracts import ProbabilisticMatches, validate


def _base_frame(with_veto: bool = False, veto_value: str | None = None) -> pd.DataFrame:
    base = {
        "PATID_A": ["P1"],
        "PATID_B": ["P2"],
        "match_source": ["model"],
        "score": [0.7],
        "match_weight": [1.0],
        "classification_tier": ["human_review"],
        "source_blocks": ["B1"],
        "n_blocks": [1],
    }
    if with_veto:
        base["veto_reason"] = [veto_value]
    return pd.DataFrame(base)


def test_validates_without_veto_reason_column():
    """enhanced_2 output: no veto column at all. Must validate."""
    df = _base_frame(with_veto=False)
    out = validate(df, ProbabilisticMatches)
    assert "veto_reason" not in out.columns


def test_validates_with_null_veto_reason():
    """enhanced output with no veto applied: column present, all null."""
    df = _base_frame(with_veto=True, veto_value=None)
    out = validate(df, ProbabilisticMatches)
    assert "veto_reason" in out.columns
    assert pd.isna(out["veto_reason"].iloc[0])


def test_validates_with_populated_veto_reason():
    """enhanced output with a veto applied."""
    df = _base_frame(with_veto=True, veto_value="ssn_conflict")
    out = validate(df, ProbabilisticMatches)
    assert out["veto_reason"].iloc[0] == "ssn_conflict"
