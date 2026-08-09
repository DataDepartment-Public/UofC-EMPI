"""Unit tests for src.models.fs_matcher.train's reviewer-label merge
(`_merge_reviewer_labels`) -- reviewer-confirmed labels
(scripts/export_reviewer_labels.py's output) should win over the primary
label source for any pair present in both.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.fs_matcher.train import _merge_reviewer_labels


def _write_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_no_reviewer_labels_is_a_no_op():
    silver = pd.DataFrame({"PATID_A": ["P1"], "PATID_B": ["P2"], "silver_label": [1]})
    out = _merge_reviewer_labels(silver, None, "silver_label")
    pd.testing.assert_frame_equal(out, silver)


def test_reviewer_labels_are_added(tmp_path):
    silver = pd.DataFrame({"PATID_A": ["P1"], "PATID_B": ["P2"], "silver_label": [1]})
    reviewer_path = _write_csv(tmp_path / "reviewer.csv", [
        {"PATID_A": "P4", "PATID_B": "P5", "reviewer_label": 0},
    ])
    out = _merge_reviewer_labels(silver, reviewer_path, "silver_label")
    got = {(r.PATID_A, r.PATID_B, r.silver_label) for r in out.itertuples(index=False)}
    assert got == {("P1", "P2", 1), ("P4", "P5", 0)}


def test_reviewer_labels_win_on_conflicting_pair(tmp_path):
    """The same pair, disagreeing labels -- reviewer confirmation is
    higher-trust and should override the silver-label proxy."""
    silver = pd.DataFrame({"PATID_A": ["P1"], "PATID_B": ["P2"], "silver_label": [1]})
    reviewer_path = _write_csv(tmp_path / "reviewer.csv", [
        {"PATID_A": "P1", "PATID_B": "P2", "reviewer_label": 0},
    ])
    out = _merge_reviewer_labels(silver, reviewer_path, "silver_label")
    assert len(out) == 1
    assert out.iloc[0]["silver_label"] == 0


def test_pair_order_does_not_matter_for_conflict_detection(tmp_path):
    """Silver has (P1, P2); reviewer has (P2, P1) -- must still be
    recognized as the same pair and resolved, not treated as distinct rows."""
    silver = pd.DataFrame({"PATID_A": ["P1"], "PATID_B": ["P2"], "silver_label": [1]})
    reviewer_path = _write_csv(tmp_path / "reviewer.csv", [
        {"PATID_A": "P2", "PATID_B": "P1", "reviewer_label": 0},
    ])
    out = _merge_reviewer_labels(silver, reviewer_path, "silver_label")
    assert len(out) == 1
    assert out.iloc[0]["silver_label"] == 0
    assert (out.iloc[0]["PATID_A"], out.iloc[0]["PATID_B"]) == ("P1", "P2")
