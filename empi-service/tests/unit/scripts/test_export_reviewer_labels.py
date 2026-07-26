"""Unit tests for scripts/export_reviewer_labels.py's pair-derivation logic.

`scripts/` has no `__init__.py` (it's standalone CLI tools, not a package),
so the module under test is loaded by file path rather than imported
normally -- mirrors how the script itself is actually invoked
(`python scripts/export_reviewer_labels.py`), not as a package member.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "export_reviewer_labels.py"
_spec = importlib.util.spec_from_file_location("export_reviewer_labels", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
export_reviewer_labels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_reviewer_labels)

derive_labeled_pairs = export_reviewer_labels.derive_labeled_pairs
canonical_pair = export_reviewer_labels._canonical_pair


def _row(action, ts, patids, mid="M-1", related_patids=None):
    return {
        "action": action, "ts_utc": ts, "patids": patids, "mid": mid,
        "related_patids": related_patids,
    }


def _pairs_as_set(df):
    return {(r.PATID_A, r.PATID_B, r.reviewer_label) for r in df.itertuples(index=False)}


def test_canonical_pair_is_order_independent():
    assert canonical_pair("P2", "P1") == canonical_pair("P1", "P2") == ("P1", "P2")


def test_merge_generates_positive_pairs_across_new_and_prior_members():
    rows = [_row("merge", "2026-01-01T00:00:00Z", "P3", related_patids="P1,P2")]
    df, skipped = derive_labeled_pairs(rows)
    assert skipped == 0
    assert _pairs_as_set(df) == {
        ("P1", "P2", 1), ("P1", "P3", 1), ("P2", "P3", 1),
    }


def test_merge_with_multiple_new_patids_pairs_everyone():
    rows = [_row("merge", "2026-01-01T00:00:00Z", "P4,P5", related_patids="P1,P2")]
    df, _ = derive_labeled_pairs(rows)
    assert _pairs_as_set(df) == {
        ("P1", "P2", 1), ("P1", "P4", 1), ("P1", "P5", 1),
        ("P2", "P4", 1), ("P2", "P5", 1), ("P4", "P5", 1),
    }


def test_merge_with_no_prior_members_still_works():
    """First-ever merge into a previously-singleton mid -- related_patids
    would be empty/None (no prior members to snapshot)."""
    rows = [_row("merge", "2026-01-01T00:00:00Z", "P1,P2", related_patids=None)]
    df, skipped = derive_labeled_pairs(rows)
    assert skipped == 0
    assert _pairs_as_set(df) == {("P1", "P2", 1)}


def test_unmerge_generates_negative_pairs_against_who_stayed():
    rows = [_row("unmerge", "2026-01-01T00:00:00Z", "P2", related_patids="P1")]
    df, skipped = derive_labeled_pairs(rows)
    assert skipped == 0
    assert _pairs_as_set(df) == {("P1", "P2", 0)}


def test_unmerge_against_multiple_remaining_members():
    rows = [_row("unmerge", "2026-01-01T00:00:00Z", "P4", related_patids="P1,P2,P3")]
    df, _ = derive_labeled_pairs(rows)
    assert _pairs_as_set(df) == {
        ("P1", "P4", 0), ("P2", "P4", 0), ("P3", "P4", 0),
    }


def test_unmerge_with_no_related_patids_is_skipped_not_guessed():
    """Pre-migration rows (before related_patids existed) can't be
    reconstructed from entity_member's current-state-only table -- dropped,
    with the skip counted so it's visible how much history is unusable."""
    rows = [_row("unmerge", "2026-01-01T00:00:00Z", "P2", related_patids=None)]
    df, skipped = derive_labeled_pairs(rows)
    assert skipped == 1
    assert df.empty


def test_dismiss_generates_a_direct_negative_pair():
    rows = [_row("dismiss", "2026-01-01T00:00:00Z", "P4,P5")]
    df, skipped = derive_labeled_pairs(rows)
    assert skipped == 0
    assert _pairs_as_set(df) == {("P4", "P5", 0)}


def test_malformed_dismiss_row_is_skipped_not_crashed():
    rows = [_row("dismiss", "2026-01-01T00:00:00Z", "P4")]  # only one patid
    df, skipped = derive_labeled_pairs(rows)
    assert skipped == 0
    assert df.empty


def test_split_action_is_ignored():
    """Declared in AuditLogRow's Literal but never emitted by audit.py --
    should be silently skipped, not crash on an unknown action."""
    rows = [_row("split", "2026-01-01T00:00:00Z", "P1,P2")]
    df, skipped = derive_labeled_pairs(rows)
    assert skipped == 0
    assert df.empty


def test_conflicting_events_resolve_last_write_wins_by_timestamp():
    """A pair confirmed by an earlier dismiss (negative) then later merged
    (positive) -- the more recent event should win, regardless of list
    order in the input."""
    rows = [
        _row("merge", "2026-01-02T00:00:00Z", "P2", related_patids="P1"),
        _row("dismiss", "2026-01-01T00:00:00Z", "P1,P2"),
    ]
    df, _ = derive_labeled_pairs(rows)
    assert _pairs_as_set(df) == {("P1", "P2", 1)}

    # Same events, reverse order in the input -- result must be identical,
    # since resolution is by ts_utc, not input order.
    df_reversed, _ = derive_labeled_pairs(list(reversed(rows)))
    assert _pairs_as_set(df_reversed) == {("P1", "P2", 1)}


def test_output_columns_match_silver_label_shape():
    rows = [_row("dismiss", "2026-01-01T00:00:00Z", "P4,P5")]
    df, _ = derive_labeled_pairs(rows)
    assert list(df.columns) == ["PATID_A", "PATID_B", "reviewer_label"]
