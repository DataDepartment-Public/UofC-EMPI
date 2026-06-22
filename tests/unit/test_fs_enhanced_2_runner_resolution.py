"""Unit tests for the runner's path-resolution + version-tag fallbacks.

The runner supports both naming conventions present in the repo:
- standalone CLI         — `<stem>_v<N>_<YYYY_MM_DD>.parquet`
- pipeline orchestrator  — `<stem>_<run_id>.parquet`  (UTC timestamp)

These tests use a tmp_path fixture so we don't touch real data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models.experiments.fs_splink_enhanced_2.run_real_enhanced_2 import (
    _derive_data_version,
    _resolve_with_fallback,
)


# ── _resolve_with_fallback ───────────────────────────────────────────────────
def test_resolves_standalone_versioned(tmp_path: Path):
    (tmp_path / "non_matches_v1_2026_06_10.parquet").touch()
    (tmp_path / "non_matches_v4_2026_06_11.parquet").touch()
    (tmp_path / "non_matches_v2_2026_06_10.parquet").touch()
    result = _resolve_with_fallback(
        tmp_path, ("non_matches_v*_*.parquet", "non_matches_*Z.parquet"),
    )
    assert result.name == "non_matches_v4_2026_06_11.parquet"


def test_resolves_run_id_when_no_versioned_match(tmp_path: Path):
    (tmp_path / "non_matches_20260620T100000Z.parquet").touch()
    (tmp_path / "non_matches_20260621T120000Z.parquet").touch()
    result = _resolve_with_fallback(
        tmp_path, ("non_matches_v*_*.parquet", "non_matches_*Z.parquet"),
    )
    assert result.name == "non_matches_20260621T120000Z.parquet"


def test_prefers_versioned_over_run_id_when_both_present(tmp_path: Path):
    """When both naming conventions coexist, prefer the deterministic
    versioned one (most projects keep these aligned; ambiguous files should
    fail loudly only if neither matches)."""
    (tmp_path / "non_matches_v4_2026_06_11.parquet").touch()
    (tmp_path / "non_matches_20260621T120000Z.parquet").touch()
    result = _resolve_with_fallback(
        tmp_path, ("non_matches_v*_*.parquet", "non_matches_*Z.parquet"),
    )
    assert result.name == "non_matches_v4_2026_06_11.parquet"


def test_searches_multiple_directories_for_candidate_pairs(tmp_path: Path):
    d1 = tmp_path / "data" / "blocking"
    d2 = tmp_path / "src" / "features" / "outputs" / "blocking"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    (d2 / "candidate_pairs_v3_2026_06_11.parquet").touch()
    result = _resolve_with_fallback(
        (d1, d2),
        ("candidate_pairs_v*_*.parquet", "candidate_pairs_*Z.parquet"),
    )
    # d1 has no candidates; d2 hit on first pass (versioned).
    assert result.parent == d2


def test_raises_when_no_match_anywhere(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No files matching"):
        _resolve_with_fallback(
            tmp_path, ("non_matches_v*_*.parquet", "non_matches_*Z.parquet"),
        )


# ── _derive_data_version ──────────────────────────────────────────────────────
def test_derive_data_version_standalone():
    assert _derive_data_version(Path("non_matches_v4_2026_06_11.parquet")) == "v4_2026_06_11"


def test_derive_data_version_pipeline_run_id():
    assert _derive_data_version(Path("non_matches_20260621T120000Z.parquet")) == "20260621T120000Z"


def test_derive_data_version_unknown_falls_back_to_stem():
    assert _derive_data_version(Path("non_matches_custom.parquet")) == "non_matches_custom"
