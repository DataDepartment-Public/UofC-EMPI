"""Unit tests for src.models.fs_matcher.train's RunManifest-based input
resolution (`_resolve_from_manifest`), added to close two risks in the old
directory-globbing default: a stale versioned-CLI file silently beating a
fresher orchestrator run, and picking run_blocking.py's narrower 8-block-only
candidate pool instead of the orchestrator's stacked-blocker output.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models.fs_matcher.train import _resolve_from_manifest


def _artifact(path: str) -> dict:
    return {"path": path, "rows": 1, "sha256": "0" * 64}


def _write_manifest(
    runs_dir: Path, run_id: str, *, cleaned_path: str, pool_path: str,
) -> Path:
    manifest = {
        "run_id": run_id,
        "created_utc": "2026-01-01T00:00:00Z",
        "git_sha": None,
        "raw_input": _artifact("data/raw/MDM_Population.csv"),
        "cleaned": _artifact(cleaned_path),
        "candidate_pairs": _artifact(pool_path),
        "matches": _artifact("data/auto_merge/matches_x.parquet"),
        "non_matches": _artifact("data/non_matches/non_matches_x.parquet"),
    }
    p = runs_dir / f"run_{run_id}.json"
    p.write_text(json.dumps(manifest))
    return p


def _settings(root: Path) -> SimpleNamespace:
    return SimpleNamespace(runs_dir=root / "data" / "runs", project_root=root)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_resolves_exact_manifest_paths_over_a_stale_versioned_file(tmp_path):
    settings = _settings(tmp_path)
    settings.runs_dir.mkdir(parents=True)

    # A stale standalone-CLI versioned file sitting in the same directory,
    # deliberately given a NEWER mtime than the manifest-referenced file, to
    # prove manifest resolution ignores directory mtimes entirely.
    stale = tmp_path / "data" / "processed" / "MDM_Population_cleaned_v9_2020_01_01.parquet"
    _touch(stale)

    real_cleaned = "data/processed/MDM_Population_cleaned_20260617T043941Z.parquet"
    real_pool = "data/blocking/candidate_pairs_20260617T043941Z.parquet"
    _touch(tmp_path / real_cleaned)
    _touch(tmp_path / real_pool)
    time.sleep(0.01)
    stale.touch()  # now strictly newer than the manifest-referenced files

    _write_manifest(
        settings.runs_dir, "20260617T043941Z",
        cleaned_path=real_cleaned, pool_path=real_pool,
    )

    cleaned_path, pool_path = _resolve_from_manifest(settings)
    assert cleaned_path == tmp_path / real_cleaned
    assert pool_path == tmp_path / real_pool


def test_resolves_latest_manifest_by_run_id(tmp_path):
    settings = _settings(tmp_path)
    settings.runs_dir.mkdir(parents=True)

    for run_id, tag in [("20260101T000000Z", "old"), ("20260601T000000Z", "new")]:
        cleaned = f"data/processed/cleaned_{tag}.parquet"
        pool = f"data/blocking/pairs_{tag}.parquet"
        _touch(tmp_path / cleaned)
        _touch(tmp_path / pool)
        _write_manifest(settings.runs_dir, run_id, cleaned_path=cleaned, pool_path=pool)

    cleaned_path, pool_path = _resolve_from_manifest(settings)
    assert cleaned_path.name == "cleaned_new.parquet"
    assert pool_path.name == "pairs_new.parquet"


def test_run_id_pins_a_specific_non_latest_manifest(tmp_path):
    settings = _settings(tmp_path)
    settings.runs_dir.mkdir(parents=True)

    for run_id, tag in [("20260101T000000Z", "old"), ("20260601T000000Z", "new")]:
        cleaned = f"data/processed/cleaned_{tag}.parquet"
        pool = f"data/blocking/pairs_{tag}.parquet"
        _touch(tmp_path / cleaned)
        _touch(tmp_path / pool)
        _write_manifest(settings.runs_dir, run_id, cleaned_path=cleaned, pool_path=pool)

    cleaned_path, pool_path = _resolve_from_manifest(settings, run_id="20260101T000000Z")
    assert cleaned_path.name == "cleaned_old.parquet"
    assert pool_path.name == "pairs_old.parquet"


def test_raises_when_no_manifest_present(tmp_path):
    settings = _settings(tmp_path)
    settings.runs_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No run manifests found"):
        _resolve_from_manifest(settings)


def test_raises_when_explicit_run_id_not_found(tmp_path):
    settings = _settings(tmp_path)
    settings.runs_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="20261231T235959Z"):
        _resolve_from_manifest(settings, run_id="20261231T235959Z")


def test_raises_when_manifest_referenced_file_is_missing(tmp_path):
    settings = _settings(tmp_path)
    settings.runs_dir.mkdir(parents=True)
    # Manifest references files that were never created on disk (simulates
    # cleanup/retention deleting the underlying parquet after the run).
    _write_manifest(
        settings.runs_dir, "20260101T000000Z",
        cleaned_path="data/processed/gone.parquet",
        pool_path="data/blocking/also_gone.parquet",
    )
    with pytest.raises(FileNotFoundError, match="lineage broken"):
        _resolve_from_manifest(settings)
