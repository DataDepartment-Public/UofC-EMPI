"""Unit tests for src/models/model_cache.py — the mtime-keyed in-memory
cache backing POST /admin/models/reload's no-downtime model hot-swap."""

from __future__ import annotations

import pytest

from src.models import model_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    model_cache.invalidate()
    yield
    model_cache.invalidate()


def test_get_or_load_calls_loader_once_for_unchanged_file(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("v1")
    calls = []

    def loader(p):
        calls.append(p)
        return p.read_text()

    first = model_cache.get_or_load("k", path, loader)
    second = model_cache.get_or_load("k", path, loader)

    assert first == "v1"
    assert second == "v1"
    assert len(calls) == 1


def test_get_or_load_reloads_when_mtime_changes(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("v1")
    calls = []

    def loader(p):
        calls.append(p.read_text())
        return p.read_text()

    assert model_cache.get_or_load("k", path, loader) == "v1"

    # Bump mtime forward explicitly -- some filesystems have coarse mtime
    # resolution, so a same-second rewrite could otherwise land on an
    # identical mtime and flake this test.
    import os

    path.write_text("v2")
    newer = path.stat().st_mtime + 1
    os.utime(path, (newer, newer))

    assert model_cache.get_or_load("k", path, loader) == "v2"
    assert calls == ["v1", "v2"]


def test_get_or_load_different_keys_are_independent(tmp_path):
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text("a")
    path_b.write_text("b")

    assert model_cache.get_or_load("a", path_a, lambda p: p.read_text()) == "a"
    assert model_cache.get_or_load("b", path_b, lambda p: p.read_text()) == "b"
    assert set(model_cache.status().keys()) == {"a", "b"}


def test_invalidate_one_key_leaves_others_cached(tmp_path):
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text("a")
    path_b.write_text("b")
    model_cache.get_or_load("a", path_a, lambda p: p.read_text())
    model_cache.get_or_load("b", path_b, lambda p: p.read_text())

    removed = model_cache.invalidate("a")

    assert removed == ["a"]
    assert set(model_cache.status().keys()) == {"b"}


def test_invalidate_all_clears_everything_and_returns_removed_keys(tmp_path):
    path_a = tmp_path / "a.json"
    path_a.write_text("a")
    model_cache.get_or_load("a", path_a, lambda p: p.read_text())

    removed = model_cache.invalidate()

    assert removed == ["a"]
    assert model_cache.status() == {}


def test_invalidate_unknown_key_is_a_no_op():
    assert model_cache.invalidate("does-not-exist") == []


def test_status_reports_path_and_mtime(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("v1")
    model_cache.get_or_load("k", path, lambda p: p.read_text())

    entry = model_cache.status()["k"]

    assert entry["path"] == str(path)
    assert entry["mtime"] == pytest.approx(path.stat().st_mtime)
