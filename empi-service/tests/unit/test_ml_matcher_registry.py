"""Unit tests for the ML matcher registry: active-model resolution + deploy
gate. Mirrors test_fs_matcher_registry.py — the two registries are literal
duplicates of the same logic, retargeted at ml_* settings (see
src/models/ml_matcher/registry.py's module docstring for why)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models.ml_matcher import registry as R


def _settings(model_dir: Path, active=None, margin=0.02):
    return SimpleNamespace(
        ml_active_model=active, ml_model_dir=model_dir, ml_deploy_gate_margin=margin,
    )


def _write_model(model_dir: Path, name: str, precision: float, recall: float) -> Path:
    mp = model_dir / name
    mp.write_text("{}")
    R.meta_path_for(mp).write_text(json.dumps(
        {"test_metrics": {"metrics_auto_merge": {"precision": precision, "recall": recall}}}
    ))
    return mp


# ─── resolution ───────────────────────────────────────────────────────────────
def test_resolve_returns_none_when_store_empty(tmp_path):
    assert R.resolve_active_model(_settings(tmp_path)) is None


def test_resolve_falls_back_to_latest_by_mtime(tmp_path):
    _write_model(tmp_path, "ml_model_A.json", 0.78, 0.95)
    b = _write_model(tmp_path, "ml_model_B.json", 0.79, 0.95)
    # B written last -> newest mtime.
    assert R.resolve_active_model(_settings(tmp_path)).name == b.name


def test_active_pointer_takes_precedence_over_mtime(tmp_path):
    a = _write_model(tmp_path, "ml_model_A.json", 0.78, 0.95)
    _write_model(tmp_path, "ml_model_B.json", 0.79, 0.95)
    R.promote(a, _settings(tmp_path))  # point active.json at A
    assert R.resolve_active_model(_settings(tmp_path)).name == a.name


def test_explicit_override_takes_precedence(tmp_path):
    a = _write_model(tmp_path, "ml_model_A.json", 0.78, 0.95)
    _write_model(tmp_path, "ml_model_B.json", 0.79, 0.95)
    st = _settings(tmp_path, active=a)
    assert R.resolve_active_model(st).name == a.name


def test_meta_path_for_is_sibling_sidecar(tmp_path):
    assert R.meta_path_for(tmp_path / "ml_model_X.json").name == "ml_model_X.meta.json"


# ─── deploy gate ──────────────────────────────────────────────────────────────
def test_gate_passes_first_promotion(tmp_path):
    a = _write_model(tmp_path, "ml_model_A.json", 0.60, 0.60)
    reason = R.promote(a, _settings(tmp_path))
    assert "first promotion" in reason
    assert (tmp_path / R.ACTIVE_POINTER).exists()


def test_gate_refuses_regression_beyond_margin(tmp_path):
    a = _write_model(tmp_path, "ml_model_A.json", 0.78, 0.95)
    R.promote(a, _settings(tmp_path))
    b = _write_model(tmp_path, "ml_model_B.json", 0.60, 0.95)  # precision drop 0.18 > margin
    with pytest.raises(R.DeployGateError, match="refused"):
        R.promote(b, _settings(tmp_path))
    # active pointer unchanged -> still A
    assert R.resolve_active_model(_settings(tmp_path)).name == a.name


def test_gate_allows_within_margin(tmp_path):
    a = _write_model(tmp_path, "ml_model_A.json", 0.78, 0.95)
    R.promote(a, _settings(tmp_path))
    b = _write_model(tmp_path, "ml_model_B.json", 0.77, 0.94)  # within 0.02 margin
    reason = R.promote(b, _settings(tmp_path))
    assert "PASS" in reason
    assert R.resolve_active_model(_settings(tmp_path)).name == b.name


def test_force_promote_overrides_failed_gate(tmp_path):
    a = _write_model(tmp_path, "ml_model_A.json", 0.78, 0.95)
    R.promote(a, _settings(tmp_path))
    b = _write_model(tmp_path, "ml_model_B.json", 0.50, 0.50)
    R.promote(b, _settings(tmp_path), force=True)
    assert R.resolve_active_model(_settings(tmp_path)).name == b.name


def test_passes_deploy_gate_pure_function():
    better = {"test_metrics": {"metrics_auto_merge": {"precision": 0.80, "recall": 0.95}}}
    active = {"test_metrics": {"metrics_auto_merge": {"precision": 0.78, "recall": 0.95}}}
    ok, _ = R.passes_deploy_gate(better, active, 0.02)
    assert ok
    worse = {"test_metrics": {"metrics_auto_merge": {"precision": 0.70, "recall": 0.95}}}
    ok, _ = R.passes_deploy_gate(worse, active, 0.02)
    assert not ok
    ok, why = R.passes_deploy_gate(worse, None, 0.02)
    assert ok and "first promotion" in why
