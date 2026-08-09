"""Unit tests for the non-match gate registry: active-model resolution +
deploy gate. Mirrors test_ml_matcher_registry.py — the registries are the same
logic retargeted at gate_* settings and the gate's own held-out metric block
(`test_metrics.gate_at_threshold`, the shape the notebook's export cell
writes)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models.nonmatch_gate import registry as R


def _settings(model_dir: Path, active=None, margin=0.02):
    return SimpleNamespace(
        gate_active_model=active, gate_model_dir=model_dir,
        gate_deploy_gate_margin=margin,
    )


def _write_model(model_dir: Path, name: str, precision: float, recall: float) -> Path:
    mp = model_dir / name
    mp.write_text("{}")
    R.meta_path_for(mp).write_text(json.dumps(
        {"test_metrics": {"gate_at_threshold": {
            "plausible_precision": precision, "plausible_recall": recall,
        }}}
    ))
    return mp


# ─── resolution ───────────────────────────────────────────────────────────────
def test_resolve_returns_none_when_store_empty(tmp_path):
    assert R.resolve_active_model(_settings(tmp_path)) is None


def test_resolve_falls_back_to_latest_by_mtime(tmp_path):
    _write_model(tmp_path, "nonmatch_gate_A.pkl", 0.96, 0.99)
    b = _write_model(tmp_path, "nonmatch_gate_B.pkl", 0.96, 0.99)
    assert R.resolve_active_model(_settings(tmp_path)).name == b.name


def test_resolve_ignores_other_model_families(tmp_path):
    """The gate store must not pick up an ML matcher or FS artifact."""
    _write_model(tmp_path, "ml_model_A.pkl", 0.96, 0.99)
    assert R.resolve_active_model(_settings(tmp_path)) is None


def test_active_pointer_takes_precedence_over_mtime(tmp_path):
    a = _write_model(tmp_path, "nonmatch_gate_A.pkl", 0.96, 0.99)
    _write_model(tmp_path, "nonmatch_gate_B.pkl", 0.96, 0.99)
    R.promote(a, _settings(tmp_path))
    assert R.resolve_active_model(_settings(tmp_path)).name == a.name


def test_explicit_override_takes_precedence(tmp_path):
    a = _write_model(tmp_path, "nonmatch_gate_A.pkl", 0.96, 0.99)
    _write_model(tmp_path, "nonmatch_gate_B.pkl", 0.96, 0.99)
    assert R.resolve_active_model(_settings(tmp_path, active=a)).name == a.name


def test_missing_override_resolves_to_none(tmp_path):
    st = _settings(tmp_path, active=tmp_path / "gone.pkl")
    assert R.resolve_active_model(st) is None


def test_malformed_pointer_falls_back_to_latest(tmp_path):
    a = _write_model(tmp_path, "nonmatch_gate_A.pkl", 0.96, 0.99)
    (tmp_path / R.ACTIVE_POINTER).write_text("not json")
    assert R.resolve_active_model(_settings(tmp_path)).name == a.name


def test_meta_path_for_is_sibling_sidecar(tmp_path):
    assert R.meta_path_for(tmp_path / "nonmatch_gate_X.pkl").name == "nonmatch_gate_X.meta.json"


# ─── deploy gate ──────────────────────────────────────────────────────────────
def test_first_promotion_always_passes(tmp_path):
    a = _write_model(tmp_path, "nonmatch_gate_A.pkl", 0.96, 0.99)
    assert "first promotion" in R.promote(a, _settings(tmp_path))


def test_recall_regression_beyond_the_margin_is_refused(tmp_path):
    """Recall is the operationally critical metric: a pair the gate drops is
    discarded for the rest of the run."""
    a = _write_model(tmp_path, "nonmatch_gate_A.pkl", 0.96, 0.999)
    R.promote(a, _settings(tmp_path))
    b = _write_model(tmp_path, "nonmatch_gate_B.pkl", 0.96, 0.90)
    with pytest.raises(R.DeployGateError):
        R.promote(b, _settings(tmp_path))
    assert R.resolve_active_model(_settings(tmp_path)).name == a.name


def test_regression_within_the_margin_is_allowed(tmp_path):
    a = _write_model(tmp_path, "nonmatch_gate_A.pkl", 0.96, 0.999)
    R.promote(a, _settings(tmp_path))
    b = _write_model(tmp_path, "nonmatch_gate_B.pkl", 0.95, 0.99)
    R.promote(b, _settings(tmp_path))
    assert R.resolve_active_model(_settings(tmp_path)).name == b.name


def test_force_overrides_a_failing_gate(tmp_path):
    a = _write_model(tmp_path, "nonmatch_gate_A.pkl", 0.96, 0.999)
    R.promote(a, _settings(tmp_path))
    b = _write_model(tmp_path, "nonmatch_gate_B.pkl", 0.50, 0.50)
    R.promote(b, _settings(tmp_path), force=True)
    pointer = json.loads((tmp_path / R.ACTIVE_POINTER).read_text())
    assert pointer["model_file"] == b.name and pointer["forced"] is True
