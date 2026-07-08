"""Unit tests for the FS matcher Settings fields + env overrides."""

from __future__ import annotations

from src.config import Settings


def test_fs_defaults():
    s = Settings()
    assert s.fs_auto_merge_threshold == 0.95
    assert s.fs_review_floor == 0.40
    assert s.fs_deploy_gate_margin == 0.02
    assert s.fs_active_model is None
    assert s.fs_model_dir.name == "fs"
    assert s.fs_output_dir.name == "FS_output"


def test_env_overrides_thresholds(monkeypatch):
    monkeypatch.setenv("EMPI_FS_AUTO_MERGE_THRESHOLD", "0.90")
    monkeypatch.setenv("EMPI_FS_REVIEW_FLOOR", "0.30")
    monkeypatch.setenv("EMPI_FS_DEPLOY_GATE_MARGIN", "0.05")
    s = Settings()
    assert s.fs_auto_merge_threshold == 0.90
    assert s.fs_review_floor == 0.30
    assert s.fs_deploy_gate_margin == 0.05


def test_env_override_active_model_path(monkeypatch, tmp_path):
    model = tmp_path / "fs_model_x.json"
    monkeypatch.setenv("EMPI_FS_ACTIVE_MODEL", str(model))
    s = Settings()
    assert s.fs_active_model == model


def test_ensure_dirs_creates_fs_dirs(tmp_path, monkeypatch):
    s = Settings()
    monkeypatch.setattr(s, "fs_model_dir", tmp_path / "models" / "fs")
    monkeypatch.setattr(s, "fs_output_dir", tmp_path / "data" / "FS_output")
    monkeypatch.setattr(s, "matches_model_dir", tmp_path / "data" / "matches_model")
    # redirect the rest under tmp so ensure_dirs doesn't touch the repo
    for attr in ("raw_dir", "processed_dir", "blocking_dir", "matches_dir",
                 "non_matches_dir", "rejects_dir", "clusters_dir", "runs_dir"):
        monkeypatch.setattr(s, attr, tmp_path / attr)
    monkeypatch.setattr(s, "db_path", tmp_path / "empi.db")
    s.ensure_dirs()
    assert (tmp_path / "models" / "fs").is_dir()
    assert (tmp_path / "data" / "FS_output").is_dir()
    assert (tmp_path / "data" / "matches_model").is_dir()
