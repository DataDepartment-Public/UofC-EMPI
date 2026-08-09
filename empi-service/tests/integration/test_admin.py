"""Integration tests for GET/PUT /admin/thresholds (src/api/routers/admin.py,
src/api/threshold_store.py).

Follows test_api.py's fixture pattern: `main.py`'s lifespan reads the
`src.config.settings` singleton directly, so it is monkeypatched in place
rather than constructing a separate `Settings()`.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api import threshold_store
from src.api.main import app
from src.config import settings as real_settings


@pytest.fixture
def test_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(real_settings, "project_root", tmp_path)
    monkeypatch.setattr(real_settings, "raw_dir", tmp_path / "data" / "raw")
    monkeypatch.setattr(real_settings, "processed_dir", tmp_path / "data" / "processed")
    monkeypatch.setattr(real_settings, "blocking_dir", tmp_path / "data" / "blocking")
    monkeypatch.setattr(real_settings, "auto_merge_dir", tmp_path / "data" / "auto_merge")
    monkeypatch.setattr(real_settings, "non_matches_dir", tmp_path / "data" / "non_matches")
    monkeypatch.setattr(real_settings, "no_match_dir", tmp_path / "data" / "no_match")
    monkeypatch.setattr(real_settings, "clusters_dir", tmp_path / "data" / "clusters")
    monkeypatch.setattr(real_settings, "runs_dir", tmp_path / "data" / "runs")
    monkeypatch.setattr(real_settings, "db_path", tmp_path / "empi.db")
    # Reset to field defaults for every test — otherwise a prior test's (or
    # this process's .env) values leak in as the "default" a test expects.
    monkeypatch.setattr(real_settings, "gate_threshold", 0.30)
    monkeypatch.setattr(real_settings, "ml_auto_merge_threshold", 0.70)
    monkeypatch.setattr(real_settings, "fs_review_floor", 0.40)
    real_settings.ensure_dirs()
    return real_settings


@pytest.fixture
def client(test_settings):
    with TestClient(app) as c:
        yield c


class TestAdminThresholds:
    def test_get_returns_field_defaults_with_no_override_file(self, client, test_settings):
        resp = client.get("/admin/thresholds")
        assert resp.status_code == 200
        assert resp.json() == {
            "gate_threshold": 0.30,
            "ml_auto_merge_threshold": 0.70,
            "fs_review_floor": 0.40,
        }

    def test_put_updates_and_get_reflects_it(self, client, test_settings):
        resp = client.put(
            "/admin/thresholds",
            json={
                "gate_threshold": 0.25,
                "ml_auto_merge_threshold": 0.80,
                "fs_review_floor": 0.35,
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "gate_threshold": 0.25,
            "ml_auto_merge_threshold": 0.80,
            "fs_review_floor": 0.35,
        }

        # Takes effect immediately on the live settings singleton.
        assert test_settings.gate_threshold == 0.25
        assert test_settings.ml_auto_merge_threshold == 0.80
        assert test_settings.fs_review_floor == 0.35

        again = client.get("/admin/thresholds")
        assert again.json() == resp.json()

    def test_put_persists_to_disk(self, client, test_settings):
        client.put(
            "/admin/thresholds",
            json={
                "gate_threshold": 0.20,
                "ml_auto_merge_threshold": 0.75,
                "fs_review_floor": 0.30,
            },
        )
        path = threshold_store.overrides_path(test_settings)
        assert path.exists()
        assert json.loads(path.read_text()) == {
            "gate_threshold": 0.20,
            "ml_auto_merge_threshold": 0.75,
            "fs_review_floor": 0.30,
        }

    def test_put_rejects_out_of_range_values(self, client, test_settings):
        resp = client.put(
            "/admin/thresholds",
            json={
                "gate_threshold": 1.5,
                "ml_auto_merge_threshold": 0.70,
                "fs_review_floor": 0.40,
            },
        )
        assert resp.status_code == 422
        # Nothing was applied — the invalid request never reached save_thresholds.
        assert test_settings.gate_threshold == 0.30

    def test_persisted_override_survives_a_simulated_restart(self, client, test_settings):
        client.put(
            "/admin/thresholds",
            json={
                "gate_threshold": 0.15,
                "ml_auto_merge_threshold": 0.60,
                "fs_review_floor": 0.20,
            },
        )
        # Simulate a fresh process: reset the live singleton to field
        # defaults, then re-run what lifespan() does on startup.
        test_settings.gate_threshold = 0.30
        test_settings.ml_auto_merge_threshold = 0.70
        test_settings.fs_review_floor = 0.40

        threshold_store.apply_persisted_overrides(test_settings)

        assert test_settings.gate_threshold == 0.15
        assert test_settings.ml_auto_merge_threshold == 0.60
        assert test_settings.fs_review_floor == 0.20
