"""Unit tests for src/api/jobs.py — retry-with-backoff, durable status files,
and startup reconciliation (the "database persists across restarts, jobs
retry transient failures" work). HTTP-layer coverage of the GET-route
fallback onto these same status files lives in
tests/integration/test_api.py instead, since that needs the real routes.

Each test builds its own `Settings(runs_dir=tmp_path)` rather than touching
the module-level `src.config.settings` singleton — `jobs.py`'s functions
all take `settings` explicitly, so this is enough isolation without needing
`ensure_dirs()` (which would create every *other* stage directory too,
including ones that default outside tmp_path).
"""

from __future__ import annotations

import pytest

from src.api import jobs
from src.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(runs_dir=tmp_path)


@pytest.fixture(autouse=True)
def _clear_registries():
    jobs._REGISTRY.clear()
    jobs._SCORE_REGISTRY.clear()
    jobs._SCORE_RESULTS.clear()
    yield


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    monkeypatch.setattr(jobs, "_backoff_seconds", lambda attempt: 0)


class FakeBackend:
    def close(self) -> None:
        pass


class TestRunPipelineJobRetry:
    def test_retries_then_succeeds(self, settings, tmp_path, monkeypatch):
        calls = {"n": 0}

        def flaky_run_pipeline(*, raw_input, settings, run_id):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("transient database blip")

        monkeypatch.setattr(jobs, "run_pipeline", flaky_run_pipeline)
        monkeypatch.setattr(jobs, "build_index_backend", lambda settings: FakeBackend())
        monkeypatch.setattr(jobs.publish, "publish_run", lambda backend, run_id, settings: None)

        jobs.run_pipeline_job("run1", tmp_path / "raw.csv", settings)

        assert calls["n"] == 2
        assert jobs.get_status("run1")["status"] == "succeeded"
        file_status = jobs.read_run_status_file("run1", settings)
        assert file_status["status"] == "succeeded"
        assert file_status["attempt"] == 2

    def test_exhausts_retries_then_fails_permanently(self, settings, tmp_path, monkeypatch):
        def always_fails(*, raw_input, settings, run_id):
            raise RuntimeError("boom")

        monkeypatch.setattr(jobs, "run_pipeline", always_fails)

        jobs.run_pipeline_job("run2", tmp_path / "raw.csv", settings)

        status = jobs.get_status("run2")
        assert status["status"] == "failed"
        assert "boom" in status["error"]
        file_status = jobs.read_run_status_file("run2", settings)
        assert file_status["status"] == "failed"
        assert file_status["attempt"] == jobs._MAX_ATTEMPTS

    def test_mark_queued_writes_durable_status(self, settings):
        jobs.mark_queued("run3", settings)

        assert jobs.get_status("run3")["status"] == "queued"
        assert jobs.read_run_status_file("run3", settings)["status"] == "queued"


class TestScoreRecordsJobRetry:
    def test_retries_then_succeeds(self, settings, monkeypatch):
        calls = {"n": 0}
        outcome = {
            "patid": "P1", "tier": "no_match", "mid": "M-1", "match_rule": None,
            "confidence": None, "matched_patids": [],
            "fs_match_probability": None, "fs_classification_tier": None,
        }

        def flaky_score(backend, settings, records, run_id):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("transient lock")
            return [outcome]

        monkeypatch.setattr(jobs.incremental, "score_records", flaky_score)
        monkeypatch.setattr(jobs, "build_index_backend", lambda settings: FakeBackend())

        jobs.score_records_job("score1", [{"PATID": "P1"}], settings)

        assert calls["n"] == 2
        assert jobs.get_score_status("score1")["status"] == "succeeded"
        assert jobs.get_score_result("score1") == [outcome]
        assert jobs.read_score_status_file("score1", settings)["status"] == "succeeded"

    def test_exhausts_retries_then_fails_permanently(self, settings, monkeypatch):
        def always_fails(backend, settings, records, run_id):
            raise RuntimeError("boom")

        monkeypatch.setattr(jobs.incremental, "score_records", always_fails)
        monkeypatch.setattr(jobs, "build_index_backend", lambda settings: FakeBackend())

        jobs.score_records_job("score2", [{"PATID": "P1"}], settings)

        status = jobs.get_score_status("score2")
        assert status["status"] == "failed"
        assert "boom" in status["error"]
        assert jobs.read_score_status_file("score2", settings)["attempt"] == jobs._MAX_ATTEMPTS


class TestReconcileInterruptedJobs:
    def test_marks_orphaned_running_and_queued_as_failed(self, settings):
        jobs.mark_queued("orphan-queued", settings)
        jobs._touch("orphan-running", "running", settings)
        jobs._touch("orphan-done", "succeeded", settings)

        count = jobs.reconcile_interrupted_jobs(settings)

        assert count == 2
        assert jobs.read_run_status_file("orphan-queued", settings)["status"] == "failed"
        assert "restart" in jobs.read_run_status_file("orphan-queued", settings)["error"]
        assert jobs.read_run_status_file("orphan-running", settings)["status"] == "failed"
        assert jobs.read_run_status_file("orphan-done", settings)["status"] == "succeeded"

    def test_catches_orphaned_score_jobs_too(self, settings):
        jobs.mark_score_queued("score-orphan", settings)

        count = jobs.reconcile_interrupted_jobs(settings)

        assert count == 1
        assert jobs.read_score_status_file("score-orphan", settings)["status"] == "failed"

    def test_returns_zero_when_runs_dir_missing(self, tmp_path):
        settings = Settings(runs_dir=tmp_path / "does-not-exist")
        assert jobs.reconcile_interrupted_jobs(settings) == 0

    def test_returns_zero_when_nothing_orphaned(self, settings):
        jobs._touch("done", "succeeded", settings)
        assert jobs.reconcile_interrupted_jobs(settings) == 0
