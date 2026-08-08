"""Integration tests for the FastAPI service (src/api/) against a temp DB and
data directory — no live uvicorn process, just Starlette's TestClient.

`main.py`'s `lifespan` reads the module-level `src.config.settings` singleton
directly (lifespan hooks aren't part of FastAPI's dependency-injection graph,
so `app.dependency_overrides` never reaches it). The `test_settings` fixture
therefore monkeypatches that singleton's fields in place, rather than
constructing a separate `Settings()` and only overriding the `Depends`-based
paths — otherwise `lifespan` would silently create/touch the real
`data/empi.db` on every test run. `jobs.run_pipeline_job` is monkeypatched to
a fast fake in the tests that don't need a real pipeline run (the pipeline
itself is already covered by tests/unit + tests/integration for it).
"""

import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import jobs
from src.api.backends import sql_backend
from src.api.backends.index_backend import build_index_backend
from src.api.main import app
from src.config import Settings, settings as real_settings
from src.contracts import ArtifactRef, RunManifest
from src.models import model_cache
from src.models.fs_matcher import registry as fs_registry


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
    real_settings.ensure_dirs()
    # lifespan no longer creates the schema implicitly (src/api/main.py) —
    # a fresh temp DB is exactly the "new environment" case scripts/init_db.py
    # exists for, so tests now do explicitly what that script does once.
    conn = sql_backend.get_connection(real_settings.db_path)
    try:
        sql_backend.init_db(conn)
    finally:
        conn.close()
    return real_settings


@pytest.fixture
def parquet_test_settings(test_settings, monkeypatch):
    """Same temp layout as `test_settings`, but `EMPI_INDEX_BACKEND=parquet`
    — proves `records.py`/`dashboard.py` work identically against
    `ParquetIndexBackend` via `get_backend`, not just SQLite."""
    monkeypatch.setattr(test_settings, "index_backend", "parquet")
    monkeypatch.setattr(
        test_settings, "local_index_dir", test_settings.project_root / "data" / "local_index"
    )
    test_settings.ensure_dirs()
    return test_settings


@pytest.fixture
def client(test_settings):
    jobs._REGISTRY.clear()
    with TestClient(app) as c:
        yield c


def _publish_fixture_run(settings: Settings, run_id: str = "r1") -> None:
    """Write a tiny manifest + Parquet run and publish it — the shortcut most
    /records, /clusters, /audit tests need without running the real pipeline.

    P1<->P2 auto-merge (SSN_DOB); P3 true singleton; P4<->P5 review-tier
    candidate (NAME_DOB_SEX) — exercises the review-queue/raw-data routes too.
    """
    from src.api.ingest import publish

    cleaned = pd.DataFrame({
        "PATID": ["P1", "P2", "P3", "P4", "P5"],
        "FirstNM_clean": ["Jane", "Jane", "John", "Amy", "Amy"],
        "LastNM_clean": ["Doe", "Doe", "Smith", "Lee", "Lee"],
        "BirthDT_clean": pd.to_datetime(
            ["1990-01-01", "1990-01-01", "1985-05-05", "1975-03-03", "1975-03-03"]
        ),
        "SSN_clean": ["123456789", "123456789", None, None, None],
        "last_4_SSN": ["6789", "6789", None, None, None],
        "Email_clean": [None, None, None, None, None],
        "ZipCD_clean_base": ["60601", "60601", None, None, None],
        "AddressLine1_clean": [None, None, None, None, None],
        "SexAtBirthDSC_clean": ["FEMALE", "FEMALE", "MALE", "FEMALE", "FEMALE"],
        "Phones_set": [set(), set(), set(), set(), set()],
        "FirstNM_raw": ["JANE", "JANE", "JOHN", "AMY", "AMY"],
        "SSN_raw": ["123-45-6789", "123456789", None, None, None],
        "valid_record": [True, True, True, True, True],
    })
    cleaned_path = settings.processed_dir / f"cleaned_{run_id}.parquet"
    cleaned.to_parquet(cleaned_path, index=False)

    matches = pd.DataFrame({
        "PATID_A": ["P1"], "PATID_B": ["P2"],
        "match_rule": ["SSN_DOB"], "confidence": [1.0],
        "rules_fired": ["SSN_DOB"], "is_suspicious": [False],
        "high_fanout_ssn": [False], "cluster_id": [0],
        "source_blocks": ["B1"], "n_blocks": [1],
    })
    matches_path = settings.auto_merge_dir / f"matches_{run_id}.parquet"
    matches.to_parquet(matches_path, index=False)

    non_matches = pd.DataFrame({
        "PATID_A": ["P4"], "PATID_B": ["P5"],
        "source_blocks": ["B3"], "n_blocks": [1],
    })
    non_matches_path = settings.non_matches_dir / f"non_matches_{run_id}.parquet"
    non_matches.to_parquet(non_matches_path, index=False)

    review_evidence = pd.DataFrame({
        "PATID_A": ["P4"], "PATID_B": ["P5"],
        "match_rule": ["NAME_DOB_SEX"], "confidence": [0.98],
        "rules_fired": ["NAME_DOB_SEX"], "is_suspicious": [False],
        "high_fanout_ssn": [False], "source_blocks": ["B3"], "n_blocks": [1],
    })
    review_evidence_path = settings.non_matches_dir / f"review_evidence_{run_id}.parquet"
    review_evidence.to_parquet(review_evidence_path, index=False)

    clusters = pd.DataFrame({
        "PATID": ["P1", "P2", "P3", "P4", "P5"], "cluster_id": [0, 0, 1, 2, 3],
    })
    clusters_path = settings.clusters_dir / f"clusters_{run_id}.parquet"
    clusters.to_parquet(clusters_path, index=False)

    def ref(path, rows):
        return ArtifactRef(
            path=str(path.relative_to(settings.project_root)), rows=rows, sha256="x"
        )

    manifest = RunManifest(
        run_id=run_id, created_utc="2026-07-01T00:00:00Z",
        raw_input=ref(cleaned_path, 5), cleaned=ref(cleaned_path, 5),
        candidate_pairs=ref(matches_path, 1), matches=ref(matches_path, 1),
        non_matches=ref(non_matches_path, 1),
        review_evidence=ref(review_evidence_path, 1),
        clusters=ref(clusters_path, 5),
        counts={
            "raw_rows": 5, "valid_records": 5, "candidate_pairs": 2, "matches": 1,
            "non_matches": 1,
        },
    )
    (settings.runs_dir / f"run_{run_id}.json").write_text(manifest.model_dump_json())

    backend = build_index_backend(settings)
    try:
        publish.publish_run(backend, run_id, settings)
    finally:
        backend.close()


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ready(self, client):
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["checks"]["db"] is True


class TestAdmin:
    """POST /admin/models/reload + GET /admin/models/status — the explicit,
    no-downtime cutover for a promoted FS/ML model (src/models/model_cache.py)."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        # model_cache is a process-wide singleton -- isolate each test from
        # whatever another test (or another test module) left cached.
        model_cache.invalidate()
        yield
        model_cache.invalidate()

    @pytest.fixture(autouse=True)
    def _isolated_model_dirs(self, test_settings, monkeypatch):
        # test_settings (module-level fixture) doesn't redirect fs_model_dir/
        # ml_model_dir -- no other test class needed to write model artifacts.
        # These tests promote real models, so without this they'd write
        # active.json/fs_model_*.json straight into the real models/fs on disk.
        monkeypatch.setattr(test_settings, "fs_model_dir", test_settings.project_root / "models" / "fs")
        monkeypatch.setattr(test_settings, "ml_model_dir", test_settings.project_root / "models" / "ml")
        test_settings.fs_model_dir.mkdir(parents=True, exist_ok=True)
        test_settings.ml_model_dir.mkdir(parents=True, exist_ok=True)

    def test_status_with_no_models_or_cache(self, client, test_settings):
        resp = client.get("/admin/models/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"cached": {}, "fs_active_model": None, "ml_active_model": None}

    def test_reload_with_empty_cache_invalidates_nothing(self, client, test_settings):
        resp = client.post("/admin/models/reload")
        assert resp.status_code == 200
        assert resp.json()["invalidated"] == []

    def test_status_reports_cached_entry_and_reload_clears_it(self, client, test_settings):
        model_path = test_settings.fs_model_dir / "fs_model_1.json"
        model_path.write_text("{}")
        model_cache.get_or_load("fs_matcher", model_path, lambda p: {"loaded": True})

        status = client.get("/admin/models/status").json()
        assert "fs_matcher" in status["cached"]
        assert status["cached"]["fs_matcher"]["path"] == str(model_path)

        reload_resp = client.post("/admin/models/reload")
        assert reload_resp.json()["invalidated"] == ["fs_matcher"]

        status_after = client.get("/admin/models/status").json()
        assert status_after["cached"] == {}

    def test_reload_reports_current_active_model_meta(self, client, test_settings):
        fs_path = test_settings.fs_model_dir / "fs_model_1.json"
        fs_path.write_text("{}")
        fs_registry.meta_path_for(fs_path).write_text(
            json.dumps({"model_file": fs_path.name, "test_metrics": {}})
        )
        fs_registry.promote(fs_path, test_settings)

        resp = client.post("/admin/models/reload")
        assert resp.status_code == 200
        body = resp.json()
        assert body["fs_active_model"]["model_file"] == fs_path.name
        assert body["ml_active_model"] is None

    def test_reload_forces_next_load_to_see_updated_file(self, test_settings):
        # Not a client test -- exercises the cache's own self-invalidating
        # mtime behavior alongside the endpoint's eager invalidate(), the two
        # mechanisms described in model_cache.py's docstring.
        model_path = test_settings.fs_model_dir / "fs_model_1.json"
        model_path.write_text("v1")
        loads = []

        def loader(p):
            loads.append(p.read_text())
            return p.read_text()

        model_cache.get_or_load("fs_matcher", model_path, loader)
        model_cache.get_or_load("fs_matcher", model_path, loader)
        assert loads == ["v1"]  # second call was a cache hit, no reload

        model_cache.invalidate("fs_matcher")
        model_cache.get_or_load("fs_matcher", model_path, loader)
        assert loads == ["v1", "v1"]  # explicit invalidate forced a reload


class TestRuns:
    def test_create_run_requires_exactly_one_source(self, client, test_settings):
        resp = client.post("/runs", data={})
        assert resp.status_code == 422

    def test_create_run_with_input_path(self, client, test_settings, monkeypatch):
        raw = test_settings.raw_dir / "tiny.csv"
        raw.write_text("PATID\nP1\n")

        calls = []
        monkeypatch.setattr(
            jobs, "run_pipeline_job",
            lambda run_id, raw_input, settings: calls.append(run_id),
        )

        resp = client.post("/runs", data={"input_path": str(raw)})
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "queued"
        assert body["run_id"]

    def test_get_unknown_run_404s(self, client):
        resp = client.get("/runs/does-not-exist")
        assert resp.status_code == 404

    def test_get_run_returns_manifest_for_completed_run(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/runs/r1")
        assert resp.status_code == 200
        assert resp.json()["run_id"] == "r1"
        assert resp.json()["status"] == "succeeded"

    def test_list_runs_includes_completed(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/runs")
        assert resp.status_code == 200
        run_ids = [r["run_id"] for r in resp.json()]
        assert "r1" in run_ids


class TestRunsStatusFileFallback:
    """The in-memory registry (`jobs._REGISTRY`) is gone the instant the
    process restarts; the durable `.status.json` file is what's left. These
    simulate that by writing the file then dropping the registry entry, the
    same gap a real crash/redeploy leaves."""

    def test_get_run_falls_back_to_status_file_after_restart(self, client, test_settings):
        jobs.mark_queued("interrupted-run", test_settings)
        jobs._touch("interrupted-run", "failed", test_settings, error="Interrupted by restart")
        jobs._REGISTRY.pop("interrupted-run")

        resp = client.get("/runs/interrupted-run")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert "Interrupted" in body["error"]

    def test_list_runs_includes_run_known_only_via_status_file(self, client, test_settings):
        jobs.mark_queued("interrupted-run-2", test_settings)
        jobs._REGISTRY.pop("interrupted-run-2")

        resp = client.get("/runs")
        run_ids = [r["run_id"] for r in resp.json()]
        assert "interrupted-run-2" in run_ids

    def test_list_runs_does_not_double_count_status_file_as_manifest(
        self, client, test_settings
    ):
        """`run_<id>.status.json` also matches the glob `run_*.json` used for
        manifests -- regression test for a real bug this caught: without
        excluding `.status.json` names, a run only known via its status file
        showed up twice, once under a mangled id ending in `.status`."""
        _publish_fixture_run(test_settings, "r1")
        jobs.mark_queued("r1", test_settings)  # r1 also has a status file now

        resp = client.get("/runs")
        run_ids = [r["run_id"] for r in resp.json()]
        assert run_ids.count("r1") == 1
        assert not any(rid.endswith(".status") for rid in run_ids)


class TestStartupReconciliation:
    def test_lifespan_marks_orphaned_status_file_as_interrupted(self, test_settings):
        jobs.mark_queued("crashed-run", test_settings)
        jobs._REGISTRY.clear()  # simulate a fresh process with nothing in memory

        with TestClient(app):
            pass  # lifespan's reconciliation runs on context entry

        status = jobs.read_run_status_file("crashed-run", test_settings)
        assert status["status"] == "failed"
        assert "restart" in status["error"]


class TestRecords:
    def test_list_records(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/records")
        assert resp.status_code == 200
        body = resp.json()
        # matched pair (1) + true singleton (1) + review-tier singletons (2)
        assert body["total"] == 4
        mids = {item["mid"] for item in body["items"]}
        assert len(mids) == 4

    def test_review_status_entity_has_candidates(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/records", params={"origin": "review"})
        body = resp.json()
        assert body["total"] == 2
        entity = body["items"][0]
        assert len(entity["review_candidates"]) == 1
        assert entity["review_candidates"][0]["match_rule"] == "NAME_DOB_SEX"

    def test_get_raw_record(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/records/P1/raw")
        assert resp.status_code == 200
        assert resp.json()["fields"]["FirstNM_raw"] == "JANE"

    def test_get_raw_record_unknown_404s(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/records/P-nope/raw")
        assert resp.status_code == 404

    def test_get_raw_record_logs_view_raw_audit_entry(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get(
            "/records/P1/raw", headers={"X-Reviewer-Id": "reviewer.jclark"}
        )
        assert resp.status_code == 200

        backend = build_index_backend(real_settings)
        try:
            rows = backend.list_audit_log(limit=10)
        finally:
            backend.close()
        view_rows = [r for r in rows if r["action"] == "view_raw"]
        assert len(view_rows) == 1
        assert view_rows[0]["user"] == "reviewer.jclark"
        assert view_rows[0]["patids"] == "P1"

    def test_get_raw_record_without_header_logs_unknown_user(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/records/P1/raw")
        assert resp.status_code == 200

        backend = build_index_backend(real_settings)
        try:
            rows = backend.list_audit_log(limit=10)
        finally:
            backend.close()
        view_rows = [r for r in rows if r["action"] == "view_raw"]
        assert len(view_rows) == 1
        assert view_rows[0]["user"] == "unknown"

    def test_get_raw_record_unknown_404_not_logged(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        client.get("/records/P-nope/raw")

        backend = build_index_backend(real_settings)
        try:
            rows = backend.list_audit_log(limit=10)
        finally:
            backend.close()
        assert [r for r in rows if r["action"] == "view_raw"] == []

    def test_view_raw_excluded_from_list_audit(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        client.get("/records/P1/raw", headers={"X-Reviewer-Id": "reviewer.jclark"})
        resp = client.get("/audit")
        assert resp.status_code == 200
        assert all(row["action"] != "view_raw" for row in resp.json())

    def test_get_clean_ssn(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/records/P1/ssn-clean")
        assert resp.status_code == 200
        # SSN_raw is "123-45-6789"; clean_ssn strips formatting -> the
        # pipeline-matched value, distinct from the raw source string.
        assert resp.json() == {"patid": "P1", "ssn": "123456789"}

    def test_get_clean_ssn_unknown_404s(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/records/P-nope/ssn-clean")
        assert resp.status_code == 404

    def test_get_clean_ssn_logs_view_ssn_clean_audit_entry(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get(
            "/records/P1/ssn-clean", headers={"X-Reviewer-Id": "reviewer.jclark"}
        )
        assert resp.status_code == 200

        backend = build_index_backend(real_settings)
        try:
            rows = backend.list_audit_log(limit=10)
        finally:
            backend.close()
        view_rows = [r for r in rows if r["action"] == "view_ssn_clean"]
        assert len(view_rows) == 1
        assert view_rows[0]["user"] == "reviewer.jclark"
        assert view_rows[0]["patids"] == "P1"

    def test_get_clean_ssn_unknown_404_not_logged(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        client.get("/records/P-nope/ssn-clean")

        backend = build_index_backend(real_settings)
        try:
            rows = backend.list_audit_log(limit=10)
        finally:
            backend.close()
        assert [r for r in rows if r["action"] == "view_ssn_clean"] == []

    def test_view_ssn_clean_excluded_from_list_audit(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        client.get(
            "/records/P1/ssn-clean", headers={"X-Reviewer-Id": "reviewer.jclark"}
        )
        resp = client.get("/audit")
        assert resp.status_code == 200
        assert all(row["action"] != "view_ssn_clean" for row in resp.json())

    def test_search_filters(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/records", params={"search": "Smith"})
        assert resp.json()["total"] == 1

    def test_is_merged_filter(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/records", params={"is_merged": "true"})
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["is_merged"] is True

    def test_get_cluster(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        mid = client.get("/records", params={"is_merged": "true"}).json()["items"][0]["mid"]
        resp = client.get(f"/clusters/{mid}")
        assert resp.status_code == 200
        assert len(resp.json()["members"]) == 2

    def test_get_unknown_cluster_404s(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/clusters/M-999999")
        assert resp.status_code == 404


def _incoming(patid, first, last, birth, sex=None):
    return {
        "PATID": patid, "FirstNM": first, "LastNM": last, "BirthDT": birth,
        "SexAtBirthDSC": sex,
    }


class TestRecordsScore:
    """POST /records/score end-to-end: the HTTP layer around
    `src/api/ingest/incremental.py` (already unit-tested in isolation in
    `tests/unit/api/test_incremental.py`) — background job, polling, and that the
    outcome is visible through the normal read routes."""

    def test_review_tier_match_visible_through_normal_routes(
        self, client, test_settings, monkeypatch
    ):
        _publish_fixture_run(test_settings, "r1")
        # No active FS model for this test — isolate from whatever real
        # artifact happens to be committed under models/fs/.
        monkeypatch.setattr(
            test_settings, "fs_model_dir", test_settings.project_root / "no_fs_model"
        )

        # Same name/DOB/sex as P4/P5 (NAME_DOB_SEX, review-tier) — no
        # SSN/email/phone corroboration, so this must not auto-merge.
        resp = client.post(
            "/records/score",
            json={"records": [_incoming("P6", "Amy", "Lee", "1975-03-03", "FEMALE")]},
        )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
        assert resp.json()["status"] == "queued"

        # TestClient runs BackgroundTasks synchronously before returning, so
        # the job is already done by the time we poll.
        result = client.get(f"/records/score/{run_id}")
        assert result.status_code == 200
        body = result.json()
        assert body["status"] == "succeeded"
        assert len(body["outcomes"]) == 1
        outcome = body["outcomes"][0]
        assert outcome["patid"] == "P6"
        assert outcome["tier"] == "human_review"
        mid = outcome["mid"]

        entity = client.get(f"/clusters/{mid}").json()
        assert entity["origin"] == "review"
        assert entity["is_merged"] is False
        assert any(m["patid"] == "P6" for m in entity["members"])
        assert len(entity["review_candidates"]) >= 1

    def test_unknown_score_run_id_404s(self, client, test_settings):
        resp = client.get("/records/score/does-not-exist")
        assert resp.status_code == 404

    def test_requires_at_least_one_record(self, client, test_settings):
        resp = client.post("/records/score", json={"records": []})
        assert resp.status_code == 422

    def test_get_score_result_falls_back_to_status_file_after_restart(
        self, client, test_settings
    ):
        jobs.mark_score_queued("interrupted-score", test_settings)
        jobs._touch_score(
            "interrupted-score", "failed", test_settings, error="Interrupted by restart"
        )
        jobs._SCORE_REGISTRY.pop("interrupted-score")

        resp = client.get("/records/score/interrupted-score")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert "Interrupted" in body["error"]


class TestAudit:
    def test_merge_requires_reviewer_header(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.post("/audit/merge", json={"mid": "M-000001", "patids": ["P3"]})
        assert resp.status_code == 401

    def test_merge_unknown_mid_404s(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.post(
            "/audit/merge",
            json={"mid": "M-999999", "patids": ["P3"]},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert resp.status_code == 404

    def test_merge_then_audit_feed(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        resp = client.post(
            "/audit/merge",
            json={"mid": matched_mid, "patids": ["P3"]},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert {m["patid"] for m in body["entity"]["members"]} == {"P1", "P2", "P3"}

        audit_resp = client.get("/audit")
        assert audit_resp.status_code == 200
        assert audit_resp.json()[0]["action"] == "merge"

    def test_merge_records_related_patids(self, client, test_settings):
        """related_patids captures the entity's membership BEFORE this merge
        — the snapshot a later export (e.g. for retraining labels) needs,
        since entity_member only ever holds current state."""
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        resp = client.post(
            "/audit/merge",
            json={"mid": matched_mid, "patids": ["P3"]},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert resp.status_code == 200

        audit_resp = client.get("/audit")
        row = audit_resp.json()[0]
        assert row["action"] == "merge"
        assert row["patids"] == "P3"
        assert set(row["related_patids"].split(",")) == {"P1", "P2"}

    def test_merge_multiple_patids_records_all_prior_members(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        resp = client.post(
            "/audit/merge",
            json={"mid": matched_mid, "patids": ["P4", "P5"]},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert resp.status_code == 200

        row = client.get("/audit").json()[0]
        assert set(row["patids"].split(",")) == {"P4", "P5"}
        assert set(row["related_patids"].split(",")) == {"P1", "P2"}

    def test_unmerge_creates_new_entity(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        resp = client.post(
            "/audit/unmerge",
            json={"mid": matched_mid, "patid": "P2"},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["new_mid"] != matched_mid
        assert [m["patid"] for m in body["entity"]["members"]] == ["P2"]

        # The old entity now has only P1 left and is no longer "merged".
        old = client.get(f"/clusters/{matched_mid}").json()
        assert [m["patid"] for m in old["members"]] == ["P1"]
        assert old["is_merged"] is False

    def test_unmerge_records_who_stayed_behind(self, client, test_settings):
        """related_patids on an unmerge row is who remained in the original
        entity -- captured at the moment of the split, not reconstructed
        later (audit_log.mid itself only ever holds the NEW standalone mid)."""
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        resp = client.post(
            "/audit/unmerge",
            json={"mid": matched_mid, "patid": "P2"},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert resp.status_code == 200

        row = client.get("/audit").json()[0]
        assert row["action"] == "unmerge"
        assert row["patids"] == "P2"
        assert row["related_patids"] == "P1"

    def test_dismiss_does_not_set_related_patids(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.post(
            "/audit/dismiss",
            json={"patid_a": "P4", "patid_b": "P5"},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert resp.status_code == 200

        row = client.get("/audit").json()[0]
        assert row["action"] == "dismiss"
        assert row["related_patids"] is None

    def test_unmerge_wrong_mid_404s(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.post(
            "/audit/unmerge",
            json={"mid": "M-999999", "patid": "P1"},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert resp.status_code == 404

    def test_unmerge_is_sticky_across_republish(self, client, test_settings):
        """The reconciliation contract end-to-end: an unmerged PATID must not
        be silently re-merged by a later publish of the same run."""
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        client.post(
            "/audit/unmerge",
            json={"mid": matched_mid, "patid": "P2"},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )

        from src.api.ingest import publish

        backend = build_index_backend(test_settings)
        try:
            publish.publish_run(backend, "r1", test_settings)
        finally:
            backend.close()

        resp = client.get("/records", params={"search": "Jane"})
        p2_entities = [
            item for item in resp.json()["items"]
            for m in item["members"] if m["patid"] == "P2"
        ]
        assert len(p2_entities) == 1
        assert len(p2_entities[0]["members"]) == 1  # still standalone

    def test_undo_requires_reviewer_header(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.post("/audit/1/undo")
        assert resp.status_code == 401

    def test_undo_unknown_audit_id_404s(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.post(
            "/audit/999999/undo", headers={"X-Reviewer-Id": "reviewer.jclark"}
        )
        assert resp.status_code == 404

    def test_undo_dismiss_action_400s(self, client, test_settings):
        """`dismiss` never mutates entities/members, so there's nothing for
        `/undo` to reverse — it must reject the action outright, not silently
        no-op."""
        _publish_fixture_run(test_settings, "r1")
        dismiss_resp = client.post(
            "/audit/dismiss",
            json={"patid_a": "P4", "patid_b": "P5"},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        audit_id = dismiss_resp.json()["audit_id"]
        resp = client.post(
            f"/audit/{audit_id}/undo", headers={"X-Reviewer-Id": "reviewer.jclark"}
        )
        assert resp.status_code == 400

    def test_undo_merge_unmerges_each_patid(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        merge_resp = client.post(
            "/audit/merge",
            json={"mid": matched_mid, "patids": ["P3"]},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        audit_id = merge_resp.json()["audit_id"]

        resp = client.post(
            f"/audit/{audit_id}/undo", headers={"X-Reviewer-Id": "reviewer.jclark"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["reversed_action"] == "merge"
        assert len(body["new_mids"]) == 1

        p3_mid = body["new_mids"][0]
        p3_entity = client.get(f"/clusters/{p3_mid}").json()
        assert [m["patid"] for m in p3_entity["members"]] == ["P3"]

        matched = client.get(f"/clusters/{matched_mid}").json()
        assert {m["patid"] for m in matched["members"]} == {"P1", "P2"}

        audit_rows = client.get("/audit").json()
        merge_row = next(r for r in audit_rows if r["id"] == audit_id)
        assert merge_row["undone"] is True

    def test_undo_unmerge_remerges_patid(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        unmerge_resp = client.post(
            "/audit/unmerge",
            json={"mid": matched_mid, "patid": "P2"},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        audit_id = unmerge_resp.json()["audit_id"]

        resp = client.post(
            f"/audit/{audit_id}/undo", headers={"X-Reviewer-Id": "reviewer.jclark"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["reversed_action"] == "unmerge"
        assert body["entity"]["mid"] == matched_mid
        assert {m["patid"] for m in body["entity"]["members"]} == {"P1", "P2"}

        audit_rows = client.get("/audit").json()
        unmerge_row = next(r for r in audit_rows if r["id"] == audit_id)
        assert unmerge_row["undone"] is True

    def test_undo_twice_400s(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        merge_resp = client.post(
            "/audit/merge",
            json={"mid": matched_mid, "patids": ["P3"]},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        audit_id = merge_resp.json()["audit_id"]
        client.post(
            f"/audit/{audit_id}/undo", headers={"X-Reviewer-Id": "reviewer.jclark"}
        )
        resp = client.post(
            f"/audit/{audit_id}/undo", headers={"X-Reviewer-Id": "reviewer.jclark"}
        )
        assert resp.status_code == 400

    def test_undo_unmerge_predating_migration_400s(self, client, test_settings):
        """An `unmerge` row written before `prev_mid` existed (or by anything
        that doesn't set it) can't be reversed — there's nowhere recorded to
        re-merge the patid into."""
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        backend = build_index_backend(test_settings)
        try:
            backend.begin()
            audit_id = backend.insert_audit_log(
                ts_utc="2026-01-01T00:00:00Z", user="legacy", action="unmerge",
                patids="P2", mid=matched_mid,
                prev_state="Merged", next_state="Unmerged", run_id="r1",
            )
            backend.commit()
        finally:
            backend.close()

        resp = client.post(
            f"/audit/{audit_id}/undo", headers={"X-Reviewer-Id": "reviewer.jclark"}
        )
        assert resp.status_code == 400


class TestDashboard:
    def test_summary(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        resp = client.get("/dashboard/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 5
        assert body["duplicate_clusters"] == 1
        assert body["matched_records"] == 2
        assert body["needs_review_records"] == 2
        assert body["status_counts"]["auto_match"] == 2
        assert body["status_counts"]["needs_review"] == 2
        assert body["status_counts"]["no_match"] == 1
        assert body["last_run_id"] == "r1"
        assert "SSN_DOB" in body["confidence_thresholds"]
        assert len(body["history"]) == 1

    def test_summary_reflects_manual_merge(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        client.post(
            "/audit/merge",
            json={"mid": matched_mid, "patids": ["P3"]},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        resp = client.get("/dashboard/summary")
        assert resp.json()["manual_merge_actions"] == 1

    def test_summary_empty_state(self, client, test_settings):
        resp = client.get("/dashboard/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 0
        assert body["last_run_id"] is None


class TestRecordsAndDashboardAgainstParquetBackend:
    """`GET /records`, `GET /clusters/{mid}`, `GET /records/{patid}/raw`, and
    `GET /dashboard/summary` run against `ParquetIndexBackend`
    (`EMPI_INDEX_BACKEND=parquet`) — the same fixture run as
    `TestRecords`/`TestDashboard`'s SQLite coverage, proving `records.py`/
    `dashboard.py` are backend-agnostic via `get_backend`, not hardcoded to
    SQLite. Audit parity (`/audit/merge`, `/audit/unmerge`) is covered
    separately in `TestAuditAgainstParquetBackend` below."""

    def test_list_records(self, client, parquet_test_settings):
        _publish_fixture_run(parquet_test_settings, "r1")
        resp = client.get("/records")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 4
        mids = {item["mid"] for item in body["items"]}
        assert len(mids) == 4

    def test_review_status_entity_has_candidates(self, client, parquet_test_settings):
        _publish_fixture_run(parquet_test_settings, "r1")
        resp = client.get("/records", params={"origin": "review"})
        body = resp.json()
        assert body["total"] == 2
        entity = body["items"][0]
        assert len(entity["review_candidates"]) == 1
        assert entity["review_candidates"][0]["match_rule"] == "NAME_DOB_SEX"

    def test_get_raw_record(self, client, parquet_test_settings):
        _publish_fixture_run(parquet_test_settings, "r1")
        resp = client.get("/records/P1/raw")
        assert resp.status_code == 200
        assert resp.json()["fields"]["FirstNM_raw"] == "JANE"

    def test_get_clean_ssn(self, client, parquet_test_settings):
        _publish_fixture_run(parquet_test_settings, "r1")
        resp = client.get("/records/P1/ssn-clean")
        assert resp.status_code == 200
        assert resp.json() == {"patid": "P1", "ssn": "123456789"}

    def test_search_filters(self, client, parquet_test_settings):
        _publish_fixture_run(parquet_test_settings, "r1")
        resp = client.get("/records", params={"search": "Smith"})
        assert resp.json()["total"] == 1

    def test_get_cluster(self, client, parquet_test_settings):
        _publish_fixture_run(parquet_test_settings, "r1")
        mid = client.get("/records", params={"is_merged": "true"}).json()["items"][0]["mid"]
        resp = client.get(f"/clusters/{mid}")
        assert resp.status_code == 200
        assert len(resp.json()["members"]) == 2

    def test_dashboard_summary(self, client, parquet_test_settings):
        _publish_fixture_run(parquet_test_settings, "r1")
        resp = client.get("/dashboard/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 5
        assert body["duplicate_clusters"] == 1
        assert body["matched_records"] == 2
        assert body["needs_review_records"] == 2
        assert body["status_counts"]["auto_match"] == 2
        assert body["status_counts"]["needs_review"] == 2
        assert body["status_counts"]["no_match"] == 1
        # No audit_log table in local mode yet — always 0 until Phase 3.
        assert body["manual_merge_actions"] == 0

    def test_incremental_score_visible_through_records(
        self, client, parquet_test_settings, monkeypatch
    ):
        """The two write paths this operationalization work targets — batch
        publish and incremental scoring — land in the same Parquet index and
        are both visible through the same read routes."""
        _publish_fixture_run(parquet_test_settings, "r1")
        monkeypatch.setattr(
            parquet_test_settings, "fs_model_dir",
            parquet_test_settings.project_root / "no_fs_model",
        )
        resp = client.post(
            "/records/score",
            json={"records": [_incoming("P6", "Amy", "Lee", "1975-03-03", "FEMALE")]},
        )
        run_id = resp.json()["run_id"]
        result = client.get(f"/records/score/{run_id}").json()
        assert result["status"] == "succeeded"
        mid = result["outcomes"][0]["mid"]

        entity = client.get(f"/clusters/{mid}").json()
        assert any(m["patid"] == "P6" for m in entity["members"])


class TestAuditAgainstParquetBackend:
    """`POST /audit/merge`, `POST /audit/unmerge`, `GET /audit` against
    `ParquetIndexBackend` — mirrors `TestAudit`'s SQLite coverage, proving
    `audit.py` is backend-agnostic via `get_backend`."""

    def test_merge_then_audit_feed(self, client, parquet_test_settings):
        _publish_fixture_run(parquet_test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        resp = client.post(
            "/audit/merge",
            json={"mid": matched_mid, "patids": ["P3"]},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert {m["patid"] for m in body["entity"]["members"]} == {"P1", "P2", "P3"}

        audit_resp = client.get("/audit")
        assert audit_resp.status_code == 200
        assert audit_resp.json()[0]["action"] == "merge"
        assert set(audit_resp.json()[0]["related_patids"].split(",")) == {"P1", "P2"}

        # Now reflected in the dashboard summary too.
        summary = client.get("/dashboard/summary").json()
        assert summary["manual_merge_actions"] == 1

    def test_unmerge_creates_new_entity(self, client, parquet_test_settings):
        _publish_fixture_run(parquet_test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        resp = client.post(
            "/audit/unmerge",
            json={"mid": matched_mid, "patid": "P2"},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["new_mid"] != matched_mid
        assert [m["patid"] for m in body["entity"]["members"]] == ["P2"]

        old = client.get(f"/clusters/{matched_mid}").json()
        assert [m["patid"] for m in old["members"]] == ["P1"]
        assert old["is_merged"] is False

    def test_unmerge_is_sticky_across_republish(self, client, parquet_test_settings):
        """The reconciliation contract end-to-end on the Parquet backend: an
        unmerged PATID must not be silently re-merged by a later publish of
        the same run."""
        _publish_fixture_run(parquet_test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )
        client.post(
            "/audit/unmerge",
            json={"mid": matched_mid, "patid": "P2"},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )

        from src.api.ingest import publish

        backend = build_index_backend(parquet_test_settings)
        try:
            publish.publish_run(backend, "r1", parquet_test_settings)
        finally:
            backend.close()

        resp = client.get("/records", params={"search": "Jane"})
        p2_entities = [
            item for item in resp.json()["items"]
            for m in item["members"] if m["patid"] == "P2"
        ]
        assert len(p2_entities) == 1
        assert len(p2_entities[0]["members"]) == 1  # still standalone

    def test_undo_merge_and_unmerge(self, client, parquet_test_settings):
        """`/audit/{id}/undo` against `ParquetIndexBackend` — mirrors
        `TestAudit`'s SQLite undo coverage."""
        _publish_fixture_run(parquet_test_settings, "r1")
        matched_mid = next(
            item["mid"] for item in client.get("/records").json()["items"]
            if item["is_merged"]
        )

        merge_resp = client.post(
            "/audit/merge",
            json={"mid": matched_mid, "patids": ["P3"]},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        merge_audit_id = merge_resp.json()["audit_id"]
        undo_merge = client.post(
            f"/audit/{merge_audit_id}/undo",
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert undo_merge.status_code == 200
        assert undo_merge.json()["reversed_action"] == "merge"
        matched = client.get(f"/clusters/{matched_mid}").json()
        assert {m["patid"] for m in matched["members"]} == {"P1", "P2"}

        unmerge_resp = client.post(
            "/audit/unmerge",
            json={"mid": matched_mid, "patid": "P2"},
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        unmerge_audit_id = unmerge_resp.json()["audit_id"]
        undo_unmerge = client.post(
            f"/audit/{unmerge_audit_id}/undo",
            headers={"X-Reviewer-Id": "reviewer.jclark"},
        )
        assert undo_unmerge.status_code == 200
        body = undo_unmerge.json()
        assert body["reversed_action"] == "unmerge"
        assert {m["patid"] for m in body["entity"]["members"]} == {"P1", "P2"}


class TestSqliteWriteConcurrency:
    """SQLite allows one writer at a time and, without `busy_timeout`, fails a
    blocked writer immediately rather than waiting (`sql_backend.get_connection`).

    That bites hardest on the routes that *look* like reads: `GET
    /records/{patid}/raw` appends a `view_raw` audit row on every call, so a UI
    that opens several records at once — the Patient Registry's raw comparison
    panel does exactly this, one request per cluster member — fired N
    concurrent writes and got `OperationalError: database is locked` back on
    all but one. The failures rendered as blank columns, which a data steward
    reads as "the source system had nothing" rather than "the request failed".
    """

    def test_concurrent_raw_views_all_succeed(self, client, test_settings):
        _publish_fixture_run(test_settings, "r1")
        patids = ["P1", "P2", "P3", "P4", "P5"]

        def _view(patid: str) -> int:
            return client.get(
                f"/records/{patid}/raw",
                headers={"X-Reviewer-Id": "reviewer.klkendall"},
            ).status_code

        with ThreadPoolExecutor(max_workers=len(patids)) as pool:
            statuses = list(pool.map(_view, patids * 3))

        assert all(s == 200 for s in statuses), (
            f"expected every concurrent raw view to succeed, got {statuses}"
        )
        # Every one of them is PHI access that has to be on the record — a
        # write silently lost to lock contention would be an audit gap, not
        # just a UI glitch.
        actions = [row["action"] for row in _audit_rows(test_settings)]
        assert actions.count("view_raw") == len(patids) * 3


def _audit_rows(settings) -> list[dict]:
    conn = sql_backend.get_connection(settings.db_path)
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM audit_log").fetchall()]
    finally:
        conn.close()


class TestParquetBackendConcurrency:
    """Real overlapping requests against the Parquet backend, dispatched from
    multiple threads through the actual ASGI app (not just a direct call to
    `get_backend`) — `deps._PARQUET_BACKEND_LOCK` must prevent a crash/corrupt
    read even under genuine concurrent load. `test_api_deps.py` covers the
    lock's acquire/release mechanics in isolation; this is the end-to-end
    proof it's wired into the real request path."""

    def test_concurrent_merges_do_not_corrupt_state(self, client, parquet_test_settings):
        _publish_fixture_run(parquet_test_settings, "r1")
        items = client.get("/records").json()["items"]
        review_mids = [item["mid"] for item in items if item["origin"] == "review"]
        assert len(review_mids) == 2  # P4, P5 — each its own singleton

        def _merge(mid: str) -> int:
            resp = client.post(
                "/audit/merge",
                json={"mid": review_mids[0], "patids": [
                    m["patid"] for e in items if e["mid"] == mid for m in e["members"]
                ]},
                headers={"X-Reviewer-Id": f"reviewer.{mid}"},
            )
            return resp.status_code

        with ThreadPoolExecutor(max_workers=8) as pool:
            statuses = list(pool.map(_merge, review_mids * 4))

        assert all(s == 200 for s in statuses)
        # Every merge targeted review_mids[0] — it must end up with both
        # PATIDs as members exactly once each, never duplicated or dropped.
        final = client.get(f"/clusters/{review_mids[0]}").json()
        assert {m["patid"] for m in final["members"]} == {"P4", "P5"}
        assert len(final["members"]) == 2
