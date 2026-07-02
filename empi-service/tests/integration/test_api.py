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

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import jobs, store
from src.api.main import app
from src.config import Settings, settings as real_settings
from src.contracts import ArtifactRef, RunManifest


@pytest.fixture
def test_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(real_settings, "project_root", tmp_path)
    monkeypatch.setattr(real_settings, "raw_dir", tmp_path / "data" / "raw")
    monkeypatch.setattr(real_settings, "processed_dir", tmp_path / "data" / "processed")
    monkeypatch.setattr(real_settings, "blocking_dir", tmp_path / "data" / "blocking")
    monkeypatch.setattr(real_settings, "matches_dir", tmp_path / "data" / "matches")
    monkeypatch.setattr(real_settings, "non_matches_dir", tmp_path / "data" / "non_matches")
    monkeypatch.setattr(real_settings, "rejects_dir", tmp_path / "data" / "rejects")
    monkeypatch.setattr(real_settings, "clusters_dir", tmp_path / "data" / "clusters")
    monkeypatch.setattr(real_settings, "runs_dir", tmp_path / "data" / "runs")
    monkeypatch.setattr(real_settings, "db_path", tmp_path / "empi.db")
    real_settings.ensure_dirs()
    return real_settings


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
    from src.api import publish

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
    matches_path = settings.matches_dir / f"matches_{run_id}.parquet"
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

    conn = store.get_connection(settings.db_path)
    try:
        store.init_db(conn)
        publish.publish_run(conn, run_id, settings)
    finally:
        conn.close()


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_ready(self, client):
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["checks"]["db"] is True


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

        conn = store.get_connection(test_settings.db_path)
        try:
            from src.api import publish
            publish.publish_run(conn, "r1", test_settings)
        finally:
            conn.close()

        resp = client.get("/records", params={"search": "Jane"})
        p2_entities = [
            item for item in resp.json()["items"]
            for m in item["members"] if m["patid"] == "P2"
        ]
        assert len(p2_entities) == 1
        assert len(p2_entities[0]["members"]) == 1  # still standalone


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
