"""Unit tests for src/api/publish.py against a hand-built manifest + Parquet
fixtures — no need to run the real pipeline.

Coverage:
    - A fresh publish creates entities/members/record_attrs for a run with a
      matched pair and a singleton.
    - Re-publishing the same run is idempotent (upsert, not duplicate).
    - Reconciliation: once a PATID is reviewer-locked (appears in audit_log),
      a later publish never repoints its entity_member.mid — instead it
      writes an entity_suggestion row. (docs/API-Design.md §2, "sticky unmerge".)
"""

import sqlite3

import pandas as pd
import pytest

from src.api import publish, store
from src.config import Settings
from src.contracts import ArtifactRef, RunManifest


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    store.init_db(c)
    yield c
    c.close()


@pytest.fixture
def fixture_settings(tmp_path):
    settings = Settings(project_root=tmp_path)
    settings.runs_dir = tmp_path / "data" / "runs"
    settings.clusters_dir = tmp_path / "data" / "clusters"
    settings.matches_dir = tmp_path / "data" / "matches"
    settings.processed_dir = tmp_path / "data" / "processed"
    settings.ensure_dirs()
    return settings


def _write_run(settings: Settings, run_id: str):
    cleaned = pd.DataFrame({
        "PATID": ["P1", "P2", "P3"],
        "FirstNM_clean": ["Jane", "Jane", "John"],
        "LastNM_clean": ["Doe", "Doe", "Smith"],
        "BirthDT_clean": pd.to_datetime(["1990-01-01", "1990-01-01", "1985-05-05"]),
        "SSN_clean": ["123456789", "123456789", None],
        "last_4_SSN": ["6789", "6789", None],
        "Email_clean": [None, None, None],
        "ZipCD_clean_base": ["60601", "60601", None],
        "AddressLine1_clean": [None, None, None],
        "SexAtBirthDSC_clean": ["FEMALE", "FEMALE", "MALE"],
        "valid_record": [True, True, True],
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

    clusters = pd.DataFrame({"PATID": ["P1", "P2", "P3"], "cluster_id": [0, 0, 1]})
    clusters_path = settings.clusters_dir / f"clusters_{run_id}.parquet"
    clusters.to_parquet(clusters_path, index=False)

    def ref(path, rows):
        return ArtifactRef(path=str(path.relative_to(settings.project_root)), rows=rows, sha256="x")

    manifest = RunManifest(
        run_id=run_id, created_utc="2026-07-01T00:00:00Z",
        raw_input=ref(cleaned_path, 3), cleaned=ref(cleaned_path, 3),
        candidate_pairs=ref(matches_path, 1), matches=ref(matches_path, 1),
        non_matches=ref(matches_path, 0), clusters=ref(clusters_path, 3),
        counts={},
    )
    (settings.runs_dir / f"run_{run_id}.json").write_text(manifest.model_dump_json())
    return manifest


class TestPublishRun:
    def test_fresh_publish_creates_entities(self, conn, fixture_settings):
        _write_run(fixture_settings, "r1")
        counts = publish.publish_run(conn, "r1", fixture_settings)

        assert counts["clusters_seen"] == 2
        assert counts["entities_upserted"] == 2
        assert counts["members_upserted"] == 3
        assert counts["locked_skipped"] == 0

        matched = store.get_entity_mid_for_patid(conn, "P1")
        assert matched == store.get_entity_mid_for_patid(conn, "P2")
        singleton_mid = store.get_entity_mid_for_patid(conn, "P3")
        assert singleton_mid != matched

        matched_entity = store.get_entity(conn, matched)
        assert matched_entity["entity"]["is_merged"] == 1
        assert matched_entity["entity"]["confidence"] == 1.0
        singleton_entity = store.get_entity(conn, singleton_mid)
        assert singleton_entity["entity"]["is_merged"] == 0

    def test_record_attrs_denormalized(self, conn, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(conn, "r1", fixture_settings)
        row = conn.execute(
            "SELECT * FROM record_attrs WHERE patid='P1'"
        ).fetchone()
        assert row["first_name"] == "Jane"
        assert row["ssn_last4"] == "6789"
        assert row["birth_date"] == "1990-01-01"

    def test_republish_is_idempotent(self, conn, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(conn, "r1", fixture_settings)
        mid_before = store.get_entity_mid_for_patid(conn, "P1")

        publish.publish_run(conn, "r1", fixture_settings)
        mid_after = store.get_entity_mid_for_patid(conn, "P1")

        assert mid_before == mid_after
        n_entities = conn.execute("SELECT COUNT(*) AS n FROM entity").fetchone()["n"]
        assert n_entities == 2

    def test_reviewer_locked_patid_not_repointed(self, conn, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(conn, "r1", fixture_settings)

        # Reviewer splits P2 out into its own entity (simulating POST /audit/unmerge).
        store.upsert_entity(conn, "M-999999", "r1", "none", False, None, "t0")
        store.upsert_entity_member(conn, "P2", "M-999999", True, "reviewer.jclark", "t0")
        store.insert_audit_log(
            conn, ts_utc="t0", user="reviewer.jclark", action="unmerge",
            patids="P2", mid="M-999999", prev_state="Merged",
            next_state="Unmerged", run_id="r1",
        )
        conn.commit()

        counts = publish.publish_run(conn, "r1", fixture_settings)

        # P2 stays put in its reviewer-created entity — never repointed back to P1's.
        assert store.get_entity_mid_for_patid(conn, "P2") == "M-999999"
        assert counts["locked_skipped"] == 1

        suggestion = conn.execute(
            "SELECT * FROM entity_suggestion WHERE patid='P2'"
        ).fetchone()
        assert suggestion is not None
        assert suggestion["run_id"] == "r1"

    def test_missing_manifest_raises(self, conn, fixture_settings):
        with pytest.raises(FileNotFoundError):
            publish.publish_run(conn, "does-not-exist", fixture_settings)
