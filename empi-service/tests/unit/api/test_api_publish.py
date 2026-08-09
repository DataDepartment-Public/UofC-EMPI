"""Unit tests for src/api/ingest/publish.py against a hand-built manifest + Parquet
fixtures — no need to run the real pipeline.

Coverage:
    - A fresh publish creates entities/members/record_attrs for a run with a
      matched pair, a true singleton, and a review-tier candidate pair.
    - Re-publishing the same run is idempotent (upsert, not duplicate).
    - Reconciliation: once a PATID is reviewer-locked (appears in audit_log),
      a later publish never repoints its entity_member.mid — instead it
      writes an entity_suggestion row. (docs/API-Design.md §2, "sticky unmerge".)
    - Review-tier pairs (non_matches + review_evidence) become
      review_candidate rows and upgrade a singleton's origin to 'review'.
    - Raw fields land in record_raw.
"""

import sqlite3

import pandas as pd
import pytest

from src.api.ingest import publish
from src.api.backends import sql_backend
from src.api.backends.index_backend import SqlIndexBackend
from src.api.backends.parquet_backend import ParquetIndexBackend
from src.config import Settings
from src.contracts import ArtifactRef, RunManifest


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    sql_backend.init_db(c)
    yield c
    c.close()


@pytest.fixture
def backend(conn):
    return SqlIndexBackend(conn)


@pytest.fixture
def fixture_settings(tmp_path):
    settings = Settings(project_root=tmp_path)
    settings.runs_dir = tmp_path / "data" / "runs"
    settings.clusters_dir = tmp_path / "data" / "clusters"
    settings.auto_merge_dir = tmp_path / "data" / "auto_merge"
    settings.non_matches_dir = tmp_path / "data" / "non_matches"
    settings.processed_dir = tmp_path / "data" / "processed"
    settings.ml_output_dir = tmp_path / "data" / "ml_output"
    settings.ensure_dirs()
    return settings


def _write_run(settings: Settings, run_id: str):
    # P1<->P2 auto-merge (SSN_DOB); P3 true singleton; P4<->P5 review-tier
    # candidate (NAME_DOB_SEX) — P4/P5 never appear in `matches`.
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
        "high_fanout_ssn": [False],
        "source_blocks": ["B3"], "n_blocks": [1],
    })
    review_evidence_path = settings.non_matches_dir / f"review_evidence_{run_id}.parquet"
    review_evidence.to_parquet(review_evidence_path, index=False)

    clusters = pd.DataFrame({
        "PATID": ["P1", "P2", "P3", "P4", "P5"], "cluster_id": [0, 0, 1, 2, 3],
    })
    clusters_path = settings.clusters_dir / f"clusters_{run_id}.parquet"
    clusters.to_parquet(clusters_path, index=False)

    def ref(path, rows):
        return ArtifactRef(path=str(path.relative_to(settings.project_root)), rows=rows, sha256="x")

    manifest = RunManifest(
        run_id=run_id, created_utc="2026-07-01T00:00:00Z",
        raw_input=ref(cleaned_path, 5), cleaned=ref(cleaned_path, 5),
        candidate_pairs=ref(matches_path, 1), matches=ref(matches_path, 1),
        non_matches=ref(non_matches_path, 1),
        review_evidence=ref(review_evidence_path, 1),
        clusters=ref(clusters_path, 5),
        counts={},
    )
    (settings.runs_dir / f"run_{run_id}.json").write_text(manifest.model_dump_json())
    return manifest


def _write_run_with_gate_drop(settings: Settings, run_id: str):
    # Same P1-P5 shape as `_write_run`, plus P6<->P7: a non_matches pair the
    # Stage-4.25 gate confidently scores TIER_NO_MATCH — it must not become
    # a review_candidate row, unlike P4<->P5 (gate tier human_review).
    cleaned = pd.DataFrame({
        "PATID": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
        "FirstNM_clean": ["Jane", "Jane", "John", "Amy", "Amy", "Bob", "Bob"],
        "LastNM_clean": ["Doe", "Doe", "Smith", "Lee", "Lee", "Ray", "Ray"],
        "BirthDT_clean": pd.to_datetime([
            "1990-01-01", "1990-01-01", "1985-05-05", "1975-03-03", "1975-03-03",
            "1970-06-06", "1970-06-06",
        ]),
        "SSN_clean": ["123456789", "123456789", None, None, None, None, None],
        "last_4_SSN": ["6789", "6789", None, None, None, None, None],
        "Email_clean": [None] * 7,
        "ZipCD_clean_base": ["60601", "60601", None, None, None, None, None],
        "AddressLine1_clean": [None] * 7,
        "SexAtBirthDSC_clean": ["FEMALE", "FEMALE", "MALE", "FEMALE", "FEMALE", "MALE", "MALE"],
        "Phones_set": [set()] * 7,
        "FirstNM_raw": ["JANE", "JANE", "JOHN", "AMY", "AMY", "BOB", "BOB"],
        "SSN_raw": ["123-45-6789", "123456789", None, None, None, None, None],
        "valid_record": [True] * 7,
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
        "PATID_A": ["P4", "P6"], "PATID_B": ["P5", "P7"],
        "source_blocks": ["B3", "B4"], "n_blocks": [1, 1],
    })
    non_matches_path = settings.non_matches_dir / f"non_matches_{run_id}.parquet"
    non_matches.to_parquet(non_matches_path, index=False)

    review_evidence = pd.DataFrame({
        "PATID_A": ["P4"], "PATID_B": ["P5"],
        "match_rule": ["NAME_DOB_SEX"], "confidence": [0.98],
        "rules_fired": ["NAME_DOB_SEX"], "is_suspicious": [False],
        "high_fanout_ssn": [False],
        "source_blocks": ["B3"], "n_blocks": [1],
    })
    review_evidence_path = settings.non_matches_dir / f"review_evidence_{run_id}.parquet"
    review_evidence.to_parquet(review_evidence_path, index=False)

    gate_results = pd.DataFrame({
        "PATID_A": ["P4", "P6"], "PATID_B": ["P5", "P7"],
        "model_name": ["nonmatch_gate", "nonmatch_gate"],
        "score": [0.91, 0.001],
        "predicted_tier": ["human_review", "no_match"],
    })
    gate_results_path = settings.non_matches_dir / f"gate_results_{run_id}.parquet"
    gate_results.to_parquet(gate_results_path, index=False)

    clusters = pd.DataFrame({
        "PATID": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
        "cluster_id": [0, 0, 1, 2, 3, 4, 5],
    })
    clusters_path = settings.clusters_dir / f"clusters_{run_id}.parquet"
    clusters.to_parquet(clusters_path, index=False)

    def ref(path, rows):
        return ArtifactRef(path=str(path.relative_to(settings.project_root)), rows=rows, sha256="x")

    manifest = RunManifest(
        run_id=run_id, created_utc="2026-07-01T00:00:00Z",
        raw_input=ref(cleaned_path, 7), cleaned=ref(cleaned_path, 7),
        candidate_pairs=ref(matches_path, 1), matches=ref(matches_path, 1),
        non_matches=ref(non_matches_path, 2),
        review_evidence=ref(review_evidence_path, 1),
        gate_results=ref(gate_results_path, 2),
        clusters=ref(clusters_path, 7),
        counts={},
    )
    (settings.runs_dir / f"run_{run_id}.json").write_text(manifest.model_dump_json())
    return manifest


def _write_run_with_ml_scores(settings: Settings, run_id: str):
    # Same P1-P5 shape as `_write_run`, but P4<->P5 has no rule confirmation
    # (no review_evidence row) — it only carries an ML matcher score, the
    # common case that motivated threading `ml_features` into
    # `_review_candidate_rows` (most review-queue pairs have no rule at
    # all, only an ML score).
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

    ml_features = pd.DataFrame({
        "PATID_A": ["P4"], "PATID_B": ["P5"],
        "match_probability": [0.24], "classification_tier": ["human_review"],
    })
    ml_features_path = settings.ml_output_dir / f"ml_features_{run_id}.parquet"
    ml_features.to_parquet(ml_features_path, index=False)

    clusters = pd.DataFrame({
        "PATID": ["P1", "P2", "P3", "P4", "P5"], "cluster_id": [0, 0, 1, 2, 3],
    })
    clusters_path = settings.clusters_dir / f"clusters_{run_id}.parquet"
    clusters.to_parquet(clusters_path, index=False)

    def ref(path, rows):
        return ArtifactRef(path=str(path.relative_to(settings.project_root)), rows=rows, sha256="x")

    manifest = RunManifest(
        run_id=run_id, created_utc="2026-07-01T00:00:00Z",
        raw_input=ref(cleaned_path, 5), cleaned=ref(cleaned_path, 5),
        candidate_pairs=ref(matches_path, 1), matches=ref(matches_path, 1),
        non_matches=ref(non_matches_path, 1),
        ml_features=ref(ml_features_path, 1),
        clusters=ref(clusters_path, 5),
        counts={},
    )
    (settings.runs_dir / f"run_{run_id}.json").write_text(manifest.model_dump_json())
    return manifest


class TestPublishRun:
    def test_fresh_publish_creates_entities(self, conn, backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        counts = publish.publish_run(backend, "r1", fixture_settings)

        assert counts["clusters_seen"] == 4
        assert counts["entities_upserted"] == 4
        assert counts["members_upserted"] == 5
        assert counts["locked_skipped"] == 0
        assert counts["review_candidates"] == 1

        matched = sql_backend.get_entity_mid_for_patid(conn, "P1")
        assert matched == sql_backend.get_entity_mid_for_patid(conn, "P2")
        singleton_mid = sql_backend.get_entity_mid_for_patid(conn, "P3")
        assert singleton_mid != matched

        matched_entity = sql_backend.get_entity(conn, matched)
        assert matched_entity["entity"]["is_merged"] == 1
        assert matched_entity["entity"]["confidence"] == 1.0
        assert matched_entity["entity"]["match_rule"] == "SSN_DOB"
        singleton_entity = sql_backend.get_entity(conn, singleton_mid)
        assert singleton_entity["entity"]["is_merged"] == 0
        assert singleton_entity["entity"]["origin"] == "none"

    def test_review_candidate_upgrades_origin(self, conn, backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(backend, "r1", fixture_settings)

        p4_mid = sql_backend.get_entity_mid_for_patid(conn, "P4")
        p4_entity = sql_backend.get_entity(conn, p4_mid)
        assert p4_entity["entity"]["origin"] == "review"

        candidates = sql_backend.review_candidates_for_patid(conn, "P4")
        assert len(candidates) == 1
        assert candidates[0]["match_rule"] == "NAME_DOB_SEX"
        assert candidates[0]["confidence"] == 0.98

    def test_review_candidate_carries_ml_score(self, conn, backend, fixture_settings):
        _write_run_with_ml_scores(fixture_settings, "r-ml")
        publish.publish_run(backend, "r-ml", fixture_settings)

        candidates = sql_backend.review_candidates_for_patid(conn, "P4")
        assert len(candidates) == 1
        # No rule fired for P4<->P5 in this fixture — confidence stays null,
        # but the ML matcher's score must still land on the row.
        assert candidates[0]["confidence"] is None
        assert candidates[0]["ml_match_probability"] == pytest.approx(0.24)
        assert candidates[0]["ml_classification_tier"] == "human_review"

        # The queue list/sort/filter fall back to the ML score when there's
        # no rule confidence (sql_backend.list_review_candidates' COALESCE).
        rows, total = sql_backend.list_review_candidates(conn, reviewed=False)
        assert total == 1
        assert rows[0]["ml_match_probability"] == pytest.approx(0.24)
        filtered_rows, filtered_total = sql_backend.list_review_candidates(
            conn, reviewed=False, confidence_min=0.2
        )
        assert filtered_total == 1
        assert filtered_rows[0]["patid_a"] == "P4"

    def test_gate_dropped_pair_resolves_to_none_not_review(self, conn, backend, fixture_settings):
        _write_run_with_gate_drop(fixture_settings, "r2")
        counts = publish.publish_run(backend, "r2", fixture_settings)

        # Only P4<->P5 (gate tier human_review) becomes a review_candidate
        # row — P6<->P7 (gate tier no_match) is excluded despite being in
        # non_matches, since the gate already resolved it confidently.
        assert counts["review_candidates"] == 1

        p4_mid = sql_backend.get_entity_mid_for_patid(conn, "P4")
        assert sql_backend.get_entity(conn, p4_mid)["entity"]["origin"] == "review"

        p6_mid = sql_backend.get_entity_mid_for_patid(conn, "P6")
        p7_mid = sql_backend.get_entity_mid_for_patid(conn, "P7")
        assert sql_backend.get_entity(conn, p6_mid)["entity"]["origin"] == "none"
        assert sql_backend.get_entity(conn, p7_mid)["entity"]["origin"] == "none"
        assert sql_backend.review_candidates_for_patid(conn, "P6") == []

    def test_record_attrs_denormalized(self, conn, backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(backend, "r1", fixture_settings)
        row = conn.execute(
            "SELECT * FROM record_attrs WHERE patid='P1'"
        ).fetchone()
        assert row["first_name"] == "Jane"
        assert row["ssn_last4"] == "6789"
        assert row["birth_date"] == "1990-01-01"

    def test_raw_fields_denormalized(self, conn, backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(backend, "r1", fixture_settings)
        raw_json = sql_backend.get_record_raw(conn, "P1")
        assert raw_json is not None
        assert "JANE" in raw_json

    def test_republish_is_idempotent(self, conn, backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(backend, "r1", fixture_settings)
        mid_before = sql_backend.get_entity_mid_for_patid(conn, "P1")

        publish.publish_run(backend, "r1", fixture_settings)
        mid_after = sql_backend.get_entity_mid_for_patid(conn, "P1")

        assert mid_before == mid_after
        n_entities = conn.execute("SELECT COUNT(*) AS n FROM entity").fetchone()["n"]
        assert n_entities == 4
        n_candidates = conn.execute(
            "SELECT COUNT(*) AS n FROM review_candidate"
        ).fetchone()["n"]
        assert n_candidates == 1  # replaced, not duplicated

    def test_reviewer_locked_patid_not_repointed(self, conn, backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(backend, "r1", fixture_settings)

        # Reviewer splits P2 out into its own entity (simulating POST /audit/unmerge).
        sql_backend.upsert_entity(conn, "M-999999", "r1", "none", False, None, "t0")
        sql_backend.upsert_entity_member(conn, "P2", "M-999999", True, "reviewer.jclark", "t0")
        sql_backend.insert_audit_log(
            conn, ts_utc="t0", user="reviewer.jclark", action="unmerge",
            patids="P2", mid="M-999999", prev_state="Merged",
            next_state="Unmerged", run_id="r1",
        )
        conn.commit()

        counts = publish.publish_run(backend, "r1", fixture_settings)

        # P2 stays put in its reviewer-created entity — never repointed back to P1's.
        assert sql_backend.get_entity_mid_for_patid(conn, "P2") == "M-999999"
        assert counts["locked_skipped"] == 1

        suggestion = conn.execute(
            "SELECT * FROM entity_suggestion WHERE patid='P2'"
        ).fetchone()
        assert suggestion is not None
        assert suggestion["run_id"] == "r1"

    def test_missing_manifest_raises(self, conn, backend, fixture_settings):
        with pytest.raises(FileNotFoundError):
            publish.publish_run(backend, "does-not-exist", fixture_settings)


@pytest.fixture
def parquet_backend(tmp_path):
    b = ParquetIndexBackend(tmp_path / "local_index")
    yield b
    b.close()


class TestPublishAgainstParquetBackend:
    """`publish_run` run entirely against `ParquetIndexBackend` — same
    fixture (`_write_run`) as `TestPublishRun`'s SQLite coverage, proving
    `publish.py` really is backend-agnostic (see `src/api/index_backend.py`).
    Locked-PATID/sticky-unmerge scenarios aren't covered here yet: Parquet
    local mode has no `audit_log` table until the audit-parity phase lands
    (`ParquetIndexBackend.locked_patids()` is hardcoded empty today)."""

    def test_same_counts_as_sqlite_backend(self, backend, parquet_backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        sql_counts = publish.publish_run(backend, "r1", fixture_settings)
        parquet_counts = publish.publish_run(parquet_backend, "r1", fixture_settings)
        assert sql_counts == parquet_counts

    def test_fresh_publish_creates_entities(self, parquet_backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        counts = publish.publish_run(parquet_backend, "r1", fixture_settings)

        assert counts["clusters_seen"] == 4
        assert counts["entities_upserted"] == 4
        assert counts["members_upserted"] == 5
        assert counts["locked_skipped"] == 0
        assert counts["review_candidates"] == 1

        matched = parquet_backend.get_entity_mid_for_patid("P1")
        assert matched == parquet_backend.get_entity_mid_for_patid("P2")
        singleton_mid = parquet_backend.get_entity_mid_for_patid("P3")
        assert singleton_mid != matched

        matched_entity = parquet_backend.get_entity(matched)
        assert matched_entity["entity"]["is_merged"] == 1
        assert matched_entity["entity"]["confidence"] == 1.0
        assert matched_entity["entity"]["match_rule"] == "SSN_DOB"
        singleton_entity = parquet_backend.get_entity(singleton_mid)
        assert singleton_entity["entity"]["is_merged"] == 0
        assert singleton_entity["entity"]["origin"] == "none"

    def test_review_candidate_upgrades_origin(self, parquet_backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(parquet_backend, "r1", fixture_settings)

        p4_mid = parquet_backend.get_entity_mid_for_patid("P4")
        p4_entity = parquet_backend.get_entity(p4_mid)
        assert p4_entity["entity"]["origin"] == "review"

        rc = parquet_backend._tables["review_candidate"]
        candidates = rc[(rc["patid_a"] == "P4") | (rc["patid_b"] == "P4")]
        assert len(candidates) == 1
        assert candidates.iloc[0]["match_rule"] == "NAME_DOB_SEX"
        assert candidates.iloc[0]["confidence"] == 0.98
        # Batch publish never FS-scores a run — only incremental scoring does.
        assert pd.isna(candidates.iloc[0]["fs_match_probability"])

    def test_record_attrs_denormalized(self, parquet_backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(parquet_backend, "r1", fixture_settings)
        row = parquet_backend._tables["record_attrs"]
        row = row[row["patid"] == "P1"].iloc[0]
        assert row["first_name"] == "Jane"
        assert row["ssn_last4"] == "6789"
        assert row["birth_date"] == "1990-01-01"

    def test_raw_fields_denormalized(self, parquet_backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(parquet_backend, "r1", fixture_settings)
        row = parquet_backend._tables["record_raw"]
        raw_json = row[row["patid"] == "P1"].iloc[0]["raw_json"]
        assert "JANE" in raw_json

    def test_republish_is_idempotent(self, parquet_backend, fixture_settings):
        _write_run(fixture_settings, "r1")
        publish.publish_run(parquet_backend, "r1", fixture_settings)
        mid_before = parquet_backend.get_entity_mid_for_patid("P1")

        publish.publish_run(parquet_backend, "r1", fixture_settings)
        mid_after = parquet_backend.get_entity_mid_for_patid("P1")

        assert mid_before == mid_after
        assert len(parquet_backend._tables["entity"]) == 4
        assert len(parquet_backend._tables["review_candidate"]) == 1  # replaced, not duplicated

    def test_missing_manifest_raises(self, parquet_backend, fixture_settings):
        with pytest.raises(FileNotFoundError):
            publish.publish_run(parquet_backend, "does-not-exist", fixture_settings)
