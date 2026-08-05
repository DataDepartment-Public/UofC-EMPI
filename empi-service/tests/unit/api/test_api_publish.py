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
from src.contracts import (
    ArtifactRef, RunManifest, TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH,
)


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
        # No deterministic rule confirmed a review candidate — by construction
        # the rule/confidence slots are NULL. A reviewer UI must render them as
        # absent, never as a 0% score.
        assert candidates[0]["match_rule"] is None
        assert candidates[0]["confidence"] is None

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
        assert candidates.iloc[0]["match_rule"] is None
        assert candidates.iloc[0]["confidence"] is None
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


# ── The review queue reflects the LAST stage to decide ──────────────────────────
class TestReviewQueueIsPostGateAndML:
    """Stage 3 writes `non_matches` before Stages 4.25/4.5 run, so publishing it
    as-is would queue pairs the gate discarded and pairs the ML matcher already
    merged (its auto_merge tier forms real merge edges). The queue must be the
    last stage's human_review tier."""

    def _pool(self, pairs, gate=None, ml=None):
        nm = pd.DataFrame({
            "PATID_A": [a for a, _ in pairs], "PATID_B": [b for _, b in pairs],
            "source_blocks": ["B3"] * len(pairs), "n_blocks": [1] * len(pairs),
        })
        return publish._final_review_pool(nm, gate, ml)

    def _results(self, rows):
        return pd.DataFrame({
            "PATID_A": [a for a, _, _, _ in rows],
            "PATID_B": [b for _, b, _, _ in rows],
            "model_name": ["m"] * len(rows),
            "score": [s for _, _, _, s in rows],
            "predicted_tier": [t for _, _, t, _ in rows],
        })

    def test_gate_drops_are_excluded(self):
        gate = self._results([
            ("P1", "P2", TIER_HUMAN_REVIEW, 0.8), ("P3", "P4", TIER_NO_MATCH, 0.01),
        ])
        kept, scores = self._pool([("P1", "P2"), ("P3", "P4")], gate=gate)
        assert list(zip(kept["PATID_A"], kept["PATID_B"])) == [("P1", "P2")]
        # The gate's score is P(plausible), a different question — not surfaced.
        assert scores == {}

    def test_ml_auto_merges_are_excluded_and_ml_wins_over_the_gate(self):
        gate = self._results([
            ("P1", "P2", TIER_HUMAN_REVIEW, 0.8), ("P3", "P4", TIER_HUMAN_REVIEW, 0.7),
        ])
        ml = self._results([
            ("P1", "P2", TIER_AUTO_MERGE, 0.97), ("P3", "P4", TIER_HUMAN_REVIEW, 0.42),
        ])
        kept, scores = self._pool([("P1", "P2"), ("P3", "P4")], gate=gate, ml=ml)
        # P1/P2 passed the gate but the ML matcher merged it — it is not a
        # review candidate, it is already an entity.
        assert list(zip(kept["PATID_A"], kept["PATID_B"])) == [("P3", "P4")]
        assert scores == {frozenset(("P3", "P4")): 0.42}

    def test_pair_order_does_not_matter(self):
        ml = self._results([("P2", "P1", TIER_HUMAN_REVIEW, 0.5)])
        kept, scores = self._pool([("P1", "P2")], ml=ml)
        assert len(kept) == 1
        assert scores == {frozenset(("P1", "P2")): 0.5}

    def test_ungated_run_publishes_stage3_unfiltered(self):
        # No gate or ML model active: Stage 3's pool is all the information the
        # run has, so it is published whole rather than emptied.
        kept, scores = self._pool([("P1", "P2"), ("P3", "P4")])
        assert len(kept) == 2
        assert scores == {}

    def test_ml_score_reaches_the_review_candidate_row(self):
        ml = self._results([("P4", "P5", TIER_HUMAN_REVIEW, 0.42)])
        kept, scores = self._pool([("P4", "P5")], ml=ml)
        rows, patids = publish._review_candidate_rows(kept, scores, "r1", "t0")
        assert len(rows) == 1
        a, b, match_rule, confidence, evidence, blocks, run_id, now = rows[0]
        assert (a, b) == ("P4", "P5")
        assert match_rule is None and evidence is None  # no rule confirmed it
        assert confidence == 0.42
        assert patids == {"P4", "P5"}
