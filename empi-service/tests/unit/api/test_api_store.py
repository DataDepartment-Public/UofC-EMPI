"""Unit tests for src/api/backends/sql_backend.py — schema init, CRUD, and the reviewer-lock
logic src/api/ingest/publish.py relies on for reconciliation."""

import sqlite3

import pytest

from src.api.backends import sql_backend


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    sql_backend.init_db(c)
    yield c
    c.close()


class TestSchema:
    def test_init_db_is_idempotent(self, conn):
        sql_backend.init_db(conn)  # second call must not raise
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "entity", "entity_member", "audit_log",
            "record_attrs", "entity_suggestion",
        } <= tables


class TestEntityUpsert:
    def test_upsert_entity_then_member_then_get(self, conn):
        sql_backend.upsert_entity(conn, "M-1", "run1", "deterministic", True, 0.99, "t0")
        sql_backend.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        sql_backend.upsert_entity_member(conn, "P2", "M-1", False, "pipeline", "t0")
        conn.commit()

        detail = sql_backend.get_entity(conn, "M-1")
        assert detail["entity"]["mid"] == "M-1"
        assert {m["patid"] for m in detail["members"]} == {"P1", "P2"}

    def test_upsert_entity_overwrites_on_conflict(self, conn):
        sql_backend.upsert_entity(conn, "M-1", "run1", "none", False, None, "t0")
        sql_backend.upsert_entity(conn, "M-1", "run2", "deterministic", True, 0.9, "t1")
        conn.commit()
        row = conn.execute("SELECT * FROM entity WHERE mid='M-1'").fetchone()
        assert row["run_id"] == "run2"
        assert row["is_merged"] == 1

    def test_get_entity_missing_returns_none(self, conn):
        assert sql_backend.get_entity(conn, "M-999") is None


class TestMidSequence:
    def test_next_mid_starts_at_1(self, conn):
        assert sql_backend.next_mid(conn) == "M-000001"

    def test_next_mid_increments_past_existing(self, conn):
        sql_backend.upsert_entity(conn, "M-000005", "r", "none", False, None, "t0")
        conn.commit()
        assert sql_backend.next_mid(conn) == "M-000006"


class TestLocking:
    def test_locked_patids_empty_when_no_audit(self, conn):
        assert sql_backend.locked_patids(conn) == set()

    def test_locked_patids_from_audit_log(self, conn):
        sql_backend.insert_audit_log(
            conn, ts_utc="t0", user="u", action="unmerge",
            patids="P1", mid="M-2", prev_state="Merged",
            next_state="Unmerged", run_id="r1",
        )
        sql_backend.insert_audit_log(
            conn, ts_utc="t0", user="u", action="merge",
            patids="P2,P3", mid="M-3", prev_state="Needs review",
            next_state="Merged", run_id="r1",
        )
        conn.commit()
        assert sql_backend.locked_patids(conn) == {"P1", "P2", "P3"}


class TestListEntities:
    def _seed(self, conn):
        sql_backend.upsert_entity(conn, "M-1", "r1", "deterministic", True, 0.9, "t0")
        sql_backend.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        sql_backend.upsert_record_attrs(
            conn, "P1", first_name="Jane", last_name="Doe", birth_date="1990-01-01",
            ssn_last4="1234", email=None, zip_code=None, address1=None, sex=None,
            run_id="r1",
        )
        sql_backend.upsert_entity(conn, "M-2", "r1", "none", False, None, "t0")
        sql_backend.upsert_entity_member(conn, "P2", "M-2", True, "pipeline", "t0")
        sql_backend.upsert_record_attrs(
            conn, "P2", first_name="John", last_name="Smith", birth_date=None,
            ssn_last4=None, email=None, zip_code=None, address1=None, sex=None,
            run_id="r1",
        )
        conn.commit()

    def test_is_merged_filter(self, conn):
        self._seed(conn)
        rows, total = sql_backend.list_entities(conn, is_merged=True)
        assert total == 1
        assert rows[0]["mid"] == "M-1"

    def test_origin_filter(self, conn):
        self._seed(conn)
        rows, total = sql_backend.list_entities(conn, origin="none")
        assert total == 1
        assert rows[0]["mid"] == "M-2"

    def test_ssn_last4_filter(self, conn):
        self._seed(conn)
        rows, total = sql_backend.list_entities(conn, ssn_last4="1234")
        assert total == 1
        assert rows[0]["mid"] == "M-1"

    def test_birth_date_filter(self, conn):
        self._seed(conn)
        rows, total = sql_backend.list_entities(conn, birth_date="1990-01-01")
        assert total == 1
        assert rows[0]["mid"] == "M-1"

    def test_search_by_last_name(self, conn):
        self._seed(conn)
        rows, total = sql_backend.list_entities(conn, search="Smith")
        assert total == 1
        assert rows[0]["mid"] == "M-2"

    def test_pagination(self, conn):
        self._seed(conn)
        rows, total = sql_backend.list_entities(conn, page=1, page_size=1)
        assert total == 2
        assert len(rows) == 1


class TestAuditLog:
    def test_insert_and_list(self, conn):
        sql_backend.insert_audit_log(
            conn, ts_utc="t0", user="u", action="merge", patids="P1,P2",
            mid="M-1", prev_state="Needs review", next_state="Merged", run_id="r1",
        )
        conn.commit()
        rows = sql_backend.list_audit_log(conn)
        assert len(rows) == 1
        assert rows[0]["action"] == "merge"

    def test_since_filters(self, conn):
        sql_backend.insert_audit_log(
            conn, ts_utc="2026-01-01T00:00:00", user="u", action="merge",
            patids="P1", mid="M-1", prev_state="a", next_state="b", run_id="r1",
        )
        sql_backend.insert_audit_log(
            conn, ts_utc="2026-06-01T00:00:00", user="u", action="merge",
            patids="P2", mid="M-2", prev_state="a", next_state="b", run_id="r1",
        )
        conn.commit()
        rows = sql_backend.list_audit_log(conn, since="2026-03-01T00:00:00")
        assert len(rows) == 1
        assert rows[0]["patids"] == "P2"


class TestRecordRaw:
    def test_bulk_upsert_and_get(self, conn):
        sql_backend.upsert_entity(conn, "M-1", "r1", "none", False, None, "t0")
        sql_backend.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        conn.commit()
        sql_backend.upsert_record_raw_bulk(
            conn, [("P1", '{"FirstNM_raw": "JANE"}', "r1")]
        )
        conn.commit()
        assert sql_backend.get_record_raw(conn, "P1") == '{"FirstNM_raw": "JANE"}'

    def test_get_missing_returns_none(self, conn):
        assert sql_backend.get_record_raw(conn, "P-nope") is None


class TestReviewCandidate:
    def test_replace_review_candidates_for_run(self, conn):
        sql_backend.replace_review_candidates_for_run(
            conn, "r1",
            [(
                "P1", "P2", "NAME_DOB_SEX", 0.98, "NAME_DOB_SEX", "B3", "r1", "t0",
                None, None,
            )],
        )
        conn.commit()
        rows = sql_backend.review_candidates_for_patid(conn, "P1")
        assert len(rows) == 1
        assert rows[0]["patid_b"] == "P2"
        assert sql_backend.review_candidates_for_patid(conn, "P2")[0]["patid_a"] == "P1"

    def test_replace_review_candidates_for_run_carries_ml_score(self, conn):
        sql_backend.replace_review_candidates_for_run(
            conn, "r1",
            [("P1", "P2", None, None, None, "B3", "r1", "t0", 0.24, "human_review")],
        )
        conn.commit()
        rows = sql_backend.review_candidates_for_patid(conn, "P1")
        assert rows[0]["ml_match_probability"] == 0.24
        assert rows[0]["ml_classification_tier"] == "human_review"

    def test_replace_clears_stale_rows_for_same_run(self, conn):
        sql_backend.replace_review_candidates_for_run(
            conn, "r1", [("P1", "P2", None, None, None, "B3", "r1", "t0", None, None)]
        )
        conn.commit()
        sql_backend.replace_review_candidates_for_run(conn, "r1", [])  # nothing this time
        conn.commit()
        assert sql_backend.review_candidates_for_patid(conn, "P1") == []

    def test_patids_with_review_candidates(self, conn):
        sql_backend.replace_review_candidates_for_run(
            conn, "r1", [("P1", "P2", None, None, None, "B3", "r1", "t0", None, None)]
        )
        conn.commit()
        assert sql_backend.patids_with_review_candidates(conn) == {"P1", "P2"}


class TestDashboardSummary:
    def test_aggregates_reflect_live_state(self, conn):
        sql_backend.upsert_entity(conn, "M-1", "r1", "deterministic", True, 0.99, "t0")
        sql_backend.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        sql_backend.upsert_entity_member(conn, "P2", "M-1", False, "pipeline", "t0")
        sql_backend.upsert_entity(conn, "M-2", "r1", "review", False, None, "t0")
        sql_backend.upsert_entity_member(conn, "P3", "M-2", True, "pipeline", "t0")
        sql_backend.upsert_entity(conn, "M-3", "r1", "none", False, None, "t0")
        sql_backend.upsert_entity_member(conn, "P4", "M-3", True, "pipeline", "t0")
        sql_backend.insert_audit_log(
            conn, ts_utc="t0", user="u", action="merge", patids="P5",
            mid="M-1", prev_state="a", next_state="b", run_id="r1",
        )
        conn.commit()

        summary = sql_backend.dashboard_summary(conn)
        assert summary["total_records"] == 4
        assert summary["duplicate_clusters"] == 1
        assert summary["matched_records"] == 2
        assert summary["needs_review_records"] == 1
        assert summary["no_match_records"] == 1
        assert summary["manual_merge_actions"] == 1
        assert summary["manual_unmerge_actions"] == 0

    def test_empty_db(self, conn):
        summary = sql_backend.dashboard_summary(conn)
        assert summary["total_records"] == 0
        assert summary["duplicate_clusters"] == 0
