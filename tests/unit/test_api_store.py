"""Unit tests for src/api/store.py — schema init, CRUD, and the reviewer-lock
logic publish.py relies on for reconciliation."""

import sqlite3

import pytest

from src.api import store


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    store.init_db(c)
    yield c
    c.close()


class TestSchema:
    def test_init_db_is_idempotent(self, conn):
        store.init_db(conn)  # second call must not raise
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
        store.upsert_entity(conn, "M-1", "run1", "deterministic", True, 0.99, "t0")
        store.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        store.upsert_entity_member(conn, "P2", "M-1", False, "pipeline", "t0")
        conn.commit()

        detail = store.get_entity(conn, "M-1")
        assert detail["entity"]["mid"] == "M-1"
        assert {m["patid"] for m in detail["members"]} == {"P1", "P2"}

    def test_upsert_entity_overwrites_on_conflict(self, conn):
        store.upsert_entity(conn, "M-1", "run1", "none", False, None, "t0")
        store.upsert_entity(conn, "M-1", "run2", "deterministic", True, 0.9, "t1")
        conn.commit()
        row = conn.execute("SELECT * FROM entity WHERE mid='M-1'").fetchone()
        assert row["run_id"] == "run2"
        assert row["is_merged"] == 1

    def test_get_entity_missing_returns_none(self, conn):
        assert store.get_entity(conn, "M-999") is None


class TestMidSequence:
    def test_next_mid_starts_at_1(self, conn):
        assert store.next_mid(conn) == "M-000001"

    def test_next_mid_increments_past_existing(self, conn):
        store.upsert_entity(conn, "M-000005", "r", "none", False, None, "t0")
        conn.commit()
        assert store.next_mid(conn) == "M-000006"


class TestLocking:
    def test_locked_patids_empty_when_no_audit(self, conn):
        assert store.locked_patids(conn) == set()

    def test_locked_patids_from_audit_log(self, conn):
        store.insert_audit_log(
            conn, ts_utc="t0", user="u", action="unmerge",
            patids="P1", mid="M-2", prev_state="Merged",
            next_state="Unmerged", run_id="r1",
        )
        store.insert_audit_log(
            conn, ts_utc="t0", user="u", action="merge",
            patids="P2,P3", mid="M-3", prev_state="Needs review",
            next_state="Merged", run_id="r1",
        )
        conn.commit()
        assert store.locked_patids(conn) == {"P1", "P2", "P3"}


class TestListEntities:
    def _seed(self, conn):
        store.upsert_entity(conn, "M-1", "r1", "deterministic", True, 0.9, "t0")
        store.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        store.upsert_record_attrs(
            conn, "P1", first_name="Jane", last_name="Doe", birth_date="1990-01-01",
            ssn_last4="1234", email=None, zip_code=None, address1=None, sex=None,
            run_id="r1",
        )
        store.upsert_entity(conn, "M-2", "r1", "none", False, None, "t0")
        store.upsert_entity_member(conn, "P2", "M-2", True, "pipeline", "t0")
        store.upsert_record_attrs(
            conn, "P2", first_name="John", last_name="Smith", birth_date=None,
            ssn_last4=None, email=None, zip_code=None, address1=None, sex=None,
            run_id="r1",
        )
        conn.commit()

    def test_status_filter(self, conn):
        self._seed(conn)
        rows, total = store.list_entities(conn, status="merged")
        assert total == 1
        assert rows[0]["mid"] == "M-1"

    def test_search_by_last_name(self, conn):
        self._seed(conn)
        rows, total = store.list_entities(conn, search="Smith")
        assert total == 1
        assert rows[0]["mid"] == "M-2"

    def test_pagination(self, conn):
        self._seed(conn)
        rows, total = store.list_entities(conn, page=1, page_size=1)
        assert total == 2
        assert len(rows) == 1


class TestAuditLog:
    def test_insert_and_list(self, conn):
        store.insert_audit_log(
            conn, ts_utc="t0", user="u", action="merge", patids="P1,P2",
            mid="M-1", prev_state="Needs review", next_state="Merged", run_id="r1",
        )
        conn.commit()
        rows = store.list_audit_log(conn)
        assert len(rows) == 1
        assert rows[0]["action"] == "merge"

    def test_since_filters(self, conn):
        store.insert_audit_log(
            conn, ts_utc="2026-01-01T00:00:00", user="u", action="merge",
            patids="P1", mid="M-1", prev_state="a", next_state="b", run_id="r1",
        )
        store.insert_audit_log(
            conn, ts_utc="2026-06-01T00:00:00", user="u", action="merge",
            patids="P2", mid="M-2", prev_state="a", next_state="b", run_id="r1",
        )
        conn.commit()
        rows = store.list_audit_log(conn, since="2026-03-01T00:00:00")
        assert len(rows) == 1
        assert rows[0]["patids"] == "P2"
