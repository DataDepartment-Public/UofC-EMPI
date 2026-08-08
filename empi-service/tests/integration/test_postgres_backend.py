"""Integration tests for `src/api/backends/postgres_backend.py` — schema init,
CRUD, and the dialect-translation spots most likely to have gone wrong porting
from `sql_backend.py` (SQLite) to Postgres: `ILIKE` case-insensitivity,
`RETURNING id` in place of `cur.lastrowid`, `ON CONFLICT ... DO NOTHING` in
place of `INSERT OR IGNORE`, and `information_schema`-based column migration.

Mirrors `tests/unit/api/test_api_store.py`'s test cases one-for-one where the
underlying function exists in both modules. Two intentionally excluded:
`patids_with_review_candidates` and the singular `upsert_record_attrs` serve
`src/api/ingest/publish.py`'s batch path, which is out of scope for this
module (see its docstring) — `_seed` here uses `upsert_record_attrs_bulk`
instead.

Needs a real Postgres reachable via `EMPI_TEST_POSTGRES_DSN` (a libpq
connection string/keyword string, e.g.
`"host=/tmp/empi_pg_test port=55432 dbname=empi_test user=postgres"`) — skips
entirely otherwise. This is deliberately *not* exercised through
`postgres_backend.get_connection()`, which only knows how to authenticate via
an Azure AD token (`DefaultAzureCredential`) and has no meaning outside
Azure; these tests connect directly to isolate SQL-dialect correctness from
Azure-specific auth.
"""

import os

import psycopg
import pytest
from psycopg.rows import dict_row

from src.api.backends import postgres_backend

_DSN = os.environ.get("EMPI_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(
    not _DSN, reason="EMPI_TEST_POSTGRES_DSN not set — no Postgres to test against"
)


@pytest.fixture
def conn():
    # Unlike sql_backend's tests (sqlite3.connect(":memory:"), a fresh
    # isolated DB per test for free), every test here points at the same
    # real, durable database — TRUNCATE gives each test the clean slate
    # test_api_store.py's in-memory SQLite gets automatically. RESTART
    # IDENTITY also resets audit_log/review_candidate's id sequences, which
    # some assertions below depend on starting at 1.
    c = psycopg.connect(_DSN, row_factory=dict_row)
    postgres_backend.init_db(c)
    c.execute(
        "TRUNCATE entity, entity_member, audit_log, record_attrs, record_raw, "
        "review_candidate, entity_suggestion, block_key, cleaned_attrs "
        "RESTART IDENTITY CASCADE"
    )
    c.commit()
    yield c
    c.close()


class TestSchema:
    def test_init_db_is_idempotent(self, conn):
        postgres_backend.init_db(conn)  # second call must not raise
        tables = {
            r["table_name"]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ).fetchall()
        }
        assert {
            "entity", "entity_member", "audit_log",
            "record_attrs", "entity_suggestion",
        } <= tables

    def test_column_migration_adds_missing_columns(self, conn):
        # _COLUMN_MIGRATIONS' columns are part of SCHEMA_SQL already in this
        # fresh DB, so re-running _ensure_columns must be a no-op, not an error.
        postgres_backend._ensure_columns(conn)
        cols = {
            r["column_name"]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'review_candidate'"
            ).fetchall()
        }
        assert {"fs_match_probability", "fs_classification_tier"} <= cols


class TestEntityUpsert:
    def test_upsert_entity_then_member_then_get(self, conn):
        postgres_backend.upsert_entity(conn, "M-1", "run1", "deterministic", True, 0.99, "t0")
        postgres_backend.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        postgres_backend.upsert_entity_member(conn, "P2", "M-1", False, "pipeline", "t0")
        conn.commit()

        detail = postgres_backend.get_entity(conn, "M-1")
        assert detail["entity"]["mid"] == "M-1"
        assert {m["patid"] for m in detail["members"]} == {"P1", "P2"}

    def test_upsert_entity_overwrites_on_conflict(self, conn):
        postgres_backend.upsert_entity(conn, "M-1", "run1", "none", False, None, "t0")
        postgres_backend.upsert_entity(conn, "M-1", "run2", "deterministic", True, 0.9, "t1")
        conn.commit()
        row = conn.execute("SELECT * FROM entity WHERE mid='M-1'").fetchone()
        assert row["run_id"] == "run2"
        assert row["is_merged"] == 1

    def test_get_entity_missing_returns_none(self, conn):
        assert postgres_backend.get_entity(conn, "M-999") is None

    def test_upsert_entities_bulk(self, conn):
        postgres_backend.upsert_entities_bulk(
            conn,
            [
                ("M-1", "r1", "none", 0, None, None, None, "t0"),
                ("M-2", "r1", "deterministic", 1, 0.95, "SSN_DOB", "SSN_DOB", "t0"),
            ],
        )
        conn.commit()
        assert postgres_backend.get_entity(conn, "M-2")["entity"]["match_rule"] == "SSN_DOB"


class TestMidSequence:
    def test_next_mid_starts_at_1(self, conn):
        assert postgres_backend.next_mid(conn) == "M-000001"

    def test_next_mid_increments_past_existing(self, conn):
        postgres_backend.upsert_entity(conn, "M-000005", "r", "none", False, None, "t0")
        conn.commit()
        assert postgres_backend.next_mid(conn) == "M-000006"


class TestLocking:
    def test_locked_patids_empty_when_no_audit(self, conn):
        assert postgres_backend.locked_patids(conn) == set()

    def test_locked_patids_from_audit_log(self, conn):
        postgres_backend.insert_audit_log(
            conn, ts_utc="t0", user="u", action="unmerge",
            patids="P1", mid="M-2", prev_state="Merged",
            next_state="Unmerged", run_id="r1",
        )
        postgres_backend.insert_audit_log(
            conn, ts_utc="t0", user="u", action="merge",
            patids="P2,P3", mid="M-3", prev_state="Needs review",
            next_state="Merged", run_id="r1",
        )
        conn.commit()
        assert postgres_backend.locked_patids(conn) == {"P1", "P2", "P3"}


class TestListEntities:
    def _seed(self, conn):
        postgres_backend.upsert_entity(conn, "M-1", "r1", "deterministic", True, 0.9, "t0")
        postgres_backend.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        postgres_backend.upsert_record_attrs_bulk(conn, [(
            "P1", "Jane", "Doe", "1990-01-01", "1234", None, None, None, None, None, "r1",
        )])
        postgres_backend.upsert_entity(conn, "M-2", "r1", "none", False, None, "t0")
        postgres_backend.upsert_entity_member(conn, "P2", "M-2", True, "pipeline", "t0")
        postgres_backend.upsert_record_attrs_bulk(conn, [(
            "P2", "John", "Smith", None, None, None, None, None, None, None, "r1",
        )])
        conn.commit()

    def test_is_merged_filter(self, conn):
        self._seed(conn)
        rows, total = postgres_backend.list_entities(conn, is_merged=True)
        assert total == 1
        assert rows[0]["mid"] == "M-1"

    def test_origin_filter(self, conn):
        self._seed(conn)
        rows, total = postgres_backend.list_entities(conn, origin="none")
        assert total == 1
        assert rows[0]["mid"] == "M-2"

    def test_ssn_last4_filter(self, conn):
        self._seed(conn)
        rows, total = postgres_backend.list_entities(conn, ssn_last4="1234")
        assert total == 1
        assert rows[0]["mid"] == "M-1"

    def test_birth_date_filter(self, conn):
        self._seed(conn)
        rows, total = postgres_backend.list_entities(conn, birth_date="1990-01-01")
        assert total == 1
        assert rows[0]["mid"] == "M-1"

    def test_search_is_case_insensitive(self, conn):
        # The dialect-risk case: SQLite's LIKE is case-insensitive for ASCII
        # by default, Postgres's isn't — this only passes if postgres_backend
        # actually uses ILIKE (not LIKE) for the search filter.
        self._seed(conn)
        rows, total = postgres_backend.list_entities(conn, search="smith")
        assert total == 1
        assert rows[0]["mid"] == "M-2"

        rows, total = postgres_backend.list_entities(conn, search="SMITH")
        assert total == 1
        assert rows[0]["mid"] == "M-2"

    def test_pagination(self, conn):
        self._seed(conn)
        rows, total = postgres_backend.list_entities(conn, page=1, page_size=1)
        assert total == 2
        assert len(rows) == 1


class TestAuditLog:
    def test_insert_and_list(self, conn):
        # RETURNING id + fetchone() stands in for sqlite3's cur.lastrowid —
        # this is the test that catches a broken port of that substitution.
        audit_id = postgres_backend.insert_audit_log(
            conn, ts_utc="t0", user="u", action="merge", patids="P1,P2",
            mid="M-1", prev_state="Needs review", next_state="Merged", run_id="r1",
        )
        conn.commit()
        assert isinstance(audit_id, int) and audit_id > 0
        rows = postgres_backend.list_audit_log(conn)
        assert len(rows) == 1
        assert rows[0]["action"] == "merge"

    def test_since_filters(self, conn):
        postgres_backend.insert_audit_log(
            conn, ts_utc="2026-01-01T00:00:00", user="u", action="merge",
            patids="P1", mid="M-1", prev_state="a", next_state="b", run_id="r1",
        )
        postgres_backend.insert_audit_log(
            conn, ts_utc="2026-06-01T00:00:00", user="u", action="merge",
            patids="P2", mid="M-2", prev_state="a", next_state="b", run_id="r1",
        )
        conn.commit()
        rows = postgres_backend.list_audit_log(conn, since="2026-03-01T00:00:00")
        assert len(rows) == 1
        assert rows[0]["patids"] == "P2"


class TestRecordRaw:
    def test_bulk_upsert_and_get(self, conn):
        postgres_backend.upsert_entity(conn, "M-1", "r1", "none", False, None, "t0")
        postgres_backend.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        conn.commit()
        postgres_backend.upsert_record_raw_bulk(
            conn, [("P1", '{"FirstNM_raw": "JANE"}', "r1")]
        )
        conn.commit()
        assert postgres_backend.get_record_raw(conn, "P1") == '{"FirstNM_raw": "JANE"}'

    def test_get_missing_returns_none(self, conn):
        assert postgres_backend.get_record_raw(conn, "P-nope") is None


class TestReviewCandidate:
    def test_replace_review_candidates_for_run(self, conn):
        postgres_backend.replace_review_candidates_for_run(
            conn, "r1",
            [(
                "P1", "P2", "NAME_DOB_SEX", 0.98, "NAME_DOB_SEX", "B3", "r1", "t0",
                None, None,
            )],
        )
        conn.commit()
        rows = postgres_backend.review_candidates_for_patid(conn, "P1")
        assert len(rows) == 1
        assert rows[0]["patid_b"] == "P2"
        assert postgres_backend.review_candidates_for_patid(conn, "P2")[0]["patid_a"] == "P1"

    def test_replace_review_candidates_for_run_carries_ml_score(self, conn):
        postgres_backend.replace_review_candidates_for_run(
            conn, "r1",
            [("P1", "P2", None, None, None, "B3", "r1", "t0", 0.24, "human_review")],
        )
        conn.commit()
        rows = postgres_backend.review_candidates_for_patid(conn, "P1")
        assert rows[0]["ml_match_probability"] == 0.24
        assert rows[0]["ml_classification_tier"] == "human_review"

    def test_replace_clears_stale_rows_for_same_run(self, conn):
        postgres_backend.replace_review_candidates_for_run(
            conn, "r1", [("P1", "P2", None, None, None, "B3", "r1", "t0", None, None)]
        )
        conn.commit()
        postgres_backend.replace_review_candidates_for_run(conn, "r1", [])
        conn.commit()
        assert postgres_backend.review_candidates_for_patid(conn, "P1") == []

    def test_list_review_candidates_default_excludes_reviewed(self, conn):
        postgres_backend.upsert_entity(conn, "M-1", "r1", "none", False, None, "t0")
        postgres_backend.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        postgres_backend.upsert_entity(conn, "M-2", "r1", "none", False, None, "t0")
        postgres_backend.upsert_entity_member(conn, "P2", "M-2", True, "pipeline", "t0")
        postgres_backend.replace_review_candidates_for_run(
            conn, "r1", [("P1", "P2", None, 0.7, None, "B3", "r1", "t0", None, None)]
        )
        conn.commit()

        rows, total = postgres_backend.list_review_candidates(conn, reviewed=False)
        assert total == 1
        assert rows[0]["patid_a"] == "P1"


class TestDashboardSummary:
    def test_aggregates_reflect_live_state(self, conn):
        postgres_backend.upsert_entity(conn, "M-1", "r1", "deterministic", True, 0.99, "t0")
        postgres_backend.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        postgres_backend.upsert_entity_member(conn, "P2", "M-1", False, "pipeline", "t0")
        postgres_backend.upsert_entity(conn, "M-2", "r1", "review", False, None, "t0")
        postgres_backend.upsert_entity_member(conn, "P3", "M-2", True, "pipeline", "t0")
        postgres_backend.upsert_entity(conn, "M-3", "r1", "none", False, None, "t0")
        postgres_backend.upsert_entity_member(conn, "P4", "M-3", True, "pipeline", "t0")
        postgres_backend.insert_audit_log(
            conn, ts_utc="t0", user="u", action="merge", patids="P5",
            mid="M-1", prev_state="a", next_state="b", run_id="r1",
        )
        conn.commit()

        summary = postgres_backend.dashboard_summary(conn)
        assert summary["total_records"] == 4
        assert summary["duplicate_clusters"] == 1
        assert summary["matched_records"] == 2
        assert summary["needs_review_records"] == 1
        assert summary["no_match_records"] == 1
        assert summary["manual_merge_actions"] == 1
        assert summary["manual_unmerge_actions"] == 0

    def test_empty_db(self, conn):
        summary = postgres_backend.dashboard_summary(conn)
        assert summary["total_records"] == 0
        assert summary["duplicate_clusters"] == 0


class TestBlockKey:
    def test_add_block_keys_ignores_duplicates(self, conn):
        # ON CONFLICT DO NOTHING stands in for sqlite3's INSERT OR IGNORE —
        # this is the test that catches a broken port of that substitution.
        postgres_backend.add_block_keys(conn, [("B1", "hash123", "P1")])
        postgres_backend.add_block_keys(conn, [("B1", "hash123", "P1")])  # duplicate
        conn.commit()
        hits = postgres_backend.lookup_block_candidates(
            conn,
            {"B1": "hash123", "B3": None, "B4": None, "B6": None, "B7": None, "B8": None, "B9": None},
            phones=[], threshold=500,
        )
        assert hits == {"P1": {"B1"}}

    def test_replace_block_keys_full_rebuild(self, conn):
        postgres_backend.add_block_keys(conn, [("B1", "old", "P1")])
        conn.commit()
        postgres_backend.replace_block_keys(conn, [("B1", "new", "P1")])
        conn.commit()
        assert postgres_backend.lookup_block_candidates(
            conn, {"B1": "old"}, phones=[], threshold=500
        ) == {}
        assert postgres_backend.lookup_block_candidates(
            conn, {"B1": "new"}, phones=[], threshold=500
        ) == {"P1": {"B1"}}


class TestCleanedAttrs:
    def test_round_trip(self, conn):
        postgres_backend.upsert_cleaned_attrs(conn, (
            "P1", "JANE", "DOE", "1990-01-01", "234567891", "7891",
            None, None, None, None, "[]", "r1",
        ))
        conn.commit()
        rows = postgres_backend.get_cleaned_attrs(conn, ["P1"])
        assert len(rows) == 1
        assert rows[0]["first_nm"] == "JANE"

    def test_replace_is_full_rebuild(self, conn):
        postgres_backend.upsert_cleaned_attrs(conn, (
            "P1", "JANE", "DOE", "1990-01-01", "234567891", "7891",
            None, None, None, None, "[]", "r1",
        ))
        conn.commit()
        postgres_backend.replace_cleaned_attrs(conn, [(
            "P2", "JOHN", "SMITH", None, None, None,
            None, None, None, None, "[]", "r2",
        )])
        conn.commit()
        assert postgres_backend.get_cleaned_attrs(conn, ["P1"]) == []
        assert len(postgres_backend.get_cleaned_attrs(conn, ["P2"])) == 1


class TestReassignEntityMembers:
    def test_absorbs_members_and_deletes_old_entities(self, conn):
        postgres_backend.upsert_entity(conn, "M-1", "r1", "none", False, None, "t0")
        postgres_backend.upsert_entity(conn, "M-2", "r1", "none", False, None, "t0")
        postgres_backend.upsert_entity_member(conn, "P1", "M-1", True, "pipeline", "t0")
        postgres_backend.upsert_entity_member(conn, "P2", "M-2", True, "pipeline", "t0")
        conn.commit()

        postgres_backend.reassign_entity_members(conn, ["M-1", "M-2"], "M-1", "t1")
        conn.commit()

        assert postgres_backend.get_entity_mid_for_patid(conn, "P2") == "M-1"
        assert postgres_backend.get_entity(conn, "M-2") is None
