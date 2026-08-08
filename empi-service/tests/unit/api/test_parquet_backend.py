"""Unit tests for `src/api/backends/parquet_backend.py` (`ParquetIndexBackend`) and
the local-mode CLI (`src/api/ingest/local_score.py`).

Two layers:
    - Direct backend tests: CRUD/round-trip/rollback against the Parquet
      files themselves, independent of `incremental.py`.
    - `incremental.score_records` run entirely against `ParquetIndexBackend`
      — the same scenarios `tests/unit/test_incremental.py` proves against
      SQLite, confirming `incremental.py` really is backend-agnostic.
"""

import pandas as pd
import pytest

from src.api.ingest import incremental
from src.api.ingest.local_score import score_local
from src.api.backends.parquet_backend import ParquetIndexBackend
from src.config import Settings


@pytest.fixture
def backend(tmp_path):
    b = ParquetIndexBackend(tmp_path / "index")
    yield b
    b.close()


def _raw(patid, first, last, birth, ssn=None, sex=None):
    return {
        "PATID": patid, "FirstNM": first, "LastNM": last, "BirthDT": birth,
        "SSN": ssn, "SexAtBirthDSC": sex,
    }


class TestBackendCrud:
    def test_locked_patids_empty_with_no_audit_log(self, backend):
        assert backend.locked_patids() == set()

    def test_next_mid_starts_at_1(self, backend):
        assert backend.next_mid() == "M-000001"

    def test_entity_and_member_round_trip(self, backend):
        backend.begin()
        backend.upsert_entity(
            "M-000001", "r1", "deterministic", is_merged=True,
            confidence=0.99, updated_utc="t0", match_rule="SSN_DOB", evidence="SSN_DOB",
        )
        backend.upsert_entity_member("P1", "M-000001", is_primary=True, added_by="pipeline", updated_utc="t0")
        backend.upsert_entity_member("P2", "M-000001", is_primary=False, added_by="pipeline", updated_utc="t0")
        backend.commit()

        assert backend.get_entity_mid_for_patid("P1") == "M-000001"
        entity = backend.get_entity("M-000001")
        assert entity["entity"]["confidence"] == 0.99
        assert entity["entity"]["match_rule"] == "SSN_DOB"
        assert {m["patid"] for m in entity["members"]} == {"P1", "P2"}

    def test_get_entity_unknown_returns_none(self, backend):
        assert backend.get_entity("M-999999") is None

    def test_block_key_lookup(self, backend):
        backend.begin()
        backend.add_block_keys([("B1", "hash123", "P1"), ("B4", "DOE|1990|JAN", "P1")])
        backend.commit()

        hits = backend.lookup_block_candidates(
            {"B1": "hash123", "B3": None, "B4": None, "B6": None, "B7": None, "B8": None, "B9": None},
            phones=[], threshold=500,
        )
        assert hits == {"P1": {"B1"}}

    def test_cleaned_attrs_round_trip_no_nan_leak(self, backend):
        backend.begin()
        backend.upsert_cleaned_attrs((
            "P1", "JANE", "DOE", "1990-01-01", "234567891", "7891",
            None, None, None, None, "[]", "r1",
        ))
        backend.commit()
        rows = backend.get_cleaned_attrs(["P1"])
        assert len(rows) == 1
        # A never-set column must come back as None, not NaN (NaN is
        # truthy — would silently break incremental.py's `if row["x"]:`).
        assert rows[0]["email"] is None

    def test_reassign_entity_members_bridges_and_deletes_absorbed(self, backend):
        backend.begin()
        backend.upsert_entity("M-000001", "r1", "deterministic", True, 1.0, "t0")
        backend.upsert_entity("M-000002", "r1", "deterministic", True, 1.0, "t0")
        backend.upsert_entity_member("P1", "M-000001", True, "pipeline", "t0")
        backend.upsert_entity_member("P2", "M-000002", True, "pipeline", "t0")
        backend.commit()

        backend.begin()
        backend.reassign_entity_members(["M-000002"], "M-000001", "t1")
        backend.commit()

        assert backend.get_entity("M-000002") is None
        entity = backend.get_entity("M-000001")
        assert {m["patid"] for m in entity["members"]} == {"P1", "P2"}

    def test_rollback_discards_uncommitted_changes(self, backend):
        backend.begin()
        backend.upsert_entity("M-000001", "r1", "none", False, None, "t0")
        backend.commit()

        backend.begin()
        backend.upsert_entity("M-000002", "r1", "none", False, None, "t0")
        backend.rollback()

        assert backend.get_entity("M-000001") is not None
        assert backend.get_entity("M-000002") is None

    def test_commit_persists_to_disk_for_a_new_backend_instance(self, tmp_path):
        data_dir = tmp_path / "index"
        b1 = ParquetIndexBackend(data_dir)
        b1.begin()
        b1.upsert_entity("M-000001", "r1", "none", False, None, "t0")
        b1.upsert_entity_member("P1", "M-000001", True, "pipeline", "t0")
        b1.commit()
        b1.close()

        b2 = ParquetIndexBackend(data_dir)
        assert b2.get_entity_mid_for_patid("P1") == "M-000001"
        b2.close()

    def test_insert_review_candidates(self, backend):
        backend.begin()
        backend.insert_review_candidates([
            ("P1", "P2", "NAME_DOB_SEX", 0.98, "NAME_DOB_SEX", "B3", "r1", "t0", None, None),
        ])
        backend.commit()
        df = backend._tables["review_candidate"]
        assert len(df) == 1
        assert df.iloc[0]["patid_a"] == "P1"


class TestBulkPublishMethods:
    """The `*_bulk`/`replace_*` methods `src/api/publish.py` uses — a real
    batch publish touches every entity in a run at once, unlike
    `incremental.py`'s row-at-a-time calls tested above."""

    def test_max_mid_sequence_and_all_entity_member_mids(self, backend):
        assert backend.max_mid_sequence() == 0
        backend.begin()
        backend.upsert_entities_bulk([
            ("M-000001", "r1", "deterministic", 1, 0.99, "SSN_DOB", "SSN_DOB", "t0"),
            ("M-000007", "r1", "none", 0, None, None, None, "t0"),
        ])
        backend.upsert_entity_members_bulk([
            ("P1", "M-000001", 1, "pipeline", "t0"),
            ("P2", "M-000007", 1, "pipeline", "t0"),
        ])
        backend.commit()
        assert backend.max_mid_sequence() == 7
        assert backend.next_mid() == "M-000008"
        assert backend.all_entity_member_mids() == {"P1": "M-000001", "P2": "M-000007"}

    def test_upsert_entities_bulk_replaces_existing_rows(self, backend):
        backend.begin()
        backend.upsert_entities_bulk([("M-000001", "r1", "none", 0, None, None, None, "t0")])
        backend.commit()
        backend.begin()
        backend.upsert_entities_bulk([
            ("M-000001", "r2", "deterministic", 1, 0.99, "SSN_DOB", "SSN_DOB", "t1"),
        ])
        backend.commit()
        entity = backend.get_entity("M-000001")
        assert entity["entity"]["run_id"] == "r2"
        assert entity["entity"]["is_merged"] == 1

    def test_upsert_record_attrs_and_raw_bulk(self, backend):
        backend.begin()
        backend.upsert_record_attrs_bulk([
            ("P1", "Jane", "Doe", "1990-01-01", "6789", None, None, None, None, None, "r1"),
        ])
        backend.upsert_record_raw_bulk([("P1", '{"FirstNM_raw": "JANE"}', "r1")])
        backend.commit()
        attrs = backend._tables["record_attrs"]
        assert attrs[attrs["patid"] == "P1"].iloc[0]["first_name"] == "Jane"
        raw = backend._tables["record_raw"]
        assert "JANE" in raw[raw["patid"] == "P1"].iloc[0]["raw_json"]

    def test_upsert_suggestions_bulk(self, backend):
        backend.begin()
        backend.upsert_suggestions_bulk([("P1", "r1", "SUGGESTED-r1-0", "t0")])
        backend.commit()
        suggestions = backend._tables["entity_suggestion"]
        assert suggestions[suggestions["patid"] == "P1"].iloc[0]["suggested_mid"] == "SUGGESTED-r1-0"

    def test_replace_review_candidates_for_run_replaces_wholesale(self, backend):
        backend.begin()
        backend.replace_review_candidates_for_run(
            "r1",
            [(
                "P1", "P2", "NAME_DOB_SEX", 0.98, "NAME_DOB_SEX", "B3", "r1", "t0",
                None, None,
            )],
        )
        backend.commit()
        df = backend._tables["review_candidate"]
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["fs_match_probability"])

        # A second publish of the same run_id with a different pair replaces
        # the first wholesale — it must not linger as a stale suggestion.
        backend.begin()
        backend.replace_review_candidates_for_run(
            "r1",
            [(
                "P3", "P4", "NAME_DOB_ADDRESS", 0.97, "NAME_DOB_ADDRESS", "B7", "r1", "t1",
                None, None,
            )],
        )
        backend.commit()
        df = backend._tables["review_candidate"]
        assert len(df) == 1
        assert df.iloc[0]["patid_a"] == "P3"

    def test_replace_review_candidates_for_run_carries_ml_score(self, backend):
        backend.begin()
        backend.replace_review_candidates_for_run(
            "r1", [("P1", "P2", None, None, None, "B3", "r1", "t0", 0.24, "human_review")]
        )
        backend.commit()
        df = backend._tables["review_candidate"]
        assert df.iloc[0]["ml_match_probability"] == 0.24
        assert df.iloc[0]["ml_classification_tier"] == "human_review"

    def test_replace_cleaned_attrs_full_rebuild(self, backend):
        backend.begin()
        backend.upsert_cleaned_attrs((
            "P1", "JANE", "DOE", "1990-01-01", "234567891", "7891",
            None, None, None, None, "[]", "r1",
        ))
        backend.commit()
        backend.begin()
        backend.replace_cleaned_attrs([
            ("P2", "JOHN", "SMITH", "1985-05-05", None, None,
             None, None, None, None, "[]", "r2"),
        ])
        backend.commit()
        # P1 is gone — replace_cleaned_attrs is a full rebuild, not an upsert.
        assert backend.get_cleaned_attrs(["P1"]) == []
        assert len(backend.get_cleaned_attrs(["P2"])) == 1

    def test_replace_block_keys_full_rebuild(self, backend):
        backend.begin()
        backend.add_block_keys([("B1", "hash123", "P1")])
        backend.commit()
        backend.begin()
        backend.replace_block_keys([("B1", "hash456", "P2")])
        backend.commit()
        # P1's key is gone — replace_block_keys is a full rebuild, not an append.
        hits = backend.lookup_block_candidates(
            {"B1": "hash123", "B3": None, "B4": None, "B6": None, "B7": None, "B8": None, "B9": None},
            phones=[], threshold=500,
        )
        assert hits == {}
        hits2 = backend.lookup_block_candidates(
            {"B1": "hash456", "B3": None, "B4": None, "B6": None, "B7": None, "B8": None, "B9": None},
            phones=[], threshold=500,
        )
        assert hits2 == {"P2": {"B1"}}


def _seed_three_entities(backend):
    """P1 (merged with P2), P3 (singleton, review), P4 (singleton, none) —
    for `list_entities`/`dashboard_summary`/`review_candidates_for_patid`
    unit coverage that doesn't need a full `publish_run`."""
    backend.begin()
    backend.upsert_entities_bulk([
        ("M-000001", "r1", "deterministic", 1, 0.99, "SSN_DOB", "SSN_DOB", "t0"),
        ("M-000002", "r1", "review", 0, None, None, None, "t1"),
        ("M-000003", "r1", "none", 0, None, None, None, "t2"),
    ])
    backend.upsert_entity_members_bulk([
        ("P1", "M-000001", 1, "pipeline", "t0"),
        ("P2", "M-000001", 0, "pipeline", "t0"),
        ("P3", "M-000002", 1, "pipeline", "t1"),
        ("P4", "M-000003", 1, "pipeline", "t2"),
    ])
    backend.upsert_record_attrs_bulk([
        ("P1", "Jane", "Doe", "1990-01-01", "6789", None, None, None, None, None, "r1"),
        ("P2", "Jane", "Doe", "1990-01-01", "6789", None, None, None, None, None, "r1"),
        ("P3", "Amy", "Lee", "1975-03-03", "1234", None, None, None, None, None, "r1"),
        ("P4", "John", "Smith", "1985-05-05", "4321", None, None, None, None, None, "r1"),
    ])
    backend.insert_review_candidates([
        ("P3", "P5", "NAME_DOB_SEX", 0.98, "NAME_DOB_SEX", "B3", "r1", "t1", None, None),
    ])
    backend.commit()


class TestReadSideMethods:
    """`list_entities`, `dashboard_summary`, `get_record_raw`,
    `review_candidates_for_patid` — the dashboard-read parity work
    (docs/Data-Contract.md Stage 6). Integration coverage against the real
    FastAPI routes lives in tests/integration/test_api.py; these are the
    isolated edge cases (pagination, individual filters) that's lighter to
    prove directly against the backend."""

    def test_list_entities_pagination(self, backend):
        _seed_three_entities(backend)
        page1, total = backend.list_entities(page=1, page_size=2)
        assert total == 3
        assert len(page1) == 2
        page2, _ = backend.list_entities(page=2, page_size=2)
        assert len(page2) == 1
        assert {r["mid"] for r in page1} | {r["mid"] for r in page2} == {
            "M-000001", "M-000002", "M-000003",
        }

    def test_list_entities_member_count(self, backend):
        _seed_three_entities(backend)
        rows, _ = backend.list_entities(is_merged=True)
        assert len(rows) == 1
        assert rows[0]["member_count"] == 2

    def test_list_entities_birth_date_filter(self, backend):
        _seed_three_entities(backend)
        rows, total = backend.list_entities(birth_date="1975-03-03")
        assert total == 1
        assert rows[0]["mid"] == "M-000002"

    def test_list_entities_ssn_last4_filter(self, backend):
        _seed_three_entities(backend)
        rows, total = backend.list_entities(ssn_last4="4321")
        assert total == 1
        assert rows[0]["mid"] == "M-000003"

    def test_list_entities_updated_after_before(self, backend):
        _seed_three_entities(backend)
        rows, total = backend.list_entities(updated_after="t1", updated_before="t1")
        assert total == 1
        assert rows[0]["mid"] == "M-000002"

    def test_list_entities_search_matches_name_and_mid(self, backend):
        _seed_three_entities(backend)
        by_name, _ = backend.list_entities(search="smith")
        assert {r["mid"] for r in by_name} == {"M-000003"}
        by_mid, _ = backend.list_entities(search="M-000001")
        assert {r["mid"] for r in by_mid} == {"M-000001"}

    def test_list_entities_search_by_full_name(self, backend):
        """Parity with `sql_backend`: a full name spans `first_name` +
        `last_name`, so the query is tokenized rather than matched whole."""
        _seed_three_entities(backend)
        rows, total = backend.list_entities(search="John Smith")
        assert total == 1
        assert rows[0]["mid"] == "M-000003"

    def test_list_entities_search_full_name_order_insensitive(self, backend):
        _seed_three_entities(backend)
        _, total = backend.list_entities(search="smith john")
        assert total == 1

    def test_list_entities_search_tokens_are_anded(self, backend):
        """Amy is Lee and John is Smith — "Amy Smith" is nobody."""
        _seed_three_entities(backend)
        _, total = backend.list_entities(search="Amy Smith")
        assert total == 0

    def test_list_entities_blank_search_does_not_filter(self, backend):
        _seed_three_entities(backend)
        _, total = backend.list_entities(search="   ")
        assert total == 3

    def test_list_entities_search_treats_query_as_text_not_regex(self, backend):
        """A name is not a pattern: an unbalanced paren must return no rows,
        not raise `re.error` out of the dashboard's registry search."""
        _seed_three_entities(backend)
        _, total = backend.list_entities(search="Smith (")
        assert total == 0

    def test_dashboard_summary_aggregates(self, backend):
        _seed_three_entities(backend)
        summary = backend.dashboard_summary()
        assert summary["total_records"] == 4
        assert summary["duplicate_clusters"] == 1
        assert summary["matched_records"] == 2
        assert summary["needs_review_records"] == 1
        assert summary["no_match_records"] == 1
        assert summary["auto_match_records"] == 2
        assert summary["manual_merge_actions"] == 0
        assert summary["manual_unmerge_actions"] == 0

    def test_get_record_raw_hit_and_miss(self, backend):
        backend.begin()
        backend.upsert_record_raw_bulk([("P1", '{"FirstNM_raw": "JANE"}', "r1")])
        backend.commit()
        assert backend.get_record_raw("P1") == '{"FirstNM_raw": "JANE"}'
        assert backend.get_record_raw("P-nope") is None

    def test_review_candidates_for_patid_joins_both_sides(self, backend):
        _seed_three_entities(backend)
        candidates = backend.review_candidates_for_patid("P3")
        assert len(candidates) == 1
        c = candidates[0]
        assert c["patid_a"] == "P3" and c["patid_b"] == "P5"
        assert c["a_first_name"] == "Amy"
        # P5 never got a record_attrs row — its display fields are None, not NaN.
        assert c["b_first_name"] is None

    def test_review_candidates_for_patid_no_match_returns_empty(self, backend):
        _seed_three_entities(backend)
        assert backend.review_candidates_for_patid("P1") == []


class TestAuditLog:
    """`audit_log`/`insert_audit_log`/`list_audit_log`/`locked_patids` — the
    reviewer merge/unmerge parity work (docs/Data-Contract.md Stage 6)."""

    def test_insert_audit_log_assigns_incrementing_ids(self, backend):
        backend.begin()
        id1 = backend.insert_audit_log(
            ts_utc="t0", user="reviewer.jclark", action="merge",
            patids="P1,P2", mid="M-000001", prev_state="Needs review",
            next_state="Merged", run_id="r1",
        )
        id2 = backend.insert_audit_log(
            ts_utc="t1", user="reviewer.jclark", action="unmerge",
            patids="P2", mid="M-000002", prev_state="Merged",
            next_state="Unmerged", run_id="r1",
        )
        backend.commit()
        assert id1 == 1
        assert id2 == 2

    def test_list_audit_log_ordering_and_since(self, backend):
        backend.begin()
        backend.insert_audit_log(
            ts_utc="t0", user="jclark", action="merge", patids="P1,P2",
            mid="M-000001", prev_state="Needs review", next_state="Merged", run_id="r1",
        )
        backend.insert_audit_log(
            ts_utc="t1", user="jclark", action="unmerge", patids="P2",
            mid="M-000002", prev_state="Merged", next_state="Unmerged", run_id="r1",
        )
        backend.commit()

        rows = backend.list_audit_log()
        assert [r["action"] for r in rows] == ["unmerge", "merge"]  # newest first

        since_rows = backend.list_audit_log(since="t0")
        assert len(since_rows) == 1
        assert since_rows[0]["action"] == "unmerge"

        limited = backend.list_audit_log(limit=1)
        assert len(limited) == 1

    def test_locked_patids_reflects_audit_log(self, backend):
        assert backend.locked_patids() == set()
        backend.begin()
        backend.insert_audit_log(
            ts_utc="t0", user="jclark", action="unmerge", patids="P1,P2",
            mid="M-000001", prev_state="Merged", next_state="Unmerged", run_id="r1",
        )
        backend.commit()
        assert backend.locked_patids() == {"P1", "P2"}


@pytest.fixture
def settings(tmp_path):
    s = Settings(project_root=tmp_path, index_backend="parquet")
    s.local_index_dir = tmp_path / "local_index"
    s.fs_model_dir = tmp_path / "fs_model_empty"
    s.ensure_dirs()
    return s


class TestIncrementalAgainstParquetBackend:
    """Same scenarios as tests/unit/test_incremental.py's SqlIndexBackend
    cases, run against ParquetIndexBackend — proves incremental.py doesn't
    special-case either backend."""

    def test_second_call_on_a_fresh_backend_auto_merges_into_first(self, settings):
        backend1 = ParquetIndexBackend(settings.local_index_dir)
        try:
            first = incremental.score_records(
                backend1, settings,
                [_raw("P1", "JANE", "DOE", "1990-01-01", ssn="234-56-7891")],
                run_id="r1",
            )
        finally:
            backend1.close()
        assert first[0]["tier"] == "no_match"
        mid1 = first[0]["mid"]

        # A brand-new backend instance over the same data_dir — proves the
        # index really persisted to disk and was reloaded correctly, not
        # just still sitting in the first backend's memory. Same SSN + DOB
        # as P1 -> SSN_DOB auto-merge despite the different name (SSN_DOB
        # doesn't require name agreement).
        backend2 = ParquetIndexBackend(settings.local_index_dir)
        try:
            second = incremental.score_records(
                backend2, settings,
                [_raw("P2", "DIFFERENT", "NAME", "1990-01-01", ssn="234-56-7891")],
                run_id="r2",
            )
        finally:
            backend2.close()
        assert second[0]["tier"] == "auto_merge"
        assert second[0]["mid"] == mid1

    def test_review_tier_writes_candidate_and_review_origin(self, settings):
        backend = ParquetIndexBackend(settings.local_index_dir)
        try:
            incremental.score_records(
                backend, settings,
                [_raw("P4", "AMY", "LEE", "1975-03-03", sex="FEMALE")],
                run_id="r1",
            )
            outcomes = incremental.score_records(
                backend, settings,
                [_raw("P5", "AMY", "LEE", "1975-03-03", sex="FEMALE")],
                run_id="r2",
            )
        finally:
            backend.close()
        outcome = outcomes[0]
        assert outcome["tier"] == "human_review"
        entity = ParquetIndexBackend(settings.local_index_dir)
        try:
            got = entity.get_entity(outcome["mid"])
        finally:
            entity.close()
        assert got["entity"]["origin"] == "review"


class TestLocalScoreCli:
    def test_score_local_round_trips_through_disk(self, settings):
        outcomes = score_local(
            [_raw("P1", "JANE", "DOE", "1990-01-01", ssn="234-56-7891")],
            settings,
        )
        assert outcomes[0]["tier"] == "no_match"

        outcomes2 = score_local(
            [_raw("P2", "OTHER", "NAME", "1990-01-01", ssn="234-56-7891")],
            settings,
        )
        assert outcomes2[0]["tier"] == "auto_merge"
        assert outcomes2[0]["mid"] == outcomes[0]["mid"]
