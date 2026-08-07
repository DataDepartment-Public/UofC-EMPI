"""`IndexBackend` — the storage interface both `src/api/ingest/incremental.py`
(single/few-record scoring) and `src/api/ingest/publish.py` (full-batch publish)
read and write through, with two implementations:

* `SqlIndexBackend` — a thin, engine-agnostic adapter over a DB-API-style
  connection and a *store module* exposing `sql_backend.py`'s function surface
  (`lookup_block_candidates`, `upsert_entity`, ...). Defaults to
  `src.api.backends.sql_backend` (SQLite today), but the connection is only ever typed as
  `Any` and the store module is injected, not imported by name — swapping in
  a Postgres-flavored store module later (same function signatures, different
  SQL/placeholder style inside) needs no change here. Used by the FastAPI
  service (the "operationalize" path, `POST /records/score` and `POST /runs`).
* `ParquetIndexBackend` (`src/api/backends/parquet_backend.py`) — the same operations
  against local Parquet files, read into memory and written back on commit.
  Used by the local-mode CLIs (`src/api/ingest/local_score.py`,
  `src/api/ingest/publish_local.py`), and by the API too when
  `EMPI_INDEX_BACKEND=parquet`.

Both expose the exact same method surface so callers never branch on which
one they're talking to — see the `IndexBackend` Protocol below. The
row-at-a-time methods (`upsert_entity`, `add_block_keys`, ...) serve
`incremental.py`; the `*_bulk`/`replace_*` methods serve `publish.py`, which
touches every entity in a run at once and needs bulk operations for the same
performance reason `sql_backend.py`'s own bulk functions exist (see that module's
PERFORMANCE note).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IndexBackend(Protocol):
    """Every storage operation `incremental.py` (row-at-a-time) and
    `publish.py` (bulk) need."""

    def begin(self) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...

    def locked_patids(self) -> set[str]: ...

    def lookup_block_candidates(
        self, keys: dict[str, str | None], phones: Iterable[str], threshold: int
    ) -> dict[str, set[str]]: ...

    def get_cleaned_attrs(self, patids: list[str]) -> list[dict]: ...
    def add_block_keys(self, rows: list[tuple[str, str, str]]) -> None: ...
    def upsert_cleaned_attrs(self, row: tuple) -> None: ...

    def get_entity_mid_for_patid(self, patid: str) -> str | None: ...
    def next_mid(self) -> str: ...
    def get_entity(self, mid: str) -> dict | None: ...

    def upsert_entity(
        self,
        mid: str,
        run_id: str,
        origin: str,
        is_merged: bool,
        confidence: float | None,
        updated_utc: str,
        match_rule: str | None = None,
        evidence: str | None = None,
    ) -> None: ...

    def upsert_entity_member(
        self, patid: str, mid: str, is_primary: bool, added_by: str, updated_utc: str
    ) -> None: ...

    def reassign_entity_members(
        self, from_mids: list[str], to_mid: str, updated_utc: str
    ) -> None: ...

    def upsert_suggestion(
        self, patid: str, run_id: str, suggested_mid: str, created_utc: str
    ) -> None: ...

    def insert_review_candidates(self, rows: list[tuple]) -> None: ...

    # ── Bulk operations (batch publish — src/api/ingest/publish.py) ────────────────
    def max_mid_sequence(self) -> int: ...
    def all_entity_member_mids(self) -> dict[str, str]: ...
    def upsert_entities_bulk(self, rows: list[tuple]) -> None: ...
    def upsert_entity_members_bulk(self, rows: list[tuple]) -> None: ...
    def upsert_record_attrs_bulk(self, rows: list[tuple]) -> None: ...
    def upsert_record_raw_bulk(self, rows: list[tuple]) -> None: ...
    def upsert_suggestions_bulk(self, rows: list[tuple]) -> None: ...
    def replace_review_candidates_for_run(
        self, run_id: str, rows: list[tuple]
    ) -> None: ...
    def replace_cleaned_attrs(self, rows: list[tuple]) -> None: ...
    def replace_block_keys(self, rows: list[tuple[str, str, str]]) -> None: ...

    # ── Read side (dashboard routes — src/api/routers/records.py, dashboard.py) ─
    def list_entities(
        self,
        *,
        search: str | None = None,
        origin: str | None = None,
        is_merged: bool | None = None,
        birth_date: str | None = None,
        ssn_last4: str | None = None,
        updated_after: str | None = None,
        updated_before: str | None = None,
        confidence_min: float | None = None,
        confidence_max: float | None = None,
        sort: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict], int]: ...

    def dashboard_summary(self) -> dict: ...
    def get_record_raw(self, patid: str) -> str | None: ...
    def review_candidates_for_patid(self, patid: str) -> list[dict]: ...
    def list_review_candidates(
        self,
        *,
        confidence_min: float | None = None,
        confidence_max: float | None = None,
        reviewed: bool | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict], int]: ...

    # ── Reviewer audit log (src/api/routers/audit.py) ───────────────────────
    def insert_audit_log(
        self,
        *,
        ts_utc: str,
        user: str,
        action: str,
        patids: str,
        mid: str,
        prev_state: str,
        next_state: str,
        run_id: str | None,
        related_patids: str | None = None,
    ) -> int: ...

    def list_audit_log(
        self, *, limit: int = 100, since: str | None = None
    ) -> list[dict]: ...

    def get_audit_log_row(self, audit_id: int) -> dict | None: ...


class SqlIndexBackend:
    """`IndexBackend` over any DB-API-style connection + a store module.

    `conn` is typed `Any` deliberately — this class never imports `sqlite3`
    or touches connection internals directly, only calls
    `store_module.<fn>(conn, ...)`. `store_module` defaults to the project's
    current SQLite-flavored `src.api.backends.sql_backend`; a future Postgres (or other
    engine) deployment would ship its own module with the same function
    names/signatures (its own placeholder style, DDL, etc. live there, not
    here) and pass it in instead — this adapter is unchanged either way.
    """

    def __init__(self, conn: Any, store_module: Any | None = None) -> None:
        if store_module is None:
            from src.api.backends import sql_backend as store_module
        self.conn = conn
        self._store = store_module

    def begin(self) -> None:
        self.conn.execute("BEGIN")

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def close(self) -> None:
        self.conn.close()

    def locked_patids(self) -> set[str]:
        return self._store.locked_patids(self.conn)

    def lookup_block_candidates(self, keys, phones, threshold):
        return self._store.lookup_block_candidates(self.conn, keys, phones, threshold)

    def get_cleaned_attrs(self, patids: list[str]) -> list[dict]:
        return self._store.get_cleaned_attrs(self.conn, patids)

    def add_block_keys(self, rows: list[tuple[str, str, str]]) -> None:
        self._store.add_block_keys(self.conn, rows)

    def upsert_cleaned_attrs(self, row: tuple) -> None:
        self._store.upsert_cleaned_attrs(self.conn, row)

    def get_entity_mid_for_patid(self, patid: str) -> str | None:
        return self._store.get_entity_mid_for_patid(self.conn, patid)

    def next_mid(self) -> str:
        return self._store.next_mid(self.conn)

    def get_entity(self, mid: str) -> dict | None:
        return self._store.get_entity(self.conn, mid)

    def upsert_entity(
        self, mid, run_id, origin, is_merged, confidence, updated_utc,
        match_rule=None, evidence=None,
    ) -> None:
        self._store.upsert_entity(
            self.conn, mid, run_id, origin, is_merged, confidence, updated_utc,
            match_rule=match_rule, evidence=evidence,
        )

    def upsert_entity_member(self, patid, mid, is_primary, added_by, updated_utc) -> None:
        self._store.upsert_entity_member(
            self.conn, patid, mid, is_primary, added_by, updated_utc
        )

    def reassign_entity_members(self, from_mids, to_mid, updated_utc) -> None:
        self._store.reassign_entity_members(self.conn, from_mids, to_mid, updated_utc)

    def upsert_suggestion(self, patid, run_id, suggested_mid, created_utc) -> None:
        self._store.upsert_suggestion(self.conn, patid, run_id, suggested_mid, created_utc)

    def insert_review_candidates(self, rows: list[tuple]) -> None:
        self._store.insert_review_candidates(self.conn, rows)

    def max_mid_sequence(self) -> int:
        return self._store.max_mid_sequence(self.conn)

    def all_entity_member_mids(self) -> dict[str, str]:
        return self._store.all_entity_member_mids(self.conn)

    def upsert_entities_bulk(self, rows: list[tuple]) -> None:
        self._store.upsert_entities_bulk(self.conn, rows)

    def upsert_entity_members_bulk(self, rows: list[tuple]) -> None:
        self._store.upsert_entity_members_bulk(self.conn, rows)

    def upsert_record_attrs_bulk(self, rows: list[tuple]) -> None:
        self._store.upsert_record_attrs_bulk(self.conn, rows)

    def upsert_record_raw_bulk(self, rows: list[tuple]) -> None:
        self._store.upsert_record_raw_bulk(self.conn, rows)

    def upsert_suggestions_bulk(self, rows: list[tuple]) -> None:
        self._store.upsert_suggestions_bulk(self.conn, rows)

    def replace_review_candidates_for_run(self, run_id: str, rows: list[tuple]) -> None:
        self._store.replace_review_candidates_for_run(self.conn, run_id, rows)

    def replace_cleaned_attrs(self, rows: list[tuple]) -> None:
        self._store.replace_cleaned_attrs(self.conn, rows)

    def replace_block_keys(self, rows: list[tuple[str, str, str]]) -> None:
        self._store.replace_block_keys(self.conn, rows)

    def list_entities(
        self,
        *,
        search=None, origin=None, is_merged=None, birth_date=None,
        ssn_last4=None, updated_after=None, updated_before=None,
        confidence_min=None, confidence_max=None, sort=None,
        page=1, page_size=50,
    ) -> tuple[list[dict], int]:
        return self._store.list_entities(
            self.conn, search=search, origin=origin, is_merged=is_merged,
            birth_date=birth_date, ssn_last4=ssn_last4,
            updated_after=updated_after, updated_before=updated_before,
            confidence_min=confidence_min, confidence_max=confidence_max,
            sort=sort, page=page, page_size=page_size,
        )

    def dashboard_summary(self) -> dict:
        return self._store.dashboard_summary(self.conn)

    def get_record_raw(self, patid: str) -> str | None:
        return self._store.get_record_raw(self.conn, patid)

    def review_candidates_for_patid(self, patid: str) -> list[dict]:
        return self._store.review_candidates_for_patid(self.conn, patid)

    def list_review_candidates(
        self,
        *,
        confidence_min=None, confidence_max=None, reviewed=None, search=None,
        page=1, page_size=50,
    ) -> tuple[list[dict], int]:
        return self._store.list_review_candidates(
            self.conn, confidence_min=confidence_min, confidence_max=confidence_max,
            reviewed=reviewed, search=search, page=page, page_size=page_size,
        )

    def insert_audit_log(
        self, *, ts_utc, user, action, patids, mid, prev_state, next_state, run_id,
        related_patids=None,
    ) -> int:
        return self._store.insert_audit_log(
            self.conn, ts_utc=ts_utc, user=user, action=action, patids=patids,
            mid=mid, prev_state=prev_state, next_state=next_state, run_id=run_id,
            related_patids=related_patids,
        )

    def list_audit_log(self, *, limit: int = 100, since: str | None = None) -> list[dict]:
        return self._store.list_audit_log(self.conn, limit=limit, since=since)

    def get_audit_log_row(self, audit_id: int) -> dict | None:
        return self._store.get_audit_log_row(self.conn, audit_id)


def build_index_backend(settings: Any) -> IndexBackend:
    """Construct the backend `settings.index_backend` selects
    (`"sqlite"` -> `SqlIndexBackend` over `sql_backend.py` / `settings.db_path`,
    `"parquet"` -> `ParquetIndexBackend` over `settings.local_index_dir`,
    `"postgres"` -> `SqlIndexBackend` over `postgres_backend.py` /
    `settings.postgres_*`, e.g. Azure Database for PostgreSQL — see
    terraform/postgres.tf).
    Caller owns the returned backend's lifecycle — always `close()` it.

    Assumes the schema already exists — this function only connects, it
    never creates or alters tables. Run `python scripts/init_db.py` once
    per environment (a new local DB, a new Azure Postgres instance, or
    after a code change adds a column to `_COLUMN_MIGRATIONS`) instead of
    relying on this to fix it implicitly; it used to, on every single call
    here, which meant on nearly every request. See docs/API-Design.md.
    """
    backend_name = getattr(settings, "index_backend", "sqlite")
    if backend_name == "parquet":
        from src.api.backends.parquet_backend import ParquetIndexBackend

        return ParquetIndexBackend(settings.local_index_dir)

    if backend_name == "postgres":
        from src.api.backends import postgres_backend

        if not settings.postgres_host or not settings.postgres_user:
            raise RuntimeError(
                "index_backend='postgres' requires EMPI_POSTGRES_HOST and "
                "EMPI_POSTGRES_USER to be set."
            )
        pg_conn = postgres_backend.get_connection(
            settings.postgres_host,
            settings.postgres_port,
            settings.postgres_db,
            settings.postgres_user,
        )
        return SqlIndexBackend(pg_conn, store_module=postgres_backend)

    from src.api.backends import sql_backend

    conn = sql_backend.get_connection(settings.db_path)
    return SqlIndexBackend(conn, store_module=sql_backend)


__all__ = ["IndexBackend", "SqlIndexBackend", "build_index_backend"]
