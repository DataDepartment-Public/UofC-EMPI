"""Postgres layer for the resolved-output database — the Azure deployment's
counterpart to `sql_backend.py`'s SQLite implementation, targeting Azure
Database for PostgreSQL Flexible Server (see `terraform/postgres.tf`).

Implements exactly the function surface `IndexBackend`/`SqlIndexBackend`
(`index_backend.py`) call through `SqlIndexBackend`'s injected `store_module`
— i.e. what the FastAPI service (`POST /records/score`, `POST /runs`, the
dashboard read routes) needs via `EMPI_INDEX_BACKEND=postgres`. It does
*not* port `sql_backend.py`'s few pipeline-batch helpers that sit outside
that Protocol (`patids_with_review_candidates`, `member_count`, the singular
`upsert_record_attrs`) — those serve `src/api/ingest/publish.py`'s full-batch
CLI path, which stays on SQLite/Parquet for now; porting the live API
surface first is the smaller, lower-risk slice, and the batch path doesn't
need to move to stay useful.

Schema mirrors `sql_backend.SCHEMA_SQL` table-for-table; differences are all
dialect, not design:
  * `?` placeholders -> `%s` (psycopg/libpq style).
  * `INTEGER PRIMARY KEY AUTOINCREMENT` -> `GENERATED ALWAYS AS IDENTITY`.
  * `INSERT OR IGNORE` -> `INSERT ... ON CONFLICT (...) DO NOTHING`.
  * `cur.lastrowid` -> `INSERT ... RETURNING id` + `fetchone()`.
  * `LIKE` -> `ILIKE` for the search filters that are documented as
    case-insensitive (SQLite's `LIKE` is case-insensitive for ASCII by
    default; Postgres's is case-sensitive — `ILIKE` is what actually
    preserves the original behavior, not a stylistic change).
  * `PRAGMA table_info` -> `information_schema.columns` for `_ensure_columns`.
`ON CONFLICT ... DO UPDATE SET excluded.col` and `ORDER BY ... NULLS LAST`
are valid, unchanged Postgres syntax (SQLite borrowed both from Postgres).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.api.backends.search_terms import search_tokens
from src.preprocessing.blocking import HASHED_BLOCKS, hash_block_key  # noqa: F401

PgConnection = psycopg.Connection[dict[str, Any]]

_AAD_TOKEN_SCOPE = "https://ossrdbms-aad.database.windows.net/.default"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entity (
    mid          TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    origin       TEXT NOT NULL,
    is_merged    INTEGER NOT NULL,
    confidence   REAL,
    match_rule   TEXT,
    evidence     TEXT,
    updated_utc  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_member (
    patid        TEXT PRIMARY KEY,
    mid          TEXT NOT NULL REFERENCES entity(mid),
    is_primary   INTEGER NOT NULL,
    added_by     TEXT NOT NULL,
    updated_utc  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_member_mid ON entity_member(mid);

-- related_patids: see sql_backend.py's audit_log DDL comment -- same column,
-- same meaning, kept identical across both backends.
CREATE TABLE IF NOT EXISTS audit_log (
    id             INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts_utc         TEXT NOT NULL,
    "user"         TEXT NOT NULL,
    action         TEXT NOT NULL,
    patids         TEXT NOT NULL,
    mid            TEXT NOT NULL,
    prev_state     TEXT NOT NULL,
    next_state     TEXT NOT NULL,
    run_id         TEXT,
    related_patids TEXT,
    prev_mid       TEXT,
    undo_of        INTEGER
);

CREATE TABLE IF NOT EXISTS record_attrs (
    patid        TEXT PRIMARY KEY REFERENCES entity_member(patid),
    first_name   TEXT,
    last_name    TEXT,
    birth_date   TEXT,
    ssn_last4    TEXT,
    email        TEXT,
    zip_code     TEXT,
    address1     TEXT,
    sex          TEXT,
    phone        TEXT,
    middle_name  TEXT,
    suffix       TEXT,
    city         TEXT,
    -- JSON array of every cleaned phone (`Phones_set`), not just the primary
    -- `phone` above — see `publish._attrs_row`.
    phones       TEXT,
    run_id       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS record_raw (
    patid        TEXT PRIMARY KEY REFERENCES entity_member(patid),
    raw_json     TEXT NOT NULL,
    run_id       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_candidate (
    id           INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    patid_a      TEXT NOT NULL,
    patid_b      TEXT NOT NULL,
    match_rule   TEXT,
    confidence   REAL,
    evidence     TEXT,
    source_blocks TEXT,
    run_id       TEXT NOT NULL,
    created_utc  TEXT NOT NULL,
    fs_match_probability   REAL,
    fs_classification_tier TEXT,
    ml_match_probability   REAL,
    ml_classification_tier TEXT,
    UNIQUE(patid_a, patid_b, run_id)
);
CREATE INDEX IF NOT EXISTS idx_review_candidate_a ON review_candidate(patid_a);
CREATE INDEX IF NOT EXISTS idx_review_candidate_b ON review_candidate(patid_b);

CREATE TABLE IF NOT EXISTS entity_suggestion (
    patid          TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    suggested_mid  TEXT NOT NULL,
    created_utc    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS block_key (
    block_id   TEXT NOT NULL,
    key_value  TEXT NOT NULL,
    patid      TEXT NOT NULL,
    PRIMARY KEY (block_id, key_value, patid)
);
CREATE INDEX IF NOT EXISTS idx_block_key_lookup ON block_key(block_id, key_value);
CREATE INDEX IF NOT EXISTS idx_block_key_patid ON block_key(patid);

CREATE TABLE IF NOT EXISTS cleaned_attrs (
    patid        TEXT PRIMARY KEY,
    first_nm     TEXT,
    last_nm      TEXT,
    birth_dt     TEXT,
    ssn          TEXT,
    ssn_last4    TEXT,
    email        TEXT,
    zip_base     TEXT,
    address1     TEXT,
    sex          TEXT,
    phones_json  TEXT,
    run_id       TEXT NOT NULL
);
"""


def get_connection(host: str, port: int, dbname: str, user: str) -> PgConnection:
    """One AAD-authenticated Postgres connection. Callers own its lifecycle —
    this module never pools connections (mirrors `sql_backend.get_connection`).

    No stored password: `user` is this app's own Azure AD identity (see
    `terraform/postgres.tf`'s AAD administrator registration), and the token
    used as the connection password is fetched fresh on every call via
    `DefaultAzureCredential` — resolves to the App Service's managed identity
    in Azure, falls back through `az login`'s credentials for local testing
    against a real server. Tokens are short-lived (a few hours); fetching one
    per connection rather than caching sidesteps expiry entirely, matching
    the per-request connection lifecycle the rest of this module assumes.

    Deliberately *not* `autocommit=True`: `SqlIndexBackend.begin()`
    (`index_backend.py`) issues a raw `BEGIN` and later calls `conn.commit()`/
    `conn.rollback()` — psycopg3 refuses manual `.commit()`/`.rollback()`
    calls once a connection is in autocommit mode (they only make sense
    against a transaction psycopg itself is tracking). Leaving autocommit at
    its default (off) means psycopg already tracks a transaction from the
    first statement on the connection, so the explicit `BEGIN` either starts
    it (the common case — `begin()` runs before any other query) or is a
    harmless no-op if something already queried first (Postgres logs a
    NOTICE, doesn't error); either way `.commit()`/`.rollback()` then work
    exactly as psycopg expects.
    """
    from azure.identity import DefaultAzureCredential

    token = DefaultAzureCredential().get_token(_AAD_TOKEN_SCOPE).token
    return psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=token,
        sslmode="require",
        row_factory=dict_row,
    )


_COLUMN_MIGRATIONS: dict[str, dict[str, str]] = {
    "review_candidate": {
        "fs_match_probability": "REAL",
        "fs_classification_tier": "TEXT",
        "ml_match_probability": "REAL",
        "ml_classification_tier": "TEXT",
    },
    "audit_log": {
        "related_patids": "TEXT",
        "prev_mid": "TEXT",
        "undo_of": "INTEGER",
    },
    # Display-only fields added for the feature-comparison table; NULL on an
    # existing database until the run is re-published. See
    # `sql_backend._COLUMN_MIGRATIONS`.
    "record_attrs": {
        "middle_name": "TEXT",
        "suffix": "TEXT",
        "city": "TEXT",
        "phones": "TEXT",
    },
}


def _ensure_columns(conn: PgConnection) -> None:
    for table, columns in _COLUMN_MIGRATIONS.items():
        existing = {
            row["column_name"]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (table,),
            )
        }
        for name, coltype in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}")


def init_db(conn: PgConnection) -> None:
    """Create every table if it doesn't already exist. Idempotent."""
    conn.execute(SCHEMA_SQL)
    _ensure_columns(conn)
    conn.commit()


def _executemany(conn: PgConnection, sql: str, rows: list[tuple]) -> None:
    """psycopg has no `Connection.executemany` shorthand (unlike sqlite3) —
    only `Cursor.executemany`. Every `*_bulk`/`replace_*` function below
    routes through this instead of repeating the cursor boilerplate."""
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(sql, rows)


# ── Locking (reconciliation) ────────────────────────────────────────────────
def locked_patids(conn: PgConnection) -> set[str]:
    rows = conn.execute("SELECT patids FROM audit_log").fetchall()
    locked: set[str] = set()
    for row in rows:
        locked.update(p for p in row["patids"].split(",") if p)
    return locked


# ── Entity / membership ─────────────────────────────────────────────────────
def get_entity_mid_for_patid(conn: PgConnection, patid: str) -> str | None:
    row = conn.execute(
        "SELECT mid FROM entity_member WHERE patid = %s", (patid,)
    ).fetchone()
    return row["mid"] if row else None


def max_mid_sequence(conn: PgConnection) -> int:
    rows = conn.execute("SELECT mid FROM entity").fetchall()
    best = 0
    for row in rows:
        mid = row["mid"]
        if mid.startswith("M-") and mid[2:].isdigit():
            best = max(best, int(mid[2:]))
    return best


def next_mid(conn: PgConnection) -> str:
    return f"M-{max_mid_sequence(conn) + 1:06d}"


_ENTITY_UPSERT_SQL = """
    INSERT INTO entity
        (mid, run_id, origin, is_merged, confidence, match_rule, evidence, updated_utc)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT(mid) DO UPDATE SET
        run_id=excluded.run_id, origin=excluded.origin,
        is_merged=excluded.is_merged, confidence=excluded.confidence,
        match_rule=excluded.match_rule, evidence=excluded.evidence,
        updated_utc=excluded.updated_utc
"""

_MEMBER_UPSERT_SQL = """
    INSERT INTO entity_member (patid, mid, is_primary, added_by, updated_utc)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT(patid) DO UPDATE SET
        mid=excluded.mid, is_primary=excluded.is_primary,
        added_by=excluded.added_by, updated_utc=excluded.updated_utc
"""

_ATTRS_UPSERT_SQL = """
    INSERT INTO record_attrs
        (patid, first_name, last_name, birth_date, ssn_last4, email,
         zip_code, address1, sex, phone, middle_name, suffix, city, phones,
         run_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT(patid) DO UPDATE SET
        first_name=excluded.first_name, last_name=excluded.last_name,
        birth_date=excluded.birth_date, ssn_last4=excluded.ssn_last4,
        email=excluded.email, zip_code=excluded.zip_code,
        address1=excluded.address1, sex=excluded.sex, phone=excluded.phone,
        middle_name=excluded.middle_name, suffix=excluded.suffix,
        city=excluded.city, phones=excluded.phones,
        run_id=excluded.run_id
"""

_SUGGESTION_UPSERT_SQL = """
    INSERT INTO entity_suggestion (patid, run_id, suggested_mid, created_utc)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT(patid) DO UPDATE SET
        run_id=excluded.run_id, suggested_mid=excluded.suggested_mid,
        created_utc=excluded.created_utc
"""


def upsert_entity(
    conn: PgConnection,
    mid: str,
    run_id: str,
    origin: str,
    is_merged: bool,
    confidence: float | None,
    updated_utc: str,
    match_rule: str | None = None,
    evidence: str | None = None,
) -> None:
    conn.execute(
        _ENTITY_UPSERT_SQL,
        (mid, run_id, origin, int(is_merged), confidence, match_rule, evidence, updated_utc),
    )


def upsert_entities_bulk(conn: PgConnection, rows: list[tuple]) -> None:
    """`rows` are `(mid, run_id, origin, is_merged, confidence, match_rule,
    evidence, updated_utc)` tuples (`is_merged` already coerced to 0/1)."""
    _executemany(conn, _ENTITY_UPSERT_SQL, rows)


def upsert_entity_member(
    conn: PgConnection,
    patid: str,
    mid: str,
    is_primary: bool,
    added_by: str,
    updated_utc: str,
) -> None:
    conn.execute(_MEMBER_UPSERT_SQL, (patid, mid, int(is_primary), added_by, updated_utc))


def upsert_entity_members_bulk(conn: PgConnection, rows: list[tuple]) -> None:
    _executemany(conn, _MEMBER_UPSERT_SQL, rows)


def upsert_record_attrs_bulk(conn: PgConnection, rows: list[tuple]) -> None:
    _executemany(conn, _ATTRS_UPSERT_SQL, rows)


_RAW_UPSERT_SQL = """
    INSERT INTO record_raw (patid, raw_json, run_id)
    VALUES (%s, %s, %s)
    ON CONFLICT(patid) DO UPDATE SET
        raw_json=excluded.raw_json, run_id=excluded.run_id
"""


def upsert_record_raw_bulk(conn: PgConnection, rows: list[tuple]) -> None:
    """`rows` are `(patid, raw_json, run_id)`."""
    _executemany(conn, _RAW_UPSERT_SQL, rows)


def get_record_raw(conn: PgConnection, patid: str) -> str | None:
    row = conn.execute(
        "SELECT raw_json FROM record_raw WHERE patid = %s", (patid,)
    ).fetchone()
    return row["raw_json"] if row else None


_REVIEW_CANDIDATE_INSERT_FULL_SQL = """
    INSERT INTO review_candidate
        (patid_a, patid_b, match_rule, confidence, evidence, source_blocks,
         run_id, created_utc, fs_match_probability, fs_classification_tier)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT(patid_a, patid_b, run_id) DO UPDATE SET
        match_rule=excluded.match_rule, confidence=excluded.confidence,
        evidence=excluded.evidence, source_blocks=excluded.source_blocks,
        created_utc=excluded.created_utc,
        fs_match_probability=excluded.fs_match_probability,
        fs_classification_tier=excluded.fs_classification_tier
"""


def insert_review_candidates(conn: PgConnection, rows: list[tuple]) -> None:
    """Append-only review-candidate insert carrying the FS score columns —
    used by incremental scoring, which always mints a fresh `run_id` per
    call, so there is nothing to delete first (unlike
    `replace_review_candidates_for_run`). `rows` are `(patid_a, patid_b,
    match_rule, confidence, evidence, source_blocks, run_id, created_utc,
    fs_match_probability, fs_classification_tier)`."""
    _executemany(conn, _REVIEW_CANDIDATE_INSERT_FULL_SQL, rows)


_REVIEW_CANDIDATE_INSERT_SQL = """
    INSERT INTO review_candidate
        (patid_a, patid_b, match_rule, confidence, evidence, source_blocks,
         run_id, created_utc, ml_match_probability, ml_classification_tier)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT(patid_a, patid_b, run_id) DO UPDATE SET
        match_rule=excluded.match_rule, confidence=excluded.confidence,
        evidence=excluded.evidence, source_blocks=excluded.source_blocks,
        created_utc=excluded.created_utc,
        ml_match_probability=excluded.ml_match_probability,
        ml_classification_tier=excluded.ml_classification_tier
"""


def replace_review_candidates_for_run(
    conn: PgConnection, run_id: str, rows: list[tuple]
) -> None:
    """Replace this run's review-candidate rows wholesale (delete + bulk
    insert) — a review pair no longer in the latest run's output should
    disappear rather than linger as a stale suggestion. `rows` are
    `(patid_a, patid_b, match_rule, confidence, evidence, source_blocks,
    run_id, created_utc, ml_match_probability, ml_classification_tier)`."""
    conn.execute("DELETE FROM review_candidate WHERE run_id = %s", (run_id,))
    _executemany(conn, _REVIEW_CANDIDATE_INSERT_SQL, rows)


#: `record_attrs` display columns aliased `a_*`/`b_*`, shared by the two pair
#: readers below — the counterpart of `sql_backend._PAIR_ATTR_SELECT`, and
#: kept in step with it so both backends return the same payload keys.
_PAIR_ATTR_SELECT = ",\n".join(
    f"               ra_{side}.{col} AS {side}_{col}"
    for side in ("a", "b")
    for col in (
        "first_name", "last_name", "birth_date", "ssn_last4", "email",
        "zip_code", "address1", "sex", "phone", "middle_name", "suffix",
        "city", "phones",
    )
)


def review_candidates_for_patid(conn: PgConnection, patid: str) -> list[dict]:
    """Every review-candidate pair touching `patid`, joined to `record_attrs`
    on *both* sides — the UI needs the other PATID's name/SSN/DOB to render
    the expanded-row candidate list, not just its ID."""
    rows = conn.execute(
        f"""
        SELECT rc.*,
{_PAIR_ATTR_SELECT}
        FROM review_candidate rc
        LEFT JOIN record_attrs ra_a ON ra_a.patid = rc.patid_a
        LEFT JOIN record_attrs ra_b ON ra_b.patid = rc.patid_b
        WHERE rc.patid_a = %s OR rc.patid_b = %s
        ORDER BY COALESCE(rc.confidence, rc.ml_match_probability) DESC NULLS LAST, rc.id
        """,
        (patid, patid),
    ).fetchall()
    return [dict(r) for r in rows]


def list_review_candidates(
    conn: PgConnection,
    *,
    confidence_min: float | None = None,
    confidence_max: float | None = None,
    reviewed: bool | None = None,
    search: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[dict], int]:
    """Paginated, candidate-grain review queue — see
    `sql_backend.list_review_candidates` for the full filter semantics this
    mirrors exactly (case-insensitive `search` is `ILIKE` here, not `LIKE` —
    Postgres's `LIKE` is case-sensitive, unlike SQLite's)."""
    where = []
    params: list = []
    if confidence_min is not None:
        where.append("COALESCE(rc.confidence, rc.ml_match_probability) >= %s")
        params.append(confidence_min)
    if confidence_max is not None:
        where.append("COALESCE(rc.confidence, rc.ml_match_probability) <= %s")
        params.append(confidence_max)
    tokens = search_tokens(search)
    if tokens:
        # Mirrors sql_backend.list_review_candidates — all tokens on one side.
        side_a = " AND ".join(
            "(ra_a.first_name ILIKE %s OR ra_a.last_name ILIKE %s OR rc.patid_a ILIKE %s)"
            for _ in tokens
        )
        side_b = " AND ".join(
            "(ra_b.first_name ILIKE %s OR ra_b.last_name ILIKE %s OR rc.patid_b ILIKE %s)"
            for _ in tokens
        )
        where.append(f"(({side_a}) OR ({side_b}))")
        for _side in (side_a, side_b):
            for token in tokens:
                like = f"%{token}%"
                params.extend([like, like, like])

    reviewed_expr = (
        "(ema.mid = emb.mid OR EXISTS ("
        "SELECT 1 FROM audit_log al WHERE al.action = 'dismiss' "
        "AND al.patids = rc.patid_a || ',' || rc.patid_b))"
    )
    if reviewed is True:
        where.append(reviewed_expr)
    elif reviewed is False:
        where.append(f"NOT {reviewed_expr}")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    base_from = """
        FROM review_candidate rc
        JOIN entity_member ema ON ema.patid = rc.patid_a
        JOIN entity_member emb ON emb.patid = rc.patid_b
        LEFT JOIN record_attrs ra_a ON ra_a.patid = rc.patid_a
        LEFT JOIN record_attrs ra_b ON ra_b.patid = rc.patid_b
    """

    total_row = conn.execute(
        f"SELECT COUNT(*) AS n {base_from} {where_sql}", params
    ).fetchone()
    assert total_row is not None
    total = total_row["n"]

    offset = max(page - 1, 0) * page_size
    rows = conn.execute(
        f"""
        SELECT rc.*, ema.mid AS mid_a, emb.mid AS mid_b,
{_PAIR_ATTR_SELECT},
               (SELECT COUNT(*) FROM entity_member em WHERE em.mid = ema.mid) AS member_count_a,
               (SELECT COUNT(*) FROM entity_member em WHERE em.mid = emb.mid) AS member_count_b,
               {reviewed_expr} AS reviewed
        {base_from}
        {where_sql}
        ORDER BY COALESCE(rc.confidence, rc.ml_match_probability) DESC NULLS LAST, rc.id
        LIMIT %s OFFSET %s
        """,
        [*params, page_size, offset],
    ).fetchall()
    return [dict(r) for r in rows], total


def upsert_suggestion(
    conn: PgConnection,
    patid: str,
    run_id: str,
    suggested_mid: str,
    created_utc: str,
) -> None:
    conn.execute(_SUGGESTION_UPSERT_SQL, (patid, run_id, suggested_mid, created_utc))


def upsert_suggestions_bulk(conn: PgConnection, rows: list[tuple]) -> None:
    _executemany(conn, _SUGGESTION_UPSERT_SQL, rows)


def all_entity_member_mids(conn: PgConnection) -> dict[str, str]:
    return {
        row["patid"]: row["mid"]
        for row in conn.execute("SELECT patid, mid FROM entity_member")
    }


def get_entity(conn: PgConnection, mid: str) -> dict | None:
    """One entity + its members (joined to `record_attrs`), or None."""
    entity_row = conn.execute("SELECT * FROM entity WHERE mid = %s", (mid,)).fetchone()
    if entity_row is None:
        return None
    members = conn.execute(
        """
        SELECT em.patid, em.is_primary, em.added_by, em.updated_utc,
               ra.first_name, ra.last_name, ra.birth_date, ra.ssn_last4,
               ra.email, ra.zip_code, ra.address1, ra.sex, ra.phone,
               ra.middle_name, ra.suffix, ra.city, ra.phones
        FROM entity_member em
        LEFT JOIN record_attrs ra ON ra.patid = em.patid
        WHERE em.mid = %s
        ORDER BY em.patid
        """,
        (mid,),
    ).fetchall()
    return {"entity": dict(entity_row), "members": [dict(m) for m in members]}


def list_entities(
    conn: PgConnection,
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
    min_members: int | None = None,
    sort: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[dict], int]:
    """Paginated master rows (one per entity) + their member count — see
    `sql_backend.list_entities` for full filter semantics. `search` is
    `ILIKE` here (case-insensitive), matching SQLite's default `LIKE`
    behavior rather than Postgres's case-sensitive default.

    `sort` accepts the same three orderings as the SQLite store. It was
    previously missing from this signature while `SqlIndexBackend` passed it
    unconditionally, so every `GET /records` against Postgres raised
    `TypeError`; adding `min_members` here would have widened that same crash
    by one more argument, so both are implemented rather than only the new one.
    """
    where = []
    params: list = []
    if origin is not None:
        origins = origin.split(",")
        where.append(f"e.origin IN ({','.join('%s' for _ in origins)})")
        params.extend(origins)
    if is_merged is not None:
        where.append("e.is_merged = %s")
        params.append(int(is_merged))
    if updated_after:
        where.append("e.updated_utc >= %s")
        params.append(updated_after)
    if updated_before:
        where.append("e.updated_utc <= %s")
        params.append(updated_before)
    if confidence_min is not None:
        where.append("e.confidence >= %s")
        params.append(confidence_min)
    if confidence_max is not None:
        where.append("e.confidence <= %s")
        params.append(confidence_max)
    if min_members is not None:
        where.append(
            "(SELECT COUNT(*) FROM entity_member em5 WHERE em5.mid = e.mid) >= %s"
        )
        params.append(min_members)
    tokens = search_tokens(search)
    if tokens:
        # Mirrors sql_backend.list_entities — every token on the same member,
        # each free to match a different column. See `search_terms`.
        member_sql = " AND ".join(
            "(em2.patid ILIKE %s OR ra2.first_name ILIKE %s OR ra2.last_name ILIKE %s)"
            for _ in tokens
        )
        where.append(
            "(e.mid ILIKE %s OR EXISTS (SELECT 1 FROM entity_member em2 "
            "LEFT JOIN record_attrs ra2 ON ra2.patid = em2.patid "
            f"WHERE em2.mid = e.mid AND {member_sql}))"
        )
        params.append(f"%{search}%")
        for token in tokens:
            like = f"%{token}%"
            params.extend([like, like, like])
    if birth_date:
        where.append(
            "EXISTS (SELECT 1 FROM entity_member em3 "
            "JOIN record_attrs ra3 ON ra3.patid = em3.patid "
            "WHERE em3.mid = e.mid AND ra3.birth_date = %s)"
        )
        params.append(birth_date)
    if ssn_last4:
        where.append(
            "EXISTS (SELECT 1 FROM entity_member em4 "
            "JOIN record_attrs ra4 ON ra4.patid = em4.patid "
            "WHERE em4.mid = e.mid AND ra4.ssn_last4 = %s)"
        )
        params.append(ssn_last4)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    total_row = conn.execute(
        f"SELECT COUNT(*) AS n FROM entity e {where_sql}", params
    ).fetchone()
    assert total_row is not None
    total = total_row["n"]

    if sort == "confidence":
        sort_join_sql = ""
        order_sql = "e.confidence DESC NULLS LAST, e.mid"
    elif sort == "name":
        sort_join_sql = (
            "LEFT JOIN entity_member pm ON pm.mid = e.mid AND pm.is_primary "
            "LEFT JOIN record_attrs pa ON pa.patid = pm.patid"
        )
        order_sql = (
            "LOWER(pa.last_name) NULLS LAST, LOWER(pa.first_name), e.mid"
        )
    else:
        sort_join_sql = ""
        order_sql = "e.updated_utc DESC, e.mid"

    offset = max(page - 1, 0) * page_size
    rows = conn.execute(
        f"""
        SELECT e.*, COUNT(em.patid) AS member_count
        FROM entity e
        LEFT JOIN entity_member em ON em.mid = e.mid
        {sort_join_sql}
        {where_sql}
        GROUP BY e.mid{", pa.last_name, pa.first_name" if sort == "name" else ""}
        ORDER BY {order_sql}
        LIMIT %s OFFSET %s
        """,
        [*params, page_size, offset],
    ).fetchall()
    return [dict(r) for r in rows], total


def dashboard_summary(conn: PgConnection) -> dict:
    """Live KPI aggregates over the current resolved-output state — recomputed
    on every call so reviewer merge/unmerge actions are reflected
    immediately. See `sql_backend.dashboard_summary`."""
    total_records_row = conn.execute(
        "SELECT COUNT(*) AS n FROM entity_member"
    ).fetchone()
    assert total_records_row is not None
    total_records = total_records_row["n"]

    origin_counts = {
        row["origin"]: row["n"]
        for row in conn.execute(
            "SELECT origin, COUNT(*) AS n FROM entity GROUP BY origin"
        ).fetchall()
    }

    matched_members_row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM entity_member em
        JOIN entity e ON e.mid = em.mid WHERE e.is_merged = 1
        """
    ).fetchone()
    assert matched_members_row is not None
    matched_members = matched_members_row["n"]

    duplicate_clusters_row = conn.execute(
        "SELECT COUNT(*) AS n FROM entity WHERE is_merged = 1"
    ).fetchone()
    assert duplicate_clusters_row is not None
    duplicate_clusters = duplicate_clusters_row["n"]

    needs_review_members_row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM entity_member em
        JOIN entity e ON e.mid = em.mid WHERE e.origin = 'review'
        """
    ).fetchone()
    assert needs_review_members_row is not None
    needs_review_members = needs_review_members_row["n"]

    no_match_members_row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM entity_member em
        JOIN entity e ON e.mid = em.mid WHERE e.origin = 'none'
        """
    ).fetchone()
    assert no_match_members_row is not None
    no_match_members = no_match_members_row["n"]

    auto_match_members_row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM entity_member em
        JOIN entity e ON e.mid = em.mid WHERE e.origin IN ('deterministic', 'merge')
        """
    ).fetchone()
    assert auto_match_members_row is not None
    auto_match_members = auto_match_members_row["n"]

    audit_action_counts = {
        row["action"]: row["n"]
        for row in conn.execute(
            "SELECT action, COUNT(*) AS n FROM audit_log GROUP BY action"
        ).fetchall()
    }

    return {
        "total_records": total_records,
        "duplicate_clusters": duplicate_clusters,
        "matched_records": matched_members,
        "unmerged_records": total_records - matched_members,
        "needs_review_records": needs_review_members,
        "no_match_records": no_match_members,
        "auto_match_records": auto_match_members,
        "origin_counts": origin_counts,
        "manual_merge_actions": audit_action_counts.get("merge", 0),
        "manual_unmerge_actions": audit_action_counts.get("unmerge", 0),
    }


def insert_audit_log(
    conn: PgConnection,
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
    prev_mid: str | None = None,
    undo_of: int | None = None,
) -> int:
    """`RETURNING id` + `fetchone()` stands in for sqlite3's `cur.lastrowid`
    — psycopg/Postgres has no equivalent cursor attribute."""
    row = conn.execute(
        """
        INSERT INTO audit_log
            (ts_utc, "user", action, patids, mid, prev_state, next_state, run_id,
             related_patids, prev_mid, undo_of)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (ts_utc, user, action, patids, mid, prev_state, next_state, run_id,
         related_patids, prev_mid, undo_of),
    ).fetchone()
    assert row is not None
    return int(row["id"])


#: Shared with `list_audit_log` and `get_audit_log_row` — `undone` is never
#: stored, it's derived: a row is undone iff some other row's `undo_of`
#: points back at its id (see sql_backend.py's identical pattern).
_UNDONE_EXPR = "EXISTS(SELECT 1 FROM audit_log u WHERE u.undo_of = al.id)"


def list_audit_log(
    conn: PgConnection, *, limit: int = 100, since: str | None = None
) -> list[dict]:
    if since:
        rows = conn.execute(
            f"""
            SELECT al.*, {_UNDONE_EXPR} AS undone FROM audit_log al
            WHERE al.ts_utc > %s ORDER BY al.id DESC LIMIT %s
            """,
            (since, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT al.*, {_UNDONE_EXPR} AS undone FROM audit_log al
            ORDER BY al.id DESC LIMIT %s
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_audit_log_row(conn: PgConnection, audit_id: int) -> dict | None:
    """One audit_log row plus its derived `undone` flag — the reversal target
    of `POST /audit/{audit_id}/undo` (`src/api/routers/audit.py`)."""
    row = conn.execute(
        f"SELECT al.*, {_UNDONE_EXPR} AS undone FROM audit_log al WHERE al.id = %s",
        (audit_id,),
    ).fetchone()
    return dict(row) if row else None


# ── Persisted blocking index (`block_key`) ──────────────────────────────────
def replace_block_keys(conn: PgConnection, rows: list[tuple[str, str, str]]) -> None:
    """Full rebuild of `block_key` — delete-then-bulk-insert, called once per
    full pipeline publish. `rows` are `(block_id, key_value, patid)`; B1/B6
    values must already be hashed via `hash_block_key`."""
    conn.execute("DELETE FROM block_key")
    _executemany(
        conn,
        "INSERT INTO block_key (block_id, key_value, patid) VALUES (%s, %s, %s)",
        rows,
    )


def add_block_keys(conn: PgConnection, rows: list[tuple[str, str, str]]) -> None:
    """Incremental append — `ON CONFLICT DO NOTHING` since the same key can
    legitimately already exist for another patid (stands in for SQLite's
    `INSERT OR IGNORE`)."""
    _executemany(
        conn,
        "INSERT INTO block_key (block_id, key_value, patid) VALUES (%s, %s, %s) "
        "ON CONFLICT (block_id, key_value, patid) DO NOTHING",
        rows,
    )


def lookup_block_candidates(
    conn: PgConnection,
    keys: dict[str, str | None],
    phones: Iterable[str],
    threshold: int,
) -> dict[str, set[str]]:
    """`existing_patid -> {block_id, ...}` for every persisted block key a
    record's own keys/phones match. See `sql_backend.lookup_block_candidates`."""
    candidates: dict[str, set[str]] = {}

    def _add(block_id: str, patid: str) -> None:
        candidates.setdefault(patid, set()).add(block_id)

    for block_id, key_value in keys.items():
        if not key_value:
            continue
        for row in conn.execute(
            "SELECT patid FROM block_key WHERE block_id = %s AND key_value = %s "
            "LIMIT %s",
            (block_id, key_value, threshold),
        ):
            _add(block_id, row["patid"])

    for phone in phones:
        if not phone:
            continue
        for row in conn.execute(
            "SELECT patid FROM block_key WHERE block_id = 'B5' AND key_value = %s "
            "LIMIT %s",
            (phone, threshold),
        ):
            _add("B5", row["patid"])

    return candidates


# ── Persisted cleaned-attribute mirror (`cleaned_attrs`) ────────────────────
_CLEANED_ATTRS_COLUMNS = (
    "patid", "first_nm", "last_nm", "birth_dt", "ssn", "ssn_last4", "email",
    "zip_base", "address1", "sex", "phones_json", "run_id",
)
_CLEANED_ATTRS_UPSERT_SQL = f"""
    INSERT INTO cleaned_attrs ({", ".join(_CLEANED_ATTRS_COLUMNS)})
    VALUES ({", ".join("%s" for _ in _CLEANED_ATTRS_COLUMNS)})
    ON CONFLICT(patid) DO UPDATE SET
        first_nm=excluded.first_nm, last_nm=excluded.last_nm,
        birth_dt=excluded.birth_dt, ssn=excluded.ssn, ssn_last4=excluded.ssn_last4,
        email=excluded.email, zip_base=excluded.zip_base, address1=excluded.address1,
        sex=excluded.sex, phones_json=excluded.phones_json, run_id=excluded.run_id
"""


def replace_cleaned_attrs(conn: PgConnection, rows: list[tuple]) -> None:
    """Full rebuild of `cleaned_attrs` — delete-then-bulk-insert; see
    `replace_block_keys`. `rows` are ordered per `_CLEANED_ATTRS_COLUMNS`."""
    conn.execute("DELETE FROM cleaned_attrs")
    _executemany(
        conn,
        f"INSERT INTO cleaned_attrs ({', '.join(_CLEANED_ATTRS_COLUMNS)}) "
        f"VALUES ({', '.join('%s' for _ in _CLEANED_ATTRS_COLUMNS)})",
        rows,
    )


def upsert_cleaned_attrs(conn: PgConnection, row: tuple) -> None:
    """Incremental single-row upsert — see `add_block_keys`. `row` is ordered
    per `_CLEANED_ATTRS_COLUMNS`."""
    conn.execute(_CLEANED_ATTRS_UPSERT_SQL, row)


def get_cleaned_attrs(conn: PgConnection, patids: list[str]) -> list[dict]:
    """`cleaned_attrs` rows for the given PATIDs. Column names match
    `_CLEANED_ATTRS_COLUMNS`, not the pipeline's `*_clean` contract names —
    callers building a `pd.DataFrame` must rename (see
    `src/api/ingest/incremental.py`)."""
    if not patids:
        return []
    placeholders = ", ".join("%s" for _ in patids)
    rows = conn.execute(
        f"SELECT * FROM cleaned_attrs WHERE patid IN ({placeholders})", patids
    ).fetchall()
    return [dict(r) for r in rows]


# ── Entity bridging (incremental auto-merge across pre-existing entities) ───
def reassign_entity_members(
    conn: PgConnection, from_mids: list[str], to_mid: str, updated_utc: str
) -> None:
    """Absorb every member of `from_mids` into `to_mid` and delete the
    now-empty `from_mids` entity rows. See `sql_backend.reassign_entity_members`."""
    from_mids = [m for m in from_mids if m != to_mid]
    if not from_mids:
        return
    placeholders = ", ".join("%s" for _ in from_mids)
    conn.execute(
        f"UPDATE entity_member SET mid = %s, updated_utc = %s "
        f"WHERE mid IN ({placeholders})",
        (to_mid, updated_utc, *from_mids),
    )
    conn.execute(f"DELETE FROM entity WHERE mid IN ({placeholders})", from_mids)


__all__ = [
    "SCHEMA_SQL",
    "get_connection",
    "init_db",
    "locked_patids",
    "get_entity_mid_for_patid",
    "max_mid_sequence",
    "next_mid",
    "upsert_entity",
    "upsert_entities_bulk",
    "upsert_entity_member",
    "upsert_entity_members_bulk",
    "upsert_record_attrs_bulk",
    "upsert_record_raw_bulk",
    "get_record_raw",
    "upsert_suggestion",
    "upsert_suggestions_bulk",
    "replace_review_candidates_for_run",
    "insert_review_candidates",
    "review_candidates_for_patid",
    "all_entity_member_mids",
    "get_entity",
    "list_entities",
    "insert_audit_log",
    "list_audit_log",
    "get_audit_log_row",
    "dashboard_summary",
    "hash_block_key",
    "HASHED_BLOCKS",
    "replace_block_keys",
    "add_block_keys",
    "lookup_block_candidates",
    "replace_cleaned_attrs",
    "upsert_cleaned_attrs",
    "get_cleaned_attrs",
    "reassign_entity_members",
    "list_review_candidates",
]
