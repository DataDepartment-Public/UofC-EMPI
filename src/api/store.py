"""SQLite layer for the resolved-output database (`empi.db`).

Implements the schema from `docs/API-Design.md` §2 — `entity`, `entity_member`,
`audit_log` — plus two additions needed to satisfy the routes the doc promises
but doesn't fully specify the storage for:

* `record_attrs` — a few display fields (name, DOB, SSN last-4, …) denormalized
  from the cleaned dataset at publish time, so `GET /records` and
  `GET /clusters/{mid}` read *only* from SQLite (no per-request Parquet I/O).
* `entity_suggestion` — where a reviewer-locked PATID's would-be new grouping is
  recorded per §2 "Reconciliation" ("recorded as a suggestion but not
  auto-applied"). Not yet exposed by a route; `src/api/publish.py` writes it.

Reconciliation / locking (§2, §6 open decision 1 — resolved as "sticky
unmerge"): a PATID is **reviewer-locked** the moment it appears in the
`patids` column of any `audit_log` row. `publish.py` never rewrites a locked
PATID's `entity_member.mid` — only an explicit new `/audit/*` action can move
it again.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entity (
    mid          TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    origin       TEXT NOT NULL,
    is_merged    INTEGER NOT NULL,
    confidence   REAL,
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

CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT NOT NULL,
    user         TEXT NOT NULL,
    action       TEXT NOT NULL,
    patids       TEXT NOT NULL,
    mid          TEXT NOT NULL,
    prev_state   TEXT NOT NULL,
    next_state   TEXT NOT NULL,
    run_id       TEXT
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
    run_id       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_suggestion (
    patid          TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    suggested_mid  TEXT NOT NULL,
    created_utc    TEXT NOT NULL
);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open one SQLite connection. Callers own its lifecycle (one per request
    or per script run) — this module never pools connections."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create every table if it doesn't already exist. Idempotent."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()


# ── Locking (reconciliation) ────────────────────────────────────────────────
def locked_patids(conn: sqlite3.Connection) -> set[str]:
    """PATIDs a reviewer has ever directly acted on (any `audit_log.patids`).

    `publish.py` must never move these — see the module docstring.
    """
    rows = conn.execute("SELECT patids FROM audit_log").fetchall()
    locked: set[str] = set()
    for row in rows:
        locked.update(p for p in row["patids"].split(",") if p)
    return locked


# ── Entity / membership ─────────────────────────────────────────────────────
def get_entity_mid_for_patid(conn: sqlite3.Connection, patid: str) -> str | None:
    row = conn.execute(
        "SELECT mid FROM entity_member WHERE patid = ?", (patid,)
    ).fetchone()
    return row["mid"] if row else None


def max_mid_sequence(conn: sqlite3.Connection) -> int:
    """Highest numeric suffix among existing `M-<n>` mids, or 0 if none."""
    rows = conn.execute("SELECT mid FROM entity").fetchall()
    best = 0
    for row in rows:
        mid = row["mid"]
        if mid.startswith("M-") and mid[2:].isdigit():
            best = max(best, int(mid[2:]))
    return best


def next_mid(conn: sqlite3.Connection) -> str:
    return f"M-{max_mid_sequence(conn) + 1:06d}"


_ENTITY_UPSERT_SQL = """
    INSERT INTO entity (mid, run_id, origin, is_merged, confidence, updated_utc)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(mid) DO UPDATE SET
        run_id=excluded.run_id, origin=excluded.origin,
        is_merged=excluded.is_merged, confidence=excluded.confidence,
        updated_utc=excluded.updated_utc
"""

_MEMBER_UPSERT_SQL = """
    INSERT INTO entity_member (patid, mid, is_primary, added_by, updated_utc)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(patid) DO UPDATE SET
        mid=excluded.mid, is_primary=excluded.is_primary,
        added_by=excluded.added_by, updated_utc=excluded.updated_utc
"""

_ATTRS_UPSERT_SQL = """
    INSERT INTO record_attrs
        (patid, first_name, last_name, birth_date, ssn_last4, email,
         zip_code, address1, sex, run_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(patid) DO UPDATE SET
        first_name=excluded.first_name, last_name=excluded.last_name,
        birth_date=excluded.birth_date, ssn_last4=excluded.ssn_last4,
        email=excluded.email, zip_code=excluded.zip_code,
        address1=excluded.address1, sex=excluded.sex, run_id=excluded.run_id
"""

_SUGGESTION_UPSERT_SQL = """
    INSERT INTO entity_suggestion (patid, run_id, suggested_mid, created_utc)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(patid) DO UPDATE SET
        run_id=excluded.run_id, suggested_mid=excluded.suggested_mid,
        created_utc=excluded.created_utc
"""


def upsert_entity(
    conn: sqlite3.Connection,
    mid: str,
    run_id: str,
    origin: str,
    is_merged: bool,
    confidence: float | None,
    updated_utc: str,
) -> None:
    conn.execute(
        _ENTITY_UPSERT_SQL, (mid, run_id, origin, int(is_merged), confidence, updated_utc)
    )


def upsert_entities_bulk(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Bulk variant of `upsert_entity` — one `executemany` instead of N `execute`
    calls. `rows` are `(mid, run_id, origin, is_merged, confidence, updated_utc)`
    tuples (`is_merged` already coerced to 0/1). Used by `publish.py`, where a
    real run can touch tens of thousands of entities per call — see its
    module docstring for why per-row `execute` doesn't scale."""
    if rows:
        conn.executemany(_ENTITY_UPSERT_SQL, rows)


def upsert_entity_member(
    conn: sqlite3.Connection,
    patid: str,
    mid: str,
    is_primary: bool,
    added_by: str,
    updated_utc: str,
) -> None:
    conn.execute(_MEMBER_UPSERT_SQL, (patid, mid, int(is_primary), added_by, updated_utc))


def upsert_entity_members_bulk(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Bulk variant of `upsert_entity_member`; see `upsert_entities_bulk`."""
    if rows:
        conn.executemany(_MEMBER_UPSERT_SQL, rows)


def upsert_record_attrs(
    conn: sqlite3.Connection,
    patid: str,
    *,
    first_name: str | None,
    last_name: str | None,
    birth_date: str | None,
    ssn_last4: str | None,
    email: str | None,
    zip_code: str | None,
    address1: str | None,
    sex: str | None,
    run_id: str,
) -> None:
    conn.execute(
        _ATTRS_UPSERT_SQL,
        (
            patid, first_name, last_name, birth_date, ssn_last4, email,
            zip_code, address1, sex, run_id,
        ),
    )


def upsert_record_attrs_bulk(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Bulk variant of `upsert_record_attrs`; see `upsert_entities_bulk`."""
    if rows:
        conn.executemany(_ATTRS_UPSERT_SQL, rows)


def upsert_suggestion(
    conn: sqlite3.Connection,
    patid: str,
    run_id: str,
    suggested_mid: str,
    created_utc: str,
) -> None:
    conn.execute(_SUGGESTION_UPSERT_SQL, (patid, run_id, suggested_mid, created_utc))


def upsert_suggestions_bulk(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Bulk variant of `upsert_suggestion`; see `upsert_entities_bulk`."""
    if rows:
        conn.executemany(_SUGGESTION_UPSERT_SQL, rows)


def all_entity_member_mids(conn: sqlite3.Connection) -> dict[str, str]:
    """Every `patid -> mid` mapping in one query — for callers (e.g.
    `publish.py`) that would otherwise run one lookup per PATID in a loop."""
    return {
        row["patid"]: row["mid"]
        for row in conn.execute("SELECT patid, mid FROM entity_member")
    }


def get_entity(conn: sqlite3.Connection, mid: str) -> dict | None:
    """One entity + its members (joined to `record_attrs`), or None."""
    entity_row = conn.execute("SELECT * FROM entity WHERE mid = ?", (mid,)).fetchone()
    if entity_row is None:
        return None
    members = conn.execute(
        """
        SELECT em.patid, em.is_primary, em.added_by, em.updated_utc,
               ra.first_name, ra.last_name, ra.birth_date, ra.ssn_last4,
               ra.email, ra.zip_code, ra.address1, ra.sex
        FROM entity_member em
        LEFT JOIN record_attrs ra ON ra.patid = em.patid
        WHERE em.mid = ?
        ORDER BY em.patid
        """,
        (mid,),
    ).fetchall()
    return {"entity": dict(entity_row), "members": [dict(m) for m in members]}


def list_entities(
    conn: sqlite3.Connection,
    *,
    search: str | None = None,
    status: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[dict], int]:
    """Paginated master rows (one per entity) + their member count.

    `status` filters on `is_merged` ('merged' / 'unmerged'); `search` matches
    a case-insensitive substring against PATID, first name, or last name.
    Returns (rows, total_count).
    """
    where = []
    params: list = []
    if status in ("merged", "unmerged"):
        where.append("e.is_merged = ?")
        params.append(1 if status == "merged" else 0)
    if search:
        where.append(
            "EXISTS (SELECT 1 FROM entity_member em2 "
            "LEFT JOIN record_attrs ra2 ON ra2.patid = em2.patid "
            "WHERE em2.mid = e.mid AND ("
            "em2.patid LIKE ? OR ra2.first_name LIKE ? OR ra2.last_name LIKE ?))"
        )
        like = f"%{search}%"
        params.extend([like, like, like])
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM entity e {where_sql}", params
    ).fetchone()["n"]

    offset = max(page - 1, 0) * page_size
    rows = conn.execute(
        f"""
        SELECT e.*, COUNT(em.patid) AS member_count
        FROM entity e
        LEFT JOIN entity_member em ON em.mid = e.mid
        {where_sql}
        GROUP BY e.mid
        ORDER BY e.updated_utc DESC, e.mid
        LIMIT ? OFFSET ?
        """,
        [*params, page_size, offset],
    ).fetchall()
    return [dict(r) for r in rows], total


def insert_audit_log(
    conn: sqlite3.Connection,
    *,
    ts_utc: str,
    user: str,
    action: str,
    patids: str,
    mid: str,
    prev_state: str,
    next_state: str,
    run_id: str | None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO audit_log
            (ts_utc, user, action, patids, mid, prev_state, next_state, run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts_utc, user, action, patids, mid, prev_state, next_state, run_id),
    )
    return int(cur.lastrowid)


def list_audit_log(
    conn: sqlite3.Connection, *, limit: int = 100, since: str | None = None
) -> list[dict]:
    if since:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE ts_utc > ? ORDER BY id DESC LIMIT ?",
            (since, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def member_count(conn: sqlite3.Connection, mid: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM entity_member WHERE mid = ?", (mid,)
    ).fetchone()
    return int(row["n"])


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
    "upsert_record_attrs",
    "upsert_record_attrs_bulk",
    "upsert_suggestion",
    "upsert_suggestions_bulk",
    "all_entity_member_mids",
    "get_entity",
    "list_entities",
    "insert_audit_log",
    "list_audit_log",
    "member_count",
]
