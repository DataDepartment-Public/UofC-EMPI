"""SQLite layer for the resolved-output database (`empi.db`).

Implements the schema from `docs/API-Design.md` §2 — `entity`, `entity_member`,
`audit_log` — plus additions needed to satisfy the routes the doc promises but
doesn't fully specify the storage for:

* `record_attrs` — a few display fields (name, DOB, SSN last-4, …) denormalized
  from the cleaned dataset at publish time, so `GET /records` and
  `GET /clusters/{mid}` read *only* from SQLite (no per-request Parquet I/O).
* `record_raw` — the un-scrubbed `*_raw` source fields, one JSON blob per
  PATID, for the "View Raw Data" drawer (docs/Dashboard-Guide.md FR-24 /
  the pasted FR doc's FR-24).
* `review_candidate` — a review-tier deterministic-rule pair or an
  unconfirmed-but-uncertain pair the pipeline routed to the human-review
  queue (`non_matches` + `review_evidence` artifacts). Not a resolved
  membership like `entity_member` — it's the *reason* a singleton entity's
  `origin` is `'review'` instead of `'none'`, and what the Dataset page's
  expand-row shows for a needs-review record (FR-19/20/21 in the FR doc).
* `entity_suggestion` — where a reviewer-locked PATID's would-be new grouping is
  recorded per §2 "Reconciliation" ("recorded as a suggestion but not
  auto-applied"). Not yet exposed by a route; `src/api/ingest/publish.py` writes it.

`entity.match_rule` / `entity.evidence` carry the deterministic rule name and
its pipe-delimited `rules_fired` for the pair that founded a merged entity —
the "key matching features used" field the Dataset table needs (FR-21/22).
`None` for singletons (nothing fired).

Reconciliation / locking (§2, §6 open decision 1 — resolved as "sticky
unmerge"): a PATID is **reviewer-locked** the moment it appears in the
`patids` column of any `audit_log` row. `publish.py` never rewrites a locked
PATID's `entity_member.mid` — only an explicit new `/audit/*` action can move
it again.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from src.api.backends.search_terms import search_tokens

#: Re-exported so existing `store.hash_block_key` / `store.HASHED_BLOCKS` call
#: sites (`publish.py`, tests) keep working — these are pure/data, not tied
#: to SQLite, which is why `src/api/ingest/incremental.py` imports them from
#: `blocking.py` (the block-key semantics module) directly instead of this
#: (SQLite-specific) module.
from src.preprocessing.blocking import HASHED_BLOCKS, hash_block_key  # noqa: F401

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

-- related_patids is a comma-joined snapshot, captured at write time, of the
-- *other* patids relevant to reconstructing this event's implied pairs --
-- NULL for 'dismiss' (patids already holds both sides). For 'merge' it's the
-- entity's members immediately before this merge; for 'unmerge' it's who
-- stayed behind in the original entity. Exists so a later export (e.g. for
-- retraining labels) never has to reconstruct past membership from
-- entity_member's current-state-only table -- see src/api/routers/audit.py.
CREATE TABLE IF NOT EXISTS audit_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc         TEXT NOT NULL,
    user           TEXT NOT NULL,
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
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
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

-- Persisted blocking posting list — `block_id/key_value -> [patid, ...]`, the
-- on-disk equivalent of `blocking.BlockingIndex`. Rebuilt wholesale by every
-- full pipeline publish (`publish.py`) from the freshly cleaned population,
-- and incrementally appended to by `src/api/ingest/incremental.py` so a single new
-- record's own keys become discoverable by the next incremental call without
-- waiting for a full run. This is what lets incremental scoring look up
-- candidates with an indexed SQL query instead of rebuilding an in-memory
-- BlockingIndex over the whole population per request. B1 (SSN) and B6
-- (email) key values are stored as a SHA-256 hash, not plaintext — the only
-- operation ever performed on this column is equality, and both are direct
-- identifiers where hashing costs nothing functionally.
CREATE TABLE IF NOT EXISTS block_key (
    block_id   TEXT NOT NULL,
    key_value  TEXT NOT NULL,
    patid      TEXT NOT NULL,
    PRIMARY KEY (block_id, key_value, patid)
);
CREATE INDEX IF NOT EXISTS idx_block_key_lookup ON block_key(block_id, key_value);
CREATE INDEX IF NOT EXISTS idx_block_key_patid ON block_key(patid);

-- SQL-queryable mirror of the `CleanedRecords` contract (src/contracts.py),
-- one row per valid PATID. Parquet stays the immutable per-run record; this
-- table exists so incremental scoring can pull the handful of candidate
-- rows deterministic-rule evaluation needs via a point lookup instead of
-- reading a ~163k-row Parquet file per request. Rebuilt wholesale on every
-- publish alongside `block_key`, from the same `cleaned` frame.
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


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open one SQLite connection. Callers own its lifecycle (one per request
    or per script run) — this module never pools connections.

    `check_same_thread=False`: FastAPI runs a sync generator dependency's
    pre-yield code, the route handler, and its post-yield/finally code as
    separate `anyio.to_thread.run_sync` dispatches — under concurrent load
    these don't always land on the same worker thread, even though the
    connection is still only ever used sequentially within one request's
    lifetime (never concurrently across requests). Without this, `close()`
    intermittently raises `sqlite3.ProgrammingError: SQLite objects created
    in a thread can only be used in that same thread.` when it lands on a
    different thread than the one that opened the connection.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


#: Columns added to a table after its original release — `CREATE TABLE IF NOT
#: EXISTS` never alters a table that already exists, so a pre-existing local
#: `empi.db` needs these added in place rather than requiring a wipe/rebuild
#: (a full pipeline run + publish over ~163k records is expensive to redo).
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
    # Display-only fields added for the feature-comparison table. An existing
    # `empi.db` gets them as all-NULL until the run is re-published
    # (`python -m src.api.ingest.publish_local --run-id <run_id>`), which
    # upserts every `record_attrs` row.
    "record_attrs": {
        "middle_name": "TEXT",
        "suffix": "TEXT",
        "city": "TEXT",
        "phones": "TEXT",
    },
}


def _ensure_columns(conn: sqlite3.Connection) -> None:
    for table, columns in _COLUMN_MIGRATIONS.items():
        existing = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
        }
        for name, coltype in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}")


def init_db(conn: sqlite3.Connection) -> None:
    """Create every table if it doesn't already exist. Idempotent."""
    conn.executescript(SCHEMA_SQL)
    _ensure_columns(conn)
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
    INSERT INTO entity
        (mid, run_id, origin, is_merged, confidence, match_rule, evidence, updated_utc)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(mid) DO UPDATE SET
        run_id=excluded.run_id, origin=excluded.origin,
        is_merged=excluded.is_merged, confidence=excluded.confidence,
        match_rule=excluded.match_rule, evidence=excluded.evidence,
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
         zip_code, address1, sex, phone, middle_name, suffix, city, phones,
         run_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    match_rule: str | None = None,
    evidence: str | None = None,
) -> None:
    conn.execute(
        _ENTITY_UPSERT_SQL,
        (mid, run_id, origin, int(is_merged), confidence, match_rule, evidence, updated_utc),
    )


def upsert_entities_bulk(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Bulk variant of `upsert_entity` — one `executemany` instead of N `execute`
    calls. `rows` are `(mid, run_id, origin, is_merged, confidence, match_rule,
    evidence, updated_utc)` tuples (`is_merged` already coerced to 0/1). Used
    by `publish.py`, where a real run can touch tens of thousands of entities
    per call — see its module docstring for why per-row `execute` doesn't
    scale."""
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
    phone: str | None = None,
    middle_name: str | None = None,
    suffix: str | None = None,
    city: str | None = None,
    phones: str | None = None,
) -> None:
    conn.execute(
        _ATTRS_UPSERT_SQL,
        (
            patid, first_name, last_name, birth_date, ssn_last4, email,
            zip_code, address1, sex, phone, middle_name, suffix, city, phones,
            run_id,
        ),
    )


def upsert_record_attrs_bulk(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Bulk variant of `upsert_record_attrs`; see `upsert_entities_bulk`."""
    if rows:
        conn.executemany(_ATTRS_UPSERT_SQL, rows)


_RAW_UPSERT_SQL = """
    INSERT INTO record_raw (patid, raw_json, run_id)
    VALUES (?, ?, ?)
    ON CONFLICT(patid) DO UPDATE SET
        raw_json=excluded.raw_json, run_id=excluded.run_id
"""


def upsert_record_raw_bulk(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """`rows` are `(patid, raw_json, run_id)` — see `upsert_entities_bulk`."""
    if rows:
        conn.executemany(_RAW_UPSERT_SQL, rows)


def get_record_raw(conn: sqlite3.Connection, patid: str) -> str | None:
    row = conn.execute(
        "SELECT raw_json FROM record_raw WHERE patid = ?", (patid,)
    ).fetchone()
    return row["raw_json"] if row else None


_REVIEW_CANDIDATE_INSERT_SQL = """
    INSERT INTO review_candidate
        (patid_a, patid_b, match_rule, confidence, evidence, source_blocks,
         run_id, created_utc, ml_match_probability, ml_classification_tier)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(patid_a, patid_b, run_id) DO UPDATE SET
        match_rule=excluded.match_rule, confidence=excluded.confidence,
        evidence=excluded.evidence, source_blocks=excluded.source_blocks,
        created_utc=excluded.created_utc,
        ml_match_probability=excluded.ml_match_probability,
        ml_classification_tier=excluded.ml_classification_tier
"""


def replace_review_candidates_for_run(
    conn: sqlite3.Connection, run_id: str, rows: list[tuple]
) -> None:
    """Replace this run's review-candidate rows wholesale (delete + bulk
    insert) — unlike entities/members, a review pair that's no longer in the
    latest run's `non_matches` (resolved, or reclassified) should disappear
    rather than linger as a stale suggestion. `rows` are `(patid_a, patid_b,
    match_rule, confidence, evidence, source_blocks, run_id, created_utc,
    ml_match_probability, ml_classification_tier)`."""
    conn.execute("DELETE FROM review_candidate WHERE run_id = ?", (run_id,))
    if rows:
        conn.executemany(_REVIEW_CANDIDATE_INSERT_SQL, rows)


_REVIEW_CANDIDATE_INSERT_FULL_SQL = """
    INSERT INTO review_candidate
        (patid_a, patid_b, match_rule, confidence, evidence, source_blocks,
         run_id, created_utc, fs_match_probability, fs_classification_tier)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(patid_a, patid_b, run_id) DO UPDATE SET
        match_rule=excluded.match_rule, confidence=excluded.confidence,
        evidence=excluded.evidence, source_blocks=excluded.source_blocks,
        created_utc=excluded.created_utc,
        fs_match_probability=excluded.fs_match_probability,
        fs_classification_tier=excluded.fs_classification_tier
"""


def insert_review_candidates(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Append-only review-candidate insert carrying the FS score columns —
    used by incremental scoring (`src/api/ingest/incremental.py`), which always
    mints a fresh `run_id` per call, so there is nothing to delete first
    (unlike the batch path's `replace_review_candidates_for_run`). `rows` are
    `(patid_a, patid_b, match_rule, confidence, evidence, source_blocks,
    run_id, created_utc, fs_match_probability, fs_classification_tier)`."""
    if rows:
        conn.executemany(_REVIEW_CANDIDATE_INSERT_FULL_SQL, rows)


#: A pair can exist in `review_candidate` under more than one `run_id` — the
#: UNIQUE constraint is `(patid_a, patid_b, run_id)`, so a candidate that
#: stays unresolved across successive full-batch publishes gets a fresh row
#: each time rather than being replaced (only `replace_review_candidates_for_run`
#: clears rows for its OWN run_id first). Both read queries below dedupe to
#: the single most-recently-created row per `(patid_a, patid_b)` so a pair
#: is never listed twice — the invariant `list_review_candidates`'s docstring
#: already promises ("each should appear once here").
_DEDUPED_REVIEW_CANDIDATE = """
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY patid_a, patid_b ORDER BY created_utc DESC, id DESC
        ) AS rn
        FROM review_candidate
    ) WHERE rn = 1
"""

#: The `record_attrs` display columns, aliased `a_*`/`b_*` for a pair query.
#: Shared by the two pair readers below so a column added to `record_attrs`
#: can't reach one route's payload and silently miss the other's.
#: `run_id` is deliberately excluded (it describes the row's provenance, not
#: the patient) — see `parquet_backend._DISPLAY_ATTR_COLUMNS`, its counterpart.
_PAIR_ATTR_SELECT = ",\n".join(
    f"               ra_{side}.{col} AS {side}_{col}"
    for side in ("a", "b")
    for col in (
        "first_name", "last_name", "birth_date", "ssn_last4", "email",
        "zip_code", "address1", "sex", "phone", "middle_name", "suffix",
        "city", "phones",
    )
)


def review_candidates_for_patid(conn: sqlite3.Connection, patid: str) -> list[dict]:
    """Every review-candidate pair touching `patid`, joined to `record_attrs`
    on *both* sides — the UI needs the other PATID's name/SSN/DOB to render
    the expanded-row candidate list, not just its ID."""
    rows = conn.execute(
        f"""
        SELECT rc.*,
{_PAIR_ATTR_SELECT}
        FROM ({_DEDUPED_REVIEW_CANDIDATE}) rc
        LEFT JOIN record_attrs ra_a ON ra_a.patid = rc.patid_a
        LEFT JOIN record_attrs ra_b ON ra_b.patid = rc.patid_b
        WHERE rc.patid_a = ? OR rc.patid_b = ?
        ORDER BY COALESCE(rc.confidence, rc.ml_match_probability) DESC NULLS LAST, rc.id
        """,
        (patid, patid),
    ).fetchall()
    return [dict(r) for r in rows]


def list_review_candidates(
    conn: sqlite3.Connection,
    *,
    confidence_min: float | None = None,
    confidence_max: float | None = None,
    reviewed: bool | None = None,
    search: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[dict], int]:
    """Paginated, candidate-grain review queue — one row per pending pair,
    not per cluster (docs/Dashboard-Guide.md's Review Queue: the same cluster
    can surface multiple candidates, and each should appear once here).

    A pair counts as "reviewed" if either:
      * both sides currently share a `mid` (the merge already happened —
        mirrors the client-side `pendingCandidates` filter in
        `DatasetRow.tsx`), or
      * a `dismiss` audit_log entry exists for the exact `(patid_a, patid_b)`
        pair (the reviewer's "Not a match" action — `POST /audit/dismiss`).

    `reviewed=False` (the default queue) excludes both; `reviewed=True`
    surfaces only them ("Already reviewed"). Default order is
    `COALESCE(confidence, ml_match_probability)` DESC, with pairs that have
    neither (blocking-only, no rule fired and not ML-scored) last — same
    convention as `review_candidates_for_patid`.

    `confidence_min`/`confidence_max` filter that same coalesced value —
    most pairs in the queue only have an ML score (no rule fired), so
    filtering on `rc.confidence` alone would hide the entire ML-scored
    majority (`NULL >= x` is never true in SQL).
    """
    where = []
    params: list = []
    if confidence_min is not None:
        where.append("COALESCE(rc.confidence, rc.ml_match_probability) >= ?")
        params.append(confidence_min)
    if confidence_max is not None:
        where.append("COALESCE(rc.confidence, rc.ml_match_probability) <= ?")
        params.append(confidence_max)
    tokens = search_tokens(search)
    if tokens:
        # All tokens must land on one side of the pair, so a query can't be
        # satisfied by taking the first name from side A and the last from B.
        side_a = " AND ".join(
            "(ra_a.first_name LIKE ? OR ra_a.last_name LIKE ? OR rc.patid_a LIKE ?)"
            for _ in tokens
        )
        side_b = " AND ".join(
            "(ra_b.first_name LIKE ? OR ra_b.last_name LIKE ? OR rc.patid_b LIKE ?)"
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

    base_from = f"""
        FROM ({_DEDUPED_REVIEW_CANDIDATE}) rc
        JOIN entity_member ema ON ema.patid = rc.patid_a
        JOIN entity_member emb ON emb.patid = rc.patid_b
        LEFT JOIN record_attrs ra_a ON ra_a.patid = rc.patid_a
        LEFT JOIN record_attrs ra_b ON ra_b.patid = rc.patid_b
    """

    total = conn.execute(
        f"SELECT COUNT(*) AS n {base_from} {where_sql}", params
    ).fetchone()["n"]

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
        LIMIT ? OFFSET ?
        """,
        [*params, page_size, offset],
    ).fetchall()
    return [dict(r) for r in rows], total


def patids_with_review_candidates(conn: sqlite3.Connection) -> set[str]:
    """Every PATID appearing on either side of any `review_candidate` row —
    used to upgrade a singleton entity's `origin` from `'none'` to `'review'`."""
    rows = conn.execute("SELECT patid_a, patid_b FROM review_candidate").fetchall()
    out: set[str] = set()
    for row in rows:
        out.add(row["patid_a"])
        out.add(row["patid_b"])
    return out


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
               ra.email, ra.zip_code, ra.address1, ra.sex, ra.phone,
               ra.middle_name, ra.suffix, ra.city, ra.phones
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
) -> tuple[list[dict], int]:
    """Paginated master rows (one per entity) + their member count.

    Filters map onto the FR doc's search/filter list (Master Patient ID,
    patient name, masked SSN, birthdate, match status, merge status,
    last-updated date):
      * `search` — case-insensitive substring against PATID/mid/first/last name.
      * `origin` — match status ('deterministic' | 'review' | 'merge' | 'none'),
        or a comma-separated list of any of those (e.g. the Patient Registry
        view's "final clusters only" = "deterministic,merge,none", excluding
        'review' — still-pending candidates belong in the Review Queue, not
        the resolved registry).
      * `is_merged` — merge status, independent of origin (a reviewer-merge is
        `origin='merge'` AND `is_merged=1`; this lets a caller ask for "any
        merged entity" without caring how it got there).
      * `birth_date` — exact `YYYY-MM-DD` match against any member.
      * `ssn_last4` — exact 4-digit match against any member.
      * `updated_after` / `updated_before` — ISO-8601 bounds on `updated_utc`.
      * `confidence_min` / `confidence_max` — bounds on `e.confidence`. Entities
        with a NULL confidence (no deterministic rule fired) are excluded by
        either bound, same as SQL's normal NULL comparison semantics — a caller
        wanting those back should leave both bounds unset.
      * `sort` — `'confidence'` (highest first, NULLs last), `'name'` (the
        entity's primary member's last/first name, A-Z), or anything else
        including unset (`'updated'`, the default) for most-recently-updated
        first.

    Returns (rows, total_count).
    """
    where = []
    params: list = []
    if origin is not None:
        origins = origin.split(",")
        where.append(f"e.origin IN ({','.join('?' for _ in origins)})")
        params.extend(origins)
    if is_merged is not None:
        where.append("e.is_merged = ?")
        params.append(int(is_merged))
    if updated_after:
        where.append("e.updated_utc >= ?")
        params.append(updated_after)
    if updated_before:
        where.append("e.updated_utc <= ?")
        params.append(updated_before)
    if confidence_min is not None:
        where.append("e.confidence >= ?")
        params.append(confidence_min)
    if confidence_max is not None:
        where.append("e.confidence <= ?")
        params.append(confidence_max)
    tokens = search_tokens(search)
    if tokens:
        # Every token must match the *same* member, each token free to match a
        # different column of it — so "PATRICIA PATEL" finds the member whose
        # first/last names are PATRICIA/PATEL. See `search_terms`.
        member_sql = " AND ".join(
            "(em2.patid LIKE ? OR ra2.first_name LIKE ? OR ra2.last_name LIKE ?)"
            for _ in tokens
        )
        where.append(
            "(e.mid LIKE ? OR EXISTS (SELECT 1 FROM entity_member em2 "
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
            "WHERE em3.mid = e.mid AND ra3.birth_date = ?)"
        )
        params.append(birth_date)
    if ssn_last4:
        where.append(
            "EXISTS (SELECT 1 FROM entity_member em4 "
            "JOIN record_attrs ra4 ON ra4.patid = em4.patid "
            "WHERE em4.mid = e.mid AND ra4.ssn_last4 = ?)"
        )
        params.append(ssn_last4)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM entity e {where_sql}", params
    ).fetchone()["n"]

    if sort == "confidence":
        sort_join_sql = ""
        order_sql = "e.confidence IS NULL, e.confidence DESC, e.mid"
    elif sort == "name":
        sort_join_sql = (
            "LEFT JOIN entity_member pm ON pm.mid = e.mid AND pm.is_primary = 1 "
            "LEFT JOIN record_attrs pa ON pa.patid = pm.patid"
        )
        order_sql = (
            "pa.last_name IS NULL, pa.last_name COLLATE NOCASE, "
            "pa.first_name COLLATE NOCASE, e.mid"
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
        GROUP BY e.mid
        ORDER BY {order_sql}
        LIMIT ? OFFSET ?
        """,
        [*params, page_size, offset],
    ).fetchall()
    return [dict(r) for r in rows], total


def dashboard_summary(conn: sqlite3.Connection) -> dict:
    """Live KPI aggregates over the current resolved-output state — recomputed
    from `empi.db` on every call so reviewer merge/unmerge actions are
    reflected immediately (FR-40 real-time refresh). Pairs with the latest
    `RunManifest.counts` (loaded by the caller) for pipeline-level numbers
    `empi.db` doesn't have, like invalid/raw record counts.
    """
    total_records = conn.execute(
        "SELECT COUNT(*) AS n FROM entity_member"
    ).fetchone()["n"]

    origin_counts = {
        row["origin"]: row["n"]
        for row in conn.execute(
            "SELECT origin, COUNT(*) AS n FROM entity GROUP BY origin"
        ).fetchall()
    }

    matched_members = conn.execute(
        """
        SELECT COUNT(*) AS n FROM entity_member em
        JOIN entity e ON e.mid = em.mid WHERE e.is_merged = 1
        """
    ).fetchone()["n"]

    duplicate_clusters = conn.execute(
        "SELECT COUNT(*) AS n FROM entity WHERE is_merged = 1"
    ).fetchone()["n"]

    needs_review_members = conn.execute(
        """
        SELECT COUNT(*) AS n FROM entity_member em
        JOIN entity e ON e.mid = em.mid WHERE e.origin = 'review'
        """
    ).fetchone()["n"]

    no_match_members = conn.execute(
        """
        SELECT COUNT(*) AS n FROM entity_member em
        JOIN entity e ON e.mid = em.mid WHERE e.origin = 'none'
        """
    ).fetchone()["n"]

    auto_match_members = conn.execute(
        """
        SELECT COUNT(*) AS n FROM entity_member em
        JOIN entity e ON e.mid = em.mid WHERE e.origin IN ('deterministic', 'merge')
        """
    ).fetchone()["n"]

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
    related_patids: str | None = None,
    prev_mid: str | None = None,
    undo_of: int | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO audit_log
            (ts_utc, user, action, patids, mid, prev_state, next_state, run_id,
             related_patids, prev_mid, undo_of)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts_utc, user, action, patids, mid, prev_state, next_state, run_id,
         related_patids, prev_mid, undo_of),
    )
    return int(cur.lastrowid)


#: Shared with `list_audit_log` and `get_audit_log_row` — `undone` is never
#: stored, it's derived: a row is undone iff some other row's `undo_of`
#: points back at its id (see docstring on `get_audit_log_row`).
_UNDONE_EXPR = "EXISTS(SELECT 1 FROM audit_log u WHERE u.undo_of = al.id)"


def list_audit_log(
    conn: sqlite3.Connection, *, limit: int = 100, since: str | None = None
) -> list[dict]:
    if since:
        rows = conn.execute(
            f"""
            SELECT al.*, {_UNDONE_EXPR} AS undone FROM audit_log al
            WHERE al.ts_utc > ? ORDER BY al.id DESC LIMIT ?
            """,
            (since, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT al.*, {_UNDONE_EXPR} AS undone FROM audit_log al
            ORDER BY al.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_audit_log_row(conn: sqlite3.Connection, audit_id: int) -> dict | None:
    """One audit_log row plus its derived `undone` flag — the reversal target
    of `POST /audit/{audit_id}/undo` (`src/api/routers/audit.py`)."""
    row = conn.execute(
        f"SELECT al.*, {_UNDONE_EXPR} AS undone FROM audit_log al WHERE al.id = ?",
        (audit_id,),
    ).fetchone()
    return dict(row) if row else None


def member_count(conn: sqlite3.Connection, mid: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM entity_member WHERE mid = ?", (mid,)
    ).fetchone()
    return int(row["n"])


# ── Persisted blocking index (`block_key`) ──────────────────────────────────

def replace_block_keys(conn: sqlite3.Connection, rows: list[tuple[str, str, str]]) -> None:
    """Full rebuild of `block_key` — delete-then-bulk-insert, called once per
    full pipeline publish from the freshly cleaned population. `rows` are
    `(block_id, key_value, patid)`; B1/B6 values must already be hashed via
    `hash_block_key`."""
    conn.execute("DELETE FROM block_key")
    if rows:
        conn.executemany(
            "INSERT INTO block_key (block_id, key_value, patid) VALUES (?, ?, ?)",
            rows,
        )


def add_block_keys(conn: sqlite3.Connection, rows: list[tuple[str, str, str]]) -> None:
    """Incremental append — a newly-scored record's own keys become
    discoverable by future incremental calls without waiting for the next
    full-run rebuild. `INSERT OR IGNORE` since the same key can legitimately
    already exist for another patid."""
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO block_key (block_id, key_value, patid) "
            "VALUES (?, ?, ?)",
            rows,
        )


def lookup_block_candidates(
    conn: sqlite3.Connection,
    keys: dict[str, str | None],
    phones: Iterable[str],
    threshold: int,
) -> dict[str, set[str]]:
    """`existing_patid -> {block_id, ...}` for every persisted block key a
    record's own keys/phones match — the SQL-backed equivalent of
    `blocking.run_inference_blocking`, without loading a `BlockingIndex` into
    memory. `keys` maps block id (B1/B3/B4/B6/B7/B8/B9) to that block's
    computed key value or None (B1/B6 must already be hashed); `phones` are
    the record's own parsed phone numbers for B5. A key whose posting list
    exceeds `threshold` is capped, mirroring the batch blocker's governance
    cap.
    """
    candidates: dict[str, set[str]] = {}

    def _add(block_id: str, patid: str) -> None:
        candidates.setdefault(patid, set()).add(block_id)

    for block_id, key_value in keys.items():
        if not key_value:
            continue
        for row in conn.execute(
            "SELECT patid FROM block_key WHERE block_id = ? AND key_value = ? "
            "LIMIT ?",
            (block_id, key_value, threshold),
        ):
            _add(block_id, row["patid"])

    for phone in phones:
        if not phone:
            continue
        for row in conn.execute(
            "SELECT patid FROM block_key WHERE block_id = 'B5' AND key_value = ? "
            "LIMIT ?",
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
    VALUES ({", ".join("?" for _ in _CLEANED_ATTRS_COLUMNS)})
    ON CONFLICT(patid) DO UPDATE SET
        first_nm=excluded.first_nm, last_nm=excluded.last_nm,
        birth_dt=excluded.birth_dt, ssn=excluded.ssn, ssn_last4=excluded.ssn_last4,
        email=excluded.email, zip_base=excluded.zip_base, address1=excluded.address1,
        sex=excluded.sex, phones_json=excluded.phones_json, run_id=excluded.run_id
"""


def replace_cleaned_attrs(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Full rebuild of `cleaned_attrs` — delete-then-bulk-insert; see
    `replace_block_keys`. `rows` are ordered per `_CLEANED_ATTRS_COLUMNS`."""
    conn.execute("DELETE FROM cleaned_attrs")
    if rows:
        conn.executemany(
            f"INSERT INTO cleaned_attrs ({', '.join(_CLEANED_ATTRS_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _CLEANED_ATTRS_COLUMNS)})",
            rows,
        )


def upsert_cleaned_attrs(conn: sqlite3.Connection, row: tuple) -> None:
    """Incremental single-row upsert — see `add_block_keys`. `row` is ordered
    per `_CLEANED_ATTRS_COLUMNS`."""
    conn.execute(_CLEANED_ATTRS_UPSERT_SQL, row)


def get_cleaned_attrs(conn: sqlite3.Connection, patids: list[str]) -> list[dict]:
    """`cleaned_attrs` rows for the given PATIDs. Column names match
    `_CLEANED_ATTRS_COLUMNS`, not the pipeline's `*_clean` contract names —
    callers building a `pd.DataFrame` for `deterministic_rules.apply_rules`
    must rename (see `src/api/ingest/incremental.py`)."""
    if not patids:
        return []
    placeholders = ", ".join("?" for _ in patids)
    rows = conn.execute(
        f"SELECT * FROM cleaned_attrs WHERE patid IN ({placeholders})", patids
    ).fetchall()
    return [dict(r) for r in rows]


# ── Entity bridging (incremental auto-merge across pre-existing entities) ───
def reassign_entity_members(
    conn: sqlite3.Connection, from_mids: list[str], to_mid: str, updated_utc: str
) -> None:
    """Absorb every member of `from_mids` into `to_mid` and delete the
    now-empty `from_mids` entity rows.

    Used when an incrementally-scored record auto-matches existing members of
    more than one entity — those entities must be unified, the same operation
    full-batch clustering's union-find performs implicitly via
    `assign_clusters`. `to_mid` is assumed to already exist and not be in
    `from_mids`; callers choose it (by convention the lexicographically
    smallest of the touched mids, matching `publish.py`'s own tie-break).
    """
    from_mids = [m for m in from_mids if m != to_mid]
    if not from_mids:
        return
    placeholders = ", ".join("?" for _ in from_mids)
    conn.execute(
        f"UPDATE entity_member SET mid = ?, updated_utc = ? "
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
    "upsert_record_attrs",
    "upsert_record_attrs_bulk",
    "upsert_record_raw_bulk",
    "get_record_raw",
    "upsert_suggestion",
    "upsert_suggestions_bulk",
    "replace_review_candidates_for_run",
    "insert_review_candidates",
    "review_candidates_for_patid",
    "patids_with_review_candidates",
    "all_entity_member_mids",
    "get_entity",
    "list_entities",
    "insert_audit_log",
    "list_audit_log",
    "get_audit_log_row",
    "member_count",
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
]
