"""Create (or update) the resolved-output database schema — run explicitly,
once per environment. Never runs automatically.

Before this script existed, `src/api/main.py`'s `lifespan` and
`src/api/backends/index_backend.py`'s `build_index_backend()` both called
`init_db()` implicitly — on every single app boot, and on every request
that resolved a backend (`get_backend`, used by nearly every dashboard/
audit/records route, plus every background job). That meant the
`CREATE TABLE IF NOT EXISTS` + column-diff dance ran on close to every
request, not just once. The app now only ever *connects* to a database
that's assumed to already have the right schema — this script is where
that schema actually gets created or updated, run deliberately by a human,
not as an automatic side effect of starting or redeploying the app.

Idempotent either way: `CREATE TABLE IF NOT EXISTS` plus the existing
`_COLUMN_MIGRATIONS` dict (in `sql_backend.py`/`postgres_backend.py`) mean
running this against an already-current database is a safe no-op. This is
NOT a versioned migration framework — it's the same schema-setup logic
that already existed, just moved from implicit-and-constant to
explicit-and-deliberate. A real migration system (renames, backfills,
rollback) is a separate decision for infra/code owners, not built here.

Usage:
    python scripts/init_db.py                  # whichever backend EMPI_INDEX_BACKEND selects
    python scripts/init_db.py --backend sqlite  # force sqlite regardless of settings
    python scripts/init_db.py --backend postgres

Run this:
  - once, the first time you stand up a new environment (a fresh local
    SQLite file, or a new Azure Postgres instance — the latter needs the
    same VNet-connected access already documented in
    terraform/README.md's "Bootstrap the FS matcher model" section, since
    Postgres has no public ingress);
  - again, whenever a code change adds a new entry to `_COLUMN_MIGRATIONS`.

Never as part of a routine deploy — that's the whole point.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from src.config import configure_logging, settings as default_settings  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--backend", choices=("sqlite", "postgres"), default=None,
        help="Override EMPI_INDEX_BACKEND for this run (default: whatever settings resolves to; "
        "'parquet' isn't a valid choice here — ParquetIndexBackend has no schema to initialize).",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure_logging(level=args.log_level)

    backend_name = args.backend or getattr(default_settings, "index_backend", "sqlite")
    if backend_name == "parquet":
        raise SystemExit(
            "index_backend='parquet' has no schema to initialize — "
            "ParquetIndexBackend creates its files lazily on first use."
        )

    if backend_name == "postgres":
        from src.api.backends import postgres_backend

        if not default_settings.postgres_host or not default_settings.postgres_user:
            raise SystemExit(
                "backend=postgres requires EMPI_POSTGRES_HOST and EMPI_POSTGRES_USER to be set."
            )
        conn = postgres_backend.get_connection(
            default_settings.postgres_host, default_settings.postgres_port,
            default_settings.postgres_db, default_settings.postgres_user,
        )
        try:
            postgres_backend.init_db(conn)
        finally:
            conn.close()
        logger.info(
            "Schema ready: postgres %s@%s/%s",
            default_settings.postgres_user, default_settings.postgres_host, default_settings.postgres_db,
        )
        return

    from src.api.backends import sql_backend

    default_settings.ensure_dirs()
    conn = sql_backend.get_connection(default_settings.db_path)
    try:
        sql_backend.init_db(conn)
    finally:
        conn.close()
    logger.info("Schema ready: sqlite %s", default_settings.db_path)


if __name__ == "__main__":
    main()
