"""FastAPI app entry point — `uvicorn src.api.main:app`.

See docs/API-Design.md for the full route contract and
docs/Application-Architecture.md for how this fits with the (separate,
not-yet-built) Next.js front-end.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import jobs
from src.api.backends import sql_backend
from src.api.routers import admin, audit, dashboard, health, records, runs
from src.config import configure_logging, settings

logger = logging.getLogger(__name__)


def _check_schema_ready(settings) -> None:
    """Read-only startup check — verifies the configured backend's schema
    already exists. Never creates or alters anything; that's
    `scripts/init_db.py`'s job now, run explicitly and deliberately by a
    human, not implicitly on every boot (see docs/API-Design.md). Logs a
    loud, actionable error if the schema looks missing, but doesn't crash
    the app — an operator needs to be able to see this in the logs, not
    get stuck in an Azure restart loop that hides the real message.
    """
    backend_name = getattr(settings, "index_backend", "sqlite")
    if backend_name == "parquet":
        return  # ParquetIndexBackend creates its files lazily on first use — a separate, pre-existing pattern, not in scope here.

    conn: Any
    if backend_name == "postgres":
        from src.api.backends import postgres_backend

        if not settings.postgres_host or not settings.postgres_user:
            return  # same guard build_index_backend() uses — nothing to check yet
        conn = postgres_backend.get_connection(
            settings.postgres_host, settings.postgres_port,
            settings.postgres_db, settings.postgres_user,
        )
        try:
            exists = conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name = 'entity'"
            ).fetchone()
        finally:
            conn.close()
    else:
        conn = sql_backend.get_connection(settings.db_path)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='entity'"
            ).fetchone()
        finally:
            conn.close()

    if not exists:
        logger.error(
            "Database schema not found (backend=%s) — run `python "
            "scripts/init_db.py` before using this app. Not creating it "
            "automatically; see docs/API-Design.md.",
            backend_name,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings)  # idempotent — see src/config.py
    settings.ensure_dirs()
    _check_schema_ready(settings)
    jobs.reconcile_interrupted_jobs(settings)
    logger.info("eMPI API started — backend=%s", getattr(settings, "index_backend", "sqlite"))
    yield


app = FastAPI(
    title="eMPI Service API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(runs.router)
app.include_router(records.router)
app.include_router(audit.router)
app.include_router(dashboard.router)
app.include_router(admin.router)


__all__ = ["app"]
