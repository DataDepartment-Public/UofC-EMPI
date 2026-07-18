"""FastAPI app entry point — `uvicorn src.api.main:app`.

See docs/API-Design.md for the full route contract and
docs/Application-Architecture.md for how this fits with the (separate,
not-yet-built) Next.js front-end.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.backends import sql_backend
from src.api.routers import audit, dashboard, health, records, runs
from src.config import configure_logging, settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings)  # idempotent — see src/config.py
    settings.ensure_dirs()
    conn = sql_backend.get_connection(settings.db_path)
    try:
        sql_backend.init_db(conn)
    finally:
        conn.close()
    logger.info("eMPI API started — db=%s", settings.db_path)
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


__all__ = ["app"]
