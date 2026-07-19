"""FastAPI dependencies: settings, a per-request SQLite connection, a
per-request `IndexBackend`, reviewer identity. Overridden in tests via
`app.dependency_overrides` to point at a temp DB/settings instead of the
real one."""

from __future__ import annotations

import threading
from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException

from src.api.backends import sql_backend
from src.api.backends.index_backend import IndexBackend, build_index_backend
from src.config import Settings, settings as default_settings

#: Serializes Parquet-backend requests within one process. `ParquetIndexBackend`
#: was designed for one-shot CLI use (load once, commit once, exit) — with a
#: real FastAPI app driving it, sync path-operation functions (`def`, not
#: `async def`, which is every route in records.py/dashboard.py/audit.py) run
#: on Starlette's threadpool, so two overlapping requests genuinely can
#: interleave and race on the same read-modify-write-on-commit Parquet files
#: without this. SQLite needs no equivalent lock here — its own engine already
#: serializes writers. Single-process only, matching the existing constraint
#: on `jobs.py`'s in-process run/score registries (see the Dockerfile
#: docstring) — a multi-replica deployment would need a different mechanism.
_PARQUET_BACKEND_LOCK = threading.Lock()


def get_settings() -> Settings:
    return default_settings


def get_db(settings: Settings = Depends(get_settings)) -> Iterator:
    """Yield one SQLite connection for the request's lifetime, then close it.

    Schema init happens once in the app `lifespan` (`src/api/main.py`), not
    here — re-running `CREATE TABLE IF NOT EXISTS` on every request briefly
    contends for a write lock against any concurrent writer (e.g. a `/runs`
    background job's publish step) for no benefit. Only routes that need raw
    SQL should depend on this directly — `get_backend` below is the
    backend-agnostic dependency everything else should use.
    """
    conn = sql_backend.get_connection(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_backend(settings: Settings = Depends(get_settings)) -> Iterator[IndexBackend]:
    """Yield one `IndexBackend` (SQLite or Parquet local mode, per
    `settings.index_backend` — see `src/api/backends/index_backend.py`) for the
    request's lifetime, then close it. The read routes (`records.py`,
    `dashboard.py`, `audit.py`) depend on this instead of `get_db` so they
    work identically against either backend.

    On the Parquet backend, holds `_PARQUET_BACKEND_LOCK` for the whole
    request (construct -> handler runs -> close) — see that lock's docstring
    for why. A no-op on the SQLite backend.
    """
    is_parquet = getattr(settings, "index_backend", "sqlite") == "parquet"
    if is_parquet:
        _PARQUET_BACKEND_LOCK.acquire()
    try:
        backend = build_index_backend(settings)
        try:
            yield backend
        finally:
            backend.close()
    finally:
        if is_parquet:
            _PARQUET_BACKEND_LOCK.release()


def get_reviewer_id(
    x_reviewer_id: str | None = Header(default=None, alias="X-Reviewer-Id"),
) -> str:
    """The trusted reviewer-identity header — required on every /audit/* call.

    See docs/Application-Architecture.md §3 "Identity / auth": a front-end BFF
    sets this from a server-side session; the browser never sets it directly.
    """
    if not x_reviewer_id:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Reviewer-Id header (reviewer identity required).",
        )
    return x_reviewer_id


__all__ = ["get_settings", "get_db", "get_backend", "get_reviewer_id"]
