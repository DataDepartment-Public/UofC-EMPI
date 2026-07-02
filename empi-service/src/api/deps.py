"""FastAPI dependencies: settings, a per-request SQLite connection, reviewer
identity. Overridden in tests via `app.dependency_overrides` to point at a
temp DB/settings instead of the real one."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException

from src.api import store
from src.config import Settings, settings as default_settings


def get_settings() -> Settings:
    return default_settings


def get_db(settings: Settings = Depends(get_settings)) -> Iterator:
    """Yield one SQLite connection for the request's lifetime, then close it.

    Schema init happens once in the app `lifespan` (`src/api/main.py`), not
    here — re-running `CREATE TABLE IF NOT EXISTS` on every request briefly
    contends for a write lock against any concurrent writer (e.g. a `/runs`
    background job's publish step) for no benefit.
    """
    conn = store.get_connection(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


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


__all__ = ["get_settings", "get_db", "get_reviewer_id"]
