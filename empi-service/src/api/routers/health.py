"""GET /health, /health/ready — docs/API-Design.md §3 "Health"."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Response

from src.api.backends import sql_backend
from src.api.deps import get_settings
from src.api.schemas import HealthResponse, ReadyChecks, ReadyResponse
from src.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadyResponse)
def ready(response: Response, settings: Settings = Depends(get_settings)) -> ReadyResponse:
    db_ok = True
    try:
        conn = sql_backend.get_connection(settings.db_path)
        try:
            conn.execute("SELECT 1")
        finally:
            conn.close()
    except sqlite3.Error:
        db_ok = False

    dirs_ok = settings.runs_dir.exists() or _try_mkdir(settings)

    last_run_id = None
    if settings.runs_dir.exists():
        manifests = sorted(settings.runs_dir.glob("run_*.json"))
        if manifests:
            last_run_id = manifests[-1].stem.removeprefix("run_")

    checks = ReadyChecks(db=db_ok, data_dirs=dirs_ok, last_run_id=last_run_id)
    is_ready = db_ok and dirs_ok
    if not is_ready:
        response.status_code = 503
    return ReadyResponse(status="ok" if is_ready else "not_ready", checks=checks)


def _try_mkdir(settings: Settings) -> bool:
    try:
        settings.ensure_dirs()
        return True
    except OSError:
        return False


__all__ = ["router"]
