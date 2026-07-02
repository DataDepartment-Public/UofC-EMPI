"""Background run wrapper + in-memory status registry for `POST /runs`.

`run_pipeline()` is minutes-long, so the route schedules this via
`BackgroundTasks` and returns immediately. Completed runs are durable via the
`RunManifest` on disk (`data/runs/run_<id>.json`); this registry only tracks
in-flight/failed state that the manifest can't represent, per
docs/API-Design.md §3 "Runs".

Publish (Parquet → `empi.db`) runs automatically at the end of a successful
pipeline run — the API design has no separate `/publish` route, so this is the
only place it's triggered (see docs/API-Design.md §1).
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.api import publish, store
from src.config import Settings
from src.pipeline import run_pipeline

logger = logging.getLogger(__name__)

Status = Literal["queued", "running", "succeeded", "failed"]

#: run_id -> {"status": Status, "error": str | None, "updated_utc": str}
#: In-process only — a multi-worker deploy would need a shared store instead;
#: fine for the single-uvicorn-process local/capstone deployment target.
_REGISTRY: dict[str, dict] = {}


def _touch(run_id: str, status: Status, error: str | None = None) -> None:
    _REGISTRY[run_id] = {
        "status": status,
        "error": error,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def mark_queued(run_id: str) -> None:
    """Register a run as queued before scheduling its background task."""
    _touch(run_id, "queued")


def get_status(run_id: str) -> dict | None:
    return _REGISTRY.get(run_id)


def all_in_flight() -> dict[str, dict]:
    return dict(_REGISTRY)


def run_pipeline_job(run_id: str, raw_input: Path, settings: Settings) -> None:
    """Run clean→block→rules→cluster, then publish to `empi.db`. Records status."""
    _touch(run_id, "running")
    try:
        run_pipeline(raw_input=raw_input, settings=settings, run_id=run_id)
        conn = store.get_connection(settings.db_path)
        try:
            store.init_db(conn)
            publish.publish_run(conn, run_id, settings)
        finally:
            conn.close()
        _touch(run_id, "succeeded")
    except Exception as exc:  # noqa: BLE001 — surface any failure via status
        logger.exception("Run %s failed", run_id)
        _touch(run_id, "failed", error=f"{exc}\n{traceback.format_exc(limit=5)}")


__all__ = ["run_pipeline_job", "mark_queued", "get_status", "all_in_flight", "Status"]
