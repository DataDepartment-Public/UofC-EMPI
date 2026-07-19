"""Background run wrapper + in-memory status registry for `POST /runs`.

`run_pipeline()` is minutes-long, so the route schedules this via
`BackgroundTasks` and returns immediately. Completed runs are durable via the
`RunManifest` on disk (`data/runs/run_<id>.json`); this registry only tracks
in-flight/failed state that the manifest can't represent, per
docs/API-Design.md §3 "Runs".

Publish (Parquet → the resolved-output index — SQLite or the local Parquet
index, whichever `settings.index_backend` selects) runs automatically at the
end of a successful pipeline run — the API design has no separate `/publish`
route, so this is the only place it's triggered (see docs/API-Design.md §1).
"""

from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.api.ingest import incremental, publish
from src.api.backends.index_backend import build_index_backend
from src.config import Settings
from src.pipeline import run_pipeline

logger = logging.getLogger(__name__)

Status = Literal["queued", "running", "succeeded", "failed"]


def new_run_id() -> str:
    """A sortable, collision-resistant id for one run — a full pipeline run
    (`POST /runs`) or an incremental-score job (`POST /records/score`)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


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
    """Run clean→block→rules→cluster, then publish to whichever `IndexBackend`
    `settings.index_backend` selects (SQLite `empi.db` by default, or the
    local Parquet index — see `src/api/backends/index_backend.py`). Records status."""
    _touch(run_id, "running")
    backend = None
    try:
        run_pipeline(raw_input=raw_input, settings=settings, run_id=run_id)
        backend = build_index_backend(settings)
        publish.publish_run(backend, run_id, settings)
        _touch(run_id, "succeeded")
    except Exception as exc:  # noqa: BLE001 — surface any failure via status
        logger.exception("Run %s failed", run_id)
        _touch(run_id, "failed", error=f"{exc}\n{traceback.format_exc(limit=5)}")
    finally:
        if backend is not None:
            backend.close()


#: run_id -> {"status": Status, "error": str | None, "updated_utc": str}
#: Separate from `_REGISTRY` (full pipeline runs) — an incremental-score job
#: is a different kind of run (per-record outcomes, not a `RunManifest`) and
#: is not listed by `GET /runs`; see `src/api/routers/records.py`.
_SCORE_REGISTRY: dict[str, dict] = {}
#: run_id -> list[dict] — the per-record outcomes of a completed score job.
_SCORE_RESULTS: dict[str, list[dict]] = {}


def _touch_score(run_id: str, status: Status, error: str | None = None) -> None:
    _SCORE_REGISTRY[run_id] = {
        "status": status,
        "error": error,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def mark_score_queued(run_id: str) -> None:
    """Register a score job as queued before scheduling its background task."""
    _touch_score(run_id, "queued")


def get_score_status(run_id: str) -> dict | None:
    return _SCORE_REGISTRY.get(run_id)


def get_score_result(run_id: str) -> list[dict] | None:
    return _SCORE_RESULTS.get(run_id)


def score_records_job(run_id: str, records: list[dict], settings: Settings) -> None:
    """Background job for `POST /records/score` — scores each record against
    the existing population (`src/api/incremental.score_records`) and records
    status/outcomes. Mirrors `run_pipeline_job`'s status-registry pattern.

    Storage is whichever `IndexBackend` `settings.index_backend` selects
    (`build_index_backend` — SQLite/`empi.db` by default, or local Parquet
    files; see `src/api/backends/index_backend.py`)."""
    _touch_score(run_id, "running")
    backend = None
    try:
        backend = build_index_backend(settings)
        outcomes = incremental.score_records(backend, settings, records, run_id)
        _SCORE_RESULTS[run_id] = outcomes
        _touch_score(run_id, "succeeded")
    except Exception as exc:  # noqa: BLE001 — surface any failure via status
        logger.exception("Score run %s failed", run_id)
        _touch_score(run_id, "failed", error=f"{exc}\n{traceback.format_exc(limit=5)}")
    finally:
        if backend is not None:
            backend.close()


__all__ = [
    "run_pipeline_job", "mark_queued", "get_status", "all_in_flight", "Status",
    "new_run_id", "mark_score_queued", "get_score_status", "get_score_result",
    "score_records_job",
]
