"""Background run wrapper + status tracking for `POST /runs` and
`POST /records/score`.

Two layers, deliberately kept separate:

  * **In-memory registry** (`_REGISTRY`/`_SCORE_REGISTRY`) — fast, process-local,
    what `get_status`/`get_score_status` read. Gone the instant the process
    restarts.
  * **Durable status files** (`settings.runs_dir/{run,score}_<id>.status.json`)
    — written at every transition, survive a crash. Extends the existing
    `RunManifest`-to-`data/runs/run_<id>.json` pattern (`src/pipeline.py`)
    rather than adding a database table: a full pipeline run already wrote
    a durable success record there; this fills the gap for in-flight/failed
    state, which previously left no trace at all if the process died or the
    run failed. `read_run_status_file`/`read_score_status_file` are the
    fallback the GET routes use once the in-memory registry has nothing
    (the post-restart case) — see `src/api/routers/runs.py`/`records.py`.

`run_pipeline_job`/`score_records_job` retry transient failures in-process
(a few attempts with backoff) before giving up. `reconcile_interrupted_jobs`
(called from `main.py`'s `lifespan`) marks anything left "running"/"queued"
from a *previous* process as interrupted — it does not resubmit; that's
deliberately left to the caller, not automatic, since the only way to
resubmit a score job safely would mean durably storing its raw record
payload somewhere new, which this doesn't do.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.api.backends.index_backend import build_index_backend
from src.api.ingest import incremental, publish
from src.config import Settings
from src.pipeline import run_pipeline

logger = logging.getLogger(__name__)

Status = Literal["queued", "running", "succeeded", "failed"]

#: In-process attempts per job before giving up, and the backoff (seconds)
#: between them — a defense against transient errors (a momentary DB blip,
#: a transient lock) within a single process's handling of one job. Not a
#: substitute for `reconcile_interrupted_jobs`, which handles the process
#: itself dying mid-job.
_MAX_ATTEMPTS = 3


def _backoff_seconds(attempt: int) -> float:
    return float(2**attempt)  # attempt 1 -> 2s, attempt 2 -> 4s


def new_run_id() -> str:
    """A sortable, collision-resistant id for one run — a full pipeline run
    (`POST /runs`) or an incremental-score job (`POST /records/score`)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


# ── Durable status files ─────────────────────────────────────────────────────
def _status_path(settings: Settings, prefix: str, run_id: str) -> Path:
    return settings.runs_dir / f"{prefix}_{run_id}.status.json"


def _write_status_file(
    settings: Settings, prefix: str, run_id: str, status: Status,
    *, error: str | None, attempt: int,
) -> None:
    path = _status_path(settings, prefix, run_id)
    path.write_text(json.dumps({
        "run_id": run_id,
        "status": status,
        "error": error,
        "attempt": attempt,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def _read_status_file(settings: Settings, prefix: str, run_id: str) -> dict | None:
    path = _status_path(settings, prefix, run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Corrupt status file %s — treating as missing", path)
        return None


def read_run_status_file(run_id: str, settings: Settings) -> dict | None:
    """Fallback for `GET /runs/{run_id}` once `get_status` has nothing —
    the post-restart case, since `_REGISTRY` is gone the instant the
    process that was tracking it dies."""
    return _read_status_file(settings, "run", run_id)


def read_score_status_file(run_id: str, settings: Settings) -> dict | None:
    """Same as `read_run_status_file`, for score jobs."""
    return _read_status_file(settings, "score", run_id)


# ── Full pipeline runs (POST /runs) ──────────────────────────────────────────
#: run_id -> {"status": Status, "error": str | None, "updated_utc": str}
#: In-process only — a multi-worker deploy would need a shared store instead;
#: fine for the single-uvicorn-process local/capstone deployment target.
#: The durable status file (above) is what survives a restart; this is just
#: the fast path while the same process is still alive.
_REGISTRY: dict[str, dict] = {}


def _touch(
    run_id: str, status: Status, settings: Settings,
    *, error: str | None = None, attempt: int = 1,
) -> None:
    _REGISTRY[run_id] = {
        "status": status,
        "error": error,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_status_file(settings, "run", run_id, status, error=error, attempt=attempt)


def mark_queued(run_id: str, settings: Settings) -> None:
    """Register a run as queued before scheduling its background task."""
    _touch(run_id, "queued", settings)


def get_status(run_id: str) -> dict | None:
    return _REGISTRY.get(run_id)


def all_in_flight() -> dict[str, dict]:
    return dict(_REGISTRY)


def run_pipeline_job(run_id: str, raw_input: Path, settings: Settings) -> None:
    """Run clean→block→rules→cluster, then publish to whichever `IndexBackend`
    `settings.index_backend` selects (SQLite `empi.db` by default, or the
    local Parquet index — see `src/api/backends/index_backend.py`). Retries
    up to `_MAX_ATTEMPTS` times with backoff before recording a final
    "failed" status."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        _touch(run_id, "running", settings, attempt=attempt)
        backend = None
        try:
            run_pipeline(raw_input=raw_input, settings=settings, run_id=run_id)
            backend = build_index_backend(settings)
            publish.publish_run(backend, run_id, settings)
            _touch(run_id, "succeeded", settings, attempt=attempt)
            return
        except Exception as exc:
            logger.exception("Run %s failed (attempt %d/%d)", run_id, attempt, _MAX_ATTEMPTS)
            if attempt == _MAX_ATTEMPTS:
                _touch(
                    run_id, "failed", settings, attempt=attempt,
                    error=f"{exc}\n{traceback.format_exc(limit=5)}",
                )
                return
            time.sleep(_backoff_seconds(attempt))
        finally:
            if backend is not None:
                backend.close()


# ── Incremental score jobs (POST /records/score) ─────────────────────────────
#: run_id -> {"status": Status, "error": str | None, "updated_utc": str}
#: Separate from `_REGISTRY` (full pipeline runs) — an incremental-score job
#: is a different kind of run (per-record outcomes, not a `RunManifest`) and
#: is not listed by `GET /runs`; see `src/api/routers/records.py`.
_SCORE_REGISTRY: dict[str, dict] = {}
#: run_id -> list[dict] — the per-record outcomes of a completed score job.
#: In-memory only, deliberately: persisting incoming record payloads
#: durably would open a new place for PHI-bearing data to land, so a score
#: job interrupted by a crash is marked failed (see
#: `reconcile_interrupted_jobs`) rather than auto-resubmitted.
_SCORE_RESULTS: dict[str, list[dict]] = {}


def _touch_score(
    run_id: str, status: Status, settings: Settings,
    *, error: str | None = None, attempt: int = 1,
) -> None:
    _SCORE_REGISTRY[run_id] = {
        "status": status,
        "error": error,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_status_file(settings, "score", run_id, status, error=error, attempt=attempt)


def mark_score_queued(run_id: str, settings: Settings) -> None:
    """Register a score job as queued before scheduling its background task."""
    _touch_score(run_id, "queued", settings)


def get_score_status(run_id: str) -> dict | None:
    return _SCORE_REGISTRY.get(run_id)


def get_score_result(run_id: str) -> list[dict] | None:
    return _SCORE_RESULTS.get(run_id)


def score_records_job(run_id: str, records: list[dict], settings: Settings) -> None:
    """Background job for `POST /records/score` — scores each record against
    the existing population (`src/api/incremental.score_records`) and records
    status/outcomes. Mirrors `run_pipeline_job`'s retry-with-backoff.

    Storage is whichever `IndexBackend` `settings.index_backend` selects
    (`build_index_backend` — SQLite/`empi.db` by default, or local Parquet
    files; see `src/api/backends/index_backend.py`)."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        _touch_score(run_id, "running", settings, attempt=attempt)
        backend = None
        try:
            backend = build_index_backend(settings)
            outcomes = incremental.score_records(backend, settings, records, run_id)
            _SCORE_RESULTS[run_id] = outcomes
            _touch_score(run_id, "succeeded", settings, attempt=attempt)
            return
        except Exception as exc:
            logger.exception("Score run %s failed (attempt %d/%d)", run_id, attempt, _MAX_ATTEMPTS)
            if attempt == _MAX_ATTEMPTS:
                _touch_score(
                    run_id, "failed", settings, attempt=attempt,
                    error=f"{exc}\n{traceback.format_exc(limit=5)}",
                )
                return
            time.sleep(_backoff_seconds(attempt))
        finally:
            if backend is not None:
                backend.close()


# ── Startup reconciliation ────────────────────────────────────────────────────
def reconcile_interrupted_jobs(settings: Settings) -> int:
    """Call once, from `main.py`'s `lifespan`, before the app starts serving
    requests. Scans durable status files for anything left "running" or
    "queued" — these were orphaned by a *previous* process that died
    mid-job (no signal handling exists anywhere in this app to flush state
    on a clean shutdown, so a redeploy/OOM/crash always looks like this).

    Marks them "failed" with a distinguishing error message and returns the
    count reconciled. Deliberately does NOT resubmit anything — see the
    module docstring."""
    if not settings.runs_dir.exists():
        return 0
    count = 0
    for path in settings.runs_dir.glob("*.status.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("status") not in ("running", "queued"):
            continue
        path.write_text(json.dumps({
            **data,
            "status": "failed",
            "error": "Interrupted by restart — the process handling this job "
                     "stopped before it finished. Resubmit if you still need it.",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        count += 1
    if count:
        logger.warning("Reconciled %d job(s) interrupted by a previous restart", count)
    return count


__all__ = [
    "Status",
    "all_in_flight",
    "get_score_result",
    "get_score_status",
    "get_status",
    "mark_queued",
    "mark_score_queued",
    "new_run_id",
    "read_run_status_file",
    "read_score_status_file",
    "reconcile_interrupted_jobs",
    "run_pipeline_job",
    "score_records_job",
]
