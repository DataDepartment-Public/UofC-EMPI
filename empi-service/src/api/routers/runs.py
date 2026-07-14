"""POST /runs, GET /runs, GET /runs/{run_id} — docs/API-Design.md §3 "Runs".

`POST /runs` accepts the raw input either as a multipart file upload or as a
form field naming an existing path (`input_path`). FastAPI/Starlette can't mix
a `File` param with a JSON `Body` on the same route, so `input_path` travels
as a form field rather than raw JSON — the doc's `{"input_path": "..."}` shape
is preserved as a form value, not a JSON body. Exactly one of `file` /
`input_path` must be given.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile

from src.api import jobs
from src.api.deps import get_settings
from src.api.schemas import RunCreateResponse, RunSummary
from src.config import Settings
from src.contracts import RunManifest

router = APIRouter(tags=["runs"])


@router.post("/runs", response_model=RunCreateResponse, status_code=202)
def create_run(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = None,
    input_path: str | None = Form(default=None),
    settings: Settings = Depends(get_settings),
) -> RunCreateResponse:
    if (file is None) == (input_path is None):
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of a file upload or `input_path`.",
        )

    settings.ensure_dirs()
    run_id = jobs.new_run_id()

    if file is not None:
        suffix = Path(file.filename or "upload.csv").suffix or ".csv"
        raw_input = settings.raw_dir / f"upload_{run_id}{suffix}"
        raw_input.write_bytes(file.file.read())
    else:
        raw_input = Path(input_path)
        if not raw_input.is_absolute():
            raw_input = settings.project_root / raw_input
        if not raw_input.exists():
            raise HTTPException(
                status_code=422, detail=f"input_path not found: {raw_input}"
            )

    jobs.mark_queued(run_id)
    background_tasks.add_task(jobs.run_pipeline_job, run_id, raw_input, settings)
    return RunCreateResponse(run_id=run_id, status="queued")


def _load_manifest_or_none(run_id: str, settings: Settings) -> RunManifest | None:
    path = settings.runs_dir / f"run_{run_id}.json"
    if not path.exists():
        return None
    try:
        return RunManifest.model_validate_json(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return None


@router.get("/runs", response_model=list[RunSummary])
def list_runs(settings: Settings = Depends(get_settings)) -> list[RunSummary]:
    """One summary per run_id.

    A `run_<id>.json` manifest is written by `run_pipeline()` itself, *before*
    the background job's subsequent `publish_run()` call finishes — so its
    mere existence does NOT mean the job is done. The in-flight registry
    (`jobs.py`) is authoritative whenever it has a non-succeeded entry for a
    run_id; only a manifest with no registry entry at all (e.g. the server
    restarted after a prior process completed it) is taken as "succeeded".
    """
    run_ids: set[str] = set(jobs.all_in_flight())
    if settings.runs_dir.exists():
        run_ids |= {p.stem.removeprefix("run_") for p in settings.runs_dir.glob("run_*.json")}

    summaries = [_run_summary(run_id, settings) for run_id in run_ids]
    return sorted(summaries, key=lambda s: s.created_utc or "", reverse=True)


def _run_summary(run_id: str, settings: Settings) -> RunSummary:
    in_flight = jobs.get_status(run_id)
    manifest = _load_manifest_or_none(run_id, settings)

    if in_flight is not None and in_flight["status"] != "succeeded":
        return RunSummary(
            run_id=run_id,
            status=in_flight["status"],
            counts=manifest.counts if manifest else {},
            created_utc=manifest.created_utc if manifest else None,
            error=in_flight.get("error"),
        )
    if manifest is not None:
        return RunSummary(
            run_id=run_id, status="succeeded",
            counts=manifest.counts, created_utc=manifest.created_utc,
        )
    return RunSummary(run_id=run_id, status=in_flight["status"] if in_flight else "queued")


@router.get("/runs/{run_id}")
def get_run(run_id: str, settings: Settings = Depends(get_settings)) -> dict:
    in_flight = jobs.get_status(run_id)
    manifest = _load_manifest_or_none(run_id, settings)

    if manifest is None and in_flight is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")

    if in_flight is not None and in_flight["status"] != "succeeded":
        base = manifest.model_dump() if manifest else {"run_id": run_id}
        return {**base, "status": in_flight["status"], "error": in_flight.get("error")}

    if manifest is not None:
        return {**manifest.model_dump(), "status": "succeeded"}

    return {"run_id": run_id, **in_flight}


__all__ = ["router"]
