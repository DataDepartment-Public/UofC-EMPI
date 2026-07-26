"""Ops routes — not part of the reviewer-facing dashboard surface.

`POST /admin/models/reload` gives an explicit, synchronous "it's live now"
trigger after a model promotion: it drops the in-memory model cache
(`src.models.model_cache`), so the very next pipeline run or incremental
`/records/score` call re-reads whatever `active.json` currently points at.
No restart, no redeploy, no dropped requests — anything already in flight
keeps running against whatever it already loaded; only requests *after*
this call see the new model.

Without calling this, the same result still happens automatically — the
cache is keyed on the model file's mtime, so a promoted model gets picked
up the first time anything asks for it regardless. This endpoint just makes
that moment immediate and observable instead of implicit and whenever.

Protected the same way every other route in this service is: there is no
per-route auth here, matching health/records/audit/dashboard/runs — the
backend has no public ingress at all (VNet-only, see terraform/README.md),
so network isolation is the control, not an application-layer check.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_settings
from src.api.schemas import ActiveModelInfo, CachedModelInfo, ModelReloadResponse, ModelStatusResponse
from src.config import Settings
from src.models import model_cache
from src.models.fs_matcher.registry import active_model_meta as fs_active_model_meta
from src.models.ml_matcher.registry import active_model_meta as ml_active_model_meta

router = APIRouter(prefix="/admin", tags=["admin"])


def _active_model_info(settings: Settings) -> tuple[ActiveModelInfo | None, ActiveModelInfo | None]:
    fs_meta = fs_active_model_meta(settings)
    ml_meta = ml_active_model_meta(settings)
    return (
        ActiveModelInfo(**fs_meta) if fs_meta is not None else None,
        ActiveModelInfo(**ml_meta) if ml_meta is not None else None,
    )


@router.post("/models/reload", response_model=ModelReloadResponse)
def reload_models(settings: Settings = Depends(get_settings)) -> ModelReloadResponse:
    invalidated = model_cache.invalidate()
    fs_active, ml_active = _active_model_info(settings)
    return ModelReloadResponse(
        invalidated=invalidated, fs_active_model=fs_active, ml_active_model=ml_active,
    )


@router.get("/models/status", response_model=ModelStatusResponse)
def model_status(settings: Settings = Depends(get_settings)) -> ModelStatusResponse:
    fs_active, ml_active = _active_model_info(settings)
    return ModelStatusResponse(
        cached={k: CachedModelInfo(**v) for k, v in model_cache.status().items()},
        fs_active_model=fs_active,
        ml_active_model=ml_active,
    )
