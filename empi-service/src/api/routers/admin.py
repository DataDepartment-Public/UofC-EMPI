"""GET/PUT /admin/thresholds — live-tunable ML decision thresholds.

No auth (out of scope — handled elsewhere) and no `X-Reviewer-Id`/audit
trail: this is operator configuration, not a reviewer action on a specific
patient pair. See `src/api/threshold_store.py` for what changing these
actually does (applies immediately to the running process, persists to a
JSON file so it survives a restart, and only affects future scoring —
never rewrites an already-published run's tiers).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api import threshold_store
from src.api.deps import get_settings
from src.api.schemas import ThresholdSettings
from src.config import Settings

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/thresholds", response_model=ThresholdSettings)
def get_thresholds(settings: Settings = Depends(get_settings)) -> ThresholdSettings:
    return ThresholdSettings(**threshold_store.current_thresholds(settings))


@router.put("/thresholds", response_model=ThresholdSettings)
def update_thresholds(
    body: ThresholdSettings, settings: Settings = Depends(get_settings),
) -> ThresholdSettings:
    saved = threshold_store.save_thresholds(settings, body.model_dump())
    return ThresholdSettings(**saved)


__all__ = ["router"]
