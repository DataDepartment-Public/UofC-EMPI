"""Live-tunable ML decision thresholds — `GET/PUT /admin/thresholds`.

`gate_threshold`, `ml_auto_merge_threshold`, and `ml_review_floor` are
ordinary `Settings` fields (`.env`-overridable, like everything else in
`src/config.py`), but changing them today means editing `.env` and
restarting the process. This module adds a second override path: a small
JSON file, so the admin page can change them for the *running* process
immediately, and have that change survive a restart without touching
`.env`. Deliberately a standalone module rather than new `Settings` fields
— these three aren't a stage's directory/config, they're operator-tunable
decision points, and don't belong on the already-large `Settings` class.

Not authenticated (out of scope — handled elsewhere) and not audited like
`/audit/merge` etc.: this is operator configuration, not a reviewer action
on a specific patient pair.

A change here is forward-looking only. It affects the next `POST
/records/score` call or pipeline run; it never rewrites tiers already
computed and published by a prior run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

THRESHOLD_FIELDS: tuple[str, ...] = (
    "gate_threshold",
    "ml_auto_merge_threshold",
    "ml_review_floor",
)


def overrides_path(settings: Any) -> Path:
    return settings.project_root / "data" / "config" / "thresholds.json"


def apply_persisted_overrides(settings: Any) -> None:
    """Load a prior session's saved thresholds onto `settings`, if any.
    Call once at process startup — malformed or missing files are silently
    skipped, since `.env`/field defaults already give `settings` a valid
    starting point without this file."""
    path = overrides_path(settings)
    if not path.exists():
        return
    try:
        values = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable thresholds override %s: %s", path, exc)
        return
    for field in THRESHOLD_FIELDS:
        if field in values:
            setattr(settings, field, float(values[field]))
    logger.info("Applied persisted threshold overrides from %s", path)


def current_thresholds(settings: Any) -> dict[str, float]:
    """Whatever's live on `settings` right now (not necessarily what the
    override file says, if something else has changed `settings` since)."""
    return {field: getattr(settings, field) for field in THRESHOLD_FIELDS}


def save_thresholds(settings: Any, values: dict[str, float]) -> dict[str, float]:
    """Apply `values` to the live `settings` singleton (takes effect for
    every request from here on) and persist the full resulting set to
    `overrides_path`. Returns the saved values."""
    unknown = set(values) - set(THRESHOLD_FIELDS)
    if unknown:
        raise ValueError(f"Unknown threshold field(s): {sorted(unknown)}")

    for field in THRESHOLD_FIELDS:
        if field in values:
            setattr(settings, field, float(values[field]))

    path = overrides_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current_thresholds(settings), indent=2))
    logger.info("Saved threshold overrides to %s: %s", path, values)
    return current_thresholds(settings)


__all__ = [
    "THRESHOLD_FIELDS",
    "overrides_path",
    "apply_persisted_overrides",
    "current_thresholds",
    "save_thresholds",
]
