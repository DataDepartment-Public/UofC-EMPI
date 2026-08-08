"""Minimal, independent model-artifact registry helpers.

Deliberately NOT imported from `empi-service` (see `CLAUDE.md` — this repo
never calls across repos). Writes the same file-naming convention
(`<prefix>_model_<ts>.json|.pkl`, `<prefix>_model_<ts>.meta.json`,
`active.json`) as
`empi-service/src/models/{fs_matcher,ml_matcher,nonmatch_gate}/registry.py`
so a promoted artifact drops into the serving share unchanged, but the
deploy-gate logic here is its own small copy, not a shared import.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import Any

ACTIVE_POINTER = "active.json"


class DeployGateError(RuntimeError):
    """Raised when a model fails the deploy-gate and promotion is not forced."""


def meta_path_for(model_path: Path) -> Path:
    return model_path.parent / f"{model_path.stem}.meta.json"


def load_model_meta(model_path: Path) -> dict[str, Any] | None:
    mp = meta_path_for(model_path)
    if not mp.exists():
        return None
    result: dict[str, Any] = json.loads(mp.read_text())
    return result


def _pointer_model_path(model_dir: Path) -> Path | None:
    pointer = model_dir / ACTIVE_POINTER
    if not pointer.exists():
        return None
    try:
        info = json.loads(pointer.read_text())
        mp = model_dir / info["model_file"]
        return mp if mp.exists() else None
    except (json.JSONDecodeError, KeyError):
        return None


def auto_merge_pr(meta: dict[str, Any] | None) -> tuple[float, float]:
    """(precision, recall) of the ML matcher's `auto_merge` tier at its served
    threshold -- the metric shape `training/lightgbm_train.py` writes."""
    am = (meta or {}).get("test_metrics", {}).get("metrics_auto_merge", {})
    return float(am.get("precision", 0.0)), float(am.get("recall", 0.0))


def gate_pr(meta: dict[str, Any] | None) -> tuple[float, float]:
    """(precision, recall) of the non-match gate's `plausible` class at its
    served threshold -- the metric shape `training/nonmatch_gate_train.py`
    writes. Recall is what matters operationally (a drop is unrecoverable
    downstream), but both are gated the same way as the matcher's."""
    at = (meta or {}).get("test_metrics", {}).get("gate_at_threshold", {})
    return float(at.get("plausible_precision", 0.0)), float(at.get("plausible_recall", 0.0))


def passes_deploy_gate(
    new_meta: dict[str, Any],
    active_meta: dict[str, Any] | None,
    margin: float,
    *,
    metric_fn: Any = auto_merge_pr,
) -> tuple[bool, str]:
    """New held-out precision AND recall (per `metric_fn`) must each be no
    worse than the active model's by more than `margin`. No active model ->
    first promotion always passes."""
    if active_meta is None:
        return True, "no active model -- first promotion"
    np_, nr = metric_fn(new_meta)
    ap, ar = metric_fn(active_meta)
    ok = (np_ >= ap - margin) and (nr >= ar - margin)
    reason = (
        f"new P={np_:.3f}/R={nr:.3f} vs active P={ap:.3f}/R={ar:.3f} "
        f"(margin={margin:.3f}) -> {'PASS' if ok else 'FAIL'}"
    )
    return ok, reason


def promote(
    model_path: Path,
    model_dir: Path,
    deploy_gate_margin: float,
    *,
    force: bool = False,
    metric_fn: Any = auto_merge_pr,
) -> str:
    """Point `active.json` at `model_path`, subject to the deploy-gate.

    `metric_fn` selects which metric shape the deploy gate reads -- pass
    `gate_pr` when promoting a non-match gate model instead of an ML matcher.

    Returns the gate reason string. Raises `DeployGateError` if the gate
    fails and `force` is False.
    """
    from datetime import datetime

    new_meta = load_model_meta(model_path) or {}
    prev = _pointer_model_path(model_dir)
    prev_active_meta = load_model_meta(prev) if prev is not None else None

    ok, reason = passes_deploy_gate(
        new_meta, prev_active_meta, deploy_gate_margin, metric_fn=metric_fn
    )
    if not ok and not force:
        raise DeployGateError(
            f"Deploy gate refused promotion of {model_path.name}: {reason}. "
            "Re-run with force=True to override."
        )

    pointer = model_dir / ACTIVE_POINTER
    pointer.write_text(
        json.dumps(
            {
                "model_file": model_path.name,
                "promoted_utc": datetime.now(UTC).isoformat(),
                "gate": reason,
                "forced": bool(not ok and force),
            },
            indent=2,
        )
    )
    return reason


__all__ = [
    "ACTIVE_POINTER",
    "DeployGateError",
    "meta_path_for",
    "load_model_meta",
    "auto_merge_pr",
    "gate_pr",
    "passes_deploy_gate",
    "promote",
]
