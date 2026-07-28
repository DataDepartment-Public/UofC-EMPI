"""Model registry / store for the ML non-match gate.

Mirrors `src/models/ml_matcher/registry.py`'s active-model/deploy-gate
pattern exactly, retargeted at `settings.gate_model_dir` and at the gate's
own held-out metric block. Deliberately a separate copy rather than a shared
generic module, for the same reason the ML registry is (see its docstring).

Trained gate artifacts live here:

    nonmatch_gate_<ts>.pkl         # joblib dump of the fitted LightGBM
                                    # classifier; predict_proba[:, 1] is
                                    # P(plausible) — no adapter needed
    nonmatch_gate_<ts>.meta.json   # provenance + held-out gate metrics
    active.json                    # pointer to the currently-served model

The deploy gate compares `test_metrics.gate_at_threshold`'s
`plausible_precision` / `plausible_recall` — the shape the notebook's export
cell writes. Recall is the one that matters operationally: a pair the gate
drops is discarded for the rest of the run, so a recall regression is
unrecoverable downstream.

No PHI here — meta sidecars carry aggregate metrics + provenance only.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ACTIVE_POINTER = "active.json"
ARTIFACT_GLOB = "nonmatch_gate_*.pkl"


class DeployGateError(RuntimeError):
    """Raised when a model fails the deploy-gate and promotion is not forced."""


# ── Artifact path helpers ────────────────────────────────────────────────────
def meta_path_for(model_path: Path) -> Path:
    """`nonmatch_gate_<ts>.pkl` -> `nonmatch_gate_<ts>.meta.json` (sibling)."""
    model_path = Path(model_path)
    return model_path.parent / f"{model_path.stem}.meta.json"


def load_model_meta(model_path: Path) -> dict | None:
    """Load a model's `.meta.json` sidecar, or None if absent."""
    mp = meta_path_for(model_path)
    if not mp.exists():
        return None
    return json.loads(mp.read_text())


def _model_candidates(model_dir: Path) -> list[Path]:
    if not model_dir.is_dir():
        return []
    return sorted(model_dir.glob(ARTIFACT_GLOB), key=lambda p: p.stat().st_mtime)


# ── Active-model resolution ──────────────────────────────────────────────────
def resolve_active_model(settings: Any) -> Path | None:
    """Resolve the gate model the pipeline should serve.

    Precedence:
      1. `settings.gate_active_model` explicit override (returned if it exists).
      2. the `active.json` pointer in `settings.gate_model_dir`.
      3. the most-recently-modified `nonmatch_gate_*.pkl` in `gate_model_dir`.
      4. None — no model available (the pipeline then falls back to the FS
         gate, or passes the pool through ungated).
    """
    override = getattr(settings, "gate_active_model", None)
    if override is not None:
        p = Path(override)
        if p.exists():
            return p
        logger.warning("gate_active_model override does not exist: %s", p)
        return None

    model_dir = Path(settings.gate_model_dir)
    pointer = model_dir / ACTIVE_POINTER
    if pointer.exists():
        try:
            info = json.loads(pointer.read_text())
            mp = model_dir / info["model_file"]
            if mp.exists():
                return mp
            logger.warning("active.json points at missing model %s", mp)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("active.json is malformed (%s); falling back to latest", exc)

    candidates = _model_candidates(model_dir)
    return candidates[-1] if candidates else None


def active_model_meta(settings: Any) -> dict | None:
    """Meta of the currently-active gate model (None if there is none)."""
    active = resolve_active_model(settings)
    return load_model_meta(active) if active is not None else None


def load_model_artifact(model_path: Path) -> Any:
    """Load a serialized gate model — a joblib dump of the fitted LightGBM
    classifier whose `predict_proba(X)[:, 1]` is `P(plausible)`."""
    import joblib

    return joblib.load(Path(model_path))


def _pointer_model_path(settings: Any) -> Path | None:
    """The explicitly-active model — the override or the `active.json` pointer
    target, WITHOUT the latest-by-mtime fallback (so the deploy gate compares
    against a deliberately-promoted model, not just the newest file)."""
    override = getattr(settings, "gate_active_model", None)
    if override is not None:
        p = Path(override)
        return p if p.exists() else None
    model_dir = Path(settings.gate_model_dir)
    pointer = model_dir / ACTIVE_POINTER
    if not pointer.exists():
        return None
    try:
        mp = model_dir / json.loads(pointer.read_text())["model_file"]
        return mp if mp.exists() else None
    except (json.JSONDecodeError, KeyError):
        return None


# ── Deploy gate + promotion ──────────────────────────────────────────────────
def _plausible_pr(meta: dict) -> tuple[float, float]:
    at = (meta or {}).get("test_metrics", {}).get("gate_at_threshold", {})
    return float(at.get("plausible_precision", 0.0)), float(at.get("plausible_recall", 0.0))


def passes_deploy_gate(
    new_meta: dict, active_meta: dict | None, margin: float,
) -> tuple[bool, str]:
    """Decide whether `new_meta`'s model may be promoted over `active_meta`.

    Gate: new held-out plausible precision AND recall must each be no worse
    than the active model's by more than `margin`. With no active model, any
    model passes (first promotion).
    """
    if active_meta is None:
        return True, "no active model — first promotion"
    np_, nr = _plausible_pr(new_meta)
    ap, ar = _plausible_pr(active_meta)
    ok = (np_ >= ap - margin) and (nr >= ar - margin)
    reason = (
        f"new P={np_:.3f}/R={nr:.3f} vs active P={ap:.3f}/R={ar:.3f} "
        f"(margin={margin:.3f}) -> {'PASS' if ok else 'FAIL'}"
    )
    return ok, reason


def promote(model_path: Path, settings: Any, *, force: bool = False) -> str:
    """Promote `model_path` to the active pointer, subject to the deploy-gate.

    Returns the gate reason string. Raises `DeployGateError` if the gate fails
    and `force` is False.
    """
    model_path = Path(model_path)
    model_dir = Path(settings.gate_model_dir)
    new_meta = load_model_meta(model_path) or {}
    prev = _pointer_model_path(settings)
    prev_active_meta = load_model_meta(prev) if prev is not None else None

    ok, reason = passes_deploy_gate(
        new_meta, prev_active_meta, getattr(settings, "gate_deploy_gate_margin", 0.0)
    )
    if not ok and not force:
        raise DeployGateError(
            f"Deploy gate refused promotion of {model_path.name}: {reason}. "
            "Re-run with force=True to override."
        )

    pointer = model_dir / ACTIVE_POINTER
    pointer.write_text(json.dumps(
        {
            "model_file": model_path.name,
            "promoted_utc": datetime.now(timezone.utc).isoformat(),
            "gate": reason,
            "forced": bool(not ok and force),
        },
        indent=2,
    ))
    logger.info("Promoted %s to active (%s)", model_path.name, reason)
    return reason


__all__ = [
    "ACTIVE_POINTER",
    "ARTIFACT_GLOB",
    "DeployGateError",
    "meta_path_for",
    "load_model_meta",
    "load_model_artifact",
    "resolve_active_model",
    "active_model_meta",
    "passes_deploy_gate",
    "promote",
]
