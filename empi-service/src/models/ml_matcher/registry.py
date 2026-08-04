"""Model registry / store for the pluggable ML matcher.

Mirrors `src/models/fs_matcher/registry.py`'s active-model/deploy-gate
pattern exactly, retargeted at `settings.ml_model_dir`. Deliberately a
separate copy rather than a shared generic module — lower risk for this
change (zero touch to FS's tested registry), and a reasonable "rule of
three" refactor once a third model family needs the same pattern.

Trained model artifacts live here:

    ml_model_<name>_<ts>.pkl        # joblib dump of the served model — today a
                                    # v5 DirectMatchAdapter, written by the
                                    # notebook as
                                    # ml_model_confident_match_v5_<ts>.pkl.
                                    # Discovered by the ml_model_*.pkl glob, so
                                    # the descriptive middle is free-form.
    ml_model_<name>_<ts>.meta.json  # provenance + held-out test metrics (this
                                    # module reads it)
    active.json                     # pointer to the currently-served model

The pipeline resolves the **active** model at serve time via
`resolve_active_model`. Promotion is guarded by a **deploy-gate**: a
retrained model may only become active if its held-out precision/recall are
within a configured margin of the current active model's (so a degraded
retrain can't be silently deployed).

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


class DeployGateError(RuntimeError):
    """Raised when a model fails the deploy-gate and promotion is not forced."""


# ── Artifact path helpers ────────────────────────────────────────────────────
def meta_path_for(model_path: Path) -> Path:
    """`<artifact>.pkl` -> `<artifact>.meta.json` (sibling sidecar)."""
    model_path = Path(model_path)
    return model_path.parent / f"{model_path.stem}.meta.json"


def load_model_meta(model_path: Path) -> dict | None:
    """Load a model's `.meta.json` sidecar, or None if absent."""
    mp = meta_path_for(model_path)
    if not mp.exists():
        return None
    return json.loads(mp.read_text())


def _model_candidates(model_dir: Path) -> list[Path]:
    """All `ml_model_*.pkl` artifacts (the `.meta.json` sidecars are `.json`,
    so no exclusion is needed)."""
    if not model_dir.is_dir():
        return []
    return sorted(
        model_dir.glob("ml_model_*.pkl"),
        key=lambda p: p.stat().st_mtime,
    )


# ── Active-model resolution ──────────────────────────────────────────────────
def resolve_active_model(settings: Any) -> Path | None:
    """Resolve the ML model the pipeline should serve.

    Precedence:
      1. `settings.ml_active_model` explicit override (returned if it exists).
      2. the `active.json` pointer in `settings.ml_model_dir`.
      3. the most-recently-modified `ml_model_*.json` in `ml_model_dir`.
      4. None — no model available (the pipeline then skips the ML stage).
    """
    override = getattr(settings, "ml_active_model", None)
    if override is not None:
        p = Path(override)
        if p.exists():
            return p
        logger.warning("ml_active_model override does not exist: %s", p)
        return None

    model_dir = Path(settings.ml_model_dir)
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
    """Meta of the currently-active model (None if there is no active model)."""
    active = resolve_active_model(settings)
    return load_model_meta(active) if active is not None else None


def load_model_artifact(model_path: Path) -> Any:
    """Load a serialized model from `model_path` into an `MLModel`-compatible
    object (`fit`/`predict_proba`).

    The active ML model is the LightGBM v5 confident-match classifier,
    serialized with joblib as a `ml_model_confident_match_v5_<ts>.pkl` artifact
    whose payload is a `src.models.ml_matcher.lightgbm_v5.DirectMatchAdapter`
    (which wraps the fitted estimator and passes `predict_proba` through
    unchanged, because its class 1 is already the confident match). This
    function is the pipeline's one call site for "load whatever `active.json`
    points at."

    Artifacts from the retired class-1-is-*ambiguous* generation pickled a
    wrapper from a module that no longer exists, so they now raise
    `ModuleNotFoundError` here rather than being served with inverted
    semantics. That is intentional: a loud failure beats a silent one. See
    `docs/ML-Model-LightGBM-v5.md`.
    """
    import joblib

    return joblib.load(Path(model_path))


def _pointer_model_path(settings: Any) -> Path | None:
    """The explicitly-active model — the override or the `active.json` pointer
    target, WITHOUT the latest-by-mtime fallback. Used by the deploy gate so
    "active" means a deliberately-promoted model, not just the newest file.
    """
    override = getattr(settings, "ml_active_model", None)
    if override is not None:
        p = Path(override)
        return p if p.exists() else None
    model_dir = Path(settings.ml_model_dir)
    pointer = model_dir / ACTIVE_POINTER
    if not pointer.exists():
        return None
    try:
        mp = model_dir / json.loads(pointer.read_text())["model_file"]
        return mp if mp.exists() else None
    except (json.JSONDecodeError, KeyError):
        return None


# ── Deploy gate + promotion ──────────────────────────────────────────────────
def _auto_merge_pr(meta: dict) -> tuple[float, float]:
    am = (meta or {}).get("test_metrics", {}).get("metrics_auto_merge", {})
    return float(am.get("precision", 0.0)), float(am.get("recall", 0.0))


def passes_deploy_gate(
    new_meta: dict, active_meta: dict | None, margin: float,
) -> tuple[bool, str]:
    """Decide whether `new_meta`'s model may be promoted over `active_meta`.

    Gate: new held-out auto_merge precision AND recall must each be no worse
    than the active model's by more than `margin`. With no active model, any
    model passes (first promotion).
    """
    if active_meta is None:
        return True, "no active model — first promotion"
    np_, nr = _auto_merge_pr(new_meta)
    ap, ar = _auto_merge_pr(active_meta)
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
    model_dir = Path(settings.ml_model_dir)
    new_meta = load_model_meta(model_path) or {}
    # Compare against the pointer-active model only (not the mtime fallback, which
    # would resolve the just-written model itself and trivially self-compare).
    prev = _pointer_model_path(settings)
    prev_active_meta = load_model_meta(prev) if prev is not None else None

    ok, reason = passes_deploy_gate(
        new_meta, prev_active_meta, getattr(settings, "ml_deploy_gate_margin", 0.0)
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
    "DeployGateError",
    "meta_path_for",
    "load_model_meta",
    "load_model_artifact",
    "resolve_active_model",
    "active_model_meta",
    "passes_deploy_gate",
    "promote",
]
