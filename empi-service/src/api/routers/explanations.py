"""GET /explanations/... — per-pair SHAP waterfalls for the non-match gate
(Stage 4.25) and the ML matcher (Stage 4.5).

READ-ONLY, AND DELIBERATELY NOT BACKED BY THE INDEX
---------------------------------------------------
Every other read route goes through `IndexBackend`, because it serves the
*mutable, reviewer-editable* projection of a run. Explanations are the
opposite: they are immutable evidence about a decision a specific model made
during a specific run. So this router resolves the run's `RunManifest` and
reads the pipeline's Parquet artifact directly — the same artifact whose
sha256 the manifest records.

That choice is the whole point. No model is loaded in the request path,
nothing is recomputed, and promoting a new model tomorrow cannot silently
change the explanation shown for a decision made today. See
`src/models/explanations.py` for why on-demand recomputation was rejected.

The artifacts are sorted by pair key at write time, so the single-pair lookup
below is a Parquet predicate pushdown against row-group statistics rather
than a full scan.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_settings
from src.api.schemas import PairExplanation
from src.config import Settings
from src.contracts import RunManifest
from src.models.explanations import build_payload, explanation_feature_names

logger = logging.getLogger(__name__)

router = APIRouter(tags=["explanations"])

#: model name -> (RunManifest field holding the artifact, settings attr for
#: the model's decision threshold). The threshold is passed through to the
#: payload so the UI can draw the decision boundary on the axis.
_MODELS: dict[str, tuple[str, str | None]] = {
    "nonmatch_gate": ("gate_explanations", "gate_threshold"),
    "ml_matcher": ("ml_explanations", "ml_auto_merge_threshold"),
}


def _load_manifest(run_id: str, settings: Settings) -> RunManifest | None:
    path = settings.runs_dir / f"run_{run_id}.json"
    if not path.exists():
        return None
    try:
        return RunManifest.model_validate_json(path.read_text())
    except (json.JSONDecodeError, ValueError):
        logger.warning("Malformed run manifest: %s", path)
        return None


def _latest_run_with(field: str, settings: Settings) -> RunManifest | None:
    """Newest run whose manifest carries `field`.

    Run ids are UTC timestamps, so a reverse filename sort is chronological.
    Used when the caller does not pin a `run_id` — the reviewer UI knows the
    run behind an entity and should pass it, but a bare pair lookup still
    resolves to the most recent run that explained that model.
    """
    if not settings.runs_dir.is_dir():
        return None
    for path in sorted(settings.runs_dir.glob("run_*.json"), reverse=True):
        manifest = _load_manifest(path.stem.removeprefix("run_"), settings)
        if manifest is not None and getattr(manifest, field, None) is not None:
            return manifest
    return None


def _read_pair(artifact: Path, patid_a: str, patid_b: str) -> pd.Series | None:
    """The one explanation row for a pair, or None.

    Pairs are canonicalized (`PATID_A < PATID_B`) everywhere upstream, so the
    caller's argument order is normalized here rather than making the UI care.
    """
    a, b = (patid_a, patid_b) if patid_a < patid_b else (patid_b, patid_a)
    try:
        frame = pd.read_parquet(
            artifact, filters=[("PATID_A", "==", a), ("PATID_B", "==", b)],
        )
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable explanation artifact %s: %s", artifact, exc)
        raise HTTPException(
            status_code=503, detail="Explanation artifact could not be read."
        ) from exc
    if frame.empty:
        return None
    return frame.iloc[0]


@router.get(
    "/explanations/{model_name}/{patid_a}/{patid_b}",
    response_model=PairExplanation,
)
def get_pair_explanation(
    model_name: str,
    patid_a: str,
    patid_b: str,
    run_id: str | None = Query(
        default=None,
        description="Run whose decision to explain. Defaults to the most "
        "recent run that produced explanations for this model — pass the "
        "entity's run_id to guarantee the explanation matches the score "
        "shown beside it.",
    ),
    settings: Settings = Depends(get_settings),
) -> PairExplanation:
    """The waterfall payload for one pair under one model.

    404 when the model is unknown, the run has no explanation artifact
    (explanations disabled, or the stage was skipped), or the pair was never
    scored by that model — which is a normal outcome, not an error: pairs the
    gate dropped never reach the ML matcher, and deterministic auto-merges and
    rule rejects are never scored by either model. Those have rule provenance
    (`match_rule` / `rules_fired`) as their explanation instead.
    """
    if model_name not in _MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_name}'. Expected one of "
                   f"{sorted(_MODELS)}.",
        )
    field, threshold_attr = _MODELS[model_name]

    manifest = (
        _load_manifest(run_id, settings) if run_id is not None
        else _latest_run_with(field, settings)
    )
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=(f"No run found with explanations for '{model_name}'."
                    if run_id is None else f"Unknown run_id '{run_id}'."),
        )

    ref = getattr(manifest, field, None)
    if ref is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{manifest.run_id}' has no explanations for "
                   f"'{model_name}' (stage skipped or explanations disabled).",
        )

    artifact = settings.project_root / ref.path
    if not artifact.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Explanation artifact missing on disk: {ref.path}",
        )

    row = _read_pair(artifact, patid_a, patid_b)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Pair ({patid_a}, {patid_b}) was not scored by "
                   f"'{model_name}' in run '{manifest.run_id}'.",
        )

    payload = build_payload(
        row,
        threshold=getattr(settings, threshold_attr, None) if threshold_attr else None,
        run_id=manifest.run_id,
        model_file=row.get("model_file"),
        top_n=settings.explanation_top_n,
        feature_names=explanation_feature_names(row.to_frame().T),
    )
    return PairExplanation(**payload)


__all__ = ["router"]
