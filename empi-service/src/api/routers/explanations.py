"""GET /explanations/... — per-pair SHAP waterfalls for the non-match gate
(Stage 4.25) and the ML matcher (Stage 4.5).

READ-ONLY, AND DELIBERATELY NOT BACKED BY THE INDEX
---------------------------------------------------
Every other read route goes through `IndexBackend`, because it serves the
*mutable, reviewer-editable* projection of a run. Explanations are the
opposite: they are immutable evidence about a decision a specific model made
during a specific run. So this router resolves the run's `RunManifest` and
reads the pipeline's Parquet artifact directly — the same artifact whose
sha256 the manifest records. The manifest/artifact resolution itself lives in
`src/api/run_artifacts.py`, shared with `routers/cluster_pairs.py`.

That choice is the whole point. No model is loaded in the request path,
nothing is recomputed, and promoting a new model tomorrow cannot silently
change the explanation shown for a decision made today. See
`src/models/explanations.py` for why on-demand recomputation was rejected.

The artifacts are sorted by pair key at write time, so the single-pair lookup
below is a Parquet predicate pushdown against row-group statistics rather
than a full scan.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.deps import get_settings
from src.api.run_artifacts import latest_run_with, load_manifest, read_pair
from src.api.schemas import PairExplanation
from src.config import Settings
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
        load_manifest(run_id, settings) if run_id is not None
        else latest_run_with(field, settings)
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

    row = read_pair(artifact, patid_a, patid_b)
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
