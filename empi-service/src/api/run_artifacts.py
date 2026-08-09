"""Resolving a run's immutable Parquet artifacts from its `RunManifest`.

Shared by the two read routes that deliberately bypass `IndexBackend`:
`routers/explanations.py` (per-pair SHAP) and `routers/cluster_pairs.py`
(per-cluster pairwise decision trace). Both serve *evidence about what a
specific run decided*, which the index — the mutable, reviewer-editable
projection — cannot answer: publishing collapses a cluster's deterministic
pairs into one best-pair evidence string and deletes the gate's drops
entirely.

The rule these helpers exist to keep is the same one `explanations.py`
documents at length: resolve the artifact through the manifest whose sha256
recorded it, never re-derive it. Promoting a new model tomorrow must not
change what the dashboard says about a decision made today.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from src.config import Settings
from src.contracts import PATID_A, PATID_B, RunManifest

logger = logging.getLogger(__name__)


def load_manifest(run_id: str, settings: Settings) -> RunManifest | None:
    """The manifest for `run_id`, or None if absent or malformed."""
    path = settings.runs_dir / f"run_{run_id}.json"
    if not path.exists():
        return None
    try:
        return RunManifest.model_validate_json(path.read_text())
    except (json.JSONDecodeError, ValueError):
        logger.warning("Malformed run manifest: %s", path)
        return None


def latest_run_with(field: str, settings: Settings) -> RunManifest | None:
    """Newest run whose manifest carries `field`.

    Run ids are UTC timestamps, so a reverse filename sort is chronological.
    Used when the caller does not pin a `run_id` — a caller that knows the run
    behind an entity should pass it, but a bare lookup still resolves to the
    most recent run that produced that artifact.
    """
    if not settings.runs_dir.is_dir():
        return None
    for path in sorted(settings.runs_dir.glob("run_*.json"), reverse=True):
        manifest = load_manifest(path.stem.removeprefix("run_"), settings)
        if manifest is not None and getattr(manifest, field, None) is not None:
            return manifest
    return None


def artifact_path(
    manifest: RunManifest, field: str, settings: Settings
) -> Path | None:
    """On-disk path for one of the manifest's artifact refs.

    None covers both "the run never produced this" (the ref is absent — a
    skipped stage, or a model that wasn't active) and "the ref is there but
    the file is gone". Callers that must distinguish those check the ref
    themselves; callers assembling a best-effort view treat both as no data.
    """
    ref = getattr(manifest, field, None)
    if ref is None:
        return None
    path = settings.project_root / ref.path
    return path if path.exists() else None


def read_pairs(artifact: Path, patids: Sequence[str]) -> pd.DataFrame:
    """Rows of a pair-keyed artifact where *both* sides are in `patids`.

    One predicate-pushdown scan per artifact rather than one lookup per pair:
    a 40-member cluster is 780 pairs, and 780 `read_parquet` calls against the
    same file would dominate the request. `PATID_B` can't go in the pushdown
    filter as an `and` alongside `PATID_A` (the pair is unordered — a row
    matches if A and B are *both* members, not if they sit in particular
    columns), so it is filtered in pandas after the read.

    Raises 503 rather than 404 on an unreadable file: the manifest says the
    artifact exists, so failing to read it is a server-side fault, not a
    missing resource. Matches `explanations.py`'s handling.
    """
    wanted = list(dict.fromkeys(patids))
    try:
        frame = pd.read_parquet(artifact, filters=[(PATID_A, "in", wanted)])
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable run artifact %s: %s", artifact, exc)
        raise HTTPException(
            status_code=503, detail="Run artifact could not be read."
        ) from exc
    if frame.empty:
        return frame
    return frame[frame[PATID_B].isin(set(wanted))]


def read_pairs_touching(artifact: Path, patids: Sequence[str]) -> pd.DataFrame:
    """Rows where *either* side is in `patids`.

    The counterpart to `read_pairs`, which requires both. This one answers
    "what was this record ever compared against", including the records that
    ended up in a different cluster — the only way to explain why a singleton
    is a singleton, since by definition it has no within-cluster pairs.

    The filter is a disjunction, which pyarrow expresses as a list of
    conjunctions (DNF): `[[A in ...], [B in ...]]` reads as A-matches OR
    B-matches. A pair where both sides are members satisfies both disjuncts
    but is still returned once — pyarrow unions row groups, it does not
    duplicate rows.
    """
    wanted = list(dict.fromkeys(patids))
    try:
        return pd.read_parquet(
            artifact, filters=[[(PATID_A, "in", wanted)], [(PATID_B, "in", wanted)]]
        )
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable run artifact %s: %s", artifact, exc)
        raise HTTPException(
            status_code=503, detail="Run artifact could not be read."
        ) from exc


def read_pair(artifact: Path, patid_a: str, patid_b: str) -> pd.Series | None:
    """The one row for a single pair, or None.

    Pairs are canonicalized (`PATID_A < PATID_B`) everywhere upstream, so the
    caller's argument order is normalized here rather than making the UI care.
    """
    a, b = (patid_a, patid_b) if patid_a < patid_b else (patid_b, patid_a)
    try:
        frame = pd.read_parquet(
            artifact, filters=[(PATID_A, "==", a), (PATID_B, "==", b)],
        )
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable run artifact %s: %s", artifact, exc)
        raise HTTPException(
            status_code=503, detail="Run artifact could not be read."
        ) from exc
    if frame.empty:
        return None
    return frame.iloc[0]


__all__ = [
    "artifact_path",
    "latest_run_with",
    "load_manifest",
    "read_pair",
    "read_pairs",
    "read_pairs_touching",
]
