"""GET /clusters/{mid}/pairs — the pairwise decision trace behind a cluster.

WHY THIS DOESN'T COME FROM THE INDEX
------------------------------------
`GET /clusters/{mid}` answers "which records are grouped here". It cannot
answer "why", because publishing throws that away on purpose: `publish.py`
collapses every deterministic pair in a cluster into the single
highest-confidence pair's `(confidence, match_rule, evidence)` on the entity
row, and `_review_candidate_rows` deletes every pair the gate tiered
`no_match`. The pairs that were rejected, dropped, or merged by a rule other
than the winning one leave no trace in the DB at all.

They do survive in the run's immutable Parquet artifacts, which is what this
route reads — the same manifest-resolution pattern `routers/explanations.py`
uses and documents, shared via `src/api/run_artifacts.py`. Reading them costs
one predicate-pushdown scan per artifact per request; persisting the same
information would mean a new table across three backends and a republish of
every existing run, for a view a reviewer opens one cluster at a time.

WHAT IT DELIBERATELY DOESN'T DO
-------------------------------
It does not inline SHAP waterfalls. The UI expands one pair at a time and
fetches `GET /explanations/{model}/{a}/{b}` for that pair; inlining would
turn a 40-member cluster into 780 waterfalls nobody asked for.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from itertools import combinations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.backends.index_backend import IndexBackend
from src.api.deps import get_backend, get_settings
from src.api.run_artifacts import (
    artifact_path,
    latest_run_with,
    load_manifest,
    read_pairs_touching,
)
from src.api.schemas import ClusterExternalPair, ClusterPair, ClusterPairsResponse
from src.config import Settings
from src.contracts import PATID_A, PATID_B, TIER_AUTO_MERGE, TIER_NO_MATCH

logger = logging.getLogger(__name__)

router = APIRouter(tags=["clusters"])

#: The manifest fields this route reads, and the columns it keeps from each.
#: Restricting the projection matters: `cleaned`-joined frames like
#: `ml_features` carry a dozen similarity columns this view has no use for.
#: `matches_model` (FS) is deliberately *not* here: FS scores the full
#: pre-gate candidate pool, so folding it into this dict would flood
#: `external_pairs` with FS-only rows a reviewer gets no other stage detail
#: for. It's read separately, only for the clustering graph below.
_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "matches": ("match_rule", "confidence", "rules_fired", "source_blocks", "n_blocks"),
    "rejects": ("reject_rule", "n_contradictions"),
    "gate_results": ("score", "predicted_tier"),
    "matches_ml": ("score", "predicted_tier"),
    "candidate_pairs": ("source_blocks", "n_blocks"),
}

#: A pair's verdict counts as its own merge edge — the reason clustering
#: unioned its two ends needs no further explanation.
_EDGE_VERDICTS = frozenset({"auto_merge_rule", "ml_auto_merge"})

#: `entity_member.added_by` for a record the pipeline placed. Anything else is
#: a reviewer id (`audit.py` writes the `X-Reviewer-Id` value there).
_PIPELINE = "pipeline"


def _pair_index(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[tuple[str, str], dict]:
    """`{(A, B): {col: value}}` for the columns that exist on `frame`.

    Missing columns are skipped rather than raising: a run written before a
    column existed is old, not broken, and the caller already treats every
    stage field as nullable.
    """
    if frame.empty:
        return {}
    present = [c for c in columns if c in frame.columns]
    out: dict[tuple[str, str], dict] = {}
    for row in frame[[PATID_A, PATID_B, *present]].to_dict("records"):
        key = (row[PATID_A], row[PATID_B])
        out[key] = {c: _scalar(row[c]) for c in present}
    return out


def _scalar(value: object) -> object:
    """numpy/pandas scalar -> a JSON-serializable Python one, NaN -> None."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _verdict(
    *,
    in_matches: bool,
    in_rejects: bool,
    ml_tier: str | None,
    gate_tier: str | None,
    blocked: bool,
) -> str:
    """The single label for a pair, resolved most-decisive stage first.

    Order is not arbitrary. A pair a deterministic rule auto-merged was never
    scored by either model, so `auto_merge_rule` can't collide with the ML
    verdicts; but a pair the gate passed *and* the matcher scored has both a
    gate row and an ML row, and the ML decision is the one that determined
    whether an edge was formed. Reading `gate_dropped` off such a pair would
    be actively wrong, so ML is checked first and `gate_dropped` is reached
    only when the gate is where the pair stopped.
    """
    if in_matches:
        return "auto_merge_rule"
    if in_rejects:
        return "reject"
    if ml_tier is not None:
        return "ml_auto_merge" if ml_tier == TIER_AUTO_MERGE else "ml_human_review"
    if gate_tier == TIER_NO_MATCH:
        return "gate_dropped"
    return "blocked_undecided" if blocked else "not_compared"


def _build_merge_graph(
    pairs: list[ClusterPair],
    fs_edges: dict[tuple[str, str], dict],
) -> dict[str, set[str]]:
    """The same graph `assign_clusters` unions edges into, restricted to this
    cluster's members: an adjacency list over every pair that actually
    *formed* a merge — `verdict` in `_EDGE_VERDICTS`, a reviewer merge
    (`joined_by == "reviewer"`, which attaches a member to the whole entity
    rather than to one counterpart, so it fans out to every other member),
    and any row of `fs_edges` tiered `auto_merge` (the caller passes an
    empty dict unless `settings.fs_feeds_clustering` was on for this run).
    Used to find *why* a pair that formed no edge of its own still ended up
    in the same component.
    """
    graph: dict[str, set[str]] = defaultdict(set)

    def _connect(a: str, b: str) -> None:
        graph[a].add(b)
        graph[b].add(a)

    for pair in pairs:
        if pair.verdict in _EDGE_VERDICTS or pair.joined_by == "reviewer":
            _connect(pair.patid_a, pair.patid_b)

    for (a, b), row in fs_edges.items():
        if row.get("classification_tier") == TIER_AUTO_MERGE:
            _connect(a, b)

    return graph


def _shortest_path(graph: dict[str, set[str]], start: str, end: str) -> list[str] | None:
    """BFS shortest path between two members of the same connected component.

    `graph` never holds a direct `start`-`end` edge for a call site that
    matters (the caller only asks for pairs that formed no edge of their
    own), so any path found is genuinely transitive — at least one
    intermediate member. Returns `None` if the two aren't connected in
    `graph` at all, which can happen when tracing a run other than the one
    that produced the cluster's current membership.
    """
    if start not in graph or end not in graph:
        return None
    visited = {start}
    queue: deque[list[str]] = deque([[start]])
    while queue:
        path = queue.popleft()
        for neighbor in graph[path[-1]]:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None


def _counterpart(backend: IndexBackend, patid: str) -> dict:
    """Display fields + current cluster for a record outside this cluster.

    The artifacts carry PATIDs only, and "PAT004821 was rejected" tells a
    reviewer nothing they can act on — they need the name and birthdate to
    judge whether the rejection looks right.

    Resolved through the index (two small lookups per counterpart) rather
    than from the run's `cleaned` artifact, because the *current* cluster is
    part of the answer: a reviewer wants to know that the record the pipeline
    declined has since been merged somewhere else. Counterparts number in the
    single digits per cluster — blocking generated ~0.5 candidate pairs per
    record — so this stays well short of needing a bulk backend method.
    """
    other_mid = backend.get_entity_mid_for_patid(patid)
    if other_mid is None:
        return {}
    record = backend.get_entity(other_mid)
    if record is None:
        return {"mid": other_mid}
    member = next(
        (m for m in record["members"] if m["patid"] == patid), None
    )
    return {"mid": other_mid, **(member or {})}


@router.get("/clusters/{mid}/pairs", response_model=ClusterPairsResponse)
def get_cluster_pairs(
    mid: str,
    run_id: str | None = Query(
        default=None,
        description="Run whose decisions to trace. Defaults to the run that "
        "published this entity, which is what the reviewer is looking at.",
    ),
    backend: IndexBackend = Depends(get_backend),
    settings: Settings = Depends(get_settings),
) -> ClusterPairsResponse:
    """What the pipeline decided about this cluster, from two directions.

    `pairs` is every unordered pair of members — how the cluster was built.
    `external_pairs` is every comparison between a member and a record that
    ended up elsewhere — where it stopped. A singleton returns an empty
    `pairs` (nothing to compare) but can still have a full `external_pairs`
    list, which is the only available answer to "why is this patient alone?".
    """
    record = backend.get_entity(mid)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown cluster: {mid}")

    members = [m["patid"] for m in record["members"]]
    added_by = {m["patid"]: (m.get("added_by") or _PIPELINE) for m in record["members"]}
    member_ts = {m["patid"]: m.get("updated_utc") for m in record["members"]}

    thresholds = {
        "gate_threshold": float(settings.gate_threshold),
        "ml_auto_merge_threshold": float(settings.ml_auto_merge_threshold),
    }

    manifest = (
        load_manifest(run_id, settings) if run_id is not None
        else load_manifest(record["entity"]["run_id"], settings)
        or latest_run_with("clusters", settings)
    )
    if run_id is not None and manifest is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id '{run_id}'.")

    # Read each artifact once for the whole member set, keeping any row that
    # touches a member on *either* side — one pass serves both the internal
    # pairs and the external ones. A stage the run skipped, or an artifact
    # deleted from disk, yields an empty index; the pairs are still
    # enumerated, with that stage's fields null.
    indexes: dict[str, dict[tuple[str, str], dict]] = {}
    found_any = False
    for field, columns in _ARTIFACTS.items():
        path = artifact_path(manifest, field, settings) if manifest else None
        if path is None:
            indexes[field] = {}
            continue
        found_any = True
        indexes[field] = _pair_index(read_pairs_touching(path, members), columns)

    member_set = set(members)

    def _stages(key: tuple[str, str]) -> dict:
        """Every stage's row for one pair, plus the derived verdict."""
        det = indexes["matches"].get(key, {})
        rej = indexes["rejects"].get(key, {})
        gate = indexes["gate_results"].get(key, {})
        ml = indexes["matches_ml"].get(key, {})
        blk = indexes["candidate_pairs"].get(key, {})
        blocked = key in indexes["candidate_pairs"] or key in indexes["matches"]
        return {
            "verdict": _verdict(
                in_matches=bool(det),
                in_rejects=bool(rej),
                ml_tier=ml.get("predicted_tier"),
                gate_tier=gate.get("predicted_tier"),
                blocked=blocked,
            ),
            "blocked": blocked,
            "source_blocks": blk.get("source_blocks") or det.get("source_blocks"),
            "n_blocks": blk.get("n_blocks") or det.get("n_blocks"),
            "match_rule": det.get("match_rule"),
            "rules_fired": det.get("rules_fired"),
            "confidence": det.get("confidence"),
            "reject_rule": rej.get("reject_rule"),
            "n_contradictions": rej.get("n_contradictions"),
            "gate_score": gate.get("score"),
            "gate_tier": gate.get("predicted_tier"),
            "ml_score": ml.get("score"),
            "ml_tier": ml.get("predicted_tier"),
        }

    pairs: list[ClusterPair] = []
    for a, b in combinations(sorted(members), 2):
        key = (a, b)  # `sorted` already gives the canonical `PATID_A < PATID_B`

        # `added_by` is the per-record answer to "pipeline or reviewer?", so a
        # pair inherits "reviewer" from whichever side a reviewer merged in.
        # Prefer the side that names a reviewer when reporting who and when.
        reviewer_side = next(
            (p for p in (a, b) if added_by.get(p, _PIPELINE) != _PIPELINE), None
        )

        pairs.append(
            ClusterPair(
                patid_a=a,
                patid_b=b,
                **_stages(key),
                joined_by="reviewer" if reviewer_side else "pipeline",
                reviewer_id=added_by.get(reviewer_side) if reviewer_side else None,
                reviewer_ts_utc=member_ts.get(reviewer_side) if reviewer_side else None,
            )
        )

    # ── Clustering (stage 6) ────────────────────────────────────────────────
    # Every pair above that formed no merge edge of its own still shares a
    # cluster with its counterpart — so something else connected them. Walk
    # the same graph `assign_clusters` unions edges into, restricted to this
    # cluster, and record the chain of members that did. FS's `matches_model`
    # is read here rather than via `_ARTIFACTS` (see the comment on that
    # dict) and only when it can actually matter — `fs_feeds_clustering` off
    # (the default) means FS forms no clustering edges at all.
    fs_edges: dict[tuple[str, str], dict] = {}
    if settings.fs_feeds_clustering:
        fs_path = artifact_path(manifest, "matches_model", settings) if manifest else None
        if fs_path is not None:
            fs_edges = _pair_index(
                read_pairs_touching(fs_path, members), ("classification_tier",)
            )
    merge_graph = _build_merge_graph(pairs, fs_edges)
    for pair in pairs:
        if pair.verdict not in _EDGE_VERDICTS and pair.joined_by != "reviewer":
            pair.cluster_path = _shortest_path(merge_graph, pair.patid_a, pair.patid_b)

    # ── External comparisons ────────────────────────────────────────────────
    # Every pair key any artifact produced that straddles the cluster
    # boundary. Driven by the artifacts rather than by enumerating non-members
    # on purpose: the pipeline only ever considered a few records per patient,
    # and listing the other ~7,000 it never looked at would bury the handful
    # that actually matter.
    external_keys = {
        key
        for index in indexes.values()
        for key in index
        if (key[0] in member_set) != (key[1] in member_set)
    }

    external: list[ClusterExternalPair] = []
    for key in sorted(external_keys):
        member_patid, other_patid = (
            key if key[0] in member_set else (key[1], key[0])
        )
        other = _counterpart(backend, other_patid)
        external.append(
            ClusterExternalPair(
                patid_a=key[0],
                patid_b=key[1],
                **_stages(key),
                member_patid=member_patid,
                other_patid=other_patid,
                other_mid=other.get("mid"),
                other_first_name=other.get("first_name"),
                other_last_name=other.get("last_name"),
                other_birth_date=other.get("birth_date"),
                other_ssn_last4=other.get("ssn_last4"),
            )
        )

    return ClusterPairsResponse(
        mid=mid,
        run_id=manifest.run_id if manifest else None,
        artifacts_available=found_any,
        members=sorted(members),
        thresholds=thresholds,
        pairs=pairs,
        external_pairs=external,
    )


__all__ = ["router"]
