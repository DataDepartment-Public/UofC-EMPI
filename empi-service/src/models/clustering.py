"""Terminal clustering stage — stage 5 in `docs/Data-Contract.md`.

Groups confirmed matches into connected-component clusters and assigns every
valid record (including records no rule ever matched) a `cluster_id`.

PIPELINE POSITION:
    raw → clean → block → rules (`src/models/deterministic_rules.py`)
        → THIS MODULE → `contracts.ClusterAssignments`

Only auto-merge-tier matches (`AUTO_MERGE_RULES`) are treated as merge edges
today — review-tier confirmations and unconfirmed pairs stay out of automatic
clustering (see `docs/Data-Contract.md` stage 4/5). Once a probabilistic model
stage is adopted for production auto-merge, its edges union in here too.

PUBLIC API:
    assign_clusters(matches)                 -> dict[str, int]  (PATID -> cluster)
    build_cluster_assignments(matches, clean) -> pd.DataFrame    (contracts.ClusterAssignments)
"""

from __future__ import annotations

import pandas as pd

from src.contracts import PATID, PATID_A, PATID_B, VALID_RECORD

__all__ = ["assign_clusters", "build_cluster_assignments"]


def assign_clusters(matches: pd.DataFrame) -> dict[str, int]:
    """Group confirmed matches into connected-component clusters.

    Treats every confirmed pair as an undirected edge and runs union-find to
    assign each PATID a cluster id. PATIDs not appearing in `matches` are not
    included (singletons) — see `build_cluster_assignments` for the full,
    singleton-inclusive assignment. Cluster ids are deterministic: the
    smallest PATID in a component (by sort order) seeds its id ordering.

    Returns
    -------
    dict[str, int]
        PATID -> integer cluster id (ids are contiguous starting at 0).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Keep the lexicographically smaller root for deterministic output.
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        parent[hi] = lo

    for a, b in zip(matches[PATID_A], matches[PATID_B]):
        union(a, b)

    roots = sorted({find(p) for p in parent})
    root_to_id = {root: i for i, root in enumerate(roots)}
    return {patid: root_to_id[find(patid)] for patid in parent}


def build_cluster_assignments(
    matches: pd.DataFrame, cleaned: pd.DataFrame
) -> pd.DataFrame:
    """Terminal clustering output: one row per valid record, incl. singletons.

    Every `PATID` with `valid_record == True` in `cleaned` gets a cluster id —
    matched PATIDs from their connected component (`assign_clusters`),
    everyone else a fresh singleton id. Singleton ids continue the contiguous
    numbering after the matched clusters and are assigned in sorted PATID
    order, so the output is deterministic across runs.

    Returns
    -------
    pd.DataFrame
        Columns `PATID`, `cluster_id` — validate against
        `contracts.ClusterAssignments`.
    """
    valid_patids = pd.Index(
        cleaned.loc[cleaned[VALID_RECORD].astype(bool), PATID].astype(str).unique()
    )

    pair_clusters = assign_clusters(matches) if not matches.empty else {}
    assigned = pd.Series(pair_clusters, dtype="int64", name="cluster_id")
    assigned.index.name = PATID

    unassigned = sorted(valid_patids.difference(assigned.index))
    next_id = int(assigned.max()) + 1 if len(assigned) else 0
    singletons = pd.Series(
        range(next_id, next_id + len(unassigned)),
        index=unassigned,
        dtype="int64",
        name="cluster_id",
    )
    singletons.index.name = PATID

    full = pd.concat([assigned, singletons])
    return full.rename_axis(PATID).reset_index()[[PATID, "cluster_id"]]
