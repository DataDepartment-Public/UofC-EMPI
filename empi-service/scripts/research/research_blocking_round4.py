"""Round-4 blocking research: hub analysis, community detection, SNM, stacking.

Analysis-only studies, read-only on data/:

1. **Hub / degree-distribution analysis.** Characterize the 8-block candidate graph's
   degree distribution — find and classify high-degree "hub" nodes (placeholder-SSN
   chains, common-name+DOB clusters). Confirm whether CNP top-10 pruning tames them
   without creating misses on legitimately high-degree records.

2. **Community detection for recall.** Run connected-components and Louvain community
   detection on the 8-block graph; expand each community to all within-community pairs;
   measure recall and candidate count vs 2-hop transitive expansion and q-gram.

3. **Sorted Neighborhood Method (SNM).** Single-pass and multi-pass SNM on different
   sort keys; fixed and adaptive windows. Compare to q-gram on silver+synthetic.

4. **Meta-blocking stacking.** Run block-then-q-gram → CNP/ARCS combined; report the
   stacked recall/precision/candidate curve — the two were never combined before.

**Super-string blocking was evaluated and dropped** — the only scalable variant
(MinHash LSH) blows up on low-entropy concatenations (e.g. lastname|year|sex collide
into pathologically large buckets), and the all-pairs cosine form is the same O(N²)
class as the retired exact pass. Code retained behind `--with-superstring` for the
record, off by default.

Usage:
    uv run python scripts/research/research_blocking_round4.py [--skip-real] [--skip-synth]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import configure_logging
from src.preprocessing.blocking import _filter_valid_records, run_batch_blocking
from scripts.eval_against_labels import _canon, _pairset, _synthetic_records
from scripts.research.research_blocking_labels import _rules_R
from scripts.research.research_blocking_scalable import (
    arcs_weights,
    block_then_qgram,
    cnp_prune,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _recall(c: set, truth: set) -> float | None:
    return round(len(c & truth) / len(truth), 4) if truth else None


def _pq(c: set, truth: set) -> float | None:
    return round(len(c & truth) / len(c), 4) if c else None


def _rr(c: set, n: int) -> float:
    return round(1 - len(c) / n, 6) if n else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hub / degree-distribution analysis
# ─────────────────────────────────────────────────────────────────────────────

def hub_degree_analysis(
    b8: pd.DataFrame,
    records: pd.DataFrame,
    silver_true: set[tuple],
    R: set[tuple],
    k: int = 10,
) -> dict:
    """
    Analyze the degree distribution of the 8-block candidate graph and evaluate
    whether CNP top-k pruning tames high-degree hub nodes without losing real matches.
    """
    # Build adjacency
    adj: dict[str, set] = defaultdict(set)
    for a, b in zip(b8["PATID_A"], b8["PATID_B"]):
        adj[a].add(b)
        adj[b].add(a)

    degrees = {node: len(nbrs) for node, nbrs in adj.items()}
    deg_vals = sorted(degrees.values(), reverse=True)

    # Percentile thresholds
    deg_array = np.array(deg_vals)
    p90 = float(np.percentile(deg_array, 90))
    p95 = float(np.percentile(deg_array, 95))
    p99 = float(np.percentile(deg_array, 99))

    # Classify hubs
    high_deg_nodes = [n for n, d in degrees.items() if d >= p95]

    # Annotate hubs: check if they are placeholder-SSN chains, common-name nodes
    # We look at whether hub nodes appear on many silver-True vs silver-False pairs
    silver_true_flat: set[str] = set()
    for a, b in silver_true:
        silver_true_flat.add(a)
        silver_true_flat.add(b)

    hub_stats = []
    for node in sorted(high_deg_nodes, key=lambda n: -degrees[n])[:50]:
        nbrs = adj[node]
        pairs_in_silver_true = sum(
            1 for nb in nbrs if _canon(node, nb) in silver_true
        )
        hub_stats.append({
            "PATID": node,
            "degree": degrees[node],
            "pairs_in_silver_true": pairs_in_silver_true,
            "silver_true_rate": round(pairs_in_silver_true / len(nbrs), 3) if nbrs else 0,
        })

    # After CNP pruning, check how many hub edges survive
    weights = arcs_weights(b8)
    pruned = cnp_prune(weights, k)

    hub_pruned = {}
    for h in hub_stats[:20]:
        node = h["PATID"]
        orig_deg = h["degree"]
        surviving_edges = sum(
            1 for nb in adj[node] if _canon(node, nb) in pruned
        )
        # Silver-true edges surviving
        silver_surviving = sum(
            1 for nb in adj[node]
            if _canon(node, nb) in pruned and _canon(node, nb) in silver_true
        )
        silver_orig = h["pairs_in_silver_true"]
        hub_pruned[node] = {
            "original_degree": orig_deg,
            "surviving_edges": surviving_edges,
            "pct_edges_kept": round(surviving_edges / orig_deg, 3) if orig_deg else 0,
            "silver_true_orig": silver_orig,
            "silver_true_surviving": silver_surviving,
            "silver_recall_on_hub": round(silver_surviving / silver_orig, 3) if silver_orig else None,
        }

    # Degree distribution summary
    dist_summary = {
        "total_nodes": len(degrees),
        "total_edges": len(b8),
        "max_degree": int(deg_array.max()),
        "median_degree": float(np.median(deg_array)),
        "mean_degree": round(float(deg_array.mean()), 2),
        "p90_degree": float(p90),
        "p95_degree": float(p95),
        "p99_degree": float(p99),
        "nodes_above_p95": len(high_deg_nodes),
        "nodes_degree_1": int((deg_array == 1).sum()),
        "nodes_degree_2_5": int(((deg_array >= 2) & (deg_array <= 5)).sum()),
        "nodes_degree_6_20": int(((deg_array >= 6) & (deg_array <= 20)).sum()),
        "nodes_degree_gt20": int((deg_array > 20).sum()),
    }

    # Degree buckets for the distribution curve
    buckets = Counter()
    for d in deg_vals:
        if d == 1:
            buckets["1"] += 1
        elif d <= 5:
            buckets["2-5"] += 1
        elif d <= 10:
            buckets["6-10"] += 1
        elif d <= 20:
            buckets["11-20"] += 1
        elif d <= 50:
            buckets["21-50"] += 1
        elif d <= 100:
            buckets["51-100"] += 1
        else:
            buckets[">100"] += 1

    # Global silver recall of pruned set (sanity check)
    global_silver_recall = _recall(pruned, silver_true)
    global_silver_pq = _pq(pruned, silver_true)

    return {
        "distribution": dist_summary,
        "degree_buckets": dict(buckets),
        "top50_hubs": hub_stats,
        "hub_pruning_detail": hub_pruned,
        "cnp_top10_global": {
            "candidates": len(pruned),
            "silver_recall": global_silver_recall,
            "silver_PQ": global_silver_pq,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Community detection for recall
# ─────────────────────────────────────────────────────────────────────────────

def community_detection_recall(
    b8: pd.DataFrame,
    R: set[tuple],
    silver_true: set[tuple],
    residual: set[tuple],
) -> dict:
    """
    Run connected-components on the 8-block graph and expand to all within-community
    pairs. Compare recall/candidates to 2-hop expansion and the 8-block baseline.
    """
    import networkx as nx

    G = nx.Graph()
    for a, b in zip(b8["PATID_A"], b8["PATID_B"]):
        G.add_edge(a, b)

    # Connected components
    components = list(nx.connected_components(G))
    n_components = len(components)
    sizes = [len(c) for c in components]
    size_arr = np.array(sizes)

    # Expand: within each component, all pairs are candidates
    B_orig = _pairset(b8)
    cc_new: set[tuple] = set()
    cc_all: set[tuple] = set()
    for comp in components:
        if len(comp) >= 2:
            for a, b in combinations(sorted(comp), 2):
                p = _canon(a, b)
                cc_all.add(p)
                if p not in B_orig:
                    cc_new.add(p)

    # Louvain (via networkx >= 3.0 community module)
    try:
        from networkx.algorithms.community import louvain_communities
        louvain_parts = louvain_communities(G, seed=42)
        louvain_new: set[tuple] = set()
        louvain_all: set[tuple] = set()
        for part in louvain_parts:
            for a, b in combinations(sorted(part), 2):
                p = _canon(a, b)
                louvain_all.add(p)
                if p not in B_orig:
                    louvain_new.add(p)
        louvain_available = True
    except Exception as e:
        louvain_available = False
        louvain_new = set()
        louvain_all = set()
        louvain_parts = []
        logger.warning("Louvain unavailable: %s", e)

    component_summary = {
        "n_components": n_components,
        "singleton_components": int((size_arr == 1).sum()),
        "size_2_5": int(((size_arr >= 2) & (size_arr <= 5)).sum()),
        "size_6_20": int(((size_arr >= 6) & (size_arr <= 20)).sum()),
        "size_gt20": int((size_arr > 20).sum()),
        "max_component_size": int(size_arr.max()) if len(size_arr) else 0,
        "median_component_size": float(np.median(size_arr)) if len(size_arr) else 0,
    }

    results = {
        "component_summary": component_summary,
        "connected_components": {
            "new_candidates": len(cc_new),
            "total_candidates_in_cc_pairs": len(cc_all),
            "residual_recovered": _recall(cc_all, residual),
            "rules_PC": _recall(cc_all, R),
            "silver_PC": _recall(cc_all, silver_true),
            "silver_PQ": _pq(cc_all, silver_true),
        },
    }

    if louvain_available:
        results["louvain"] = {
            "n_communities": len(louvain_parts),
            "new_candidates": len(louvain_new),
            "total_candidates": len(louvain_all),
            "residual_recovered": _recall(louvain_all, residual),
            "rules_PC": _recall(louvain_all, R),
            "silver_PC": _recall(louvain_all, silver_true),
            "silver_PQ": _pq(louvain_all, silver_true),
        }
    else:
        results["louvain"] = {"available": False}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Super-string blocking
# ─────────────────────────────────────────────────────────────────────────────

def _build_superstring(df: pd.DataFrame, weighting: str = "equal") -> list[str]:
    """
    Build super-strings from PII fields.

    weighting:
        'equal'    — LastNM|DOB|Sex (baseline, no duplication)
        'weighted' — LastNM|LastNM|DOB|DOB|Sex (double critical fields)
        'minimal'  — LastNM|BirthYear|Sex (drop address, use birth year only)
    """
    last = df["LastNM_clean"].fillna("").astype(str).str.lower()
    first = df["FirstNM_clean"].fillna("").astype(str).str.lower()
    dob = pd.to_datetime(df["BirthDT_clean"], errors="coerce")
    dob_str = dob.dt.strftime("%Y%m%d").fillna("")
    birth_year = dob.dt.year.fillna(0).astype(int).astype(str).replace("0", "")
    sex = df.get("SexAtBirthDSC_clean", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.lower().str[:1]

    if weighting == "equal":
        return (last + "|" + dob_str + "|" + sex).tolist()
    elif weighting == "weighted":
        # Double last name and DOB to increase their q-gram weight
        return (last + "|" + last + "|" + dob_str + "|" + dob_str + "|" + first + "|" + sex).tolist()
    elif weighting == "minimal":
        # Stable fields only: last name + birth year + sex
        return (last + "|" + last + "|" + birth_year + "|" + birth_year + "|" + sex).tolist()
    raise ValueError(weighting)


def superstring_minhash_lsh(
    df: pd.DataFrame,
    thresholds: list[float],
    weighting: str = "weighted",
    num_perm: int = 64,
) -> tuple[dict[float, set], float]:
    """MinHash LSH on super-string char q-grams (datasketch).

    Signatures are built once and reused across all thresholds (each threshold only
    re-indexes the pre-built signatures into a fresh LSH table — cheap).
    """
    from datasketch import MinHash, MinHashLSH

    strings = _build_superstring(df, weighting)
    patids = [str(p) for p in df["PATID"].to_numpy()]

    # Build MinHash signatures once (the dominant cost).
    t0 = time.time()
    minhashes = []
    for s in strings:
        m = MinHash(num_perm=num_perm)
        for ng in set(_ngrams(s, 3)):
            m.update(ng.encode("utf-8"))
        minhashes.append(m)

    results: dict[float, set] = {t: set() for t in thresholds}
    for t in thresholds:
        lsh = MinHashLSH(threshold=t, num_perm=num_perm)
        with lsh.insertion_session() as session:
            for pid, m in zip(patids, minhashes):
                session.insert(pid, m)
        for pid, m in zip(patids, minhashes):
            for cpid in lsh.query(m):
                if cpid != pid:
                    results[t].add(_canon(pid, cpid))

    elapsed = time.time() - t0
    return results, elapsed


def _ngrams(s: str, n: int) -> list[str]:
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def run_superstring_experiments(
    df: pd.DataFrame,
    R: set[tuple],
    silver_true: set[tuple],
    residual: set[tuple],
    B_orig: set[tuple],
    thresholds: list[float],
) -> dict:
    """MinHash LSH only — the single super-string variant that scales (~linear).

    All-pairs cosine on the super-string is the same O(N²) cost class as the retired
    exact pass, so it is intentionally NOT run. We sweep the three weighting schemes
    (equal / weighted / minimal) — the research question (does field duplication
    correct dilution?) is answered through the LSH backend, which is shippable.
    """
    out = {}
    lsh_thresholds = [0.4, 0.5, 0.6, 0.7]

    for weighting in ("equal", "weighted", "minimal"):
        logger.info("Super-string MinHash LSH: %s", weighting)
        try:
            # Build signatures ONCE per weighting; sweep all thresholds in one call.
            sets, elapsed = superstring_minhash_lsh(df, lsh_thresholds, weighting)
            by_t = {}
            for t in lsh_thresholds:
                cands = sets[t]
                by_t[f"{t:.2f}"] = {
                    "candidates": len(cands),
                    "rules_PC": _recall(cands, R),
                    "silver_PC": _recall(cands, silver_true),
                    "silver_PQ": _pq(cands, silver_true),
                    "residual_recovered": _recall(cands, residual),
                    "qgram_only_vs_8block": len(cands - B_orig),
                }
            out[f"minhash_lsh_{weighting}"] = {"build_s": round(elapsed, 1), "by_threshold": by_t}
        except Exception as e:
            out[f"minhash_lsh_{weighting}"] = {"error": str(e)}

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sorted Neighborhood Method (SNM)
# ─────────────────────────────────────────────────────────────────────────────

def _snm_sort_key(df: pd.DataFrame, key_type: str) -> np.ndarray:
    """Build a string sort key for SNM."""
    import jellyfish

    last = df["LastNM_clean"].fillna("").astype(str)
    first = df["FirstNM_clean"].fillna("").astype(str)
    dob = pd.to_datetime(df["BirthDT_clean"], errors="coerce")
    dob_str = dob.dt.strftime("%Y%m%d").fillna("00000000")
    sex = df.get("SexAtBirthDSC_clean", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str[:1]
    zip_code = df.get("ZipCD_clean_base", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str[:5]

    if key_type == "soundex_last_dob":
        sdx = last.apply(lambda x: jellyfish.soundex(x) if x.strip() else "Z000")
        return (sdx + dob_str).to_numpy()
    elif key_type == "soundex_first_zip":
        sdx = first.apply(lambda x: jellyfish.soundex(x) if x.strip() else "Z000")
        return (sdx + zip_code).to_numpy()
    elif key_type == "dob_sex":
        return (dob_str + sex).to_numpy()
    elif key_type == "last3_dob":
        return (last.str[:3].str.lower() + dob_str).to_numpy()
    raise ValueError(key_type)


def snm_window(
    df: pd.DataFrame,
    key_type: str,
    window_sizes: list[int],
    adaptive: bool = False,
) -> dict[int | str, set]:
    """Sorted Neighborhood Method: sort on key, slide a window, pair all within window."""
    patids = df["PATID"].to_numpy()
    sort_key = _snm_sort_key(df, key_type)

    order = np.argsort(sort_key, kind="stable")
    sorted_patids = patids[order]
    sorted_keys = sort_key[order]
    n = len(sorted_patids)

    results: dict = {}

    if adaptive:
        # Adaptive window: window size = max(min_w, density_factor * local_density)
        # Density = number of records sharing the same key prefix (first 6 chars)
        prefix_counts = Counter(k[:6] for k in sorted_keys)
        pairs: set[tuple] = set()
        i = 0
        while i < n:
            prefix = sorted_keys[i][:6]
            density = prefix_counts.get(prefix, 1)
            w = max(5, min(50, density * 2))
            end = min(i + w, n)
            window = sorted_patids[i:end]
            for a, b in combinations(window, 2):
                pairs.add(_canon(a, b))
            i += 1
        results["adaptive"] = pairs
    else:
        for w in window_sizes:
            pairs: set[tuple] = set()
            for i in range(n - 1):
                end = min(i + w, n)
                window = sorted_patids[i:end]
                for j in range(1, len(window)):
                    pairs.add(_canon(window[0], window[j]))
            results[w] = pairs

    return results


def run_snm_experiments(
    df: pd.DataFrame,
    R: set[tuple],
    silver_true: set[tuple],
    residual: set[tuple],
    B_orig: set[tuple],
) -> dict:
    out = {}
    window_sizes = [10, 20, 50]

    # Single-pass experiments
    for key_type in ("soundex_last_dob", "soundex_first_zip", "dob_sex", "last3_dob"):
        logger.info("SNM single-pass: %s", key_type)
        try:
            sets = snm_window(df, key_type, window_sizes, adaptive=False)
            by_w = {}
            for w, cands in sets.items():
                by_w[str(w)] = {
                    "candidates": len(cands),
                    "rules_PC": _recall(cands, R),
                    "silver_PC": _recall(cands, silver_true),
                    "silver_PQ": _pq(cands, silver_true),
                    "residual_recovered": _recall(cands, residual),
                    "new_vs_8block": len(cands - B_orig),
                }
            out[f"single_pass_{key_type}"] = {"by_window": by_w}
        except Exception as e:
            out[f"single_pass_{key_type}"] = {"error": str(e)}

    # Multi-pass: union of multiple sort keys, w=20
    logger.info("SNM multi-pass")
    try:
        passes = {}
        for key_type in ("soundex_last_dob", "soundex_first_zip", "dob_sex"):
            s = snm_window(df, key_type, [20], adaptive=False)
            passes[key_type] = s[20]

        union_all = set().union(*passes.values())
        # Incremental value of each pass
        cumulative: set[tuple] = set()
        incremental = {}
        for key_type, cands in passes.items():
            new = cands - cumulative
            cumulative |= cands
            incremental[key_type] = {
                "pass_candidates": len(cands),
                "new_to_union": len(new),
                "union_after": len(cumulative),
                "union_rules_PC": _recall(cumulative, R),
                "union_silver_PC": _recall(cumulative, silver_true),
                "union_residual_rec": _recall(cumulative, residual),
            }
        out["multi_pass_w20"] = {
            "total_union_candidates": len(union_all),
            "rules_PC": _recall(union_all, R),
            "silver_PC": _recall(union_all, silver_true),
            "silver_PQ": _pq(union_all, silver_true),
            "residual_recovered": _recall(union_all, residual),
            "new_vs_8block": len(union_all - B_orig),
            "incremental": incremental,
        }
    except Exception as e:
        out["multi_pass_w20"] = {"error": str(e)}

    # Adaptive window
    logger.info("SNM adaptive window")
    try:
        sets = snm_window(df, "soundex_last_dob", [], adaptive=True)
        cands = sets["adaptive"]
        out["adaptive_soundex_last_dob"] = {
            "candidates": len(cands),
            "rules_PC": _recall(cands, R),
            "silver_PC": _recall(cands, silver_true),
            "silver_PQ": _pq(cands, silver_true),
            "residual_recovered": _recall(cands, residual),
            "new_vs_8block": len(cands - B_orig),
        }
    except Exception as e:
        out["adaptive_soundex_last_dob"] = {"error": str(e)}

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Meta-blocking stacking: block-then-q-gram → CNP
# ─────────────────────────────────────────────────────────────────────────────

def stacked_pipeline(
    df: pd.DataFrame,
    b8: pd.DataFrame,
    R: set[tuple],
    silver_true: set[tuple],
    residual: set[tuple],
    thresholds: list[float],
    k_values: list[int],
) -> dict:
    """
    Union of 8-block + block-then-q-gram, then apply CNP/ARCS pruning.
    Measures the combined recall/precision curve — was never done before.
    """
    B_orig = _pairset(b8)
    weights_8block = arcs_weights(b8)

    btq_sets, btq_elapsed = block_then_qgram(df, "fulldob", thresholds)

    results = {}
    for t in thresholds:
        q_set = btq_sets[t]
        union = B_orig | q_set
        new_from_q = q_set - B_orig

        # Build an edge weight for the union: 8-block pairs carry their ARCS weight;
        # q-gram-only pairs get a weight proportional to cosine sim (approximate with
        # a fixed weight of 0.5 — they appear in only one "block").
        union_weights = dict(weights_8block)
        for p in new_from_q:
            if p not in union_weights:
                union_weights[p] = 0.5  # single-signal default

        by_k = {}
        for k in k_values:
            pruned = cnp_prune(union_weights, k)
            by_k[str(k)] = {
                "candidates": len(pruned),
                "rules_PC": _recall(pruned, R),
                "silver_PC": _recall(pruned, silver_true),
                "silver_PQ": _pq(pruned, silver_true),
                "residual_recovered": _recall(pruned, residual),
                "delta_vs_8block_pct": round(100 * (len(pruned) / len(B_orig) - 1), 1),
            }

        results[f"qgram_t{int(t*100)}"] = {
            "union_candidates": len(union),
            "q_only_pairs": len(new_from_q),
            "union_rules_PC": _recall(union, R),
            "union_silver_PC": _recall(union, silver_true),
            "union_residual_rec": _recall(union, residual),
            "after_cnp": by_k,
        }

    return {"btq_elapsed_s": round(btq_elapsed, 1), "by_threshold": results}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Residual deep-dive
# ─────────────────────────────────────────────────────────────────────────────

def residual_deep_dive(
    df: pd.DataFrame,
    R: set[tuple],
    B_orig: set[tuple],
    btq_sets: dict[float, set],
    thresholds: list[float],
) -> dict:
    """
    Characterize pairs in the residual (R - B_orig) that q-gram still misses at each
    threshold. What do the hard cases look like?
    """
    residual = R - B_orig
    patid_to_idx = {pid: i for i, pid in enumerate(df["PATID"].to_numpy())}

    out = {}
    for t in thresholds:
        q_set = btq_sets[t]
        still_missed = residual - q_set
        recovered = residual & q_set

        missed_details = []
        for a, b in list(still_missed)[:200]:
            ia = patid_to_idx.get(a)
            ib = patid_to_idx.get(b)
            if ia is None or ib is None:
                continue
            ra = df.iloc[ia]
            rb = df.iloc[ib]
            last_a = str(ra.get("LastNM_clean", ""))
            last_b = str(rb.get("LastNM_clean", ""))
            first_a = str(ra.get("FirstNM_clean", ""))
            first_b = str(rb.get("FirstNM_clean", ""))
            dob_a = str(ra.get("BirthDT_clean", ""))
            dob_b = str(rb.get("BirthDT_clean", ""))
            missed_details.append({
                "LastNM_A": last_a, "LastNM_B": last_b,
                "FirstNM_A": first_a, "FirstNM_B": first_b,
                "DOB_A": dob_a, "DOB_B": dob_b,
                "same_last": last_a.lower() == last_b.lower(),
                "same_dob": dob_a == dob_b,
            })

        # Classify still-missed pairs
        n_diff_dob = sum(1 for d in missed_details if not d["same_dob"])
        n_diff_last = sum(1 for d in missed_details if not d["same_last"])
        n_same_last = sum(1 for d in missed_details if d["same_last"])

        out[f"t{int(t*100)}"] = {
            "residual_size": len(residual),
            "recovered_by_qgram": len(recovered),
            "still_missed": len(still_missed),
            "still_missed_pct": round(len(still_missed) / len(residual), 4) if residual else 0,
            "sample_analysis": {
                "n_sampled": len(missed_details),
                "different_DOB": n_diff_dob,
                "different_LastNM": n_diff_last,
                "same_LastNM_but_missed": n_same_last,
            },
            "examples": missed_details[:10],
        }

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_real(
    cleaned: pd.DataFrame,
    silver: pd.DataFrame,
    thresholds: list[float],
    k_values: list[int],
    with_superstring: bool = False,
) -> dict:
    records = _filter_valid_records(cleaned)
    R = _rules_R(records)
    b8 = run_batch_blocking(records)
    B_orig = _pairset(b8)
    residual = R - B_orig

    silver["key"] = [_canon(a, b) for a, b in zip(silver.PATID_A, silver.PATID_B)]
    silver_true = set(silver.loc[silver.silver_label, "key"])

    n_all_pairs = len(records) * (len(records) - 1) // 2

    print(f"  Records: {len(records):,}  |R|={len(R):,}  residual={len(residual)}  "
          f"silver_true={len(silver_true):,}  8-block={len(B_orig):,}")

    # 1. Hub analysis
    print("  [1] Hub / degree-distribution analysis...")
    hub = hub_degree_analysis(b8, records, silver_true, R, k=10)

    # 2. Community detection
    print("  [2] Community detection...")
    community = community_detection_recall(b8, R, silver_true, residual)

    # 3. Super-string (dropped by default — see module docstring)
    ss = {}
    if with_superstring:
        print("  [3] Super-string experiments (--with-superstring)...")
        ss = run_superstring_experiments(records, R, silver_true, residual, B_orig, thresholds)
    else:
        ss = {"status": "dropped — non-scalable (see module docstring)"}

    # 4. SNM
    print("  [3] SNM experiments...")
    snm = run_snm_experiments(records, R, silver_true, residual, B_orig)

    # 5. Stacking
    print("  [5] Meta-blocking stacking...")
    btq_sets, _ = block_then_qgram(records, "fulldob", thresholds)
    stacking = stacked_pipeline(records, b8, R, silver_true, residual, thresholds, k_values)

    # 6. Residual deep-dive
    print("  [6] Residual deep-dive...")
    deep_dive = residual_deep_dive(records, R, B_orig, btq_sets, thresholds)

    return {
        "meta": {
            "n_records": len(records),
            "R": len(R),
            "residual": len(residual),
            "silver_true": len(silver_true),
            "B_8block": len(B_orig),
            "n_all_pairs": n_all_pairs,
        },
        "hub_analysis": hub,
        "community_detection": community,
        "superstring": ss,
        "snm": snm,
        "stacking": stacking,
        "residual_deep_dive": deep_dive,
    }


def run_synthetic(syn: pd.DataFrame, thresholds: list[float]) -> dict:
    """Lighter synthetic run: SNM + super-string recall on planted positives."""
    records = _synthetic_records(syn)
    pos = _pairset(syn[syn.label.astype(int) == 1])
    B_orig = _pairset(run_batch_blocking(records))
    residual = pos - B_orig

    print(f"  Synthetic: {len(records):,} records  pos={len(pos):,}  "
          f"missed_by_8block={len(residual):,}")

    # SNM on synthetic
    print("  SNM (synthetic)...")
    snm_res = {}
    for key_type in ("soundex_last_dob", "dob_sex"):
        sets = snm_window(records, key_type, [10, 20], adaptive=False)
        snm_res[key_type] = {
            str(w): {
                "candidates": len(c),
                "recall": _recall(c, pos),
                "residual_recovered": _recall(c, residual),
            } for w, c in sets.items()
        }

    # Multi-pass SNM
    union = set().union(*[
        snm_window(records, k, [20])[20]
        for k in ("soundex_last_dob", "soundex_first_zip", "dob_sex")
    ])
    snm_res["multi_pass_w20"] = {
        "candidates": len(union),
        "recall": _recall(union, pos),
        "residual_recovered": _recall(union, residual),
    }

    return {
        "meta": {"positives": len(pos), "missed_by_8block": len(residual)},
        "snm": snm_res,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path,
                    default=_ROOT / "data/processed/MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--silver", type=Path, default=_ROOT / "data/raw/silver_labels.csv")
    ap.add_argument("--synthetic", type=Path, default=_ROOT / "data/raw/synthetic_data.csv")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.55, 0.60])
    ap.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--with-superstring", action="store_true",
                    help="Run the dropped super-string LSH experiments (non-scalable; for the record only).")
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "data/runs/research_blocking_round4.json")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    res = {}

    if not args.skip_real:
        print("\n=== REAL DATA ===")
        cleaned = pd.read_parquet(args.cleaned)
        silver = pd.read_csv(args.silver)
        res["real"] = run_real(cleaned, silver, args.thresholds, args.k_values,
                               with_superstring=args.with_superstring)

    if not args.skip_synth:
        print("\n=== SYNTHETIC DATA ===")
        syn = pd.read_csv(args.synthetic, dtype=str, keep_default_na=True, na_values=[""])
        syn["label"] = syn["label"].astype(int)
        res["synthetic"] = run_synthetic(syn, args.thresholds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
