"""Round-3 blocking research: scalable q-gram + meta-blocking-vs-silver + graph recall.

Three analysis-only studies (no production code changed), all reusing the cached
8-block baseline, the rules-as-truth set, the silver labels, and the synthetic data:

1. **Scalable q-gram backend (block-then-q-gram).** The exact all-pairs cosine
   (`research_blocking_labels.py`) is O(N²)-ish and not shippable. Here we restrict
   the char-n-gram cosine to *within a cheap coarse block* (full DOB or birth-year)
   — sub-quadratic and production-viable — and check it reproduces the exact pass's
   recall/residual-recovery wins (§3.3 of the research doc) at a fraction of the
   compute.

2. **Meta-blocking validation vs silver.** Re-derive the CNP/ARCS node-centric
   prune of the 8-block graph and measure **silver-True retention** — the real
   (not rules) recall of the −48% candidate cut. Silver labels every 8-block pair,
   so this directly validates the "0 recall loss" claim against adjudicated labels.

3. **Graph-based recall (2-hop).** Meta-blocking can only prune; this tests whether
   graph *connectivity* can RECOVER misses: if A–B and B–C are candidate edges, add
   A–C. Measure how much of the rules residual (212) / synthetic residual (491) a
   2-hop expansion reaches, and at what candidate cost.

Usage: python scripts/research/research_blocking_scalable.py [--skip-real]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import configure_logging
from src.preprocessing.blocking import _filter_valid_records, run_batch_blocking
from scripts.eval_against_labels import _canon, _pairset, _synthetic_records
from scripts.research.research_blocking_labels import _rules_R

logger = logging.getLogger(__name__)
_B8_CACHE = Path("/tmp/empi_research/B_8block.parquet")


def _name_identity(df: pd.DataFrame) -> np.ndarray:
    f = df["FirstNM_clean"].fillna("").astype(str)
    last = df["LastNM_clean"].fillna("").astype(str)
    return (f + " " + last).str.lower().str.strip().to_numpy()


def _block_keys(df: pd.DataFrame, kind: str) -> np.ndarray:
    dob = pd.to_datetime(df["BirthDT_clean"], errors="coerce")
    if kind == "fulldob":
        return dob.dt.strftime("%Y-%m-%d").to_numpy()
    if kind == "birthyear":
        return dob.dt.year.astype("Int64").astype(str).to_numpy()
    raise ValueError(kind)


# ── 1. block-then-q-gram (scalable backend) ──────────────────────────────────────
def block_then_qgram(
    df: pd.DataFrame, block_kind: str, thresholds: list[float]
) -> tuple[dict[float, set], float]:
    """Char-n-gram cosine on names, restricted to within-block groups."""
    names = _name_identity(df)
    patids = df["PATID"].to_numpy()
    keys = _block_keys(df, block_kind)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                          lowercase=True, dtype=np.float32)
    x = vec.fit_transform(names)
    tmin = min(thresholds)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        if k and k != "<NA>" and k != "nan":
            groups[k].append(i)

    t0 = time.time()
    ai, aj, av = [], [], []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        xg = x[idx]
        sim = (xg @ xg.T).toarray()
        iu = np.triu_indices(len(idx), k=1)
        s = sim[iu]
        keep = s >= tmin
        rows = np.array(idx)[iu[0][keep]]
        cols = np.array(idx)[iu[1][keep]]
        ai.append(rows)
        aj.append(cols)
        av.append(s[keep])
    elapsed = time.time() - t0
    ai = np.concatenate(ai) if ai else np.array([], int)
    aj = np.concatenate(aj) if aj else np.array([], int)
    av = np.concatenate(av) if av else np.array([], float)

    out: dict[float, set] = {}
    for t in thresholds:
        m = av >= t
        pa, pb = patids[ai[m]], patids[aj[m]]
        out[t] = {(a, b) if a <= b else (b, a) for a, b in zip(pa, pb)}
    return out, elapsed


# ── 2. meta-blocking (CNP / ARCS) ────────────────────────────────────────────────
def arcs_weights(b8: pd.DataFrame) -> dict[tuple, float]:
    """ARCS edge weight = sum over shared blocks of 1/(comparisons in that block)."""
    comps: dict[str, int] = defaultdict(int)
    for s in b8["source_blocks"]:
        for blk in s.split("|"):
            comps[blk] += 1
    w = {}
    for a, b, s in zip(b8["PATID_A"], b8["PATID_B"], b8["source_blocks"]):
        w[_canon(a, b)] = sum(1.0 / comps[blk] for blk in s.split("|"))
    return w


def cnp_prune(weights: dict[tuple, float], k: int) -> set[tuple]:
    """Cardinality Node Pruning: keep an edge if it is in the top-k weighted edges
    of *either* endpoint (reciprocal-OR)."""
    incident: dict[str, list] = defaultdict(list)
    for (a, b), wt in weights.items():
        incident[a].append((wt, (a, b)))
        incident[b].append((wt, (a, b)))
    keep: set[tuple] = set()
    for edges in incident.values():
        edges.sort(reverse=True)
        for _, pair in edges[:k]:
            keep.add(pair)
    return keep


# ── 3. 2-hop graph expansion ─────────────────────────────────────────────────────
def two_hop(b_set: set[tuple], degree_cap: int = 40) -> set[tuple]:
    """New pairs from transitive expansion through low-degree bridge nodes."""
    adj: dict[str, set] = defaultdict(set)
    for a, b in b_set:
        adj[a].add(b)
        adj[b].add(a)
    new: set[tuple] = set()
    for nbrs in adj.values():
        if 2 <= len(nbrs) <= degree_cap:
            for a, b in combinations(sorted(nbrs), 2):
                p = (a, b) if a <= b else (b, a)
                if p not in b_set:
                    new.add(p)
    return new


# ── orchestration ────────────────────────────────────────────────────────────────
def _recall(c: set, truth: set) -> float | None:
    return round(len(c & truth) / len(truth), 4) if truth else None


def run_real(cleaned: pd.DataFrame, silver: pd.DataFrame, thresholds: list[float]) -> dict:
    records = _filter_valid_records(cleaned)
    R = _rules_R(records)
    b8 = pd.read_parquet(_B8_CACHE)
    B = _pairset(b8)
    residual = R - B
    silver["key"] = [_canon(a, b) for a, b in zip(silver.PATID_A, silver.PATID_B)]
    silver_true = set(silver.loc[silver.silver_label, "key"])

    # 1. block-then-q-gram
    btq = {}
    for kind in ("fulldob", "birthyear"):
        sets, secs = block_then_qgram(records, kind, thresholds)
        btq[kind] = {"seconds": round(secs, 1), "by_threshold": {
            f"{t:.2f}": {
                "candidates": len(sets[t]),
                "rules_PC": _recall(sets[t], R),
                "silver_PC": _recall(sets[t], silver_true),
                "residual_recovered": _recall(sets[t], residual),
            } for t in thresholds}}

    # 2. meta-blocking vs silver
    w = arcs_weights(b8)
    meta = {}
    for k in (10, 5):
        keep = cnp_prune(w, k)
        meta[f"CNP_ARCS_top{k}"] = {
            "candidates": len(keep),
            "delta_vs_B_pct": round(100 * (len(keep) / len(B) - 1), 1),
            "rules_PC": _recall(keep, R),
            "silver_PC_retained": _recall(keep, silver_true),
            "silver_PQ": round(len(keep & silver_true) / len(keep & set(silver["key"])), 4),
        }

    # 3. 2-hop graph recall
    th = two_hop(B)
    two = {
        "new_candidates": len(th),
        "residual_recovered": _recall(th, residual),
        "residual_n": len(residual),
    }
    return {"R": len(R), "residual": len(residual), "silver_true": len(silver_true),
            "block_then_qgram": btq, "meta_blocking": meta, "two_hop": two}


def run_synthetic(syn: pd.DataFrame, thresholds: list[float]) -> dict:
    records = _synthetic_records(syn)
    pos = _pairset(syn[syn.label.astype(int) == 1])
    B = _pairset(run_batch_blocking(records))
    missed = pos - B

    btq = {}
    for kind in ("fulldob", "birthyear"):
        sets, secs = block_then_qgram(records, kind, thresholds)
        btq[kind] = {"seconds": round(secs, 1), "by_threshold": {
            f"{t:.2f}": {
                "candidates": len(sets[t]),
                "recall": _recall(sets[t], pos),
                "residual_recovered": _recall(sets[t], missed),
            } for t in thresholds}}

    th = two_hop(B)
    two = {"new_candidates": len(th),
           "residual_recovered": _recall(th, missed), "residual_n": len(missed)}
    return {"positives": len(pos), "missed_by_8block": len(missed),
            "block_then_qgram": btq, "two_hop": two}


def _print(res: dict) -> None:
    if "real" in res:
        r = res["real"]
        print("=" * 80)
        print(f"  REAL  |R|={r['R']}  residual={r['residual']}  silver_true={r['silver_true']}")
        print("=" * 80)
        print("\n  [1] SCALABLE q-gram (block-then-q-gram), name-only cosine:")
        for kind, d in r["block_then_qgram"].items():
            print(f"    {kind:<10} ({d['seconds']}s):")
            for t, m in d["by_threshold"].items():
                print(f"      >={t}  cands={m['candidates']:<7} rules_PC={m['rules_PC']} "
                      f"silver_PC={m['silver_PC']} resid={m['residual_recovered']}")
        print("\n  [2] META-BLOCKING vs SILVER (real recall of the prune):")
        for name, d in r["meta_blocking"].items():
            print(f"    {name:<16} cands={d['candidates']:<7} ({d['delta_vs_B_pct']}%) "
                  f"rules_PC={d['rules_PC']} silver_PC_retained={d['silver_PC_retained']} "
                  f"silver_PQ={d['silver_PQ']}")
        print("\n  [3] 2-HOP GRAPH RECALL:")
        d = r["two_hop"]
        print(f"    new candidates={d['new_candidates']:<9} "
              f"residual recovered={d['residual_recovered']} (of {d['residual_n']})")
    if "synthetic" in res:
        s = res["synthetic"]
        print("\n" + "=" * 80)
        print(f"  SYNTHETIC  positives={s['positives']}  missed_by_8block={s['missed_by_8block']}")
        print("=" * 80)
        print("\n  [1] SCALABLE q-gram (block-then-q-gram):")
        for kind, d in s["block_then_qgram"].items():
            print(f"    {kind:<10} ({d['seconds']}s):")
            for t, m in d["by_threshold"].items():
                print(f"      >={t}  cands={m['candidates']:<7} recall={m['recall']} "
                      f"resid={m['residual_recovered']}")
        d = s["two_hop"]
        print(f"\n  [3] 2-HOP: new candidates={d['new_candidates']:<9} "
              f"residual recovered={d['residual_recovered']} (of {d['residual_n']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path, default=_ROOT / "data/processed/"
                    "MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--silver", type=Path, default=_ROOT / "data/raw/silver_labels.csv")
    ap.add_argument("--synthetic", type=Path, default=_ROOT / "data/raw/synthetic_data.csv")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.60, 0.70, 0.80])
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--out", type=Path, default=_ROOT / "data/runs/research_blocking_scalable.json")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    syn = pd.read_csv(args.synthetic, dtype=str, keep_default_na=True, na_values=[""])
    syn["label"] = syn["label"].astype(int)
    res = {"synthetic": run_synthetic(syn, args.thresholds)}
    if not args.skip_real:
        cleaned = pd.read_parquet(args.cleaned)
        silver = pd.read_csv(args.silver)
        res["real"] = run_real(cleaned, silver, args.thresholds)

    _print(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
