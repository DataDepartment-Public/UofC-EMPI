"""Round-2 of the embedding/graph blocking research: evaluate the q-gram blocker
against ALL THREE ground-truth sources — rules-as-truth (current method), the
silver labels, and the synthetic data — paralleling the 4-method evaluation
framework now documented in the guides.

Extends `Blocking-Research-Embedding-Graph.md` (Round 1 measured q-gram only
against rules-as-truth, which the doc flags as a *floor* for a typo-tolerant
blocker). The synthetic positives are built independently of blocking, so they
are the first source that can credit q-gram for matches the 8-block scheme — and
the rules — miss. Silver (the adjudicated 8-block output) adds a real-record
recall check that is not capped by what the rules confirm.

ANALYSIS ONLY. No production code changed. scikit-learn / scipy are installed in
the venv for this research but intentionally NOT added to the project manifest.

q-gram blocker (mirrors Round 1 §3): char-n-gram(2-4) TF-IDF identity vectors
(`first last YYYYMMDD`), candidate pairs = cosine ≥ threshold, extracted from the
sparse matrix product in row chunks (no embedding model, no ANN).

Usage:
    python scripts/research_blocking_labels.py [--skip-real] [--thresholds 0.50 0.55 0.60]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import configure_logging
from src.evaluation.blocking_eval import wide_candidate_pairs
from src.models.deterministic_rules import apply_rules
from src.preprocessing.blocking import _filter_valid_records, run_batch_blocking
from scripts.eval_against_labels import _canon, _pairset, _synthetic_records

logger = logging.getLogger(__name__)

_B8_CACHE = Path("/tmp/empi_research/B_8block.parquet")  # Round-1 cached baseline
_R_CACHE = _ROOT / "data/runs/_cache_rules_R_real.parquet"


# ── q-gram blocker ───────────────────────────────────────────────────────────────
def _identity(df: pd.DataFrame) -> pd.Series:
    f = df["FirstNM_clean"].fillna("").astype(str)
    last = df["LastNM_clean"].fillna("").astype(str)
    dob = pd.to_datetime(df["BirthDT_clean"], errors="coerce").dt.strftime("%Y%m%d")
    return (f + " " + last + " " + dob.fillna("")).str.lower().str.strip()


def qgram_pairs_multi(
    df: pd.DataFrame,
    thresholds: list[float],
    block: int = 10000,
    top_n: int = 100,
) -> dict[float, set[tuple[str, str]]]:
    """Canonical candidate pairs at several cosine thresholds in one pass.

    Builds the char-n-gram TF-IDF once, then extracts every above-threshold pair
    with a top-n sparse-cosine kernel (`sparse_dot_topn`, C++) over row blocks —
    so the dense similarity matrix is never materialized. Collects pairs at/above
    the *lowest* threshold once, then derives each threshold's set by filtering.

    `top_n` caps neighbours per record (memory bound); generous here so it does
    not truncate the true match in dense name+DOB neighbourhoods.
    """
    strings = _identity(df).to_numpy()
    patids = df["PATID"].to_numpy()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          min_df=2, lowercase=True, dtype=np.float32)
    x = vec.fit_transform(strings)            # L2-normalized rows
    xt = x.T.tocsr()
    n = x.shape[0]
    tmin = min(thresholds)
    ai, aj, av = [], [], []
    for start in range(0, n, block):
        c = sp_matmul_topn(x[start:start + block], xt, top_n=top_n,
                           threshold=float(tmin), sort=False).tocoo()
        gi = c.row + start
        keep = c.col > gi
        ai.append(gi[keep])
        aj.append(c.col[keep])
        av.append(c.data[keep])
    ai = np.concatenate(ai) if ai else np.array([], dtype=int)
    aj = np.concatenate(aj) if aj else np.array([], dtype=int)
    av = np.concatenate(av) if av else np.array([], dtype=float)

    out: dict[float, set[tuple[str, str]]] = {}
    for t in thresholds:
        m = av >= t
        pa, pb = patids[ai[m]], patids[aj[m]]
        out[t] = {(a, b) if a <= b else (b, a) for a, b in zip(pa, pb)}
    return out


# ── ground-truth assembly ────────────────────────────────────────────────────────
def _rules_R(records: pd.DataFrame) -> set[tuple[str, str]]:
    if _R_CACHE.exists():
        r = pd.read_parquet(_R_CACHE)
        return _pairset(r)
    logger.warning("Computing rules-as-truth R (slow ~2min)...")
    wide, recs = wide_candidate_pairs(records, method="loose")
    confirmed = apply_rules(wide, recs)
    confirmed[["PATID_A", "PATID_B"]].to_parquet(_R_CACHE, index=False)
    return _pairset(confirmed)


def _metrics_vs(c: set, truth_pos: set, c_all_labeled: set | None = None) -> dict:
    """PC (recall) and PQ (precision) of candidate set C against positives."""
    inter = len(c & truth_pos)
    pc = inter / len(truth_pos) if truth_pos else None
    # PQ denominator: labeled pairs in C (silver/synthetic only label a subset).
    denom = c & c_all_labeled if c_all_labeled is not None else c
    pq = inter / len(denom) if denom else None
    return {"candidates": len(c), "caught": inter,
            "PC": round(pc, 4) if pc is not None else None,
            "PQ": round(pq, 4) if pq is not None else None}


# ── real population ───────────────────────────────────────────────────────────────
def run_real(cleaned: pd.DataFrame, silver: pd.DataFrame, thresholds: list[float]) -> dict:
    records = _filter_valid_records(cleaned)
    R = _rules_R(records)
    B = _pairset(pd.read_parquet(_B8_CACHE)) if _B8_CACHE.exists() \
        else _pairset(run_batch_blocking(records))
    residual = R - B

    silver["key"] = [_canon(a, b) for a, b in zip(silver.PATID_A, silver.PATID_B)]
    silver_true = set(silver.loc[silver.silver_label, "key"])
    silver_labeled = set(silver["key"])

    def row(name: str, c: set) -> dict:
        vr = _metrics_vs(c, R)
        vs = _metrics_vs(c, silver_true, c_all_labeled=silver_labeled)
        return {
            "method": name, "candidates": len(c),
            "rules_PC": vr["PC"], "rules_PQ": vr["PQ"],
            "residual_recovered": round(len(c & residual) / len(residual), 4)
            if residual else None,
            "silver_PC": vs["PC"], "silver_PQ": vs["PQ"],
            "qgram_only_vs_8block": len(c - B),  # pairs silver/rules can't see
        }

    rows = [row("8-block baseline", B)]
    qg = qgram_pairs_multi(records, thresholds)
    for t in thresholds:
        rows.append(row(f"q-gram>={t:.2f}", qg[t]))
    return {"R": len(R), "residual": len(residual),
            "silver_true": len(silver_true), "rows": rows}


# ── synthetic population ─────────────────────────────────────────────────────────
def run_synthetic(syn: pd.DataFrame, thresholds: list[float]) -> dict:
    records = _synthetic_records(syn)
    pos = _pairset(syn[syn.label.astype(int) == 1])
    neg = _pairset(syn[syn.label.astype(int) == 0])
    B = _pairset(run_batch_blocking(records))
    missed_by_8block = pos - B

    def row(name: str, c: set) -> dict:
        return {
            "method": name, "candidates": len(c),
            "PC_recall": round(len(c & pos) / len(pos), 4),
            "8block_residual_recovered": round(
                len(c & missed_by_8block) / len(missed_by_8block), 4)
            if missed_by_8block else None,
            "hard_neg_surfaced": round(len(c & neg) / len(neg), 4),
        }

    rows = [row("8-block baseline", B)]
    qg = qgram_pairs_multi(records, thresholds)
    for t in thresholds:
        rows.append(row(f"q-gram>={t:.2f}", qg[t]))
    return {"positives": len(pos), "negatives": len(neg),
            "missed_by_8block": len(missed_by_8block), "rows": rows}


def _print(res: dict) -> None:
    if "real" in res:
        r = res["real"]
        print("=" * 96)
        print(f"  REAL POPULATION  |R(rules)|={r['R']}  residual={r['residual']}  "
              f"silver_true={r['silver_true']}")
        print("=" * 96)
        h = (f"  {'method':<18}{'cands':>9}{'rules_PC':>10}{'rules_PQ':>10}"
             f"{'resid_rec':>11}{'silver_PC':>11}{'silver_PQ':>11}{'qgram_only':>12}")
        print(h); print("  " + "-" * 92)
        for d in r["rows"]:
            print(f"  {d['method']:<18}{d['candidates']:>9}{str(d['rules_PC']):>10}"
                  f"{str(d['rules_PQ']):>10}{str(d['residual_recovered']):>11}"
                  f"{str(d['silver_PC']):>11}{str(d['silver_PQ']):>11}"
                  f"{d['qgram_only_vs_8block']:>12}")
    if "synthetic" in res:
        s = res["synthetic"]
        print("\n" + "=" * 96)
        print(f"  SYNTHETIC  positives={s['positives']}  negatives={s['negatives']}  "
              f"missed_by_8block={s['missed_by_8block']}")
        print("=" * 96)
        print(f"  {'method':<18}{'cands':>9}{'PC_recall':>11}"
              f"{'resid_rec(491)':>16}{'hard_neg_surf':>15}")
        print("  " + "-" * 68)
        for d in s["rows"]:
            print(f"  {d['method']:<18}{d['candidates']:>9}{str(d['PC_recall']):>11}"
                  f"{str(d['8block_residual_recovered']):>16}"
                  f"{str(d['hard_neg_surfaced']):>15}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path,
                    default=_ROOT / "data/processed/"
                    "MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--silver", type=Path, default=_ROOT / "data/raw/silver_labels.csv")
    ap.add_argument("--synthetic", type=Path, default=_ROOT / "data/raw/synthetic_data.csv")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.55, 0.60])
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "data/runs/research_blocking_labels.json")
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
