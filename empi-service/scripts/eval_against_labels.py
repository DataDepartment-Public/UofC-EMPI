"""Evaluate blocking + deterministic rules against external label sets.

Unlike `src/evaluation/{rule_eval,blocking_eval}.py` — which use the
deterministic rules themselves as a silver standard (high precision but circular,
and only a recall *lower bound*) — this script scores the same two stages against
two *independent* label sources:

  * **silver labels** (`data/raw/silver_labels.csv`) — the production blocking
    candidate set (204,805 pairs over the real MDM_Population) with a True/False
    `silver_label` adjudication per pair. Lets us measure rule precision AND
    recall on real records, and blocking pair-quality (PQ). It can NOT measure
    blocking recall, because it only labels pairs blocking already emitted.

  * **synthetic data** (`data/raw/synthetic_data.csv`) — 40,000 pre-constructed
    pairs (16k true duplicates with planted corruptions + 24k hard negatives),
    each carrying the cleaned `*_l` / `*_r` fields, a 0/1 `label`, and a
    `case_type`. Because the positives are generated independently of blocking,
    this set measures blocking pair-completeness (PC = recall) as well as rule
    precision/recall/F1, with a per-case-type breakdown.

Both are silver-grade, not gold (no hand-adjudicated ground truth yet) — see the
"Evaluation" section of docs/Blocking-Guide.md and docs/Deterministic-Rules-Guide.md.

Usage:
    python scripts/eval_against_labels.py \
        --cleaned data/processed/MDM_Population_cleaned_real_20260620.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from src.config import configure_logging
from src.models.deterministic_rules import RULES, apply_rules
from src.preprocessing.blocking import run_batch_blocking

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]

# Cleaned attribute columns the rules + blocking read (base names, no _l/_r).
_CLEAN_COLS = [
    "FirstNM_clean", "MiddleNM_clean", "LastNM_clean", "SuffixNM_clean",
    "BirthDT_clean", "SSN_clean", "last_4_SSN", "AddressLine1_clean",
    "AddressLine2_clean", "CityNM_clean", "StateCD_clean", "ZipCD_clean_base",
    "PrimaryPhoneNBR_clean", "Phone01NBR_clean", "Phone02NBR_clean",
    "Email_clean", "SexAtBirthDSC_clean",
]


def _canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pairset(df: pd.DataFrame) -> set[tuple[str, str]]:
    return {_canon(a, b) for a, b in zip(df["PATID_A"], df["PATID_B"])}


# ── classification metrics ──────────────────────────────────────────────────────
def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {
        "n_pairs": int(len(y_true)), "positives": int(y_true.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def _rule_eval(pairs: pd.DataFrame, cleaned: pd.DataFrame, label_col: str) -> dict:
    """Precision/recall/F1 of the deterministic rules against a labeled pair set."""
    confirmed = apply_rules(pairs[["PATID_A", "PATID_B"]], cleaned)
    confirmed_keys = _pairset(confirmed)
    rule_by_pair = {
        _canon(a, b): r
        for a, b, r in zip(confirmed["PATID_A"], confirmed["PATID_B"],
                           confirmed["match_rule"])
    }

    keys = [_canon(a, b) for a, b in zip(pairs["PATID_A"], pairs["PATID_B"])]
    y_pred = np.array([k in confirmed_keys for k in keys])
    y_true = pairs[label_col].to_numpy().astype(bool)
    out = _binary_metrics(y_true, y_pred)

    # Per-rule precision: among pairs each rule fired on, fraction truly positive.
    per_rule = {}
    fired_rule = pd.Series([rule_by_pair.get(k) for k in keys], index=pairs.index)
    for rule in RULES:
        mask = (fired_rule == rule.name).to_numpy()
        n = int(mask.sum())
        if not n:
            continue
        same = int(np.sum(y_true & mask))
        per_rule[rule.name] = {
            "fired": n, "true": same,
            "precision": round(same / n, 4),
        }
    out["per_rule"] = per_rule
    return out


# ── synthetic record reconstruction ─────────────────────────────────────────────
def _phones_to_set(value) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {p for p in str(value).replace(",", " ").split() if p}


def _side_records(syn: pd.DataFrame, side: str) -> pd.DataFrame:
    """Pull one side (_l or _r) of every synthetic pair into a record frame."""
    suffix = f"_{side}"
    idcol = "PATID_A" if side == "l" else "PATID_B"
    rename = {
        c: c[: -len(suffix)] for c in syn.columns if c.endswith(suffix)
    }
    rec = syn[[idcol, *rename]].rename(columns={idcol: "PATID", **rename})
    return rec


def _synthetic_records(syn: pd.DataFrame) -> pd.DataFrame:
    """80k-record table (both sides of every pair) for running blocking."""
    rec = pd.concat([_side_records(syn, "l"), _side_records(syn, "r")],
                    ignore_index=True)
    rec = rec.drop_duplicates(subset="PATID").reset_index(drop=True)
    rec["Phones_set"] = rec.get("Phones_set", pd.Series(dtype=object)).map(_phones_to_set)
    rec["valid_record"] = True
    return rec


# ── headline assembly ───────────────────────────────────────────────────────────
def evaluate_silver(cleaned: pd.DataFrame, silver: pd.DataFrame) -> dict:
    logger.info("Silver: %d labeled pairs (%d True).", len(silver),
                int(silver["silver_label"].sum()))
    rules = _rule_eval(silver, cleaned, "silver_label")

    # Blocking pair-quality: silver IS the blocking output, so PC is trivially
    # 1.0 and uninformative; PQ = fraction of emitted pairs that are true.
    n_true = int(silver["silver_label"].sum())
    blocking = {
        "candidate_pairs": int(len(silver)),
        "true_pairs": n_true,
        "pair_quality": round(n_true / len(silver), 4),
        "note": "silver == blocking output, so recall (PC) is not measurable "
                "from it; only pair-quality (PQ) is.",
    }
    return {"rules": rules, "blocking": blocking}


def evaluate_synthetic(syn: pd.DataFrame) -> dict:
    records = _synthetic_records(syn)
    logger.info("Synthetic: %d pairs -> %d records.", len(syn), len(records))

    rules = _rule_eval(syn, records, "label")

    # Per-case-type rule recall on the positives / fallout on the negatives.
    confirmed = apply_rules(syn[["PATID_A", "PATID_B"]], records)
    confirmed_keys = _pairset(confirmed)
    syn = syn.copy()
    syn["pred"] = [
        _canon(a, b) in confirmed_keys for a, b in zip(syn.PATID_A, syn.PATID_B)
    ]
    by_family = {}
    syn["family"] = np.where(syn["label"].astype(bool), "match", "non_match")
    for fam, grp in syn.groupby("family"):
        by_family[fam] = {
            "n": int(len(grp)),
            "rule_fired_pct": round(100 * grp["pred"].mean(), 2),
        }

    # Blocking pair-completeness (recall) on the independently-built positives.
    blocked = run_batch_blocking(records)
    b_keys = _pairset(blocked)
    pos = syn[syn["label"].astype(bool)]
    neg = syn[~syn["label"].astype(bool)]
    pos_keys = _pairset(pos)
    neg_keys = _pairset(neg)
    pc = len(pos_keys & b_keys) / len(pos_keys) if pos_keys else None
    blocking = {
        "records": int(len(records)),
        "candidate_pairs": int(len(blocked)),
        "positive_pairs": int(len(pos_keys)),
        "positives_caught": int(len(pos_keys & b_keys)),
        "pair_completeness": round(pc, 4) if pc is not None else None,
        "negatives_surfaced_pct": round(
            100 * len(neg_keys & b_keys) / len(neg_keys), 2) if neg_keys else None,
    }

    # Per-case-type blocking recall (positives only).
    pos_pc = {}
    for ct, grp in pos.groupby("case_type"):
        gk = _pairset(grp)
        pos_pc[ct] = {
            "n": int(len(gk)),
            "caught_pct": round(100 * len(gk & b_keys) / len(gk), 1),
        }
    blocking["by_case_type"] = dict(sorted(pos_pc.items()))

    return {"rules": rules, "blocking": blocking, "by_family": by_family}


def _print_report(silver_res: dict, syn_res: dict) -> None:
    def hdr(t):
        print("\n" + "=" * 66 + f"\n  {t}\n" + "=" * 66)

    hdr("SILVER LABELS  (real records — blocking output adjudicated True/False)")
    r = silver_res["rules"]
    print(f"  Rules vs silver: P={r['precision']}  R={r['recall']}  F1={r['f1']}")
    print(f"    pairs={r['n_pairs']}  positives={r['positives']}  "
          f"TP={r['TP']} FP={r['FP']} FN={r['FN']} TN={r['TN']}")
    print("    per-rule precision:")
    for name, d in r["per_rule"].items():
        print(f"      {name:<16} fired={d['fired']:<6} true={d['true']:<6} "
              f"P={d['precision']}")
    b = silver_res["blocking"]
    print(f"  Blocking pair-quality (PQ): {b['pair_quality']} "
          f"({b['true_pairs']}/{b['candidate_pairs']})")

    hdr("SYNTHETIC DATA  (planted duplicates + hard negatives, known labels)")
    r = syn_res["rules"]
    print(f"  Rules vs label:  P={r['precision']}  R={r['recall']}  F1={r['f1']}")
    print(f"    pairs={r['n_pairs']}  positives={r['positives']}  "
          f"TP={r['TP']} FP={r['FP']} FN={r['FN']} TN={r['TN']}")
    print("    per-rule precision:")
    for name, d in r["per_rule"].items():
        print(f"      {name:<16} fired={d['fired']:<6} true={d['true']:<6} "
              f"P={d['precision']}")
    print("    rule fire-rate by family:")
    for fam, d in syn_res["by_family"].items():
        print(f"      {fam:<10} n={d['n']:<6} fired={d['rule_fired_pct']}%")
    b = syn_res["blocking"]
    print(f"  Blocking pair-completeness (PC/recall): {b['pair_completeness']} "
          f"({b['positives_caught']}/{b['positive_pairs']})")
    print(f"    candidate pairs={b['candidate_pairs']}  "
          f"negatives surfaced={b['negatives_surfaced_pct']}%")
    print("    blocking recall by case_type (positives):")
    for ct, d in b["by_case_type"].items():
        print(f"      {ct:<28} n={d['n']:<5} caught={d['caught_pct']}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path,
                    default=_ROOT / "data/processed/"
                    "MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--silver", type=Path,
                    default=_ROOT / "data/raw/silver_labels.csv")
    ap.add_argument("--synthetic", type=Path,
                    default=_ROOT / "data/raw/synthetic_data.csv")
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "data/runs/eval_against_labels.json")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    cleaned = pd.read_parquet(args.cleaned)
    silver = pd.read_csv(args.silver)
    synthetic = pd.read_csv(args.synthetic, dtype=str, keep_default_na=True,
                            na_values=[""])
    synthetic["label"] = synthetic["label"].astype(int)

    silver_res = evaluate_silver(cleaned, silver)
    syn_res = evaluate_synthetic(synthetic)

    _print_report(silver_res, syn_res)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"silver": silver_res, "synthetic": syn_res}, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
