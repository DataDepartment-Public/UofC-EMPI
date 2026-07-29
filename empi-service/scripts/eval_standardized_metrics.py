"""Standardized precision/recall for blocking and deterministic rules across
all three label sources: synthetic data, silver labels, and gold labels.

Gold labels (`data/gold_labels/final_gold_labels_v1_2026_07_05.csv`) are the first hand-adjudicated,
independent ground truth available for the production candidate pair set
(same 204,805 pairs as the silver labels, schema
`PATID_A, PATID_B, ambiguous_pair, final_gold_label`). This script re-runs the
same methodology as `eval_against_labels.py` against all three sources side by
side, and reports rule precision/recall at both decision tiers:

  * auto-merge tier only (the pipeline's actual automatic-linkage decision)
  * all confirmed (auto-merge + review tiers combined, i.e. "any rule fired")

Usage:
    python scripts/eval_standardized_metrics.py \
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
from src.models.deterministic_rules import AUTO_MERGE_RULES, RULES, apply_rules
from src.preprocessing.blocking import run_batch_blocking

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[1]


def _canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pairset(df: pd.DataFrame) -> set[tuple[str, str]]:
    return {_canon(a, b) for a, b in zip(df["PATID_A"], df["PATID_B"])}


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
    }


def _rule_tiers(pairs: pd.DataFrame, cleaned: pd.DataFrame, label_col: str) -> dict:
    """Precision/recall at both auto-merge-only and all-confirmed decision tiers."""
    confirmed = apply_rules(pairs[["PATID_A", "PATID_B"]], cleaned)
    all_keys = _pairset(confirmed)
    auto_keys = _pairset(confirmed[confirmed["match_rule"].isin(AUTO_MERGE_RULES)])

    keys = [_canon(a, b) for a, b in zip(pairs["PATID_A"], pairs["PATID_B"])]
    y_true = pairs[label_col].to_numpy().astype(bool)
    y_pred_all = np.array([k in all_keys for k in keys])
    y_pred_auto = np.array([k in auto_keys for k in keys])

    rule_by_pair = {
        _canon(a, b): r
        for a, b, r in zip(confirmed["PATID_A"], confirmed["PATID_B"],
                           confirmed["match_rule"])
    }
    fired_rule = pd.Series([rule_by_pair.get(k) for k in keys])
    per_rule = {}
    for rule in RULES:
        mask = (fired_rule == rule.name).to_numpy()
        n = int(mask.sum())
        if not n:
            continue
        true_n = int(np.sum(y_true & mask))
        per_rule[rule.name] = {
            "tier": rule.tier, "fired": n, "true": true_n,
            "precision": round(true_n / n, 4),
        }

    return {
        "auto_merge_only": _prf(y_true, y_pred_auto),
        "all_confirmed": _prf(y_true, y_pred_all),
        "per_rule": per_rule,
    }


def evaluate_pair_labels(cleaned: pd.DataFrame, labeled: pd.DataFrame,
                          label_col: str, source: str) -> dict:
    """Silver/gold-shaped evaluation: labeled pairs ARE the blocking output,
    so only pair-quality (precision) is measurable for blocking, not recall."""
    n_true = int(labeled[label_col].sum())
    return {
        "source": source,
        "rules": _rule_tiers(labeled, cleaned, label_col),
        "blocking": {
            "candidate_pairs": int(len(labeled)),
            "true_pairs": n_true,
            "precision": round(n_true / len(labeled), 4),
            "recall": None,
            "note": "labels only cover the blocking output itself, so blocking "
                    "recall is not measurable from this source.",
        },
    }


def _phones_to_set(value) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {p for p in str(value).replace(",", " ").split() if p}


def _side_records(syn: pd.DataFrame, side: str) -> pd.DataFrame:
    suffix = f"_{side}"
    idcol = "PATID_A" if side == "l" else "PATID_B"
    rename = {c: c[: -len(suffix)] for c in syn.columns if c.endswith(suffix)}
    return syn[[idcol, *rename]].rename(columns={idcol: "PATID", **rename})


def _synthetic_records(syn: pd.DataFrame) -> pd.DataFrame:
    rec = pd.concat([_side_records(syn, "l"), _side_records(syn, "r")],
                     ignore_index=True)
    rec = rec.drop_duplicates(subset="PATID").reset_index(drop=True)
    rec["Phones_set"] = rec.get("Phones_set", pd.Series(dtype=object)).map(_phones_to_set)
    rec["valid_record"] = True
    return rec


def evaluate_synthetic(syn: pd.DataFrame) -> dict:
    records = _synthetic_records(syn)
    rules = _rule_tiers(syn, records, "label")

    blocked = run_batch_blocking(records)
    b_keys = _pairset(blocked)
    pos = syn[syn["label"].astype(bool)]
    neg = syn[~syn["label"].astype(bool)]
    pos_keys = _pairset(pos)
    neg_keys = _pairset(neg)
    caught = len(pos_keys & b_keys)
    recall = caught / len(pos_keys) if pos_keys else None
    # Precision proxy over the *labeled* universe only (candidate pairs blocking
    # emits that also appear in the synthetic label set, true vs false).
    labeled_keys = pos_keys | neg_keys
    emitted_labeled = labeled_keys & b_keys
    precision = (caught / len(emitted_labeled)) if emitted_labeled else None

    return {
        "source": "synthetic",
        "rules": rules,
        "blocking": {
            "candidate_pairs": int(len(blocked)),
            "true_pairs": int(len(pos_keys)),
            "recall": round(recall, 4) if recall is not None else None,
            "precision": round(precision, 4) if precision is not None else None,
            "note": "precision computed over the labeled universe (pos+neg "
                    "synthetic pairs) only, not all pairs blocking emits from "
                    "the reconstructed record set.",
        },
    }


def _print_table(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("BLOCKING — precision / recall by label source")
    print("=" * 78)
    print(f"{'Source':<12}{'Precision':<14}{'Recall':<14}Note")
    for r in results:
        b = r["blocking"]
        p = f"{b['precision']:.4f}" if b["precision"] is not None else "N/A"
        rec = f"{b['recall']:.4f}" if b["recall"] is not None else "N/A"
        print(f"{r['source']:<12}{p:<14}{rec:<14}{b.get('note', '')}")

    print("\n" + "=" * 78)
    print("DETERMINISTIC RULES — precision / recall by label source")
    print("=" * 78)
    for tier in ("auto_merge_only", "all_confirmed"):
        print(f"\n  [{tier}]")
        print(f"  {'Source':<12}{'Precision':<14}{'Recall':<14}")
        for r in results:
            m = r["rules"][tier]
            p = f"{m['precision']:.4f}" if m["precision"] is not None else "N/A"
            rec = f"{m['recall']:.4f}" if m["recall"] is not None else "N/A"
            print(f"  {r['source']:<12}{p:<14}{rec:<14}")

    print("\n" + "=" * 78)
    print("PER-RULE PRECISION by label source")
    print("=" * 78)
    rule_names = list(results[0]["rules"]["per_rule"].keys())
    print(f"{'Rule':<20}{'Tier':<12}" + "".join(f"{r['source']:<12}" for r in results))
    for name in rule_names:
        tier = ""
        cells = []
        for r in results:
            d = r["rules"]["per_rule"].get(name)
            if d:
                tier = d["tier"]
                cells.append(f"{d['precision']:.4f} ({d['fired']})")
            else:
                cells.append("N/A")
        print(f"{name:<20}{tier:<12}" + "".join(f"{c:<12}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path,
                    default=_ROOT / "data/processed/"
                    "MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--silver", type=Path,
                    default=_ROOT / "data/silver_labels/silver_labels_v1_2026_06_21.csv")
    ap.add_argument("--gold", type=Path,
                    default=_ROOT / "data/gold_labels/final_gold_labels_v1_2026_07_05.csv")
    ap.add_argument("--synthetic", type=Path,
                    default=_ROOT / "data/raw/synthetic_data.csv")
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "data/runs/eval_standardized_metrics.json")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    cleaned = pd.read_parquet(args.cleaned)
    silver = pd.read_csv(args.silver)
    gold = pd.read_csv(args.gold)
    synthetic = pd.read_csv(args.synthetic, dtype=str, keep_default_na=True,
                            na_values=[""])
    synthetic["label"] = synthetic["label"].astype(int)

    results = [
        evaluate_synthetic(synthetic),
        evaluate_pair_labels(cleaned, silver, "silver_label", "silver"),
        evaluate_pair_labels(cleaned, gold, "final_gold_label", "gold"),
    ]

    _print_table(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
