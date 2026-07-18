"""Does FS close the gap on the *production review segment*?

The production pipeline today is: clean → 8-block blocking → deterministic rules →
three-way split (`apply_rules` + `classify_non_matches`):

    * match  — a rule fired                          → auto-merge
    * reject — ≥3 strong-identifier contradictions    → dropped
    * review — everything else (no rule, <3 conflicts) → parked for a human

The **review segment** is exactly the slot FS is meant to fill. This harness asks the
operational question directly: of the pairs the rules leave in review, **how many are
true matches the rules missed (the "gap"), and does the FS score recover them — at
what false-merge cost?** Evaluated against BOTH label sources:

    * silver labels (real 8-block candidate pairs), and
    * synthetic data (generator-true labels, with per-corruption-type breakdown).

Read-only on data/ except reports/fs/charts + the JSON. Engine: Splink 4 / DuckDB
(local, HIPAA-safe). Deps in venv only.

Usage:
    uv run python scripts/research/research_fs_review_segment.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import configure_logging
from src.models.deterministic_rules import apply_rules, classify_non_matches
from src.preprocessing.blocking import _filter_valid_records, run_batch_blocking
from scripts.eval_against_labels import _canon, _synthetic_records
from scripts.research.research_fs_splink import _add_concat, _prep, _prep_synth
from scripts.research.research_fs_splink_r3 import _auc, _by_threshold, _ece, _lookup, _train_r3

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

_CHARTS = _ROOT / "reports/fs/charts"
_OUT = _ROOT / "data/runs/research_fs_review_segment.json"


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    """Add the phonetic/year block keys _train_r3's expanded blocking needs."""
    from src.preprocessing.blocking import _dm_primary, _soundex

    df = df.copy()
    df["birth_year"] = df["dob"].str.slice(0, 4)
    df["dm_last"] = df["last_name"].map(lambda x: _dm_primary(x) if isinstance(x, str) else None)
    df["sx_last"] = df["last_name"].map(lambda x: _soundex(x) if isinstance(x, str) else None)
    df["sx_first"] = df["first_name"].map(lambda x: _soundex(x) if isinstance(x, str) else None)
    return df


def _segment_pairs(decision_col: pd.DataFrame, value: str) -> set[tuple]:
    sub = decision_col[decision_col["decision"] == value]
    return {_canon(str(a), str(b)) for a, b in zip(sub["PATID_A"], sub["PATID_B"])}


# ─────────────────────────────────────────────────────────────────────────────
# REAL — production pipeline + silver
# ─────────────────────────────────────────────────────────────────────────────

def run_real(cleaned: pd.DataFrame, silver: pd.DataFrame) -> dict:
    records = _filter_valid_records(cleaned)
    print(f"  records={len(records):,}")

    # --- production pipeline: 8-block → rules → three-way split ---
    b8 = run_batch_blocking(records)
    matches = apply_rules(b8, records)
    classified = classify_non_matches(b8, matches, records)

    match_set = {_canon(str(a), str(b)) for a, b in zip(matches["PATID_A"], matches["PATID_B"])}
    review_set = _segment_pairs(classified, "review")
    reject_set = _segment_pairs(classified, "reject")
    print(f"  match={len(match_set):,}  review={len(review_set):,}  reject={len(reject_set):,}")

    # --- silver labels ---
    silver = silver.copy()
    silver["key"] = [_canon(str(a), str(b)) for a, b in zip(silver.PATID_A, silver.PATID_B)]
    silver_true = set(silver.loc[silver.silver_label, "key"])
    silver_neg = set(silver["key"]) - silver_true

    def seg_breakdown(s: set) -> dict:
        return {
            "n": len(s),
            "silver_true": len(s & silver_true),
            "silver_false": len(s & silver_neg),
            "unlabeled": len(s - silver_true - silver_neg),
        }

    segments = {
        "match": seg_breakdown(match_set),
        "review": seg_breakdown(review_set),
        "reject": seg_breakdown(reject_set),
    }

    # --- The gap: where do the true matches live? ---
    all_true_blocked = len(silver_true)  # silver only labels blocked (8-block) pairs
    gap = {
        "silver_true_total": all_true_blocked,
        "in_match_auto_merged": len(match_set & silver_true),
        "in_review_THE_GAP": len(review_set & silver_true),
        "in_reject_false_rejects": len(reject_set & silver_true),
        "rule_recall": round(len(match_set & silver_true) / all_true_blocked, 4),
        "review_gap_pct_of_all_true": round(len(review_set & silver_true) / all_true_blocked, 4),
        "false_merges_in_match": len(match_set & silver_neg),
    }
    print(f"  GAP: {gap['in_review_THE_GAP']:,} silver-True parked in review "
          f"({gap['review_gap_pct_of_all_true']:.1%} of all true); rule recall={gap['rule_recall']:.1%}")

    # --- train FS, score the review segment ---
    df = _augment(_add_concat(_prep(records)))
    linker, preds_sdf = _train_r3(df)
    lookup = _lookup(preds_sdf.as_pandas_dataframe())

    review_pos = {p for p in review_set & silver_true}
    review_neg = {p for p in review_set & silver_neg}
    cov_pos = sum(1 for p in review_pos if p in lookup)
    cov_neg = sum(1 for p in review_neg if p in lookup)
    pos_scores = [lookup[p] for p in review_pos if p in lookup]
    neg_scores = [lookup[p] for p in review_neg if p in lookup]
    # unscored review pairs → FS sees no evidence → treat as 0 (won't be promoted)
    pos_scores_all = [lookup.get(p, 0.0) for p in review_pos]
    neg_scores_all = [lookup.get(p, 0.0) for p in review_neg]

    fs_on_review = {
        "review_silver_true": len(review_pos),
        "review_silver_false": len(review_neg),
        "coverage_pos": round(cov_pos / len(review_pos), 4) if review_pos else None,
        "coverage_neg": round(cov_neg / len(review_neg), 4) if review_neg else None,
        "roc_auc_scored": _auc(pos_scores, neg_scores),
        "ece_scored": _ece(pos_scores, neg_scores),
        "by_threshold_scored_only": _by_threshold(pos_scores, neg_scores),
        "by_threshold_incl_unscored_as_0": _by_threshold(pos_scores_all, neg_scores_all),
    }

    # --- gap closure: pipeline recall/precision before vs after FS promotes review pairs ---
    tp_rules = len(match_set & silver_true)
    fp_rules = len(match_set & silver_neg)
    gap_closure = {}
    for t in (0.5, 0.9, 0.99):
        recovered = sum(1 for s in pos_scores_all if s >= t)        # true matches rescued from review
        induced = sum(1 for s in neg_scores_all if s >= t)          # new false merges from review
        tp_after = tp_rules + recovered
        fp_after = fp_rules + induced
        gap_closure[f"fs_ge_{t}"] = {
            "review_true_recovered": recovered,
            "review_gap_closed_pct": round(recovered / len(review_pos), 4) if review_pos else None,
            "review_false_promoted": induced,
            "review_promotion_precision": round(recovered / (recovered + induced), 4) if (recovered + induced) else None,
            "pipeline_recall_before": round(tp_rules / len(silver_true), 4),
            "pipeline_recall_after": round(tp_after / len(silver_true), 4),
            "pipeline_precision_before": round(tp_rules / (tp_rules + fp_rules), 4),
            "pipeline_precision_after": round(tp_after / (tp_after + fp_after), 4),
        }

    charts = _chart_real(segments, gap, gap_closure, fs_on_review)

    return {
        "production_segments": segments,
        "the_gap": gap,
        "fs_on_review_segment": fs_on_review,
        "gap_closure": gap_closure,
        "charts": charts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC — production pipeline + generator-true labels
# ─────────────────────────────────────────────────────────────────────────────

def run_synthetic(syn: pd.DataFrame) -> dict:
    records = _synthetic_records(syn)
    print(f"  synthetic records={len(records):,}")

    b8 = run_batch_blocking(records)
    matches = apply_rules(b8, records)
    classified = classify_non_matches(b8, matches, records)
    match_set = {_canon(str(a), str(b)) for a, b in zip(matches["PATID_A"], matches["PATID_B"])}
    review_set = _segment_pairs(classified, "review")
    reject_set = _segment_pairs(classified, "reject")
    blocked = match_set | review_set | reject_set

    pos = {_canon(str(a), str(b)) for a, b, lab in
           zip(syn.PATID_A, syn.PATID_B, syn.label.astype(int)) if lab == 1}

    df = _augment(_add_concat(_prep_synth(records)))
    linker, preds_sdf = _train_r3(df)
    lookup = _lookup(preds_sdf.as_pandas_dataframe())

    review_pos = pos & review_set
    review_pos_scores = [lookup.get(p, 0.0) for p in review_pos]

    gap = {
        "positives_total": len(pos),
        "positives_blocked": len(pos & blocked),
        "in_match_auto_merged": len(pos & match_set),
        "in_review_THE_GAP": len(pos & review_set),
        "in_reject_false_rejects": len(pos & reject_set),
        "rule_recall_of_blocked": round(len(pos & match_set) / len(pos & blocked), 4) if (pos & blocked) else None,
    }
    fs_recovery = {
        f"ge_{t}": round(sum(1 for s in review_pos_scores if s >= t) / len(review_pos), 4)
        if review_pos else None
        for t in (0.5, 0.9, 0.99)
    }

    # per-case-type recovery within the review segment
    by_case = {}
    pos_df = syn[syn.label.astype(int) == 1].copy()
    pos_df["key"] = [_canon(str(a), str(b)) for a, b in zip(pos_df.PATID_A, pos_df.PATID_B)]
    for case, grp in pos_df.groupby("case_type"):
        keys = set(grp["key"]) & review_set
        if not keys:
            continue
        scores = [lookup.get(k, 0.0) for k in keys]
        by_case[case] = {
            "n_in_review": len(keys),
            "fs_recover_ge_0.9": round(sum(1 for s in scores if s >= 0.9) / len(keys), 4),
            "median_prob": round(float(np.median(scores)), 4),
        }

    print(f"  synthetic GAP: {gap['in_review_THE_GAP']:,} planted dups in review; "
          f"FS recovers @0.9 = {fs_recovery['ge_0.9']}")

    return {
        "the_gap": gap,
        "fs_recovery_in_review": fs_recovery,
        "review_recovery_by_case_type": dict(sorted(
            by_case.items(), key=lambda kv: -kv[1]["n_in_review"])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────────────────────────────────────

def _chart_real(segments: dict, gap: dict, gap_closure: dict, fs_review: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = {}

    # (1) Where the true matches sit + what FS recovers from review
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["auto-merged\n(rules)", "REVIEW\n(the gap)", "rejected\n(dropped)"]
    vals = [gap["in_match_auto_merged"], gap["in_review_THE_GAP"], gap["in_reject_false_rejects"]]
    colors = ["#2e7d32", "#e8a33d", "#c62828"]
    bars = ax1.bar(labels, vals, color=colors)
    ax1.set_title(f"Where the {gap['silver_true_total']:,} true matches land in production\n"
                  f"(rule recall {gap['rule_recall']:.0%}; "
                  f"{gap['review_gap_pct_of_all_true']:.0%} parked in review)", fontsize=10)
    ax1.set_ylabel("silver-True pairs")
    for b, v in zip(bars, vals):
        ax1.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)

    cuts = [0.5, 0.9, 0.99]
    recov = [gap_closure[f"fs_ge_{t}"]["review_gap_closed_pct"] * 100 for t in cuts]
    prec = [gap_closure[f"fs_ge_{t}"]["review_promotion_precision"] * 100 for t in cuts]
    x = np.arange(len(cuts))
    w = 0.38
    b1 = ax2.bar(x - w / 2, recov, w, label="% of review gap recovered", color="#2e7d32")
    b2 = ax2.bar(x + w / 2, prec, w, label="precision of promotions", color="#3f6fa5")
    ax2.set_xticks(x); ax2.set_xticklabels([f"FS ≥ {c}" for c in cuts])
    ax2.set_ylim(0, 100); ax2.set_ylabel("%")
    ax2.set_title("FS on the review segment:\nhow much of the gap it closes vs how clean the promotions are",
                  fontsize=10)
    ax2.legend(fontsize=8)
    for rects in (b1, b2):
        for r in rects:
            ax2.annotate(f"{r.get_height():.0f}%", (r.get_x() + r.get_width() / 2, r.get_height()),
                         ha="center", va="bottom", fontsize=8)
    fig.suptitle("Does FS close the gap on the production review segment?", fontsize=12, weight="bold")
    fig.tight_layout()
    p = _CHARTS / "v3_review_gap_closure.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    out["review_gap_closure"] = str(p.relative_to(_ROOT))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path,
                    default=_ROOT / "data/processed/MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--silver", type=Path, default=_ROOT / "data/raw/silver_labels.csv")
    ap.add_argument("--synthetic", type=Path, default=_ROOT / "data/raw/synthetic_data.csv")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    res = {}
    print("=== REAL: production review segment + FS ===")
    cleaned = pd.read_parquet(args.cleaned)
    silver = pd.read_csv(args.silver)
    res["real"] = run_real(cleaned, silver)

    if not args.skip_synth:
        print("=== SYNTHETIC: production review segment + FS ===")
        syn = pd.read_csv(args.synthetic, dtype=str, keep_default_na=True, na_values=[""])
        syn["label"] = syn["label"].astype(int)
        res["synthetic"] = run_synthetic(syn)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
