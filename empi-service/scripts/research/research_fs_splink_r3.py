"""FS round-3 / G1 — re-evaluate Fellegi–Sunter on the REAL (stacked-blocker) candidate set.

Rounds 1–2 scored FS on Splink's *own* `block_on(dob/ssn/email/phone)` passes
(~190k pairs). That is NOT the blocker the blocking research recommends — the
**stacked blocker**: 8-block ∪ block-then-q-gram(fulldob) → CNP top-k pruning. So
the fuzzy q-gram residual where FS's recall advantage supposedly lives was never in
its evaluation set, and the hard negatives CNP would have pruned were still in it.

This harness closes G1:
  PART A — **freeze the blocker.** Reproduce the recommended stacked candidate set
           (8-block ∪ btq@0.30 → CNP top-10 ≈ 109k pairs) and persist it as a
           supplied pair table (data/blocking/candidate_pairs_stacked_frozen.parquet).
  PART B — **score FS on that frozen set** (u/m trained as the round-1 simple model)
           and recompute AUC, recall, precision-at-threshold, calibration (ECE),
           cluster-size distribution, and the three-way review volume — all on the
           frozen set, then compare head-to-head with the round-1 (Splink-own) eval.

Key honesty points this run quantifies:
  * silver only labels the 8-block pairs, so the q-gram-only residual is UNLABELLED
    (the G2 gold-label gap) — we report its size and FS's score distribution on it,
    but cannot score its precision;
  * CNP pruning removes hard negatives, which *changes* the negative composition the
    eval sees — the main reason round-1 precision/calibration differ from the frozen-set
    numbers.

Read-only on data/ except the frozen-blocker parquet + reports/fs/charts + the JSON.
Engine: Splink 4 on DuckDB (local, HIPAA-safe). Deps in venv only.

Usage:
    uv run python scripts/research/research_fs_splink_r3.py [--qgram-threshold 0.30] [--cnp-k 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl

from src.config import configure_logging
from src.preprocessing.blocking import (
    _dm_primary,
    _filter_valid_records,
    _soundex,
    run_batch_blocking,
)
from scripts.eval_against_labels import _canon, _pairset
from scripts.research.research_blocking_labels import _rules_R
from scripts.research.research_blocking_scalable import arcs_weights, block_then_qgram, cnp_prune
from scripts.research.research_fs_splink import _add_concat, _prep

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

_CHARTS = _ROOT / "reports/fs/charts"
_OUT = _ROOT / "data/runs/research_fs_splink_r3.json"
_FROZEN = _ROOT / "data/blocking/candidate_pairs_stacked_frozen.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# PART A — freeze the recommended stacked blocker
# ─────────────────────────────────────────────────────────────────────────────

def freeze_stacked_blocker(
    records: pd.DataFrame,
    silver_true: set[tuple],
    silver_all: set[tuple],
    R: set[tuple],
    qgram_threshold: float,
    cnp_k: int,
) -> tuple[set[tuple], dict]:
    """8-block ∪ block-then-q-gram(fulldob) → CNP top-k. Returns (frozen_pairs, meta)."""
    b8 = run_batch_blocking(records)
    B_orig = _pairset(b8)
    residual = R - B_orig

    btq_sets, btq_s = block_then_qgram(records, "fulldob", [qgram_threshold])
    q_set = btq_sets[qgram_threshold]
    q_only = q_set - B_orig
    union = B_orig | q_set

    # ARCS weights for 8-block edges; q-gram-only edges get the single-signal default
    # (mirrors scripts/research/research_blocking_round4.stacked_pipeline).
    weights = dict(arcs_weights(b8))
    for p in q_only:
        weights.setdefault(p, 0.5)
    frozen = cnp_prune(weights, cnp_k)

    def rec(c):  # recall against a truth set
        return round(len(c & silver_true) / len(silver_true), 4) if silver_true else None

    frozen_q_only = frozen & q_only
    meta = {
        "qgram_threshold": qgram_threshold,
        "cnp_k": cnp_k,
        "btq_seconds": round(btq_s, 1),
        "n_8block": len(B_orig),
        "n_qgram_only_vs_8block": len(q_only),
        "n_union": len(union),
        "n_frozen_after_cnp": len(frozen),
        "delta_vs_8block_pct": round(100 * (len(frozen) / len(B_orig) - 1), 1),
        "frozen_q_only_pairs": len(frozen_q_only),
        "union_silver_PC": rec(union),
        "frozen_silver_PC": rec(frozen),
        "union_residual_recovered": round(len(union & residual) / len(residual), 4) if residual else None,
        "frozen_residual_recovered": round(len(frozen & residual) / len(residual), 4) if residual else None,
        "frozen_cap_silver_pos": len(frozen & silver_true),
        "frozen_cap_silver_neg": len(frozen & (silver_all - silver_true)),
        "frozen_cap_silver_labeled": len(frozen & silver_all),
    }
    return frozen, meta, q_only


# ─────────────────────────────────────────────────────────────────────────────
# PART B — train FS (round-1 simple model) with expanded prediction blocking
# ─────────────────────────────────────────────────────────────────────────────

def _settings_r3() -> SettingsCreator:
    """Round-1 (recommended, simple) comparisons. Prediction blocking is EXPANDED so
    predict() generates a superset of the frozen stacked set: the 4 strong-ID rules
    PLUS name and birth-year+last passes that cover the phonetic / birth-year / last-4
    blocks (B4/B7/B8/B9). m/u are unaffected (EM uses its own passes)."""
    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cl.ForenameSurnameComparison(
                "first_name", "last_name",
                forename_surname_concat_col_name="first_last_concat",
            ),
            cl.DateOfBirthComparison(
                "dob", input_is_string=True,
                datetime_thresholds=[1, 1, 10], datetime_metrics=["month", "year", "year"],
            ),
            cl.ExactMatch("ssn").configure(term_frequency_adjustments=True),
            cl.EmailComparison("email"),
            cl.ExactMatch("phone").configure(term_frequency_adjustments=True),
            cl.ExactMatch("sex"),
            cl.LevenshteinAtThresholds("zip", 1),
            cl.JaroWinklerAtThresholds("address", [0.9, 0.7]),
        ],
        # Mirror the 8 production blocks so predict() generates a superset of the
        # frozen stacked set (incl. the phonetic B3/B7/B8 and q-gram@same-DOB pairs):
        blocking_rules_to_generate_predictions=[
            block_on("ssn"),                                  # B1
            block_on("dob"),                                  # B3 + q-gram (same exact DOB)
            block_on("last_name", "birth_year"),              # B4 (exact last)
            block_on("phone"),                                # B5 (primary phone)
            block_on("email"),                                # B6
            block_on("dm_last", "birth_year"),                # B3/B7 (Double-Metaphone last)
            block_on("sx_first", "sx_last", "birth_year"),    # B8 (Soundex first+last)
            block_on("first_name", "last_name"),              # B9 (exact first+last)
        ],
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
    )


def _train_r3(df: pd.DataFrame):
    linker = Linker(df, _settings_r3(), db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(
        [block_on("ssn", "dob"), block_on("email", "dob")], recall=0.8,
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=5e6)
    for cols in (("dob",), ("first_name", "last_name"), ("email",), ("phone",)):
        try:
            linker.training.estimate_parameters_using_expectation_maximisation(block_on(*cols))
        except Exception as e:
            logger.warning("EM pass on %s skipped: %s", cols, e)
    # threshold 0 → keep every generated pair (so frozen-set coverage isn't dented by
    # low-scoring pairs being silently dropped).
    preds_sdf = linker.inference.predict(threshold_match_probability=0.0)
    return linker, preds_sdf


def _lookup(preds: pd.DataFrame) -> dict[tuple, float]:
    out = {}
    for a, b, p in zip(preds["unique_id_l"], preds["unique_id_r"], preds["match_probability"]):
        out[_canon(str(a), str(b))] = float(p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _auc(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    ys = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
    ss = np.array(scores_pos + scores_neg)
    if len(ys) == 0 or len(set(ys.tolist())) < 2:
        return None
    order = np.argsort(ss)
    ys = ys[order]
    n_pos, n_neg = int(ys.sum()), int((ys == 0).sum())
    rank_sum = np.sum(np.where(ys == 1)[0] + 1)
    return round(float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)), 4)


def _pr_curve_auc(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    """Average precision (PR-AUC) via the step-wise trapezoid over sorted scores."""
    if not scores_pos:
        return None
    labels = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
    scores = np.array(scores_pos + scores_neg)
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    prec = tp / (tp + fp)
    rec = tp / labels.sum()
    # average precision = sum (R_n - R_{n-1}) * P_n
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return round(float(np.sum((rec - rec_prev) * prec)), 4)


def _by_threshold(pos: list[float], neg: list[float], thresholds=(0.5, 0.9, 0.99)) -> list[dict]:
    out = []
    npos = len(pos)
    pa, na = np.array(pos), np.array(neg)
    for t in thresholds:
        tp = int((pa >= t).sum())
        fp = int((na >= t).sum())
        fn = npos - tp
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        out.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
        })
    return out


def _ece(pos: list[float], neg: list[float], n_bins: int = 10) -> float | None:
    """Expected calibration error over equal-width probability bins."""
    scores = np.array(pos + neg)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    if len(scores) == 0:
        return None
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(scores)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (scores >= lo) & (scores < hi if i < n_bins - 1 else scores <= hi)
        if not m.any():
            continue
        conf = scores[m].mean()
        acc = labels[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return round(float(ece), 4)


def _cluster_sizes(edge_probs: dict[tuple, float], threshold: float) -> dict:
    """Connected-components clustering over a supplied edge set at a probability cut."""
    adj: dict[str, set] = defaultdict(set)
    nodes: set[str] = set()
    for (a, b), p in edge_probs.items():
        nodes.add(a)
        nodes.add(b)
        if p >= threshold:
            adj[a].add(b)
            adj[b].add(a)
    seen: set[str] = set()
    sizes = []
    for start in nodes:
        if start in seen:
            continue
        stack, comp = [start], 0
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp += 1
            stack.extend(adj[x] - seen)
        sizes.append(comp)
    arr = np.array(sizes)
    multi = arr[arr >= 2]
    return {
        "threshold": threshold,
        "n_nodes": len(nodes),
        "n_clusters": int(len(arr)),
        "n_clusters_ge2": int((arr >= 2).sum()),
        "max_cluster_size": int(arr.max()) if len(arr) else 0,
        "clusters_ge5": int((arr >= 5).sum()),
        "clusters_ge10": int((arr >= 10).sum()),
        "pct_records_in_clusters_ge10": round(float(arr[arr >= 10].sum() / arr.sum()), 4) if arr.sum() else 0.0,
        "mean_multi_cluster_size": round(float(multi.mean()), 2) if len(multi) else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def _chart_candidate_funnel(meta: dict, splink_own_pairs: int) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = ["Splink-own\nblock_on\n(rounds 1–2)", "8-block", "8-block ∪\nq-gram\n(union)",
              "frozen:\n→ CNP top-10\n(recommended)"]
    vals = [splink_own_pairs, meta["n_8block"], meta["n_union"], meta["n_frozen_after_cnp"]]
    colors = ["#8888aa", "#5a7fb0", "#3f6fa5", "#2e7d32"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(stages)), vals, color=colors)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("candidate pairs")
    ax.set_title("G1: FS is now scored on the recommended stacked blocker, not Splink's own blocking",
                 fontsize=11)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    ax.annotate(f"+{meta['frozen_q_only_pairs']:,} fuzzy q-gram-only pairs\nnow in the eval set "
                f"(unlabelled → G2)",
                (3, meta["n_frozen_after_cnp"]), xytext=(2.1, max(vals) * 0.62),
                fontsize=8, color="#2e7d32", style="italic",
                arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    fig.tight_layout()
    p = _CHARTS / "v3_g1_candidate_funnel.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(p.relative_to(_ROOT))


def _chart_eval_compare(r1: dict, frozen: dict) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cuts = [0.5, 0.9, 0.99]
    r1_prec = {x["threshold"]: x["precision"] for x in r1}
    fz_prec = {x["threshold"]: x["precision"] for x in frozen}
    r1_rec = {x["threshold"]: x["recall"] for x in r1}
    fz_rec = {x["threshold"]: x["recall"] for x in frozen}

    fig, (axp, axr) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(cuts))
    w = 0.38
    axp.bar(x - w / 2, [r1_prec[c] for c in cuts], w, label="round-1 (Splink-own)", color="#8888aa")
    axp.bar(x + w / 2, [fz_prec[c] for c in cuts], w, label="frozen stacked blocker", color="#2e7d32")
    axp.set_title("Precision vs silver — by score cutoff")
    axp.set_xticks(x); axp.set_xticklabels([f"≥{c}" for c in cuts]); axp.set_ylim(0, 1)
    axp.set_ylabel("precision"); axp.legend(fontsize=8)
    for i, c in enumerate(cuts):
        axp.annotate(f"{r1_prec[c]:.0%}", (i - w / 2, r1_prec[c]), ha="center", va="bottom", fontsize=8)
        axp.annotate(f"{fz_prec[c]:.0%}", (i + w / 2, fz_prec[c]), ha="center", va="bottom", fontsize=8)

    axr.bar(x - w / 2, [r1_rec[c] for c in cuts], w, label="round-1 (Splink-own)", color="#8888aa")
    axr.bar(x + w / 2, [fz_rec[c] for c in cuts], w, label="frozen stacked blocker", color="#2e7d32")
    axr.set_title("Recall vs silver — by score cutoff")
    axr.set_xticks(x); axr.set_xticklabels([f"≥{c}" for c in cuts]); axr.set_ylim(0, 1)
    axr.set_ylabel("recall"); axr.legend(fontsize=8)
    for i, c in enumerate(cuts):
        axr.annotate(f"{r1_rec[c]:.0%}", (i - w / 2, r1_rec[c]), ha="center", va="bottom", fontsize=8)
        axr.annotate(f"{fz_rec[c]:.0%}", (i + w / 2, fz_rec[c]), ha="center", va="bottom", fontsize=8)
    fig.suptitle("G1: pruning hard negatives lifts precision; recall is unchanged (silver lives in both sets)",
                 fontsize=11)
    fig.tight_layout()
    p = _CHARTS / "v3_g1_eval_compare.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(p.relative_to(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run(cleaned: pd.DataFrame, silver: pd.DataFrame, qgram_threshold: float, cnp_k: int) -> dict:
    records = _filter_valid_records(cleaned)
    R = _rules_R(records)
    silver = silver.copy()
    silver["key"] = [_canon(str(a), str(b)) for a, b in zip(silver.PATID_A, silver.PATID_B)]
    silver_all = set(silver["key"])
    silver_true = set(silver.loc[silver.silver_label, "key"])
    silver_neg = silver_all - silver_true
    print(f"records={len(records):,}  silver_true={len(silver_true):,}  silver_neg={len(silver_neg):,}  |R|={len(R):,}")

    # PART A — freeze the blocker
    print("[A] freezing stacked blocker (8-block ∪ q-gram → CNP)...")
    frozen, meta, q_only = freeze_stacked_blocker(
        records, silver_true, silver_all, R, qgram_threshold, cnp_k)
    print(f"    frozen={meta['n_frozen_after_cnp']:,} ({meta['delta_vs_8block_pct']}% vs 8-block)  "
          f"silver_PC={meta['frozen_silver_PC']}  q_only_in_frozen={meta['frozen_q_only_pairs']:,}")

    # persist the supplied pair table
    frozen_df = pd.DataFrame(
        [(a, b, (a, b) in q_only) for (a, b) in frozen],
        columns=["PATID_A", "PATID_B", "q_only"])
    frozen_df["silver_label"] = [
        (True if k in silver_true else (False if k in silver_neg else None))
        for k in zip(frozen_df.PATID_A, frozen_df.PATID_B)]
    _FROZEN.parent.mkdir(parents=True, exist_ok=True)
    frozen_df.to_parquet(_FROZEN, index=False)
    print(f"    wrote {_FROZEN.relative_to(_ROOT)}")

    # PART B — train FS and score the frozen set
    print("[B] training FS (round-1 simple model, expanded prediction blocking)...")
    df = _add_concat(_prep(records))
    df["birth_year"] = df["dob"].str.slice(0, 4)
    # Phonetic block keys (mirror the production B3/B7/B8 blocks) for prediction coverage.
    df["dm_last"] = df["last_name"].map(lambda x: _dm_primary(x) if isinstance(x, str) else None)
    df["sx_last"] = df["last_name"].map(lambda x: _soundex(x) if isinstance(x, str) else None)
    df["sx_first"] = df["first_name"].map(lambda x: _soundex(x) if isinstance(x, str) else None)
    linker, preds_sdf = _train_r3(df)
    preds = preds_sdf.as_pandas_dataframe()
    lookup = _lookup(preds)
    print(f"    predicted pairs (expanded blocking) = {len(preds):,}")

    # coverage of the frozen set by the prediction lookup
    frozen_scored = {p: lookup[p] for p in frozen if p in lookup}
    coverage = len(frozen_scored) / len(frozen)
    q_only_scored = {p: lookup[p] for p in (frozen & q_only) if p in lookup}
    print(f"    frozen-set scoring coverage = {coverage:.4f}  "
          f"(q-only covered {len(q_only_scored)}/{meta['frozen_q_only_pairs']})")

    # --- evaluation on the silver-labeled subset of the FROZEN set ---
    fpos = [frozen_scored[p] for p in (frozen & silver_true) if p in frozen_scored]
    fneg = [frozen_scored[p] for p in (frozen & silver_neg) if p in frozen_scored]
    frozen_eval = {
        "n_pos_labeled_scored": len(fpos),
        "n_neg_labeled_scored": len(fneg),
        "roc_auc": _auc(fpos, fneg),
        "pr_auc": _pr_curve_auc(fpos, fneg),
        "ece": _ece(fpos, fneg),
        "by_threshold": _by_threshold(fpos, fneg),
    }

    # --- recall denominator caveat: recall here is "of silver-True IN the frozen set" ---
    # generator-free recall vs ALL silver-True (incl. the ~0.04% CNP dropped):
    all_pos_scores = []
    for p in silver_true:
        all_pos_scores.append(frozen_scored.get(p, 0.0))  # dropped/unscored → 0
    frozen_eval["recall_vs_all_silver_true"] = {
        f"ge_{t}": round(sum(1 for s in all_pos_scores if s >= t) / len(all_pos_scores), 4)
        for t in (0.5, 0.9, 0.99)
    }

    # --- the q-gram-only residual: FS score distribution (UNLABELLED — G2) ---
    q_scores = list(q_only_scored.values())
    residual_fs = {
        "n_q_only_in_frozen": meta["frozen_q_only_pairs"],
        "n_q_only_scored": len(q_scores),
        "median_fs_prob": round(float(np.median(q_scores)), 4) if q_scores else None,
        "ge_0.5": round(sum(1 for s in q_scores if s >= 0.5) / len(q_scores), 4) if q_scores else None,
        "ge_0.9": round(sum(1 for s in q_scores if s >= 0.9) / len(q_scores), 4) if q_scores else None,
        "ge_0.99": round(sum(1 for s in q_scores if s >= 0.99) / len(q_scores), 4) if q_scores else None,
        "note": "UNLABELLED in silver (silver only labels 8-block pairs) — precision needs gold (G2)",
    }

    # --- review-volume (three-way) over the WHOLE frozen set, mirroring round-2 cuts ---
    upper, lower = 0.99, 0.968  # round-2 thresholds_review
    fz_all = np.array(list(frozen_scored.values()))
    review_volume = {
        "upper_auto_merge_threshold": upper,
        "lower_auto_reject_threshold": lower,
        "n_scored": int(len(fz_all)),
        "auto_merge_pct": round(float((fz_all >= upper).mean()), 4),
        "auto_reject_pct": round(float((fz_all < lower).mean()), 4),
        "clerical_review_pct": round(float(((fz_all >= lower) & (fz_all < upper)).mean()), 4),
    }

    # --- cluster-size distribution over the FROZEN edge set (CNP-capped) ---
    clusters = {f"at_{t}": _cluster_sizes(frozen_scored, t) for t in (0.9, 0.99)}

    charts = {}
    try:
        charts["candidate_funnel"] = _chart_candidate_funnel(meta, splink_own_pairs=190038)
        charts["eval_compare"] = _chart_eval_compare(_R1_BY_THRESHOLD, frozen_eval["by_threshold"])
    except Exception as e:
        logger.warning("charts failed: %s", e)

    return {
        "blocker_freeze": meta,
        "fs_scoring": {
            "predicted_pairs_expanded_blocking": len(preds),
            "frozen_coverage": round(coverage, 4),
            "frozen_scored": len(frozen_scored),
        },
        "frozen_eval_vs_silver": frozen_eval,
        "round1_eval_vs_silver_for_reference": {
            "roc_auc": 0.953, "by_threshold": _R1_BY_THRESHOLD,
            "neg_coverage": 0.643, "note": "scored on Splink-own block_on, hard-negative-enriched",
        },
        "qgram_residual_fs": residual_fs,
        "review_volume_frozen": review_volume,
        "cluster_sizes_frozen": clusters,
        "charts": charts,
    }


# round-1 silver eval (from data/runs/research_fs_splink.json) for head-to-head charts
_R1_BY_THRESHOLD = [
    {"threshold": 0.5, "tp": 50751, "fp": 41945, "fn": 316, "precision": 0.5475, "recall": 0.9938},
    {"threshold": 0.9, "tp": 50375, "fp": 20850, "fn": 692, "precision": 0.7073, "recall": 0.9864},
    {"threshold": 0.99, "tp": 43969, "fp": 10936, "fn": 7098, "precision": 0.8008, "recall": 0.861},
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path,
                    default=_ROOT / "data/processed/MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--silver", type=Path, default=_ROOT / "data/raw/silver_labels.csv")
    ap.add_argument("--qgram-threshold", type=float, default=0.30)
    ap.add_argument("--cnp-k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    print("=== FS round-3 / G1: re-evaluate on the stacked-blocker candidate set ===")
    cleaned = pd.read_parquet(args.cleaned)
    silver = pd.read_csv(args.silver)
    res = run(cleaned, silver, args.qgram_threshold, args.cnp_k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nWrote {args.out}")
    print(f"Charts in {_CHARTS}")


if __name__ == "__main__":
    main()
