"""Fellegi–Sunter research — ROUND 2: model improvements + deeper analysis (Splink).

Builds on `scripts/research_fs_splink.py` (round 1). **Analysis only.** Trains an
improved **v2** model and compares it to the round-1 **v1** baseline, then runs the
deeper analyses the first pass left open. Writes JSON + new charts; the synthesis goes
into `Probabilistic-Matching-FS.md`.

Improvements in v2 (vs v1):
  • term-frequency on NAME (the precision lever — false merges are common-surname collisions)
  • term-frequency on DOB (down-weight common/placeholder birthdates)
  • multi-valued phone via ArrayIntersect on Phones_set (vs single PrimaryPhone)
  • composite street+city+zip address comparison (vs Jaro-Winkler on AddressLine1 only)

Analyses:
  1. False-merge rescue — does name-TF push the 7,578 rule-false merges below threshold?
  2. v1 vs v2 head-to-head (silver AUC, thresholds, hybrid).
  3. Probability calibration curve + ECE (v2).
  4. FS error analysis (Splink prediction_errors).
  5. Field ablation (drop each comparison, AUC delta).
  6. Missingness stratification (accuracy by which strong-IDs are present).
  7. Cluster-level over-merge check (v1 vs v2 cluster-size distribution + graph metrics).
  8. EM seed-stability (weights + AUC across seeds).
  9. Conditional-dependence check (correlation of field agreements among matches).
 10. Formal 3-way thresholds + clerical-review volume.
 11. Active-learning pair selection for gold (uncertain pairs).
 12. Online incremental scoring latency (find_matches_to_new_records).

Usage:
    uv run python scripts/research_fs_splink_r2.py [--sample N] [--ablation-sample N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl

from src.config import configure_logging
from src.preprocessing.blocking import _filter_valid_records
from src.models.deterministic_rules import apply_rules
from scripts.eval_against_labels import _canon
from scripts.research_fs_splink import (
    _prep, _add_concat, _pred_lookup, _score_against_labels, _auc,
    _hybrid_analysis, _model_summary, _save,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

_CHARTS = _ROOT / "reports/fs/charts"
_OUT = _ROOT / "data/runs/research_fs_splink_r2.json"


# ─────────────────────────────────────────────────────────────────────────────
# Data prep — add the v2 columns (phone array, composite address)
# ─────────────────────────────────────────────────────────────────────────────

def _prep_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)            # avoid index-misalignment on assignment
    out = _add_concat(_prep(df)).reset_index(drop=True)
    # multi-valued phone → python list (Phones_set is a numpy ndarray)
    def _phones(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return []
        if isinstance(v, (list, np.ndarray)):
            return [str(x) for x in v if str(x) not in ("", "nan", "None")]
        return [p for p in str(v).replace(",", " ").split() if p and p != "nan"]
    out["phones_arr"] = df["Phones_set"].map(_phones)
    # composite address: street | city | zip (normalized)
    a1 = df["AddressLine1_clean"].astype("string").str.upper().fillna("")
    city = df["CityNM_clean"].astype("string").str.upper().fillna("")
    zp = df["ZipCD_clean_base"].astype("string").fillna("")
    comp = (a1.str.strip() + " | " + city.str.strip() + " | " + zp.str.strip())
    out["address_full"] = comp.where(a1.str.strip() != "", pd.NA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Settings — v1 (round-1 baseline) and v2 (improved)
# ─────────────────────────────────────────────────────────────────────────────

def _settings_v1() -> SettingsCreator:
    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            cl.ForenameSurnameComparison("first_name", "last_name",
                                         forename_surname_concat_col_name="first_last_concat"),
            cl.DateOfBirthComparison("dob", input_is_string=True,
                                     datetime_thresholds=[1, 1, 10],
                                     datetime_metrics=["month", "year", "year"]),
            cl.ExactMatch("ssn").configure(term_frequency_adjustments=True),
            cl.EmailComparison("email"),
            cl.ExactMatch("phone").configure(term_frequency_adjustments=True),
            cl.ExactMatch("sex"),
            cl.LevenshteinAtThresholds("zip", 1),
            cl.JaroWinklerAtThresholds("address", [0.9, 0.7]),
        ],
        blocking_rules_to_generate_predictions=[
            block_on("dob"), block_on("ssn"), block_on("email"), block_on("phone")],
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
    )


def _v2_comparisons() -> list:
    """Fresh list of v2 comparison objects (built each call so ablation can subset).

    v2 = v1 + the three improvements that are neutral-or-better in isolation:
    name-TF (neutral but motivated), multi-valued phone (small AUC win), composite
    address (neutral). **DOB-TF was tested and dropped** — term-frequency on a
    multi-level date comparison is ill-defined and collapsed the model (AUC 0.78).
    """
    return [
        cl.ForenameSurnameComparison(
            "first_name", "last_name",
            forename_surname_concat_col_name="first_last_concat",
        ).configure(term_frequency_adjustments=True),       # ← name-TF (neutral)
        cl.DateOfBirthComparison("dob", input_is_string=True,
                                 datetime_thresholds=[1, 1, 10],
                                 datetime_metrics=["month", "year", "year"]),
        cl.ExactMatch("ssn").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
        cl.ArrayIntersectAtSizes("phones_arr", [1]),        # ← multi-valued phone (+AUC)
        cl.ExactMatch("sex"),
        cl.LevenshteinAtThresholds("zip", 1),
        cl.JaroWinklerAtThresholds("address_full", [0.9, 0.7]),  # ← composite addr (neutral)
    ]


_V2_BLOCKING = [block_on("dob"), block_on("ssn"), block_on("email"), block_on("phone")]


def _settings_v2(comparisons: list | None = None) -> SettingsCreator:
    return SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons if comparisons is not None else _v2_comparisons(),
        blocking_rules_to_generate_predictions=_V2_BLOCKING,
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
    )


def _attribution_settings(change: str) -> SettingsCreator:
    """v1 baseline with exactly ONE v2 change applied — to attribute each improvement."""
    name = cl.ForenameSurnameComparison("first_name", "last_name",
                                        forename_surname_concat_col_name="first_last_concat")
    if change == "name_tf":
        name = name.configure(term_frequency_adjustments=True)
    phone = (cl.ArrayIntersectAtSizes("phones_arr", [1]) if change == "phone_array"
             else cl.ExactMatch("phone").configure(term_frequency_adjustments=True))
    addr_col = "address_full" if change == "addr_composite" else "address"
    comps = [name,
             cl.DateOfBirthComparison("dob", input_is_string=True),
             cl.ExactMatch("ssn").configure(term_frequency_adjustments=True),
             cl.EmailComparison("email"), phone, cl.ExactMatch("sex"),
             cl.LevenshteinAtThresholds("zip", 1),
             cl.JaroWinklerAtThresholds(addr_col, [0.9, 0.7])]
    return SettingsCreator(
        link_type="dedupe_only", comparisons=comps,
        blocking_rules_to_generate_predictions=[
            block_on("dob"), block_on("ssn"), block_on("email"), block_on("phone")],
        retain_intermediate_calculation_columns=True, retain_matching_columns=True)


def _improvement_attribution(df: pd.DataFrame, pos: set, neg: set, fm: set) -> dict:
    """Each improvement applied singly to v1 → isolated effect on AUC + false-merge rescue."""
    out = {}
    for change in ("baseline", "name_tf", "phone_array", "addr_composite"):
        lk, sdf = _train(df, _attribution_settings(change), f"attr_{change}")
        look = _pred_lookup(sdf.as_pandas_dataframe())
        fmp = [look.get(p, 0.0) for p in fm]
        out[change] = {"auc": _auc(look, pos, neg),
                       "fm_median_prob": round(float(np.median(fmp)), 4) if fmp else None,
                       "fm_below_0.9": round(float(np.mean([x < 0.9 for x in fmp])), 4) if fmp else None}
    out["dob_tf"] = {"note": "tested in combined v2 — collapsed AUC to ~0.78; "
                             "TF on a multi-level date comparison is ill-defined. Dropped."}
    return out


def _train(df: pd.DataFrame, settings: SettingsCreator, label: str, seed: int = 1):
    linker = Linker(df, settings, db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(
        [block_on("ssn", "dob"), block_on("email", "dob")], recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=5e6, seed=seed)
    for cols in (("dob",), ("first_name", "last_name"), ("email",)):
        try:
            linker.training.estimate_parameters_using_expectation_maximisation(block_on(*cols))
        except Exception as e:
            logger.warning("[%s] EM pass %s skipped: %s", label, cols, e)
    preds_sdf = linker.inference.predict(threshold_match_probability=0.001)
    return linker, preds_sdf


# ─────────────────────────────────────────────────────────────────────────────
# Silver / rules helpers
# ─────────────────────────────────────────────────────────────────────────────

def _silver_sets(silver: pd.DataFrame, present: set) -> tuple[set, set, pd.DataFrame]:
    silver = silver.copy()
    silver["key"] = [_canon(str(a), str(b)) for a, b in zip(silver.PATID_A, silver.PATID_B)]
    keep = [a in present and b in present
            for a, b in zip(silver.PATID_A.astype(str), silver.PATID_B.astype(str))]
    silver = silver[keep]
    pos = set(silver.loc[silver.silver_label, "key"])
    neg = set(silver.loc[~silver.silver_label, "key"])
    return pos, neg, silver


def _rule_false_merges(records: pd.DataFrame, silver: pd.DataFrame) -> dict:
    """The rule-confirmed / silver-False pairs, split by winning rule."""
    cand = silver[["PATID_A", "PATID_B"]]
    m = apply_rules(cand, records)
    slab = {_canon(str(a), str(b)): bool(x)
            for a, b, x in zip(silver.PATID_A, silver.PATID_B, silver.silver_label)}
    out = {"all": set(), "by_rule": {}}
    for a, b, rule in zip(m.PATID_A, m.PATID_B, m.match_rule):
        k = _canon(str(a), str(b))
        if slab.get(k) is False:
            out["all"].add(k)
            out["by_rule"].setdefault(rule, set()).add(k)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Charts (matplotlib for the custom analyses)
# ─────────────────────────────────────────────────────────────────────────────

def _chart_calibration(lookup: dict, pos: set, neg: set, name: str) -> tuple[str, float]:
    """Reliability curve: predicted prob vs actual fraction-true, + ECE."""
    pts = [(lookup[p], 1) for p in pos if p in lookup] + \
          [(lookup[p], 0) for p in neg if p in lookup]
    if not pts:
        return "", 0.0
    probs = np.array([p for p, _ in pts])
    ys = np.array([y for _, y in pts])
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, 9)
    xs, actual, ece, n = [], [], 0.0, len(probs)
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = probs[m].mean()
        acc = ys[m].mean()
        xs.append(conf)
        actual.append(acc)
        ece += (m.sum() / n) * abs(conf - acc)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    ax.plot(xs, actual, "o-", color="#2b8cbe", label="FS")
    ax.set_xlabel("FS predicted probability")
    ax.set_ylabel("actual fraction true (silver)")
    ax.set_title(f"Calibration — {name}  (ECE={ece:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    p = _CHARTS / f"{name}_calibration.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p.relative_to(_ROOT)), round(float(ece), 4)


def _chart_rescue(v1: dict, v2: dict, fm: set, name: str) -> str:
    """Histogram of FS prob on the rule-false merges: v1 vs the improved model."""
    a = [v1.get(p, 0.0) for p in fm]
    b = [v2.get(p, 0.0) for p in fm]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(a, bins=bins, alpha=0.55, label="round-1 model", color="#d6604d")
    ax.hist(b, bins=bins, alpha=0.55, label="with name term-frequency", color="#2b8cbe")
    ax.axvline(0.9, ls="--", color="k", lw=1)
    ax.text(0.9, ax.get_ylim()[1] * 0.9, " 0.9", fontsize=8)
    ax.set_xlabel("FS match probability on the 7,578 rule-FALSE merges")
    ax.set_ylabel("count of pairs")
    ax.set_title("Does name-weighting rescue the false merges?")
    ax.legend()
    p = _CHARTS / f"{name}_falsemerge_rescue.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p.relative_to(_ROOT))


def _chart_clusters(sizes_v1: list, sizes_v2: list) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    mx = max(max(sizes_v1, default=1), max(sizes_v2, default=1))
    bins = np.arange(1, min(mx, 30) + 2) - 0.5
    ax.hist(sizes_v1, bins=bins, alpha=0.55, label="round-1 (max 34)", color="#d6604d")
    ax.hist(sizes_v2, bins=bins, alpha=0.55, label="with phone-array (max 1,261)", color="#2b8cbe")
    ax.set_yscale("log")
    ax.set_xlabel("cluster size (records per entity) — tail truncated at 30")
    ax.set_ylabel("count (log)")
    ax.set_title("Entity cluster-size distribution (@0.9)")
    ax.legend()
    p = _CHARTS / "cluster_size_distribution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p.relative_to(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Analyses
# ─────────────────────────────────────────────────────────────────────────────

def _eval_block(lookup: dict, pos: set, neg: set, R: set) -> dict:
    ev = _score_against_labels(lookup, pos, neg)
    ev["roc_auc"] = _auc(lookup, pos, neg)
    ev["hybrid"] = _hybrid_analysis(lookup, R, pos, neg)
    return ev


def _false_merge_rescue(v1: dict, v2: dict, fm: dict, pos: set) -> dict:
    """How many rule-false merges each model pushes below threshold, + true-match safety."""
    out = {}
    allfm = fm["all"]
    for label, lk in (("v1", v1), ("v2", v2)):
        probs = [lk.get(p, 0.0) for p in allfm]
        out[label] = {
            "n_false_merges": len(allfm),
            "median_prob": round(float(np.median(probs)), 4) if probs else None,
            "below_0.5": round(np.mean([x < 0.5 for x in probs]), 4),
            "below_0.9": round(np.mean([x < 0.9 for x in probs]), 4),
            "below_0.99": round(np.mean([x < 0.99 for x in probs]), 4),
        }
    # true-match safety: do the same thresholds keep true matches high?
    for label, lk in (("v1", v1), ("v2", v2)):
        tp_probs = [lk.get(p, 0.0) for p in pos]
        out[label]["true_match_recall_at_0.9"] = round(np.mean([x >= 0.9 for x in tp_probs]), 4)
    # per-rule rescue for the two weak rules
    out["by_rule"] = {}
    for rule, pairs in fm["by_rule"].items():
        if rule not in ("NAME_DOB_SEX", "NAME_DOB_ADDRESS"):
            continue
        out["by_rule"][rule] = {
            "n": len(pairs),
            "v1_below_0.9": round(np.mean([v1.get(p, 0.0) < 0.9 for p in pairs]), 4),
            "v2_below_0.9": round(np.mean([v2.get(p, 0.0) < 0.9 for p in pairs]), 4),
            "v1_median": round(float(np.median([v1.get(p, 0.0) for p in pairs])), 4),
            "v2_median": round(float(np.median([v2.get(p, 0.0) for p in pairs])), 4),
        }
    return out


def _conditional_dependence(linker, preds: pd.DataFrame) -> dict:
    """Correlation between field 'agreement' indicators among high-prob (likely-match)
    pairs — FS assumes these are independent; large correlations break that."""
    gamma_cols = [c for c in preds.columns if c.startswith("gamma_")]
    if not gamma_cols:
        return {}
    hi = preds[preds["match_probability"] >= 0.9]
    if len(hi) < 50:
        hi = preds
    agree = (hi[gamma_cols] > 0).astype(int)
    corr = agree.corr().abs()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append((cols[i].replace("gamma_", ""), cols[j].replace("gamma_", ""), round(float(v), 3)))
    pairs.sort(key=lambda x: -x[2])
    return {"top_correlated_field_pairs": pairs[:8]}


def _missingness_strata(records: pd.DataFrame, lookup: dict, pos: set, neg: set) -> dict:
    """AUC by which strong identifiers both records carry."""
    have = {}
    for c, key in (("SSN_clean", "ssn"), ("Email_clean", "email"), ("PrimaryPhoneNBR_clean", "phone")):
        s = records.set_index("PATID")[c]
        have[key] = {pid for pid, v in s.items() if pd.notna(v) and str(v) != ""}
    out = {}
    for key, ids in have.items():
        sub_pos = {p for p in pos if p[0] in ids and p[1] in ids}
        sub_neg = {p for p in neg if p[0] in ids and p[1] in ids}
        out[f"both_have_{key}"] = {"n_pos": len(sub_pos), "n_neg": len(sub_neg),
                                   "auc": _auc(lookup, sub_pos, sub_neg)}
    # the weak stratum: neither record has SSN/email/phone → relies on name+DOB+sex
    strong = have["ssn"] | have["email"] | have["phone"]
    weak_pos = {p for p in pos if p[0] not in strong and p[1] not in strong}
    weak_neg = {p for p in neg if p[0] not in strong and p[1] not in strong}
    out["no_strong_id_either_side"] = {"n_pos": len(weak_pos), "n_neg": len(weak_neg),
                                       "auc": _auc(lookup, weak_pos, weak_neg)}
    return out


def _cluster_stats(linker, preds_sdf, thr=0.9) -> tuple[dict, list]:
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        preds_sdf, threshold_match_probability=thr)
    cdf = clusters.as_pandas_dataframe()
    cid = [c for c in cdf.columns if "cluster" in c.lower()][0]
    sizes = cdf.groupby(cid).size()
    return {
        "n_records_clustered": int(len(cdf)),
        "n_clusters": int(sizes.shape[0]),
        "max_cluster_size": int(sizes.max()),
        "clusters_size_ge_5": int((sizes >= 5).sum()),
        "clusters_size_ge_10": int((sizes >= 10).sum()),
        "pct_records_in_clusters_ge_10": round(float(sizes[sizes >= 10].sum() / len(cdf)), 4),
    }, sizes.tolist()


_ABLATION_NAMES = ["name", "dob", "ssn", "email", "phone", "sex", "zip", "address"]


def _ablation(df: pd.DataFrame, pos: set, neg: set) -> dict:
    """Drop each comparison, retrain, measure AUC delta (subsample for speed)."""
    out = {}
    lk, sdf = _train(df, _settings_v2(), "ablation_full")
    out["__full__"] = _auc(_pred_lookup(sdf.as_pandas_dataframe()), pos, neg)
    for i, name in enumerate(_ABLATION_NAMES):
        comps = _v2_comparisons()
        reduced = [comps[j] for j in range(len(comps)) if j != i]
        try:
            lk, sdf = _train(df, _settings_v2(reduced), f"ablation_drop_{name}")
            out[name] = _auc(_pred_lookup(sdf.as_pandas_dataframe()), pos, neg)
        except Exception as e:
            out[name] = None
            logger.warning("ablation drop %s failed: %s", name, e)
    return out


def _em_stability(df: pd.DataFrame, pos: set, neg: set, seeds=(1, 7, 42)) -> dict:
    aucs = []
    for sd in seeds:
        lk, sdf = _train(df, _settings_v2(), f"stability_{sd}", seed=sd)
        aucs.append(_auc(_pred_lookup(sdf.as_pandas_dataframe()), pos, neg))
    aucs = [a for a in aucs if a is not None]
    return {"seeds": list(seeds), "aucs": aucs,
            "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
            "auc_std": round(float(np.std(aucs)), 5) if aucs else None}


def _thresholds_and_review(lookup: dict, pos: set, neg: set,
                           target_match_prec=0.97, target_reject_prec=0.99) -> dict:
    """Pick upper (auto-merge) / lower (auto-reject) cutoffs; size the review band."""
    allk = list(pos | neg)
    scored = [(lookup.get(k, 0.0), 1 if k in pos else 0) for k in allk]
    scored.sort()
    probs = np.array([s for s, _ in scored])
    ys = np.array([y for _, y in scored])
    grid = np.linspace(0.5, 0.999, 50)
    upper = None
    for t in grid[::-1]:
        sel = probs >= t
        if sel.sum() and ys[sel].mean() >= target_match_prec:
            upper = float(round(t, 3))
            break
    # lower: below it, fraction that are actually non-match ≥ target
    lower = None
    for t in grid:
        sel = probs < t
        if sel.sum() and (1 - ys[sel].mean()) >= target_reject_prec:
            lower = float(round(t, 3))
    n = len(ys)
    auto_merge = int((probs >= (upper or 1)).sum())
    auto_reject = int((probs < (lower or 0)).sum())
    review = n - auto_merge - auto_reject
    return {
        "upper_auto_merge_threshold": upper, "lower_auto_reject_threshold": lower,
        "labeled_pairs": n,
        "auto_merge_pct": round(auto_merge / n, 4),
        "auto_reject_pct": round(auto_reject / n, 4),
        "clerical_review_pct": round(review / n, 4),
    }


def _active_learning(lookup: dict, pos: set, neg: set, lo=0.4, hi=0.6) -> dict:
    """Uncertain pairs (near the decision boundary) — the most informative for gold."""
    uncertain = [(k, p) for k, p in lookup.items() if lo <= p <= hi]
    labeled_unc = [(k, p) for k, p in uncertain if k in pos or k in neg]
    return {
        "n_uncertain_pairs": len(uncertain),
        "band": [lo, hi],
        "of_which_silver_labeled": len(labeled_unc),
        "silver_true_rate_in_band": round(
            np.mean([k in pos for k, _ in labeled_unc]), 4) if labeled_unc else None,
        "note": "label these first — silver agreement near 0.5 confirms they are genuinely ambiguous",
    }


def _online_latency(linker, df: pd.DataFrame, n=20) -> dict:
    sample = df.sample(min(n, len(df)), random_state=0)
    t0 = time.time()
    try:
        res = linker.inference.find_matches_to_new_records(
            sample, blocking_rules=[block_on("dob")], match_weight_threshold=-5)
        _ = res.as_pandas_dataframe()
        dt = time.time() - t0
        return {"n_query_records": len(sample), "total_s": round(dt, 2),
                "per_record_ms": round(1000 * dt / len(sample), 1)}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path,
                    default=_ROOT / "data/processed/MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--silver", type=Path, default=_ROOT / "data/raw/silver_labels.csv")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--ablation-sample", type=int, default=40000)
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    records = _filter_valid_records(pd.read_parquet(args.cleaned))
    if args.sample:
        records = records.sample(args.sample, random_state=42)
    silver = pd.read_csv(args.silver)
    df = _prep_v2(records)
    present = set(df["unique_id"])
    pos, neg, silver_p = _silver_sets(silver, present)
    R = {_canon(str(a), str(b)) for a, b in zip(
        apply_rules(silver_p[["PATID_A", "PATID_B"]], records).PATID_A,
        apply_rules(silver_p[["PATID_A", "PATID_B"]], records).PATID_B)}
    fm = _rule_false_merges(records, silver_p)
    print(f"records={len(df):,} silver_pos={len(pos):,} silver_neg={len(neg):,} "
          f"rule_false_merges={len(fm['all']):,}")

    res = {"meta": {"n_records": len(df), "silver_pos": len(pos), "silver_neg": len(neg),
                    "rule_false_merges": len(fm["all"])}}

    # --- train v1 and v2 ---
    print("Training v1 (baseline)...")
    lk1, sdf1 = _train(df, _settings_v1(), "v1")
    pred1 = sdf1.as_pandas_dataframe()
    look1 = _pred_lookup(pred1)
    print("Training v2 (improved)...")
    lk2, sdf2 = _train(df, _settings_v2(), "v2")
    pred2 = sdf2.as_pandas_dataframe()
    look2 = _pred_lookup(pred2)

    res["v1_eval"] = _eval_block(look1, pos, neg, R)
    res["v2_eval"] = _eval_block(look2, pos, neg, R)
    print(f"  v1 AUC={res['v1_eval']['roc_auc']}  v2 AUC={res['v2_eval']['roc_auc']}")

    # improvement attribution — each v2 change applied singly to v1
    print("Improvement attribution...")
    res["improvement_attribution"] = _improvement_attribution(df, pos, neg, fm["all"])
    print(f"  attribution: {[(k, v.get('auc')) for k, v in res['improvement_attribution'].items()]}")

    # 1. false-merge rescue (headline)
    res["false_merge_rescue"] = _false_merge_rescue(look1, look2, fm, pos)
    print(f"  rescue: v1 below0.9={res['false_merge_rescue']['v1']['below_0.9']} "
          f"v2 below0.9={res['false_merge_rescue']['v2']['below_0.9']}")

    # 3. calibration
    cal_path, ece = _chart_calibration(look2, pos, neg, "v2")
    res["calibration"] = {"ece": ece, "chart": cal_path}

    # 6. missingness strata
    res["missingness_strata"] = _missingness_strata(records, look2, pos, neg)

    # 9. conditional dependence
    res["conditional_dependence"] = _conditional_dependence(lk2, pred2)

    # 7. cluster over-merge (v1 vs v2)
    try:
        cs1, sizes1 = _cluster_stats(lk1, sdf1)
        cs2, sizes2 = _cluster_stats(lk2, sdf2)
        res["clusters"] = {"v1": cs1, "v2": cs2, "chart": _chart_clusters(sizes1, sizes2)}
    except Exception as e:
        logger.warning("clustering failed: %s", e)
        res["clusters"] = {"error": str(e)}

    # 10. thresholds + review volume (v2)
    res["thresholds_review"] = _thresholds_and_review(look2, pos, neg)

    # 11. active learning
    res["active_learning"] = _active_learning(look2, pos, neg)

    # 12. online latency (v2)
    res["online_latency"] = _online_latency(lk2, df)

    # rescue chart + new model charts
    res["charts"] = {
        "falsemerge_rescue": _chart_rescue(look1, look2, fm["all"], "v2"),
        "calibration": cal_path,
    }
    try:
        res["charts"]["v2_match_weights"] = _save(lk2.visualisations.match_weights_chart(), "v2_match_weights")
    except Exception as e:
        logger.warning("v2 match_weights failed: %s", e)
    for tf_col in ("first_last_concat", "last_name", "first_name"):
        try:
            res["charts"]["v2_tf_name"] = _save(
                lk2.visualisations.tf_adjustment_chart(tf_col), "v2_tf_name")
            break
        except Exception as e:
            logger.warning("v2 tf_name(%s) failed: %s", tf_col, e)
    # waterfall on a rescued false-merge (lowest-v2-prob false merge that v1 rated high)
    try:
        cand = sorted(fm["all"], key=lambda p: look2.get(p, 0.0))
        target = next((p for p in cand if look1.get(p, 0) >= 0.9 and look2.get(p, 1) < 0.9), None)
        if target:
            row = pred2[((pred2.unique_id_l == target[0]) & (pred2.unique_id_r == target[1])) |
                        ((pred2.unique_id_l == target[1]) & (pred2.unique_id_r == target[0]))]
            if len(row):
                res["charts"]["v2_rescued_waterfall"] = _save(
                    lk2.visualisations.waterfall_chart(row.to_dict("records")), "v2_rescued_waterfall")
    except Exception as e:
        logger.warning("rescued waterfall failed: %s", e)

    res["v2_model"] = _model_summary(lk2)

    # 5. ablation (subsample for speed)
    print("Ablation...")
    ab_records = records.sample(min(args.ablation_sample, len(records)), random_state=7)
    ab_df = _prep_v2(ab_records)
    ab_present = set(ab_df["unique_id"])
    ab_pos, ab_neg, _ = _silver_sets(silver, ab_present)
    try:
        res["ablation"] = _ablation(ab_df, ab_pos, ab_neg)
    except Exception as e:
        logger.warning("ablation failed: %s", e)
        res["ablation"] = {"error": str(e)}

    # 8. EM stability (subsample)
    print("EM stability...")
    try:
        res["em_stability"] = _em_stability(ab_df, ab_pos, ab_neg)
    except Exception as e:
        logger.warning("stability failed: %s", e)
        res["em_stability"] = {"error": str(e)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
