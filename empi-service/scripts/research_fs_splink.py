"""Fellegi–Sunter probabilistic matching research (Splink) — exploratory / analytical.

The next pipeline stage after the deterministic rules: a probabilistic matcher that
scores the candidate pairs the five rules cannot confidently decide. **Analysis only —
no production integration.** Read-only on `data/`; writes a JSON metrics artifact and a
folder of stakeholder-facing charts.

Engine: **Splink 4** (MoJ), DuckDB backend — runs locally, HIPAA-safe.

Studies (mirrors the scope in `to-do.md`):
  1. Comparison-vector design — multi-level comparators + term-frequency weighting.
  2. m/u estimation — unsupervised EM vs supervised-from-silver, side by side.
  3. Evaluation & calibration — ROC/PR vs silver + synthetic; threshold volumes.
  4. FS vs the deterministic rules — does FS fix NAME_DOB_SEX / placeholder-SSN?
  5. Visuals — Splink's native charts (waterfall, m/u, accuracy, dashboards),
     each captioned for a non-technical audience in the write-up.

Usage:
    uv run python scripts/research_fs_splink.py [--sample N] [--skip-synth]
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl

from src.config import configure_logging
from src.preprocessing.blocking import _filter_valid_records
from scripts.eval_against_labels import _canon, _synthetic_records
from scripts.research_blocking_labels import _rules_R

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

_CHARTS = _ROOT / "reports/fs/charts"
_OUT = _ROOT / "data/runs/research_fs_splink.json"


# ─────────────────────────────────────────────────────────────────────────────
# Data prep — map the cleaned columns onto a Splink-ready frame
# ─────────────────────────────────────────────────────────────────────────────

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Rename/normalize cleaned columns into the names the Splink settings expect."""
    out = pd.DataFrame()
    out["unique_id"] = df["PATID"].astype(str)
    out["first_name"] = df["FirstNM_clean"].astype("string").str.upper()
    out["last_name"] = df["LastNM_clean"].astype("string").str.upper()
    dob = pd.to_datetime(df["BirthDT_clean"], errors="coerce")
    out["dob"] = dob.dt.strftime("%Y-%m-%d").astype("string")
    out["ssn"] = df["SSN_clean"].astype("string")
    out["email"] = df["Email_clean"].astype("string").str.lower()
    out["phone"] = df["PrimaryPhoneNBR_clean"].astype("string")
    out["sex"] = df["SexAtBirthDSC_clean"].astype("string").str.upper()
    out["zip"] = df.get("ZipCD_clean_base", pd.Series(dtype="string")).astype("string")
    out["city"] = df.get("CityNM_clean", pd.Series(dtype="string")).astype("string").str.upper()
    out["address"] = df.get("AddressLine1_clean", pd.Series(dtype="string")).astype("string").str.upper()
    # blank → NA so Splink treats them as null, not as an agreeing empty string
    for c in out.columns:
        if c != "unique_id":
            out[c] = out[c].replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    return out


def _settings(for_em: bool = True) -> SettingsCreator:
    """Comparison vector: multi-level comparators + TF on the rare-value fields.

    The TF adjustments on SSN / name / phone are the mechanism that should down-weight
    placeholder SSNs and common names — the documented deterministic-rule failures.
    """
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
        blocking_rules_to_generate_predictions=[
            block_on("dob"),
            block_on("ssn"),
            block_on("email"),
            block_on("phone"),
        ],
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
    )


def _add_concat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["first_last_concat"] = (
        df["first_name"].fillna("") + " " + df["last_name"].fillna("")
    ).str.strip().replace({"": pd.NA})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Train a Splink model (shared by real + synthetic studies)
# ─────────────────────────────────────────────────────────────────────────────

def _train(df: pd.DataFrame, label: str):
    """Estimate λ, u (random sampling), and m (EM, several blocking passes).

    Returns (linker, predictions_splink_dataframe). Keep the SplinkDataframe — the
    histogram, dashboards and clustering need it (not a pandas frame).
    """
    linker = Linker(df, _settings(), db_api=DuckDBAPI())
    # λ: probability two random records match — seed from a tight deterministic rule.
    linker.training.estimate_probability_two_random_records_match(
        [block_on("ssn", "dob"), block_on("email", "dob")], recall=0.8,
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=5e6)
    # EM passes on different blocking columns so every comparison gets trained.
    # Blocking on a column means it can't train *that* column's m — so we vary the
    # block: name/email/phone passes train dob's fuzzy levels; the dob pass trains
    # name/ssn/etc.
    for cols in (("dob",), ("first_name", "last_name"), ("email",), ("phone",)):
        try:
            linker.training.estimate_parameters_using_expectation_maximisation(block_on(*cols))
        except Exception as e:  # a pass can fail if a block is too sparse
            logger.warning("[%s] EM pass on %s skipped: %s", label, cols, e)
    preds_sdf = linker.inference.predict(threshold_match_probability=0.001)
    return linker, preds_sdf


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pred_lookup(preds: pd.DataFrame) -> dict[tuple, float]:
    """Canonical-pair → match_probability for the predicted set."""
    out = {}
    for a, b, p in zip(preds["unique_id_l"], preds["unique_id_r"], preds["match_probability"]):
        out[_canon(str(a), str(b))] = float(p)
    return out


def _score_against_labels(lookup: dict[tuple, float], pos: set, neg: set,
                          thresholds=(0.5, 0.9, 0.99)) -> dict:
    """Precision / recall / coverage of the FS score vs a labeled pair set."""
    def metrics(thr):
        tp = sum(1 for p in pos if lookup.get(p, 0.0) >= thr)
        fp = sum(1 for p in neg if lookup.get(p, 0.0) >= thr)
        fn = len(pos) - tp
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        return {"threshold": thr, "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4) if prec is not None else None,
                "recall": round(rec, 4) if rec is not None else None}

    pos_scored = sum(1 for p in pos if p in lookup)
    neg_scored = sum(1 for p in neg if p in lookup)
    return {
        "n_pos": len(pos), "n_neg": len(neg),
        "pos_coverage": round(pos_scored / len(pos), 4) if pos else None,
        "neg_coverage": round(neg_scored / len(neg), 4) if neg else None,
        "by_threshold": [metrics(t) for t in thresholds],
    }


def _prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None}


def _hybrid_analysis(lookup: dict, R: set, pos: set, neg: set, fs_thr=0.9,
                     arbiter_veto=0.5) -> dict:
    """Four deterministic/FS architectures scored on the same silver-labeled set.

    Only pairs that are silver-labeled AND scored are counted, so every architecture
    is judged on identical ground.
    """
    def decide(pair, mode):
        in_rule = pair in R
        fs = lookup.get(pair, 0.0)
        if mode == "rules_only":
            return in_rule
        if mode == "fs_only":
            return fs >= fs_thr
        if mode == "union":            # rules → FS-on-review (FS promotes non-rule pairs)
            return in_rule or fs >= fs_thr
        if mode == "arbiter":          # FS can veto a weak rule match
            return in_rule and fs >= arbiter_veto
        raise ValueError(mode)

    out = {"fs_threshold": fs_thr, "arbiter_veto": arbiter_veto}
    for mode in ("rules_only", "fs_only", "union", "arbiter"):
        tp = sum(1 for p in pos if decide(p, mode))
        fp = sum(1 for p in neg if decide(p, mode))
        fn = len(pos) - tp
        out[mode] = _prf(tp, fp, fn)
    return out


def _auc(lookup: dict[tuple, float], pos: set, neg: set) -> float | None:
    """ROC-AUC of the match probability separating pos from neg (scored pairs only)."""
    ys, ss = [], []
    for p in pos:
        if p in lookup:
            ys.append(1)
            ss.append(lookup[p])
    for p in neg:
        if p in lookup:
            ys.append(0)
            ss.append(lookup[p])
    if not ys or len(set(ys)) < 2:
        return None
    order = np.argsort(ss)
    ys = np.array(ys)[order]
    n_pos, n_neg = int(ys.sum()), int((ys == 0).sum())
    rank_sum = np.sum(np.where(ys == 1)[0] + 1)
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return round(float(auc), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def _save(chart, name: str) -> str | None:
    """Save a Splink chart as HTML (always) and PNG (best-effort). Return PNG path."""
    _CHARTS.mkdir(parents=True, exist_ok=True)
    html = _CHARTS / f"{name}.html"
    png = _CHARTS / f"{name}.png"
    try:
        chart.save(str(html))
    except Exception as e:
        logger.warning("save html %s failed: %s", name, e)
    try:
        chart.save(str(png), scale_factor=2.0)
        return str(png.relative_to(_ROOT))
    except Exception as e:
        logger.warning("save png %s failed: %s", name, e)
        return str(html.relative_to(_ROOT)) if html.exists() else None


def _make_charts(linker: Linker, preds_sdf, preds_pd: pd.DataFrame, prefix: str,
                 waterfall_pairs: pd.DataFrame | None = None) -> dict:
    saved = {}
    viz = linker.visualisations
    try:
        saved["match_weights"] = _save(viz.match_weights_chart(), f"{prefix}_match_weights")
    except Exception as e:
        logger.warning("match_weights failed: %s", e)
    try:
        saved["m_u_parameters"] = _save(viz.m_u_parameters_chart(), f"{prefix}_m_u")
    except Exception as e:
        logger.warning("m_u failed: %s", e)
    try:
        saved["tf_adjustment"] = _save(viz.tf_adjustment_chart("ssn"), f"{prefix}_tf_ssn")
    except Exception as e:
        logger.warning("tf failed: %s", e)
    try:
        saved["param_estimates"] = _save(viz.parameter_estimate_comparisons_chart(),
                                          f"{prefix}_param_estimates")
    except Exception as e:
        logger.warning("param_estimates failed: %s", e)
    try:  # histogram needs the SplinkDataframe
        saved["match_weights_histogram"] = _save(viz.match_weights_histogram(preds_sdf),
                                                  f"{prefix}_mw_histogram")
    except Exception as e:
        logger.warning("mw_histogram failed: %s", e)
    # Waterfall — the hero visual. Pick representative records if not supplied.
    try:
        if waterfall_pairs is None:
            waterfall_pairs = _pick_waterfall(preds_pd)
        recs = waterfall_pairs.to_dict("records")
        saved["waterfall"] = _save(viz.waterfall_chart(recs), f"{prefix}_waterfall")
    except Exception as e:
        logger.warning("waterfall failed: %s", e)
    # Interactive dashboard (HTML only) — needs the SplinkDataframe
    try:
        viz.comparison_viewer_dashboard(preds_sdf, str(_CHARTS / f"{prefix}_comparison_viewer.html"),
                                        overwrite=True, num_example_rows=4)
        saved["comparison_viewer"] = str((_CHARTS / f"{prefix}_comparison_viewer.html").relative_to(_ROOT))
    except Exception as e:
        logger.warning("comparison_viewer failed: %s", e)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 1. Real-data study
# ─────────────────────────────────────────────────────────────────────────────

def run_real(cleaned: pd.DataFrame, silver: pd.DataFrame, sample: int | None) -> dict:
    records = _filter_valid_records(cleaned)
    if sample:
        records = records.sample(sample, random_state=42)
    df = _add_concat(_prep(records))
    present = set(df["unique_id"])
    logger.info("Real: %d records", len(df))

    # Silver labels restricted to records present (relevant for --sample)
    silver = silver.copy()
    silver["key"] = [_canon(str(a), str(b)) for a, b in zip(silver.PATID_A, silver.PATID_B)]
    silver = silver[[a in present and b in present
                     for a, b in zip(silver.PATID_A.astype(str), silver.PATID_B.astype(str))]]
    silver_pos = set(silver.loc[silver.silver_label, "key"])
    silver_neg = set(silver.loc[~silver.silver_label, "key"])

    linker, preds_sdf = _train(df, "real")
    preds = preds_sdf.as_pandas_dataframe()
    lookup = _pred_lookup(preds)

    # --- supervised m from silver-True pairwise labels, for the EM-vs-labels chart ---
    try:
        lab = silver[silver.silver_label][["PATID_A", "PATID_B"]].rename(
            columns={"PATID_A": "unique_id_l", "PATID_B": "unique_id_r"})
        lab["clerical_match_score"] = 1.0
        linker.table_management.register_table(lab, "silver_clerical_labels", overwrite=True)
        linker.training.estimate_m_from_pairwise_labels("silver_clerical_labels")
    except Exception as e:
        logger.warning("estimate_m_from_pairwise_labels failed: %s", e)

    # --- evaluation vs silver ---
    silver_eval = _score_against_labels(lookup, silver_pos, silver_neg)
    silver_eval["roc_auc"] = _auc(lookup, silver_pos, silver_neg)

    # --- FS vs deterministic rules: does FS catch the rule false-merges? ---
    R = _rules_R(records)  # rule-confirmed pairs (canonical)
    R_in_silver = {p for p in R if p in silver_pos or p in silver_neg}
    rule_true = {p for p in R_in_silver if p in silver_pos}
    rule_false = {p for p in R_in_silver if p in silver_neg}  # rule said match, silver says NOT
    fs_vs_rules = {
        "rule_confirmed_in_silver": len(R_in_silver),
        "rule_confirmed_silver_true": len(rule_true),
        "rule_confirmed_silver_false": len(rule_false),
        "rule_precision_silver": round(len(rule_true) / len(R_in_silver), 4) if R_in_silver else None,
    }
    # On the rule-FALSE merges, what does FS say? (low prob = FS correctly doubts them)
    if rule_false:
        fp_probs = [lookup.get(p, 0.0) for p in rule_false]
        for thr in (0.5, 0.9, 0.99):
            fs_vs_rules[f"rule_false_FS_below_{thr}"] = round(
                sum(1 for x in fp_probs if x < thr) / len(fp_probs), 4)
        fs_vs_rules["rule_false_FS_median_prob"] = round(float(np.median(fp_probs)), 4)

    # --- hybrid architectures: rules-only / FS-only / union / FS-as-arbiter ---
    hybrid = _hybrid_analysis(lookup, R, silver_pos, silver_neg)

    # --- charts: pick waterfall pairs (clear match / non-match / gray zone) ---
    waterfall = _pick_waterfall(preds)
    charts = _make_charts(linker, preds_sdf, preds, "real", waterfall)

    # accuracy charts (ROC + threshold selection) vs silver labels
    try:
        acc_lab = silver[["PATID_A", "PATID_B"]].rename(
            columns={"PATID_A": "unique_id_l", "PATID_B": "unique_id_r"}).copy()
        acc_lab["clerical_match_score"] = silver["silver_label"].astype(float).values
        linker.table_management.register_table(acc_lab, "silver_acc_labels", overwrite=True)
        charts["roc"] = _save(linker.evaluation.accuracy_analysis_from_labels_table(
            "silver_acc_labels", output_type="roc"), "real_roc")
        charts["threshold_selection"] = _save(linker.evaluation.accuracy_analysis_from_labels_table(
            "silver_acc_labels", output_type="threshold_selection",
            add_metrics=["f1"]), "real_threshold_selection")
    except Exception as e:
        logger.warning("accuracy charts failed: %s", e)

    # cluster studio on a sample of predicted clusters
    try:
        clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
            preds_sdf, threshold_match_probability=0.9)
        linker.visualisations.cluster_studio_dashboard(
            preds_sdf, clusters, str(_CHARTS / "real_cluster_studio.html"),
            sampling_method="by_cluster_size", overwrite=True)
        charts["cluster_studio"] = str((_CHARTS / "real_cluster_studio.html").relative_to(_ROOT))
    except Exception as e:
        logger.warning("cluster studio failed: %s", e)

    return {
        "n_records": len(df),
        "n_predicted_pairs": len(preds),
        "silver_labeled_present": {"pos": len(silver_pos), "neg": len(silver_neg)},
        "evaluation_vs_silver": silver_eval,
        "fs_vs_rules": fs_vs_rules,
        "hybrid_architectures": hybrid,
        "charts": charts,
        "model": _model_summary(linker),
    }


def _pick_waterfall(preds: pd.DataFrame) -> pd.DataFrame:
    """One clear match (highest weight), one clear non-match (lowest), two gray-zone
    pairs (match_weight nearest 0, i.e. probability ~0.5) — a representative spread to
    explain to a non-technical reader."""
    if preds.empty:
        return preds
    p = preds.sort_values("match_weight")
    strong = p.iloc[[-1]]
    weak = p.iloc[[0]]
    gray = preds.loc[preds["match_weight"].abs().sort_values().index[:2]]
    return pd.concat([strong, gray, weak]).drop_duplicates(subset=["unique_id_l", "unique_id_r"])


def _model_summary(linker: Linker) -> dict:
    """Pull the trained m/u and match weights into a compact JSON summary."""
    try:
        params = linker.misc.save_model_to_json()
    except Exception:
        try:
            params = linker._settings_obj.as_dict()
        except Exception:
            return {}
    out = {"comparisons": []}
    for c in params.get("comparisons", []):
        levels = []
        for lvl in c.get("comparison_levels", []):
            levels.append({
                "label": lvl.get("label_for_charts"),
                "m": round(lvl["m_probability"], 5) if lvl.get("m_probability") is not None else None,
                "u": round(lvl["u_probability"], 6) if lvl.get("u_probability") is not None else None,
            })
        out["comparisons"].append({
            "output_column": c.get("output_column_name"),
            "levels": levels,
        })
    out["probability_two_random_records_match"] = params.get(
        "probability_two_random_records_match")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic study — honest, label-true accuracy (generator-independent of FS)
# ─────────────────────────────────────────────────────────────────────────────

def run_synthetic(syn: pd.DataFrame) -> dict:
    records = _synthetic_records(syn)
    df = _add_concat(_prep_synth(records))
    logger.info("Synthetic: %d records", len(df))

    pos = set()
    neg = set()
    for a, b, lab in zip(syn.PATID_A.astype(str), syn.PATID_B.astype(str), syn.label.astype(int)):
        (pos if lab == 1 else neg).add(_canon(a, b))

    linker, preds_sdf = _train(df, "synthetic")
    preds = preds_sdf.as_pandas_dataframe()
    lookup = _pred_lookup(preds)

    syn_eval = _score_against_labels(lookup, pos, neg)
    syn_eval["roc_auc"] = _auc(lookup, pos, neg)

    # Per-case-type recall — which corruption types does FS catch?
    by_case = {}
    pos_df = syn[syn.label.astype(int) == 1]
    for case, grp in pos_df.groupby("case_type"):
        keys = {_canon(str(a), str(b)) for a, b in zip(grp.PATID_A, grp.PATID_B)}
        scored = [lookup.get(k, 0.0) for k in keys]
        by_case[case] = {
            "n": len(keys),
            "recall_at_0.9": round(sum(1 for s in scored if s >= 0.9) / len(keys), 4) if keys else None,
            "median_prob": round(float(np.median(scored)), 4) if scored else None,
        }

    charts = _make_charts(linker, preds_sdf, preds, "synthetic", _pick_waterfall(preds))
    return {
        "n_records": len(df),
        "n_predicted_pairs": len(preds),
        "labels": {"pos": len(pos), "neg": len(neg)},
        "evaluation": syn_eval,
        "recall_by_case_type": by_case,
        "charts": charts,
    }


def _prep_synth(records: pd.DataFrame) -> pd.DataFrame:
    """Synthetic records already use the *_clean column names; map to Splink frame."""
    out = pd.DataFrame()
    out["unique_id"] = records["PATID"].astype(str)
    out["first_name"] = records["FirstNM_clean"].astype("string").str.upper()
    out["last_name"] = records["LastNM_clean"].astype("string").str.upper()
    dob = pd.to_datetime(records["BirthDT_clean"], errors="coerce")
    out["dob"] = dob.dt.strftime("%Y-%m-%d").astype("string")
    out["ssn"] = records["SSN_clean"].astype("string")
    out["email"] = records["Email_clean"].astype("string").str.lower()
    out["phone"] = records["PrimaryPhoneNBR_clean"].astype("string")
    out["sex"] = records["SexAtBirthDSC_clean"].astype("string").str.upper()
    out["zip"] = records.get("ZipCD_clean_base", pd.Series(dtype="string")).astype("string")
    out["city"] = records.get("CityNM_clean", pd.Series(dtype="string")).astype("string").str.upper()
    out["address"] = records.get("AddressLine1_clean", pd.Series(dtype="string")).astype("string").str.upper()
    for c in out.columns:
        if c != "unique_id":
            out[c] = out[c].replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path,
                    default=_ROOT / "data/processed/MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--silver", type=Path, default=_ROOT / "data/raw/silver_labels.csv")
    ap.add_argument("--synthetic", type=Path, default=_ROOT / "data/raw/synthetic_data.csv")
    ap.add_argument("--sample", type=int, default=None,
                    help="Subsample the real records for a fast smoke run.")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--out", type=Path, default=_OUT)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    res = {}
    if not args.skip_real:
        print("=== REAL DATA (Fellegi–Sunter / Splink) ===")
        cleaned = pd.read_parquet(args.cleaned)
        silver = pd.read_csv(args.silver)
        res["real"] = run_real(cleaned, silver, args.sample)
        e = res["real"]["evaluation_vs_silver"]
        print(f"  records={res['real']['n_records']:,} predicted_pairs={res['real']['n_predicted_pairs']:,} "
              f"silver_AUC={e['roc_auc']}")
        print(f"  FS vs rules: {res['real']['fs_vs_rules']}")

    if not args.skip_synth:
        print("=== SYNTHETIC DATA ===")
        syn = pd.read_csv(args.synthetic, dtype=str, keep_default_na=True, na_values=[""])
        syn["label"] = syn["label"].astype(int)
        res["synthetic"] = run_synthetic(syn)
        print(f"  records={res['synthetic']['n_records']:,} AUC={res['synthetic']['evaluation']['roc_auc']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nWrote {args.out}")
    print(f"Charts in {_CHARTS}")


if __name__ == "__main__":
    main()
