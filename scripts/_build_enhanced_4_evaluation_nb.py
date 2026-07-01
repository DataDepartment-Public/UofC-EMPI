#!/usr/bin/env python3
"""Generator: writes notebooks/fellegi_sunter/fellegi_sunter_4_evaluation.ipynb from scratch.

Idempotent — overwrites on each run; byte-identical output because cell IDs are
derived from a seeded namespace UUID rather than random UUIDs.

Mirrors _inject_section_12.py conventions: md() / code() / _nid(),
json.dumps(nb, indent=1) + "\\n".

Run from the project root:
    python scripts/_build_enhanced_4_evaluation_nb.py
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks" / "fellegi_sunter" / "fellegi_sunter_4_evaluation.ipynb"
)

# Deterministic namespace so every run produces byte-identical cell IDs.
_NAMESPACE = uuid.UUID("e4e4e4e4-e4e4-e4e4-e4e4-e4e4e4e4e4e4")
_cell_seq: list[int] = [0]  # mutable counter, reset on each import


def _nid() -> str:
    _cell_seq[0] += 1
    return uuid.uuid5(_NAMESPACE, str(_cell_seq[0])).hex[:12]


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _nid(),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": _nid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook cells
# ═══════════════════════════════════════════════════════════════════════════════

CELLS = [

    # ── 1. Title ────────────────────────────────────────────────────────────
    md("""\
# Fellegi-Sunter Enhanced_4 — full-cohort evaluation

This notebook is **VM-only** and **PHI-bearing** when executed.  It mirrors the
enhanced_3 §12 evaluation suite for **enhanced_4**, which adds three structural
precision levers over enhanced_3 (addressing its ≈78 % auto_merge precision gap):

| Lever | Description |
|---|---|
| Multi-level comparisons | Explicit conflict-vs-missing mismatch levels — SSN 9-digit mismatch, JW<0.5 name conflict, etc. — so contradicting identifiers contribute real negative weight instead of falling into an uninformative "else" |
| Corroboration gate | Post-tier demotion of `auto_merge` pairs that lack a person-unique (SSN/Email) or dual-household (Phone+Address) corroborating signal |
| Grounded SSN/Email exact u | `1/n_distinct` priors pinned via `fix_u_probability=True` before training so both exact levels have a well-founded Bayes factor |

`m` stays **purely supervised** from the real-cohort silver labels; `lambda` is seeded
from the deterministic rules; `u` is from random sampling with SSN/Email exact levels
grounded.

> **VM-only.**  Needs `data/silver_labels/silver_labels_v1_2026_06_21.csv`
> (gitignored PHI) and the full candidate-pairs parquet.  Off-VM the setup cell
> prints a skip message and all subsequent cells no-op.
"""),

    # ── 2. Imports ──────────────────────────────────────────────────────────
    code("""\
# enhanced_4 eval setup — stdlib + project imports
from pathlib import Path
import sys, re, json

# Notebook lives in notebooks/fellegi_sunter/ — project root is two levels up.
PROJECT_ROOT = Path.cwd().resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from models.common.versioning import latest_versioned
from models.experiments.fs_splink_enhanced_4.fs_enhanced_4 import FSEnhanced4
from models.experiments.fs_splink_enhanced_4.corroboration_gate import (
    CORROBORATION_COL,
    DEMOTED_REASON,
)

print(f"PROJECT_ROOT : {PROJECT_ROOT}")
print(f"Python       : {sys.version.split()[0]}")
"""),

    # ── 3. Palette + styling ─────────────────────────────────────────────────
    code("""\
# Palette + presentation styling (mirrors §8.0 of fellegi_sunter_validation.ipynb)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

TIER_COLORS = {
    "no_match": "#B5B5B5",
    "human_review": "#F0BE7E",
    "auto_merge": "#88B888",
}
TIER_EDGE = {
    "no_match": "#7F7F7F",
    "human_review": "#C68A3F",
    "auto_merge": "#4F8A4F",
}
TIER_ORDER_LOW_TO_HIGH = ["no_match", "human_review", "auto_merge"]

FIGURES_DIR = Path.cwd() / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
print(f"FIGURES_DIR : {FIGURES_DIR}")
"""),

    # ── 4. Input resolution + load ───────────────────────────────────────────
    code("""\
# Input resolution + load (mirrors §2-3 of fellegi_sunter_validation.ipynb)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

try:
    cleaned_path = latest_versioned(PROCESSED_DIR, "MDM_Population_cleaned_v*_*.parquet")
    df_clean = pd.read_parquet(cleaned_path)
    _m_tag = re.search(r"(v\\d+_\\d{4}_\\d{2}_\\d{2})", cleaned_path.name)
    VERSION_TAG = _m_tag.group(1) if _m_tag else "unversioned"
    print(f"cleaned_path : {cleaned_path.name}")
    print(f"VERSION_TAG  : {VERSION_TAG}")
    print(f"df_clean     : {df_clean.shape[0]:,} records, {df_clean.shape[1]} columns")
except Exception as _load_e:
    print("INPUT SKIPPED — VM-only section.")
    print(f"  Could not resolve cleaned parquet from {PROCESSED_DIR}: {_load_e}")
    cleaned_path = None
    df_clean = None
    VERSION_TAG = "unversioned"
"""),

    # ── §12.0 md ────────────────────────────────────────────────────────────
    md("""\
## Enhanced_4 — full-cohort FS evaluation (silver-label trained)

enhanced_4 extends enhanced_3's methodology (supervised m from real-cohort silver labels,
lambda seeded from deterministic rules, u from random sampling) with three structural fixes:

| Aspect | enhanced_4 |
|---|---|
| Comparisons | Multi-level with explicit conflict-vs-missing mismatch levels per field |
| m training | `estimate_m_from_pairwise_labels` on the silver-label TRAIN split |
| lambda | `estimate_probability_two_random_records_match` (deterministic PRIOR_RULES) |
| u | `estimate_u_using_random_sampling` + SSN/Email exact levels grounded to `1/n_distinct` |
| Corroboration gate | `auto_merge` pairs demoted unless SSN/Email agree OR (Phone+Address both agree) |
| Tiers | `0.95 / 0.40` (auto_merge / review_floor) |

### Setup — train enhanced_4, score the full candidate pool, score the test split

Splits the silver labels (stratified, seed 42) into TRAIN (→ m) and TEST (held out). Trains
`FSEnhanced4` and scores the entire candidate pool with `return_linker=True` so the trained
Splink linker is available for the native charts below. Test-split metrics come from joining
the scored full-pool output to the held-out labels (the silver pairs are present in the full
pool by construction).

> **VM-only.** Needs `data/silver_labels/silver_labels_v1_2026_06_21.csv` and the full
> candidate-pairs parquet (run `run_blocking.py` first if absent).
"""),

    # ── §12.0 code ───────────────────────────────────────────────────────────
    code("""\
# enhanced_4 eval setup — train FSEnhanced4, score full candidate pool, score test split.

SILVER_PATH_E4 = PROJECT_ROOT / "data" / "silver_labels" / "silver_labels_v1_2026_06_21.csv"
E4_AM, E4_FLOOR = 0.95, 0.40


def _resolve_full_candidate_pairs_e4():
    # Resolve the full candidate-pairs parquet (VM convention: multiple search dirs).
    for _d in (
        PROJECT_ROOT / "src" / "preprocessing" / "outputs" / "blocking",
        PROJECT_ROOT / "data" / "blocking",
        PROJECT_ROOT / "src" / "features" / "outputs" / "blocking",
    ):
        if not _d.is_dir():
            continue
        try:
            return latest_versioned(_d, "candidate_pairs_v*_*.parquet")
        except (FileNotFoundError, Exception):
            _g = sorted(_d.glob("candidate_pairs_*.parquet"))
            if _g:
                return _g[-1]
    return None


E4_READY = False
if df_clean is None:
    print("enhanced_4 eval SKIPPED — cleaned frame not loaded (see input-resolution cell).")
elif not SILVER_PATH_E4.exists():
    print(f"enhanced_4 eval SKIPPED — silver labels absent ({SILVER_PATH_E4}). VM-only section.")
else:
    _cp_path = _resolve_full_candidate_pairs_e4()
    if _cp_path is None:
        print("enhanced_4 eval SKIPPED — no candidate_pairs parquet found. "
              "Generate the full blocking output first (run_blocking.py).")
    else:
        print(f"silver labels   : {SILVER_PATH_E4.name}")
        print(f"candidate pairs : {_cp_path.name}")
        _silver = pd.read_csv(SILVER_PATH_E4, dtype={"PATID_A": str, "PATID_B": str})
        # Silver labels: column `silver_label` with boolean True/False. Normalize to int.
        _src_col = "silver_label" if "silver_label" in _silver.columns else "label"
        _silver["label"] = _silver[_src_col].map(
            {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
        ).astype(int)

        # Stratified 80/20 split (seed 42) — matches run_real_enhanced_4 defaults.
        _test = pd.concat([g.sample(frac=0.2, random_state=42)
                           for _, g in _silver.groupby("label")])
        _train = _silver.drop(_test.index).reset_index(drop=True)
        _test = _test.reset_index(drop=True)
        print(f"silver split    : train {len(_train)} (pos={int((_train.label==1).sum())}) "
              f"/ test {len(_test)} (pos={int((_test.label==1).sum())})")

        _cand = pd.read_parquet(_cp_path)
        print(f"Scoring full candidate pool: {len(_cand):,} pairs ...")

        _e4_model = FSEnhanced4(labels_df=_train, include_address=True, u_max_pairs=1e6)
        try:
            e4_classified, e4_linker = _e4_model.run(
                _cand, df_clean, full_output=True, return_linker=True,
            )
        except RuntimeError as _exc:
            print(f"Retrying with u_max_pairs=1e4 ({_exc})")
            _e4_model = FSEnhanced4(labels_df=_train, include_address=True, u_max_pairs=1e4)
            e4_classified, e4_linker = _e4_model.run(
                _cand, df_clean, full_output=True, return_linker=True,
            )

        # Held-out test metrics: canonicalize test pairs, inner-join to scored pool.
        _tc = _test.copy()
        _a = _tc[["PATID_A", "PATID_B"]].min(axis=1)
        _b = _tc[["PATID_A", "PATID_B"]].max(axis=1)
        _tc["PATID_A"], _tc["PATID_B"] = _a, _b
        e4_test = e4_classified.merge(
            _tc[["PATID_A", "PATID_B", "label"]], on=["PATID_A", "PATID_B"], how="inner",
        )
        print(f"test pairs scored in full pool: {len(e4_test)}/{len(_test)}")

        _tiers = e4_classified["classification_tier"].value_counts().to_dict()
        print("Full-pool tier breakdown:", {k: int(v) for k, v in _tiers.items()})
        E4_READY = True


def _save_altair_e4(chart, name: str):
    # Save a Splink/Altair chart as HTML (always) + PNG (if vl-convert present).
    base = FIGURES_DIR / f"{name}__{VERSION_TAG}"
    try:
        chart.save(str(base) + ".html")
        print(f"Saved: {base.name}.html")
    except Exception as _e:
        print(f"  (HTML save failed for {name}: {_e})")
    try:
        chart.save(str(base) + ".png")
        print(f"Saved: {base.name}.png")
    except Exception:
        print(f"  (PNG save skipped for {name} — install vl-convert-python to enable)")
    return chart
"""),

    # ── §12.1 md ────────────────────────────────────────────────────────────
    md("""\
### Enhanced_4 — performance: waterfall, confusion matrix, score histogram

1. **Waterfall chart** (Splink) — decomposes a handful of scored pairs into per-feature
   match-weight contributions, so you can see *which* fields drove each score.
2. **Confusion matrix** (test split) — tier vs silver label on the held-out pairs
   (rows = label, cols = tier).
3. **Score histogram** — full-pool score distribution, stacked and coloured by tier,
   with the `0.40 / 0.95` threshold bands.
"""),

    # ── §12.1 code ───────────────────────────────────────────────────────────
    code("""\
# enhanced_4 — waterfall + confusion matrix + score histogram.
if not E4_READY:
    print("enhanced_4 performance charts SKIPPED — setup did not complete.")
else:
    # (1) Waterfall — representative scored records from the trained linker.
    try:
        _wf_records = e4_linker.inference.predict().as_record_dict(limit=12)
        _wf = e4_linker.visualisations.waterfall_chart(_wf_records, filter_nulls=False)
        _save_altair_e4(_wf, "enhanced_4_waterfall")
        display(_wf)
    except Exception as _e:
        print(f"waterfall skipped: {_e}")

    # (2) Confusion matrix on the held-out test split.
    _TIER_ORDER = ["no_match", "human_review", "auto_merge"]
    _cm = (pd.crosstab(e4_test["label"], e4_test["classification_tier"])
             .reindex(index=[0, 1], columns=_TIER_ORDER, fill_value=0))
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.imshow(_cm.values, cmap="Greens", vmin=0, vmax=max(_cm.values.max(), 1))
    ax.set_xticks(range(3)); ax.set_xticklabels(_TIER_ORDER, rotation=20, ha="right")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["different (0)", "same (1)"])
    ax.set_xlabel("Predicted tier"); ax.set_ylabel("Silver label")
    for (i, j), v in np.ndenumerate(_cm.values):
        ax.text(j, i, str(int(v)), ha="center", va="center",
                color="white" if v > _cm.values.max() / 2 else "#222")
    ax.set_title(f"enhanced_4 — test confusion  ·  {VERSION_TAG}", fontweight="bold")
    ax.grid(False)
    _out = FIGURES_DIR / f"enhanced_4_confusion__{VERSION_TAG}.png"
    plt.savefig(_out); plt.show(); print(f"Saved: {_out.name}")

    # Precision / recall at auto_merge (test split).
    _tp = int(((e4_test.label == 1) & (e4_test.classification_tier == "auto_merge")).sum())
    _fp = int(((e4_test.label == 0) & (e4_test.classification_tier == "auto_merge")).sum())
    _fn = int(((e4_test.label == 1) & (e4_test.classification_tier != "auto_merge")).sum())
    _prec = _tp / (_tp + _fp) if (_tp + _fp) else float("nan")
    _rec  = _tp / (_tp + _fn) if (_tp + _fn) else float("nan")
    print(f"auto_merge — precision={_prec:.1%}  recall={_rec:.1%}  (tp={_tp} fp={_fp} fn={_fn})")

    # (3) Full-pool score histogram, stacked by tier, with threshold bands.
    _bins = np.linspace(0, 1, 51)
    _centers = (_bins[:-1] + _bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _cum = np.zeros(len(_centers))
    for _t in TIER_ORDER_LOW_TO_HIGH:
        _s = e4_classified.loc[e4_classified["classification_tier"] == _t, "match_probability"]
        _h, _ = np.histogram(_s.to_numpy(), bins=_bins)
        ax.bar(_centers, _h, width=(_bins[1] - _bins[0]) * 0.95, bottom=_cum,
               color=TIER_COLORS[_t], edgecolor=TIER_EDGE[_t], linewidth=0.4,
               label=f"{_t}  ({int(_h.sum()):,})")
        _cum = _cum + _h
    ax.axvline(E4_FLOOR, color="#555", ls="--", lw=0.8)
    ax.axvline(E4_AM,    color="#555", ls="--", lw=0.8)
    ax.set_yscale("log"); ax.set_xlim(0, 1)
    ax.set_xlabel("match probability"); ax.set_ylabel("count (log)")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    ax.set_title(f"enhanced_4 — full-pool score distribution  ·  {VERSION_TAG}", fontweight="bold")
    _out = FIGURES_DIR / f"enhanced_4_score_histogram__{VERSION_TAG}.png"
    plt.savefig(_out, bbox_inches="tight"); plt.show(); print(f"Saved: {_out.name}")
"""),

    # ── NEW: corroboration gate + grounded-u (md) ────────────────────────────
    md("""\
### Enhanced_4 — corroboration gate + grounded-u summary

Summarises the two enhanced_4-specific additions:

**Grounded u** — The SSN and Email *exact* comparison levels have their u grounded to
`1 / n_distinct(col)` before training, rather than left at the Splink default (these exact
levels never appear in random pairs so random-sampling u is untrained in enhanced_3).
The grounded values are read from `_e4_model._grounded_u`.

**Corroboration gate** — After tier classification, `auto_merge` pairs are demoted to
`human_review` unless they carry a person-unique (SSN/Email agreement) or dual-household
(Phone+Address both agree) corroborating signal.  Demotion counts are reported for the full
pool and the test split.  No identifier values are printed — aggregate counts only.
"""),

    # ── NEW: corroboration gate + grounded-u (code) ──────────────────────────
    code("""\
# enhanced_4 — corroboration gate + grounded-u summary.
if not E4_READY:
    print("Gate/grounded-u summary SKIPPED — setup did not complete.")
else:
    # --- Grounded-u values used by the model --------------------------------
    print("Grounded SSN/Email exact-level u values (1 / n_distinct):")
    if _e4_model._grounded_u:
        for _comp_name, _u_val in _e4_model._grounded_u.items():
            print(f"  {_comp_name:10s}: u = {_u_val:.3g}")
    else:
        print("  (none recorded — check _ground_untrained_u ran correctly)")

    # --- Full-pool corroboration gate demotion counts -----------------------
    print()
    print("Full-pool corroboration gate (CORROBORATION_COL value_counts):")
    if CORROBORATION_COL in e4_classified.columns:
        _gate_vc = e4_classified[CORROBORATION_COL].value_counts(dropna=False)
        print(_gate_vc.to_string())
        _n_demoted_full = int((e4_classified[CORROBORATION_COL] == DEMOTED_REASON).sum())
        print(f"  pairs demoted from auto_merge to human_review (full pool): {_n_demoted_full:,}")
    else:
        print(f"  CORROBORATION_COL ({CORROBORATION_COL!r}) not present in e4_classified — "
              "check corroboration_gate import.")

    # --- Test-split corroboration gate demotion counts ----------------------
    print()
    print("Test-split corroboration gate demotion count:")
    if CORROBORATION_COL in e4_test.columns:
        _n_demoted_test = int((e4_test[CORROBORATION_COL] == DEMOTED_REASON).sum())
        print(f"  pairs demoted from auto_merge to human_review (test split): {_n_demoted_test}")
    else:
        print(f"  CORROBORATION_COL ({CORROBORATION_COL!r}) not present in e4_test")

    # --- Bar chart: corroboration column value counts (auto_merge pairs only) ---
    if CORROBORATION_COL in e4_classified.columns:
        _gate_counts = (
            e4_classified[CORROBORATION_COL].fillna("kept_auto_merge").value_counts()
        )
        fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
        _bar_colors = [
            TIER_EDGE.get("auto_merge", "#4F8A4F") if "demoted" in str(_k) else "#aaaaaa"
            for _k in _gate_counts.index
        ]
        ax.barh(range(len(_gate_counts)), _gate_counts.values,
                color=_bar_colors, edgecolor="#555", linewidth=0.5)
        ax.set_yticks(range(len(_gate_counts)))
        ax.set_yticklabels([str(_k) for _k in _gate_counts.index])
        ax.set_xlabel("pair count")
        ax.set_title(
            f"enhanced_4 — corroboration gate (auto_merge pairs only)  ·  {VERSION_TAG}",
            fontweight="bold",
        )
        _out = FIGURES_DIR / f"enhanced_4_gate_summary__{VERSION_TAG}.png"
        plt.savefig(_out, bbox_inches="tight"); plt.show(); print(f"Saved: {_out.name}")
"""),

    # ── §12.2 md ────────────────────────────────────────────────────────────
    md("""\
### Enhanced_4 — model diagnostics: Splink parameter charts

Splink's native diagnostic charts on the trained linker:

- **match_weights_chart** — the learned weight (log2 Bayes factor) for every comparison level.
- **m_u_parameters_chart** — the underlying m and u probabilities each weight is derived from.
- **parameter_estimate_comparisons_chart** — compares estimates across training passes
  (supervised m vs random-sampling u, showing the grounded SSN/Email exact levels).
- **tf_adjustment_chart** — per-value term-frequency adjustment for FirstNM, LastNM, Email.
"""),

    # ── §12.2 code ───────────────────────────────────────────────────────────
    code("""\
# enhanced_4 — Splink parameter diagnostic charts.
if not E4_READY:
    print("enhanced_4 parameter charts SKIPPED — setup did not complete.")
else:
    for _name, _fn in [
        ("enhanced_4_match_weights",
         lambda: e4_linker.visualisations.match_weights_chart()),
        ("enhanced_4_m_u_parameters",
         lambda: e4_linker.visualisations.m_u_parameters_chart()),
        ("enhanced_4_parameter_estimates",
         lambda: e4_linker.visualisations.parameter_estimate_comparisons_chart()),
    ]:
        try:
            _c = _fn()
            _save_altair_e4(_c, _name)
            display(_c)
        except Exception as _e:
            print(f"enhanced_4 {_name} skipped: {_e}")

    # tf_adjustment_chart is per-column — only the TF-enabled identity fields.
    for _col in ["FirstNM_clean", "LastNM_clean", "Email_clean"]:
        try:
            _c = e4_linker.visualisations.tf_adjustment_chart(_col)
            _save_altair_e4(_c, f"enhanced_4_tf_adjustment_{_col}")
            display(_c)
        except Exception as _e:
            print(f"tf_adjustment[{_col}] skipped: {_e}")
"""),

    # ── §12.3 md ────────────────────────────────────────────────────────────
    md("""\
### Enhanced_4 — edge/link evaluation: accuracy + threshold-selection from test labels

Registers the held-out **test split** as a Splink labels table and runs
`linker.evaluation.accuracy_analysis_from_labels_table`:

- **accuracy chart** (`output_type="accuracy"`) — precision / recall / F1 etc. as a function
  of the match-weight threshold.
- **threshold-selection tool** (`output_type="threshold_selection"`) — the interactive tool
  for picking the operating threshold from labelled truth.

The labels table uses Splink's expected schema (`unique_id_l`, `unique_id_r`,
`clerical_match_score`); `clerical_match_score` is the silver label (1.0 / 0.0).
"""),

    # ── §12.3 code ───────────────────────────────────────────────────────────
    code("""\
# enhanced_4 — accuracy + threshold-selection from the test labels table.
if not E4_READY:
    print("enhanced_4 edge evaluation SKIPPED — setup did not complete.")
else:
    try:
        _lab = _tc[["PATID_A", "PATID_B", "label"]].copy()
        _labels_tbl = pd.DataFrame({
            "unique_id_l": _lab["PATID_A"].astype(str),
            "unique_id_r": _lab["PATID_B"].astype(str),
            "source_dataset_l": "__splink__input_table_0",
            "source_dataset_r": "__splink__input_table_0",
            "clerical_match_score": _lab["label"].astype(float),
        })
        _registered = e4_linker.table_management.register_labels_table(
            _labels_tbl, overwrite=True,
        )
        for _name, _otype in [
            ("enhanced_4_accuracy", "accuracy"),
            ("enhanced_4_threshold_selection", "threshold_selection"),
        ]:
            try:
                _c = e4_linker.evaluation.accuracy_analysis_from_labels_table(
                    _registered, output_type=_otype,
                )
                _save_altair_e4(_c, _name)
                display(_c)
            except Exception as _e:
                print(f"enhanced_4 {_name} skipped: {_e}")
    except Exception as _e:
        print(f"enhanced_4 labels-table registration failed: {_e}")
        print("  Fallback: see the confusion matrix + precision/recall print above.")
"""),

    # ── §12.4 md ────────────────────────────────────────────────────────────
    md("""\
### Enhanced_4 — derived match-weight table (m / u / log2 Bayes factor)

The headline interpretability artifact: one row per comparison level with the trained m, u,
and the log2 Bayes factor (the match weight).  Saved as CSV for
`docs/Fellegi-Sunter-Enhanced_4.md` and printed as a markdown table.
"""),

    # ── §12.4 code ───────────────────────────────────────────────────────────
    code("""\
# enhanced_4 — derived match-weight table.
if not E4_READY:
    print("enhanced_4 match-weight table SKIPPED — setup did not complete.")
else:
    _settings_e4 = e4_linker.misc.save_model_to_json()
    _rows_e4 = []
    for _comp in _settings_e4.get("comparisons", []):
        _cname = _comp.get("output_column_name", "?")
        for _lvl in _comp.get("comparison_levels", []):
            if _lvl.get("is_null_level"):
                continue
            _m, _u = _lvl.get("m_probability"), _lvl.get("u_probability")
            _bf = (np.log2(_m / _u) if (_m and _u) else None)
            _rows_e4.append({
                "comparison": _cname,
                "level": _lvl.get("label_for_charts", _lvl.get("sql_condition", "?")),
                "m": _m,
                "u": _u,
                "log2_bayes_factor": (round(_bf, 3) if _bf is not None else None),
            })
    _mw_e4 = pd.DataFrame(_rows_e4)
    _out_csv = FIGURES_DIR / f"enhanced_4_match_weights__{VERSION_TAG}.csv"
    _mw_e4.to_csv(_out_csv, index=False)
    print(f"Saved: {_out_csv.name}")
    print()
    # Markdown table for the doc (falls back to plain print if tabulate absent).
    try:
        print(_mw_e4.to_markdown(index=False))
    except Exception as _e:
        print(f"(markdown render needs `tabulate`: {_e})")
        print(_mw_e4.to_string(index=False))
"""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Writer
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {NB_PATH}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
