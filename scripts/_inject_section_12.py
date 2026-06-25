"""One-off injector: append Section 12 (enhanced_3 full-cohort evaluation) to the validation notebook.

Mirrors `_inject_section_11.py`: idempotent — re-runs replace the §12 block in
place rather than duplicating. Inserts §12 cells AFTER §11 and BEFORE the
trailing "## Reviewer judgments" markdown cell.

§12 is VM-only and self-contained: it trains FSEnhanced3 on the silver-label
TRAIN split and scores the **full candidate pool** (all Alliance data through
the FS model, bypassing the deterministic-rules stage), then renders the
evaluation suite the experiment requires:

  Performance:
    (1) Splink waterfall chart        — per-feature match-weight contribution
    (2) Confusion matrix (test split) — custom matplotlib
    (3) Score histogram + thresholds  — custom matplotlib (reuses §8 palette)
  Model diagnostics (Splink linker.visualisations.*):
    (4) match_weights_chart
    (5) m_u_parameters_chart
    (6) parameter_estimate_comparisons_chart
    (7) tf_adjustment_chart (per TF field)
  Edge / link evaluation (Splink linker.evaluation.* from the test labels table):
    (8) accuracy_analysis_from_labels_table  (accuracy chart)
    (9) accuracy_analysis_from_labels_table  (threshold-selection tool)
  Reference:
    (10) derived match-weight table (m / u / log2 BF) — CSV + printed markdown

Splink charts are Altair: saved via `chart.save(...)` as HTML (always) and PNG
(if vl-convert is available). Cells are defensive — a missing silver-labels file
or a Splink API drift degrades to a clear message instead of erroring the run.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks" / "fellegi_sunter" / "fellegi_sunter_validation.ipynb"
)
SECTION_MARKER = "# §12 SECTION MARKER — do not remove"


def _nid() -> str:
    return uuid.uuid4().hex[:12]


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


SECTION_12_CELLS = [
    md("""## 12. Enhanced_3 — full-cohort FS evaluation (silver-label trained)

enhanced_3 is the deliberately-simple FS variant: seven two-level (Exact / All-other) comparisons, m trained **supervised from the real-cohort silver labels**, lambda seeded from the deterministic rules, u from random sampling. This section trains it and runs **all Alliance data through the FS model directly** — the *full candidate pool*, **bypassing** the deterministic-rules stage — then renders the full evaluation suite.

| Aspect | enhanced_3 |
|---|---|
| Comparisons | FirstNM, LastNM, BirthDT, SSN, Email, Phones, Address — each null / exact / all-other |
| m training | `estimate_m_from_pairwise_labels` on the silver-label TRAIN split |
| lambda | `estimate_probability_two_random_records_match` (deterministic PRIOR_RULES) |
| u | `estimate_u_using_random_sampling` |
| Tiers | `0.95 / 0.40` (auto_merge / review_floor) |

> **VM-only.** Needs `data/silver_labels/silver_labels_v1_2026_06_21.csv` (gitignored PHI) and the full candidate-pairs parquet. Off-VM the setup cell skips and the rest of §12 no-ops.
"""),

    md("""### 12.0 Setup — train enhanced_3, score the full candidate pool, score the test split

Splits the silver labels (stratified, seed 42) into TRAIN (→ m) and TEST (held out). Trains `FSEnhanced3` and scores the entire candidate pool with `return_linker=True` so the trained Splink linker is available for the native charts below. Test-split metrics come from joining the scored full-pool output to the held-out labels (the silver pairs are present in the full pool by construction).
"""),

    code(SECTION_MARKER + """
# §12.0 setup — train enhanced_3 + score full candidate pool + test split.
import numpy as _np12

from models.common.versioning import latest_versioned as _latest_versioned_12
from models.experiments.fs_splink_enhanced_3.fs_enhanced_3 import FSEnhanced3

# Tier palette / figure conventions are defined in §8.0; redefine defensively.
try:
    _ = TIER_COLORS  # type: ignore[name-defined]
except NameError:
    TIER_COLORS = {"no_match": "#B5B5B5", "human_review": "#F0BE7E", "auto_merge": "#88B888"}
    TIER_EDGE = {"no_match": "#7F7F7F", "human_review": "#C68A3F", "auto_merge": "#4F8A4F"}
    TIER_ORDER_LOW_TO_HIGH = ["no_match", "human_review", "auto_merge"]

E3_AM, E3_FLOOR = 0.95, 0.40
SILVER_PATH_12 = PROJECT_ROOT / "data" / "silver_labels" / "silver_labels_v1_2026_06_21.csv"

# Resolve the FULL candidate-pairs parquet (VM convention: src/preprocessing/outputs/blocking;
# also try data/blocking). enhanced_3 scores this directly — no deterministic-rules stage.
def _resolve_full_candidate_pairs_12() -> Path | None:
    for _d in (
        PROJECT_ROOT / "src" / "preprocessing" / "outputs" / "blocking",
        PROJECT_ROOT / "data" / "blocking",
        PROJECT_ROOT / "src" / "features" / "outputs" / "blocking",
    ):
        if not _d.is_dir():
            continue
        try:
            return _latest_versioned_12(_d, "candidate_pairs_v*_*.parquet")
        except (FileNotFoundError, Exception):
            _g = sorted(_d.glob("candidate_pairs_*.parquet"))
            if _g:
                return _g[-1]
    return None

E3_READY = False
if not SILVER_PATH_12.exists():
    print(f"§12 SKIPPED — silver labels absent ({SILVER_PATH_12}). VM-only section.")
else:
    _cp_path = _resolve_full_candidate_pairs_12()
    if _cp_path is None:
        print("§12 SKIPPED — no candidate_pairs parquet found. Generate the full "
              "blocking output first (run_blocking.py).")
    else:
        print(f"silver labels   : {SILVER_PATH_12.name}")
        print(f"candidate pairs : {_cp_path.name}")
        # cleaned_path / df_clean are resolved/loaded earlier in the notebook (§2-3).
        _silver = pd.read_csv(SILVER_PATH_12, dtype={"PATID_A": str, "PATID_B": str})
        _silver["label"] = _silver["label"].astype(int)

        # Stratified 80/20 split (seed 42) — matches run_real_enhanced_3 defaults.
        _test = pd.concat([g.sample(frac=0.2, random_state=42)
                           for _, g in _silver.groupby("label")])
        _train = _silver.drop(_test.index).reset_index(drop=True)
        _test = _test.reset_index(drop=True)
        print(f"silver split    : train {len(_train)} (pos={int((_train.label==1).sum())}) "
              f"/ test {len(_test)} (pos={int((_test.label==1).sum())})")

        _cand = pd.read_parquet(_cp_path)
        print(f"Scoring full candidate pool: {len(_cand):,} pairs ...")

        _e3_model = FSEnhanced3(labels_df=_train, include_address=True, u_max_pairs=1e6)
        try:
            e3_classified, e3_linker = _e3_model.run(
                _cand, df_clean, full_output=True, return_linker=True,
            )
        except RuntimeError as _exc:
            print(f"Retrying with u_max_pairs=1e4 ({_exc})")
            _e3_model = FSEnhanced3(labels_df=_train, include_address=True, u_max_pairs=1e4)
            e3_classified, e3_linker = _e3_model.run(
                _cand, df_clean, full_output=True, return_linker=True,
            )

        # Held-out test metrics: canonicalize test pairs, inner-join to scored pool.
        _tc = _test.copy()
        _a = _tc[["PATID_A", "PATID_B"]].min(axis=1)
        _b = _tc[["PATID_A", "PATID_B"]].max(axis=1)
        _tc["PATID_A"], _tc["PATID_B"] = _a, _b
        e3_test = e3_classified.merge(
            _tc[["PATID_A", "PATID_B", "label"]], on=["PATID_A", "PATID_B"], how="inner",
        )
        print(f"test pairs scored in full pool: {len(e3_test)}/{len(_test)}")

        _tiers = e3_classified["classification_tier"].value_counts().to_dict()
        print("Full-pool tier breakdown:", {k: int(v) for k, v in _tiers.items()})
        E3_READY = True


def _save_altair_12(chart, name: str):
    \"\"\"Save a Splink/Altair chart as HTML (always) + PNG (if vl-convert present).\"\"\"
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

    md("""### 12.1 Performance — waterfall, confusion matrix, score histogram

1. **Waterfall chart** (Splink) — decomposes a handful of scored pairs into the per-feature match-weight contribution, so you can see *which* of the seven fields drove each score.
2. **Confusion matrix** (test split) — tier vs silver label on the held-out pairs (rows = label, cols = tier).
3. **Score histogram** — full-pool score distribution, stacked + coloured by tier, with the `0.40 / 0.95` threshold bands.
"""),

    code("""# §12.1 — waterfall + confusion matrix + score histogram.
if not E3_READY:
    print("§12.1 SKIPPED — setup did not complete.")
else:
    # (1) Waterfall — a few representative scored records from the trained linker.
    try:
        _wf_records = e3_linker.inference.predict().as_record_dict(limit=12)
        _wf = e3_linker.visualisations.waterfall_chart(_wf_records, filter_nulls=False)
        _save_altair_12(_wf, "enhanced_3_waterfall")
        display(_wf)
    except Exception as _e:
        print(f"§12.1 waterfall skipped: {_e}")

    # (2) Confusion matrix on the held-out test split.
    _TIER_ORDER = ["no_match", "human_review", "auto_merge"]
    _cm = (pd.crosstab(e3_test["label"], e3_test["classification_tier"])
             .reindex(index=[0, 1], columns=_TIER_ORDER, fill_value=0))
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.imshow(_cm.values, cmap="Greens", vmin=0, vmax=max(_cm.values.max(), 1))
    ax.set_xticks(range(3)); ax.set_xticklabels(_TIER_ORDER, rotation=20, ha="right")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["different (0)", "same (1)"])
    ax.set_xlabel("Predicted tier"); ax.set_ylabel("Silver label")
    for (i, j), v in _np12.ndenumerate(_cm.values):
        ax.text(j, i, str(int(v)), ha="center", va="center",
                color="white" if v > _cm.values.max() / 2 else "#222")
    ax.set_title(f"enhanced_3 — test confusion  ·  {VERSION_TAG}", fontweight="bold")
    ax.grid(False)
    _out = FIGURES_DIR / f"enhanced_3_confusion__{VERSION_TAG}.png"
    plt.savefig(_out); plt.show(); print(f"Saved: {_out.name}")

    # Precision / recall at auto_merge (test split).
    _tp = int(((e3_test.label == 1) & (e3_test.classification_tier == "auto_merge")).sum())
    _fp = int(((e3_test.label == 0) & (e3_test.classification_tier == "auto_merge")).sum())
    _fn = int(((e3_test.label == 1) & (e3_test.classification_tier != "auto_merge")).sum())
    _prec = _tp / (_tp + _fp) if (_tp + _fp) else float("nan")
    _rec = _tp / (_tp + _fn) if (_tp + _fn) else float("nan")
    print(f"auto_merge — precision={_prec:.1%}  recall={_rec:.1%}  (tp={_tp} fp={_fp} fn={_fn})")

    # (3) Full-pool score histogram, stacked by tier, with threshold bands.
    _bins = _np12.linspace(0, 1, 51)
    _centers = (_bins[:-1] + _bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _cum = _np12.zeros(len(_centers))
    for _t in TIER_ORDER_LOW_TO_HIGH:
        _s = e3_classified.loc[e3_classified["classification_tier"] == _t, "match_probability"]
        _h, _ = _np12.histogram(_s.to_numpy(), bins=_bins)
        ax.bar(_centers, _h, width=(_bins[1] - _bins[0]) * 0.95, bottom=_cum,
               color=TIER_COLORS[_t], edgecolor=TIER_EDGE[_t], linewidth=0.4,
               label=f"{_t}  ({int(_h.sum()):,})")
        _cum = _cum + _h
    ax.axvline(E3_FLOOR, color="#555", ls="--", lw=0.8)
    ax.axvline(E3_AM, color="#555", ls="--", lw=0.8)
    ax.set_yscale("log"); ax.set_xlim(0, 1)
    ax.set_xlabel("match probability"); ax.set_ylabel("count (log)")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    ax.set_title(f"enhanced_3 — full-pool score distribution  ·  {VERSION_TAG}", fontweight="bold")
    _out = FIGURES_DIR / f"enhanced_3_score_histogram__{VERSION_TAG}.png"
    plt.savefig(_out, bbox_inches="tight"); plt.show(); print(f"Saved: {_out.name}")
"""),

    md("""### 12.2 Model diagnostics — Splink parameter charts

Splink's native diagnostic charts on the trained linker:

- **match_weights_chart** — the learned weight (log2 Bayes factor) for every comparison level.
- **m_u_parameters_chart** — the underlying m and u probabilities each weight is derived from.
- **parameter_estimate_comparisons_chart** — compares estimates across training passes (here: the supervised m vs the random-sampling u).
- **tf_adjustment_chart** — per-value term-frequency adjustment for the high-cardinality identity fields (FirstNM, LastNM, Email).
"""),

    code("""# §12.2 — Splink parameter diagnostic charts.
if not E3_READY:
    print("§12.2 SKIPPED — setup did not complete.")
else:
    for _name, _fn in [
        ("enhanced_3_match_weights", lambda: e3_linker.visualisations.match_weights_chart()),
        ("enhanced_3_m_u_parameters", lambda: e3_linker.visualisations.m_u_parameters_chart()),
        ("enhanced_3_parameter_estimates",
         lambda: e3_linker.visualisations.parameter_estimate_comparisons_chart()),
    ]:
        try:
            _c = _fn()
            _save_altair_12(_c, _name)
            display(_c)
        except Exception as _e:
            print(f"§12.2 {_name} skipped: {_e}")

    # tf_adjustment_chart is per-column — only the TF-enabled identity fields.
    for _col in ["FirstNM_clean", "LastNM_clean", "Email_clean"]:
        try:
            _c = e3_linker.visualisations.tf_adjustment_chart(_col)
            _save_altair_12(_c, f"enhanced_3_tf_adjustment_{_col}")
            display(_c)
        except Exception as _e:
            print(f"§12.2 tf_adjustment[{_col}] skipped: {_e}")
"""),

    md("""### 12.3 Edge / link evaluation — accuracy + threshold-selection from the test labels

Registers the held-out **test split** as a Splink labels table and runs `linker.evaluation.accuracy_analysis_from_labels_table`:

- **accuracy chart** (`output_type="accuracy"`) — precision / recall / F1 etc. as a function of the match-weight threshold.
- **threshold-selection tool** (`output_type="threshold_selection"`) — the interactive tool for picking the operating threshold from labelled truth.

The labels table uses Splink's expected schema (`unique_id_l`, `unique_id_r`, `clerical_match_score`); `clerical_match_score` is the silver label (1.0 / 0.0).
"""),

    code("""# §12.3 — accuracy + threshold-selection from the test labels table.
if not E3_READY:
    print("§12.3 SKIPPED — setup did not complete.")
else:
    try:
        # Build a Splink labels table from the held-out test split.
        _lab = _tc[["PATID_A", "PATID_B", "label"]].copy()
        _labels_tbl = pd.DataFrame({
            "unique_id_l": _lab["PATID_A"].astype(str),
            "unique_id_r": _lab["PATID_B"].astype(str),
            "source_dataset_l": "__splink__input_table_0",
            "source_dataset_r": "__splink__input_table_0",
            "clerical_match_score": _lab["label"].astype(float),
        })
        _registered = e3_linker.table_management.register_labels_table(
            _labels_tbl, overwrite=True,
        )
        for _name, _otype in [
            ("enhanced_3_accuracy", "accuracy"),
            ("enhanced_3_threshold_selection", "threshold_selection"),
        ]:
            try:
                _c = e3_linker.evaluation.accuracy_analysis_from_labels_table(
                    _registered, output_type=_otype,
                )
                _save_altair_12(_c, _name)
                display(_c)
            except Exception as _e:
                print(f"§12.3 {_name} skipped: {_e}")
    except Exception as _e:
        print(f"§12.3 labels-table registration failed: {_e}")
        print("  Fallback: see the §12.1 confusion matrix + precision/recall print.")
"""),

    md("""### 12.4 Derived match-weight table (m / u / log2 Bayes factor)

The headline interpretability artifact: one row per comparison level with the trained m, u, and the log2 Bayes factor (the match weight). Saved as CSV for `docs/Fellegi-Sunter-Enhanced_3.md` and printed as a markdown table to paste into the build guide.
"""),

    code("""# §12.4 — derived match-weight table.
if not E3_READY:
    print("§12.4 SKIPPED — setup did not complete.")
else:
    _settings = e3_linker.misc.save_model_to_json()
    _rows = []
    for _comp in _settings.get("comparisons", []):
        _cname = _comp.get("output_column_name", "?")
        for _lvl in _comp.get("comparison_levels", []):
            if _lvl.get("is_null_level"):
                continue
            _m, _u = _lvl.get("m_probability"), _lvl.get("u_probability")
            _bf = (_np12.log2(_m / _u) if (_m and _u) else None)
            _rows.append({
                "comparison": _cname,
                "level": _lvl.get("label_for_charts", _lvl.get("sql_condition", "?")),
                "m": _m, "u": _u,
                "log2_bayes_factor": (round(_bf, 3) if _bf is not None else None),
            })
    _mw = pd.DataFrame(_rows)
    _out_csv = FIGURES_DIR / f"enhanced_3_match_weights__{VERSION_TAG}.csv"
    _mw.to_csv(_out_csv, index=False)
    print(f"Saved: {_out_csv.name}")
    print()
    # Markdown table for the doc (falls back to plain print if tabulate absent).
    try:
        print(_mw.to_markdown(index=False))
    except Exception as _e:
        print(f"(markdown render needs `tabulate`: {_e})")
        print(_mw.to_string(index=False))
"""),
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # Drop any prior §12 cells (so re-running this script replaces in-place).
    keep = []
    in_section_12 = False
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and src.lstrip().startswith("## 12."):
            in_section_12 = True
            continue
        if in_section_12:
            if c["cell_type"] == "markdown" and (
                src.lstrip().startswith("## Reviewer judgments")
                or src.lstrip().startswith("## 13.")
            ):
                in_section_12 = False
                keep.append(c)
                continue
            if SECTION_MARKER in src or src.lstrip().startswith("### 12.") or "§12" in src[:200]:
                continue
            continue
        keep.append(c)

    # Insert just before "## Reviewer judgments" (or at end if absent).
    insert_at = len(keep)
    for i, c in enumerate(keep):
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and src.lstrip().startswith("## Reviewer judgments"):
            insert_at = i
            break

    new_cells = keep[:insert_at] + SECTION_12_CELLS + keep[insert_at:]
    nb["cells"] = new_cells
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {NB_PATH} with {len(new_cells)} cells "
          f"({len(SECTION_12_CELLS)} new §12 cells inserted at index {insert_at})")


if __name__ == "__main__":
    main()
