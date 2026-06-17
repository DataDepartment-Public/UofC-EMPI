"""One-off injector: append Section 10 (head-to-head) cells to the FS validation notebook.

Inserts before the trailing "Reviewer judgments" markdown cell so the labels
remain at the bottom of the file. Safe to re-run — the script detects an
existing §10 setup cell and replaces all §10 cells in-place rather than
duplicating them.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fellegi_sunter" / "fellegi_sunter_validation.ipynb"
SECTION_MARKER = "# §10 SECTION MARKER — do not remove"


import uuid


def _nid() -> str:
    return uuid.uuid4().hex[:12]


def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": _nid(), "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": _nid(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


SECTION_10_CELLS = [
    md("""## 10. Head-to-head — baseline vs combined system (rules + enhanced FS)

This section evaluates the **combined Stage-3 + Stage-4 system** against the original FS Splink baseline on the 42 manually-labeled pairs at the bottom of this notebook. The combined-system tier for a pair is:

- **`auto_merge`** if the pair was confirmed by the deterministic rules (`src/models/deterministic_rules.py`) — written to `data/matches/matches_<run_id>.parquet`.
- otherwise, the enhanced FS classification — written to `data/matches_model/matches_model_<run_id>.parquet` (`ProbabilisticMatches` contract).
- pairs the rules dropped *and* the enhanced FS never scored (e.g., not in the non_matches input) fall through to `no_match`.

The 5-column `eval_schema` contract (`models/common/eval_schema.py`) is what makes this apples-to-apples: both `fs_splink_baseline` and `fs_splink_enhanced` write the same `PATID_A | PATID_B | model_name | score | predicted_tier` projection, so a single join on `(PATID_A, PATID_B)` aligns the two models without leaking implementation details.

> The 42 labels were drawn from the baseline FS run. Some labeled pairs may have been **auto-confirmed by the deterministic rules before reaching the enhanced FS** — those count as combined-system `auto_merge` and the rules stage is responsible for them. §10.1 surfaces this attribution.
"""),
    md("""### 10.1 Confusion matrix — combined system vs 42 reviewer labels

Parses the reviewer-judgment markdown cell at the bottom of this notebook (so the matrix tracks any label edits without code changes), looks each pair up in:

1. `data/matches/matches_<run_id>.parquet` (deterministic confirmations), then
2. `data/matches_model/matches_model_<run_id>.parquet` (enhanced FS) when not rule-confirmed,
3. and the in-memory baseline `df_scored` for the baseline side of the comparison.
"""),
    code(SECTION_MARKER + """
# §10 setup — resolve artifacts, parse labels, attach combined-system tier.
import json as _json
import re as _re10

MATCHES_DIR = PROJECT_ROOT / "data" / "matches"
MATCHES_MODEL_DIR = PROJECT_ROOT / "data" / "matches_model"


def _latest_by_mtime(dir_: Path, pattern: str) -> Path | None:
    if not dir_.exists():
        return None
    paths = sorted(dir_.glob(pattern), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


det_matches_path = _latest_by_mtime(MATCHES_DIR, "matches_*.parquet")
enh_matches_path = _latest_by_mtime(MATCHES_MODEL_DIR, "matches_model_*.parquet")
if enh_matches_path is None:
    enh_matches_path = _latest_by_mtime(MATCHES_MODEL_DIR, "*.parquet")

print(f"Deterministic matches : {det_matches_path}")
print(f"Enhanced FS matches   : {enh_matches_path}")
if det_matches_path is None or enh_matches_path is None:
    raise FileNotFoundError(
        "§10 expects `python -m src.pipeline` to have been run so that both "
        "data/matches/ and data/matches_model/ contain at least one parquet."
    )

df_rules = pd.read_parquet(det_matches_path)[["PATID_A", "PATID_B"]].copy()
df_rules["rule_confirmed"] = True

df_enhanced = pd.read_parquet(enh_matches_path).copy()
# Normalise tier column name — ProbabilisticMatches uses `classification_tier`.
_enh_tier_col = "classification_tier" if "classification_tier" in df_enhanced.columns else "predicted_tier"
_enh_score_col = "score" if "score" in df_enhanced.columns else "match_probability"
df_enhanced = df_enhanced.rename(columns={_enh_tier_col: "enhanced_tier", _enh_score_col: "enhanced_score"})
_enh_keep = ["PATID_A", "PATID_B", "enhanced_tier", "enhanced_score"]
if "veto_reason" in df_enhanced.columns:
    _enh_keep.append("veto_reason")
df_enhanced = df_enhanced[_enh_keep]

# ---- Parse labels straight out of the reviewer-judgments markdown ---------
_label_re = _re.compile(
    r"PATID_A=<([0-9A-Fa-f]+)>\s*PATID_B=<([0-9A-Fa-f]+)>[^\n]*?score=<([0-9.]+)>"
    r"[^v]*?verdict:\s*<([a-z\- ]+)>",
    _re.IGNORECASE,
)
_nb = _json.loads(Path("fellegi_sunter_validation.ipynb").read_text())
_labels_md = ""
for _c in _nb["cells"]:
    _src = "".join(_c["source"])
    if _c["cell_type"] == "markdown" and _src.lstrip().startswith("## Reviewer judgments"):
        _labels_md = _src
        break
_labels: list[dict] = []
for _m in _label_re.finditer(_labels_md):
    _labels.append({
        "PATID_A": _m.group(1).upper(),
        "PATID_B": _m.group(2).upper(),
        "baseline_score": float(_m.group(3)),
        "verdict": _m.group(4).strip().lower(),
    })
df_labels = pd.DataFrame(_labels).drop_duplicates(["PATID_A", "PATID_B"]).reset_index(drop=True)
print(f"Parsed {len(df_labels)} labeled pairs from reviewer-judgments markdown")

# ---- Attach baseline tier from in-memory df_scored ------------------------
_b = df_scored[["PATID_A", "PATID_B", "classification_tier", "match_probability"]].rename(
    columns={"classification_tier": "baseline_tier", "match_probability": "baseline_score_live"}
)
df_labels = df_labels.merge(_b, on=["PATID_A", "PATID_B"], how="left")

# ---- Attach rules-stage outcome + enhanced tier ---------------------------
df_labels = df_labels.merge(df_rules, on=["PATID_A", "PATID_B"], how="left")
df_labels["rule_confirmed"] = df_labels["rule_confirmed"].fillna(False)
df_labels = df_labels.merge(df_enhanced, on=["PATID_A", "PATID_B"], how="left")

def _combined_tier(row: pd.Series) -> str:
    if row["rule_confirmed"]:
        return "auto_merge"
    if pd.notna(row.get("enhanced_tier")):
        return row["enhanced_tier"]
    return "no_match"  # dropped pre-enhanced and not rule-confirmed

df_labels["combined_tier"] = df_labels.apply(_combined_tier, axis=1)

# ---- Normalise verdict labels --------------------------------------------
_VERDICT_MAP = {"same": "same", "different": "different", "unsure": "unsure"}
df_labels["verdict_norm"] = df_labels["verdict"].map(lambda v: _VERDICT_MAP.get(v, "unsure"))
print(df_labels[["baseline_tier", "rule_confirmed", "enhanced_tier", "combined_tier", "verdict_norm"]]
      .value_counts(dropna=False).head(20))
"""),
    code("""# §10.1 — confusion matrix: baseline vs combined system on 42 labels.
from matplotlib.patches import Rectangle

_TIER_ORDER = ["no_match", "human_review", "auto_merge"]
_VERDICT_ORDER = ["different", "unsure", "same"]


def _confusion(df: pd.DataFrame, tier_col: str) -> pd.DataFrame:
    cm = pd.crosstab(df["verdict_norm"], df[tier_col]).reindex(
        index=_VERDICT_ORDER, columns=_TIER_ORDER, fill_value=0
    )
    return cm


cm_baseline = _confusion(df_labels.assign(baseline_tier=df_labels["baseline_tier"].fillna("no_match")), "baseline_tier")
cm_combined = _confusion(df_labels, "combined_tier")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
for ax, cm, title in zip(axes, [cm_baseline, cm_combined],
                          ["Baseline FS", "Combined: rules + enhanced FS"]):
    im = ax.imshow(cm.values, cmap="Greens", vmin=0, vmax=max(cm.values.max(), 1))
    ax.set_xticks(range(len(_TIER_ORDER)))
    ax.set_xticklabels(_TIER_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(_VERDICT_ORDER)))
    ax.set_yticklabels(_VERDICT_ORDER)
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Reviewer verdict")
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(cm.values):
        ax.text(j, i, str(int(v)), ha="center", va="center",
                color="white" if v > cm.values.max() / 2 else "#222")
    ax.grid(False)

fig.suptitle(
    f"Confusion matrix — 42 labeled pairs  ·  {VERSION_TAG}",
    fontsize=13, fontweight="bold", y=1.05,
)
out_path = FIGURES_DIR / f"confusion_matrix_baseline_vs_combined__{VERSION_TAG}.png"
plt.savefig(out_path)
plt.show()
print(f"Saved: {out_path}")

# ---- Quantitative summaries -----------------------------------------------
def _summarise(cm: pd.DataFrame, label: str) -> None:
    n = int(cm.values.sum())
    am_diff = int(cm.loc["different", "auto_merge"])
    am_same = int(cm.loc["same", "auto_merge"])
    am_total = int(cm["auto_merge"].sum())
    same_total = int(cm.loc["same"].sum())
    am_prec = (am_same / am_total) if am_total else float("nan")
    same_rec_am_or_hr = (cm.loc["same", "auto_merge"] + cm.loc["same", "human_review"])
    same_recall = (same_rec_am_or_hr / same_total) if same_total else float("nan")
    print(f"{label}: n={n} | auto_merge precision (same/total) = {am_prec:.2%} "
          f"({am_same}/{am_total}) | same-recall to AM∪HR = {same_recall:.2%} "
          f"({int(same_rec_am_or_hr)}/{same_total}) | false-merges (different→AM) = {am_diff}")

_summarise(cm_baseline, "Baseline      ")
_summarise(cm_combined, "Combined sys  ")
"""),
    md("""### 10.2 Tier-shift — baseline → combined system

For each labeled pair, draw a ribbon from its baseline tier to its combined-system tier. Pairs that the deterministic rules auto-confirmed are highlighted separately, since the rules stage (not the FS retraining) is what moved them.
"""),
    code("""# §10.2 — tier-shift flow over the 42 labeled pairs.
import matplotlib.patches as mpatches

_left = df_labels["baseline_tier"].fillna("no_match")
_right = df_labels["combined_tier"]

_flow = (
    pd.DataFrame({"baseline": _left, "combined": _right, "rule": df_labels["rule_confirmed"]})
    .value_counts()
    .reset_index(name="n")
)

fig, ax = plt.subplots(figsize=(10, 5.2))
_x_left, _x_right = 0.0, 1.0
_y_positions = {t: i for i, t in enumerate(_TIER_ORDER)}  # 0 = no_match (bottom)

# Stacked bar at each side (height = count in that tier).
for side, col, x in [("baseline", "baseline", _x_left), ("combined", "combined", _x_right)]:
    counts = df_labels[f"{col}_tier"].fillna("no_match").value_counts().reindex(_TIER_ORDER, fill_value=0)
    bottom = 0
    for tier in _TIER_ORDER:
        h = int(counts[tier])
        ax.barh(bottom + h / 2, 0.06, height=h, left=x - 0.03, align="center",
                color=TIER_COLORS[tier], edgecolor=TIER_EDGE[tier], linewidth=1.0)
        ax.text(x, bottom + h / 2, f"{tier} ({h})",
                ha="center", va="center", fontsize=9, color="#222")
        bottom += h

# Draw ribbons. Each row in _flow is a (baseline, combined, rule) triple with weight `n`.
_left_cursors = {t: 0 for t in _TIER_ORDER}
_right_cursors = {t: 0 for t in _TIER_ORDER}
_left_totals = df_labels["baseline_tier"].fillna("no_match").value_counts()
_right_totals = df_labels["combined_tier"].value_counts()
for _, row in _flow.sort_values("rule", ascending=True).iterrows():
    lt, rt, n = row["baseline"], row["combined"], row["n"]
    if lt not in _left_cursors or rt not in _right_cursors:
        continue
    # y-anchors: start at bottom of each tier-bar, advance the cursor by n.
    _l_base = sum(_left_totals.get(t, 0) for t in _TIER_ORDER if _TIER_ORDER.index(t) < _TIER_ORDER.index(lt))
    _r_base = sum(_right_totals.get(t, 0) for t in _TIER_ORDER if _TIER_ORDER.index(t) < _TIER_ORDER.index(rt))
    y0 = _l_base + _left_cursors[lt]
    y1 = _r_base + _right_cursors[rt]
    color = "#4A7DBF" if row["rule"] else TIER_COLORS[rt]
    alpha = 0.55 if row["rule"] else 0.35
    ax.fill_between(
        [_x_left + 0.03, _x_right - 0.03],
        [y0, y1], [y0 + n, y1 + n],
        color=color, alpha=alpha, edgecolor="none",
    )
    _left_cursors[lt] += n
    _right_cursors[rt] += n

ax.set_xlim(-0.15, 1.15)
ax.set_ylim(-0.5, len(df_labels) + 1)
ax.set_xticks([_x_left, _x_right])
ax.set_xticklabels(["Baseline FS tier", "Combined-system tier"])
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.grid(False)

legend_handles = [
    mpatches.Patch(color="#4A7DBF", alpha=0.55, label="Rule-confirmed (Stage 3)"),
    mpatches.Patch(color=TIER_COLORS["auto_merge"], alpha=0.35, label="→ auto_merge"),
    mpatches.Patch(color=TIER_COLORS["human_review"], alpha=0.35, label="→ human_review"),
    mpatches.Patch(color=TIER_COLORS["no_match"], alpha=0.35, label="→ no_match"),
]
ax.legend(handles=legend_handles, loc="upper center", ncol=4,
          bbox_to_anchor=(0.5, -0.05), fontsize=9)
ax.set_title(f"Tier shift — 42 labeled pairs  ·  {VERSION_TAG}",
             fontsize=13, fontweight="bold")

out_path = FIGURES_DIR / f"tier_shift_baseline_to_combined__{VERSION_TAG}.png"
plt.savefig(out_path)
plt.show()
print(f"Saved: {out_path}")
"""),
    md("""### 10.3 Per-veto sample (5 per rule)

Pulls 5 random pairs per `veto_reason` from the enhanced `ProbabilisticMatches` artifact. The intent is to spot-check that each deterministic veto fires on the right pattern (e.g. `ssn_conflict` pairs should have visibly different `SSN` values; `gender_conflict` should be a strict MALE↔FEMALE pair).
"""),
    code("""# §10.3 — per-veto sample (5 per rule).
from IPython.display import Markdown, display

VETOS_PER_RULE = 5

if "veto_reason" not in df_enhanced.columns:
    print("Enhanced artifact has no veto_reason column — skip §10.3.")
else:
    _vetoes = df_enhanced[df_enhanced["veto_reason"].notna()].copy()
    print(f"Vetoed pairs in enhanced artifact: {len(_vetoes):,}")
    print(_vetoes["veto_reason"].value_counts())
    print()

    for rule in sorted(_vetoes["veto_reason"].unique()):
        pool = _vetoes[_vetoes["veto_reason"] == rule]
        n_take = min(VETOS_PER_RULE, len(pool))
        sample = pool.sample(n=n_take, random_state=RANDOM_SEED).reset_index(drop=True)
        display(Markdown(f"#### Veto: `{rule}` — sampling {n_take} of {len(pool):,} pairs"))
        for i, row in sample.iterrows():
            display(Markdown(
                f"##### Pair {i+1}/{n_take}  \\n"
                f"`PATID_A={row['PATID_A']}` ↔ `PATID_B={row['PATID_B']}`  \\n"
                f"`enhanced_score={row['enhanced_score']:.4f}`  ·  "
                f"`veto_reason={row['veto_reason']}`"
            ))
            display(render_identifier_table(row["PATID_A"], row["PATID_B"]))
"""),
    md("""### 10.4 Score distribution overlay

Compares the `match_probability` distribution from the baseline FS (`df_scored`, scored on all candidate pairs) against the enhanced FS (`df_enhanced.enhanced_score`, scored on the post-rules non_matches pool). The thresholds (`REVIEW_FLOOR`, `AUTO_MERGE_THRESHOLD`) are pulled from the **enhanced** module so the shaded bands reflect the production-target cuts.

Reading guide: the enhanced model should concentrate mass below `REVIEW_FLOOR` (most non_matches are true negatives) and produce a clean spike near 1.0 for the surviving high-confidence merges. A flat band across `[0.40, 0.95]` would indicate uncalibrated middle scores.
"""),
    code("""# §10.4 — score distribution overlay (baseline vs enhanced).
from models.experiments.fs_splink_enhanced import fellegi_sunter_enhanced as fs_enh

_enh_floor = fs_enh.DEFAULT_REVIEW_FLOOR
_enh_auto = fs_enh.DEFAULT_AUTO_MERGE_THRESHOLD

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.axvspan(0.0, _enh_floor, color=TIER_COLORS["no_match"], alpha=0.18, lw=0)
ax.axvspan(_enh_floor, _enh_auto, color=TIER_COLORS["human_review"], alpha=0.22, lw=0)
ax.axvspan(_enh_auto, 1.0, color=TIER_COLORS["auto_merge"], alpha=0.22, lw=0)

_bins = np.linspace(0, 1, 41)
ax.hist(df_scored["match_probability"].to_numpy(), bins=_bins,
        density=True, histtype="step", linewidth=2.0, color="#1F4E79",
        label=f"Baseline FS  (n={len(df_scored):,})")
ax.hist(df_enhanced["enhanced_score"].to_numpy(), bins=_bins,
        density=True, histtype="step", linewidth=2.0, color="#A14B2A",
        label=f"Enhanced FS  (n={len(df_enhanced):,})")

ax.axvline(_enh_floor, color="#555", linestyle="--", linewidth=0.8)
ax.axvline(_enh_auto, color="#555", linestyle="--", linewidth=0.8)
ax.text(_enh_floor, ax.get_ylim()[1] * 0.95, f" review_floor={_enh_floor}",
        ha="left", va="top", fontsize=9, color="#444")
ax.text(_enh_auto, ax.get_ylim()[1] * 0.95, f" auto_merge={_enh_auto}",
        ha="left", va="top", fontsize=9, color="#444")

ax.set_xlabel("match_probability / score")
ax.set_ylabel("density")
ax.set_xlim(0, 1)
ax.legend(loc="upper center")
ax.set_title(f"Score distribution — baseline vs enhanced  ·  {VERSION_TAG}",
             fontsize=13, fontweight="bold")

out_path = FIGURES_DIR / f"score_distribution_overlay__{VERSION_TAG}.png"
plt.savefig(out_path)
plt.show()
print(f"Saved: {out_path}")
"""),
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # Drop any prior §10 cells (so re-running this script replaces in-place).
    keep = []
    in_section_10 = False
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and src.lstrip().startswith("## 10."):
            in_section_10 = True
            continue
        if in_section_10:
            # Continue dropping until we hit Reviewer judgments / end of §10 block.
            if c["cell_type"] == "markdown" and src.lstrip().startswith("## Reviewer judgments"):
                in_section_10 = False
                keep.append(c)
                continue
            # Drop §10 code cells (markered with SECTION_MARKER or following a §10 markdown).
            if SECTION_MARKER in src or src.lstrip().startswith("### 10.") or "§10" in src[:200]:
                continue
            # Otherwise drop too (we're still in §10 until reviewer judgments shows up).
            continue
        keep.append(c)

    # Find insertion point: just before the "## Reviewer judgments" markdown cell.
    insert_at = len(keep)
    for i, c in enumerate(keep):
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and src.lstrip().startswith("## Reviewer judgments"):
            insert_at = i
            break

    new_cells = keep[:insert_at] + SECTION_10_CELLS + keep[insert_at:]
    nb["cells"] = new_cells
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {NB_PATH} with {len(new_cells)} cells "
          f"({len(SECTION_10_CELLS)} new §10 cells inserted at index {insert_at})")


if __name__ == "__main__":
    main()
