"""One-off injector: append Section 11 (3-way FS head-to-head) cells to the validation notebook.

Mirrors `_inject_section_10.py`: idempotent — re-runs replace the §11 block in
place rather than duplicating. Inserts §11 cells AFTER §10 and BEFORE the
trailing "## Reviewer judgments" markdown cell.

Five figures, all saved to `notebooks/fellegi_sunter/figures/<name>__{VERSION_TAG}.png`:

  (1) 3-panel confusion matrix on 42 reviewer labels (combined system per model)
  (2) Silver-label recall@tier grouped bar (3 models × 3 tiers)
  (3) 3-panel score-distribution histograms on the post-rules non_matches pool
  (4) Reject-vs-review score-distribution overlay (enhanced_2 full-pool)
  (5) Per-comparison m/u horizontal grouped bars for enhanced_2
      — depends on a model-JSON artifact the runner doesn't yet write;
        emits a clear skip with patch instructions if absent.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fellegi_sunter" / "fellegi_sunter_validation.ipynb"
SECTION_MARKER = "# §11 SECTION MARKER — do not remove"


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


SECTION_11_CELLS = [
    md("""## 11. Three-way FS head-to-head — baseline vs enhanced vs enhanced_2

§10 compared the *baseline FS alone* against the *combined system* (deterministic rules + enhanced FS). §11 widens that comparison to **three FS variants**, each as the Stage-4 model inside the same combined system:

| Model | Training | Tunables | Special features |
|---|---|---|---|
| **baseline** | EM (3 sessions) | `0.90 / 0.50` (am / floor) | the original FS Splink build |
| **enhanced** | EM + manual priors (Phase E1 deterministic vetoes) | `0.95 / 0.40` | adds `Household_discount`, JW<0.5 mismatch levels, deterministic vetoes |
| **enhanced_2** | supervised m from synthetic positives | `0.95 / 0.40` | no vetoes (replaced by upstream `classify_non_matches` from Phase E2-5b); adds Middle name, Sex_positive, Phonetic-name, ZIP-base comparisons |

Each model is evaluated under the same Stage-3 pre-filter (the contradiction-counting `classify_non_matches` shipped on `develop`). The figures below answer four questions:

1. **Where does each model land on the 42 reviewer labels?** (Figure 1 — directional precision/recall.)
2. **Which model recovers known positives best?** (Figure 2 — silver-label recall, statistically powered.)
3. **How does each model spread its scores across the ambiguous review pool?** (Figure 3 — distributional separation.)
4. **Did Stage 3's contradiction-filter do what we thought it did?** (Figure 4 — pairs that Stage 3 routed to `reject` would have scored high in Stage 4.)

> §10 reads inputs from in-memory `df_scored` (baseline) + `data/matches/` + `data/matches_model/`. §11 reads the durable cross-model contract: `models/outputs/<model_name>__<data-version>.parquet` (the 5-col `eval_schema`). This is the apples-to-apples lane both runners write to.
"""),

    md("""### 11.0 Setup — resolve artifacts, parse labels, build the joins

Loads the three eval-schema parquets (baseline / enhanced / enhanced_2), the enhanced_2 full-pool parquet, the Stage-3 review/reject frames, and any silver-label parquet from `data/silver_labels/`. Also defines the `df_labels_3way` frame the confusion-matrix figure consumes.
"""),

    code(SECTION_MARKER + """
# §11 setup — resolve artifacts, parse labels, attach per-model tiers.
import json as _json11
import re as _re11

from models.common.versioning import latest_versioned

# Reuse §10 helper if available; redefine defensively.
try:
    _ = _latest_by_mtime  # type: ignore[name-defined]
except NameError:
    def _latest_by_mtime(dir_: Path, pattern: str) -> Path | None:
        if not dir_.exists():
            return None
        paths = sorted(dir_.glob(pattern), key=lambda p: p.stat().st_mtime)
        return paths[-1] if paths else None

OUTPUTS_DIR = PROJECT_ROOT / "models" / "outputs"
NON_MATCHES_DIR = PROJECT_ROOT / "data" / "non_matches"
REJECTS_DIR = PROJECT_ROOT / "data" / "rejects"
MATCHES_DIR_11 = PROJECT_ROOT / "data" / "matches"
SILVER_DIR = PROJECT_ROOT / "data" / "silver_labels"
ARTIFACTS_ENH2 = PROJECT_ROOT / "models" / "artifacts" / "fs_splink_enhanced_2"


def _resolve_eval(model_name: str, full_pool: bool = False) -> Path:
    # Only enhanced_2 has separate `_full_pool` vs non_matches variants. Baseline
    # and enhanced runners default to scoring the full candidate pool and write a
    # single output per data-version (no suffix). For those two models, both
    # full_pool=True and full_pool=False resolve to the same file — the cell then
    # filters to review-tier via an inner-join when a non_matches view is wanted.
    has_pool_variants = (model_name == "fs_splink_enhanced_2")
    if has_pool_variants:
        cands = sorted(OUTPUTS_DIR.glob(f"{model_name}__*.parquet"))
        cands = [c for c in cands
                 if c.stem.endswith("_full_pool") == full_pool
                 and "synthetic" not in c.stem]
    else:
        cands = sorted(OUTPUTS_DIR.glob(f"{model_name}__*.parquet"))
        cands = [c for c in cands
                 if not c.stem.endswith("_full_pool")
                 and "synthetic" not in c.stem]
    if not cands:
        raise FileNotFoundError(
            f"No eval-schema parquet for {model_name} (full_pool={full_pool}). "
            f"Looked under {OUTPUTS_DIR}."
        )
    # Most recent mtime wins (handles multiple data-version tags).
    return max(cands, key=lambda p: p.stat().st_mtime)


# Eval-schema parquets (5 cols: PATID_A, PATID_B, model_name, score, predicted_tier)
path_bl_full = _resolve_eval("fs_splink_baseline", full_pool=True)
path_enh_full = _resolve_eval("fs_splink_enhanced", full_pool=True)
path_enh2_nm = _resolve_eval("fs_splink_enhanced_2", full_pool=False)
path_enh2_full = _resolve_eval("fs_splink_enhanced_2", full_pool=True)

print(f"baseline  (full pool) : {path_bl_full.name}")
print(f"enhanced  (full pool) : {path_enh_full.name}")
print(f"enhanced_2 (non_matches): {path_enh2_nm.name}")
print(f"enhanced_2 (full pool) : {path_enh2_full.name}")

df_bl = pd.read_parquet(path_bl_full)
df_enh = pd.read_parquet(path_enh_full)
df_enh2 = pd.read_parquet(path_enh2_nm)
df_enh2_full = pd.read_parquet(path_enh2_full)

# Stage-3 frames (review = downstream non_matches; reject = dropped; matches = rule-confirmed)
nm_path = _latest_by_mtime(NON_MATCHES_DIR, "non_matches_*.parquet")
rj_path = _latest_by_mtime(REJECTS_DIR, "rejects_*.parquet")
mt_path = _latest_by_mtime(MATCHES_DIR_11, "matches_*.parquet")
print(f"non_matches (review): {nm_path}")
print(f"rejects             : {rj_path}")
print(f"matches (rule confirm): {mt_path}")

df_review = pd.read_parquet(nm_path)[["PATID_A", "PATID_B"]] if nm_path else pd.DataFrame(columns=["PATID_A","PATID_B"])
df_reject = pd.read_parquet(rj_path)[["PATID_A", "PATID_B"]] if rj_path else pd.DataFrame(columns=["PATID_A","PATID_B"])
df_matches11 = pd.read_parquet(mt_path)[["PATID_A","PATID_B"]] if mt_path else pd.DataFrame(columns=["PATID_A","PATID_B"])

# ── Silver labels (positives-only CSV, may be absent locally) ────────────────
df_silver = None
if SILVER_DIR.exists():
    _silver_paths = sorted(SILVER_DIR.glob("*.csv"))
    if _silver_paths:
        # Take the largest one — usually a single combined positives file
        _sp = max(_silver_paths, key=lambda p: p.stat().st_size)
        # dtype=str preserves PATID leading zeros (mirrors parquet roundtrip behaviour)
        df_silver = pd.read_csv(_sp, dtype=str)
        # Normalize column case if needed
        if "PATID_A" not in df_silver.columns:
            df_silver = df_silver.rename(columns={c: c.upper() for c in df_silver.columns if c.lower() in {"patid_a","patid_b"}})
        df_silver = df_silver[["PATID_A","PATID_B"]].drop_duplicates().reset_index(drop=True)
        print(f"silver labels      : {_sp.name}  (n={len(df_silver):,})")
    else:
        print("silver labels      : (no *.csv files in data/silver_labels/)")
else:
    print("silver labels      : (data/silver_labels/ absent — skipping recall figure)")

# ── Tier-column normalizer (eval_schema uses `predicted_tier`) ───────────────
def _norm_eval(df: pd.DataFrame, model: str) -> pd.DataFrame:
    out = df.rename(columns={
        "predicted_tier": f"tier_{model}",
        "score":          f"score_{model}",
    })
    return out[["PATID_A", "PATID_B", f"tier_{model}", f"score_{model}"]]

bl_norm = _norm_eval(df_bl, "baseline")
enh_norm = _norm_eval(df_enh, "enhanced")
enh2_norm = _norm_eval(df_enh2, "enhanced_2")
enh2_full_norm = _norm_eval(df_enh2_full, "enhanced_2")  # for silver-label join

# ── 42-label parse — reuse §10's regex inline (defensive against re-ordering) ─
_label_re_11 = _re11.compile(
    r"PATID_A=<([0-9A-Fa-f]+)>\\s*PATID_B=<([0-9A-Fa-f]+)>[^\\n]*?score=<([0-9.]+)>"
    r"[^v]*?verdict:\\s*<([a-z\\- ]+)>",
    _re11.IGNORECASE,
)
_nb_path = Path("fellegi_sunter_validation.ipynb")
_nb = _json11.loads(_nb_path.read_text())
_labels_md = ""
for _c in _nb["cells"]:
    _src = "".join(_c["source"])
    if _c["cell_type"] == "markdown" and _src.lstrip().startswith("## Reviewer judgments"):
        _labels_md = _src
        break
_labels = [{
    "PATID_A": _m.group(1).upper(),
    "PATID_B": _m.group(2).upper(),
    "verdict": _m.group(4).strip().lower(),
} for _m in _label_re_11.finditer(_labels_md)]
df_labels_3way = pd.DataFrame(_labels).drop_duplicates(["PATID_A","PATID_B"]).reset_index(drop=True)
df_labels_3way["verdict_norm"] = df_labels_3way["verdict"].map(
    lambda v: {"same":"same","different":"different","unsure":"unsure"}.get(v, "unsure")
)
print(f"Parsed {len(df_labels_3way)} labeled pairs")

# ── Attach per-model COMBINED-SYSTEM tier ────────────────────────────────────
# Combined tier rules:
#   1. rule-confirmed (in df_matches11)              → auto_merge
#   2. else, tier from this model's non_matches-only score (filter full-pool to review only for fair comparison)
#   3. else (pair is rejected or never blocked)       → no_match
def _restrict_to_review(scored: pd.DataFrame, review: pd.DataFrame, tier_col: str) -> pd.DataFrame:
    if review.empty:
        return scored
    return scored.merge(review, on=["PATID_A","PATID_B"], how="inner")

bl_review = _restrict_to_review(bl_norm, df_review, "tier_baseline")
enh_review = _restrict_to_review(enh_norm, df_review, "tier_enhanced")
# enhanced_2 default IS non_matches already → use as-is
enh2_review = enh2_norm.copy()

# Build df_labels_3way with three combined tiers
df_rules_flag = df_matches11.copy()
df_rules_flag["rule_confirmed"] = True
df_labels_3way = df_labels_3way.merge(df_rules_flag, on=["PATID_A","PATID_B"], how="left")
df_labels_3way["rule_confirmed"] = df_labels_3way["rule_confirmed"].fillna(False)

for label, frame in [("baseline", bl_review), ("enhanced", enh_review), ("enhanced_2", enh2_review)]:
    df_labels_3way = df_labels_3way.merge(frame, on=["PATID_A","PATID_B"], how="left")
    def _combine(row, lbl=label):
        if row["rule_confirmed"]:
            return "auto_merge"
        t = row.get(f"tier_{lbl}")
        if pd.notna(t):
            return t
        return "no_match"
    df_labels_3way[f"combined_{label}"] = df_labels_3way.apply(_combine, axis=1)

print(df_labels_3way[["verdict_norm","combined_baseline","combined_enhanced","combined_enhanced_2"]]
      .value_counts(dropna=False).head(10))
"""),

    md("""### 11.1 Confusion matrix — 3-panel on 42 reviewer labels

Each panel shows the **combined-system tier** (rules + that FS model) versus the reviewer verdict on the 42 labeled pairs. Rows are reviewer verdicts (`different` / `unsure` / `same`); columns are predicted tiers (`no_match` / `human_review` / `auto_merge`).

**Read direction:** for high precision you want the `different → auto_merge` cell empty; for high recall you want the `same → auto_merge` cell to dominate the `same` row.

⚠ 42 labels is a small sample. The CM tells you which model is *directionally* most calibrated — recall claims under this view are noisy. §11.2 covers recall under a higher-power view.
"""),

    code("""# §11.1 — 3-panel confusion matrix.
_TIER_ORDER_11 = ["no_match", "human_review", "auto_merge"]
_VERDICT_ORDER_11 = ["different", "unsure", "same"]
_MODELS_11 = [("baseline", "Baseline FS  (0.90 / 0.50)"),
              ("enhanced", "Enhanced FS  (0.95 / 0.40)"),
              ("enhanced_2", "Enhanced_2 FS  (0.95 / 0.40)")]


def _cm(df: pd.DataFrame, tier_col: str) -> pd.DataFrame:
    return pd.crosstab(df["verdict_norm"], df[tier_col]).reindex(
        index=_VERDICT_ORDER_11, columns=_TIER_ORDER_11, fill_value=0
    )


cms = [(label, title, _cm(df_labels_3way, f"combined_{label}")) for label, title in _MODELS_11]
_vmax = max(cm.values.max() for _, _, cm in cms)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
for ax, (label, title, cm) in zip(axes, cms):
    ax.imshow(cm.values, cmap="Greens", vmin=0, vmax=max(_vmax, 1))
    ax.set_xticks(range(len(_TIER_ORDER_11)))
    ax.set_xticklabels(_TIER_ORDER_11, rotation=20, ha="right")
    ax.set_yticks(range(len(_VERDICT_ORDER_11)))
    ax.set_yticklabels(_VERDICT_ORDER_11)
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Reviewer verdict")
    ax.set_title(title)
    for (i, j), v in np.ndenumerate(cm.values):
        ax.text(j, i, str(int(v)), ha="center", va="center",
                color="white" if v > _vmax / 2 else "#222")
    ax.grid(False)

fig.suptitle(f"3-way confusion matrix — 42 labeled pairs  ·  {VERSION_TAG}",
             fontsize=13, fontweight="bold", y=1.06)
out_path = FIGURES_DIR / f"confusion_matrix_3way__{VERSION_TAG}.png"
plt.savefig(out_path)
plt.show()
print(f"Saved: {out_path}")


# ── Quantitative summary ─────────────────────────────────────────────────────
def _summarise_3way(label: str, cm: pd.DataFrame) -> None:
    n = int(cm.values.sum())
    am_diff = int(cm.loc["different", "auto_merge"])
    am_same = int(cm.loc["same", "auto_merge"])
    am_total = int(cm["auto_merge"].sum())
    same_total = int(cm.loc["same"].sum())
    am_prec = (am_same / am_total) if am_total else float("nan")
    same_rec = ((cm.loc["same","auto_merge"] + cm.loc["same","human_review"]) / same_total) if same_total else float("nan")
    print(f"{label:11s}: n={n} | AM precision = {am_prec:.0%} ({am_same}/{am_total}) | "
          f"same-recall to AM∪HR = {same_rec:.0%} | false-merges (diff→AM) = {am_diff}")


for label, _, cm in cms:
    _summarise_3way(label, cm)
"""),

    md("""### 11.2 Silver-label recall by tier — 3 models × 3 tiers grouped bar

Silver labels are **positives-only** (Stage-3-confirmed pairs from the `data/silver_labels/` validation set, ~99% adjudicator precision). They give a statistically-powered view of one direction: **of all known positives, what fraction does each model promote to auto_merge / hold for human review / drop to no_match?**

This is not a confusion matrix — there are no negatives to test against — so don't try to read precision off it. But the recall direction is reliable here in a way the 42-label CM is not.

Silver labels are filtered out of `non_matches` by Stage 3 (because they were confirmed), so we read them off the **full-pool** scoring parquets. Pairs that aren't in the full-pool output are assumed `no_match` (e.g., never blocked).
"""),

    code("""# §11.2 — silver-label recall@tier (grouped bar).
if df_silver is None or df_silver.empty:
    print("§11.2 SKIPPED — no silver labels available locally. Re-run on the VM "
          "where `data/silver_labels/*.parquet` exists.")
else:
    def _silver_tier(model_norm_full: pd.DataFrame, model_label: str) -> pd.Series:
        m = df_silver.merge(
            model_norm_full[["PATID_A","PATID_B", f"tier_{model_label}"]],
            on=["PATID_A","PATID_B"], how="left",
        )
        return m[f"tier_{model_label}"].fillna("no_match")

    _silver_tiers = {
        "baseline":   _silver_tier(bl_norm, "baseline"),
        "enhanced":   _silver_tier(enh_norm, "enhanced"),
        "enhanced_2": _silver_tier(enh2_full_norm, "enhanced_2"),  # full-pool variant
    }

    _grouped = pd.DataFrame({
        m: t.value_counts().reindex(_TIER_ORDER_11, fill_value=0)
        for m, t in _silver_tiers.items()
    }).T
    _grouped_pct = _grouped.div(_grouped.sum(axis=1), axis=0) * 100.0

    fig, ax = plt.subplots(figsize=(11, 4.6))
    _x = np.arange(len(_grouped_pct.index))
    _width = 0.26
    for i, tier in enumerate(_TIER_ORDER_11):
        ax.bar(_x + (i - 1) * _width, _grouped_pct[tier].values, _width,
               color=TIER_COLORS[tier], edgecolor=TIER_EDGE[tier],
               label=tier, linewidth=1.0)
        for xi, v in zip(_x + (i - 1) * _width, _grouped_pct[tier].values):
            ax.text(xi, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(_x)
    ax.set_xticklabels(_grouped_pct.index)
    ax.set_ylabel("% of silver-label positives")
    ax.set_ylim(0, 105)
    ax.set_title(f"Silver-label recall@tier — n={len(df_silver):,} known-positive pairs  ·  {VERSION_TAG}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.25)

    out_path = FIGURES_DIR / f"silver_label_recall_by_tier__{VERSION_TAG}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")
    print()
    print("Silver-label tier counts (rows = model, cols = tier):")
    print(_grouped)
"""),

    md("""### 11.3 Score distribution — 3-panel histograms on the post-rules `non_matches` pool

Each panel is the score histogram a model assigns to the *review*-tier pool (post-rules `non_matches`). Log y-axis surfaces the tail. The shaded vertical bands are that model's own `[0, review_floor)` / `[review_floor, auto_merge_threshold)` / `[auto_merge_threshold, 1]` regions.

**Reading guide:** a well-calibrated model concentrates mass at the floor (most ambiguous pairs are true negatives) and produces a clean spike near 1.0 only for the genuine merges. A *uniform spread* across `[floor, auto_merge]` is the warning sign — undifferentiated middle scores.
"""),

    code("""# §11.3 — 3-panel score distribution on the post-rules non_matches pool.
# Filter each model's full-pool eval-schema to review-tier pairs for fair comparison.
_BL_FLOOR, _BL_AM = 0.50, 0.90
_EN_FLOOR, _EN_AM = 0.40, 0.95
_E2_FLOOR, _E2_AM = 0.40, 0.95

_panels = [
    ("baseline",   _restrict_to_review(bl_norm,  df_review, "tier_baseline"),   "score_baseline",   _BL_FLOOR, _BL_AM, "Baseline FS"),
    ("enhanced",   _restrict_to_review(enh_norm, df_review, "tier_enhanced"),   "score_enhanced",   _EN_FLOOR, _EN_AM, "Enhanced FS"),
    ("enhanced_2", enh2_norm,                                                    "score_enhanced_2", _E2_FLOOR, _E2_AM, "Enhanced_2 FS"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
_bins = np.linspace(0, 1, 51)

for ax, (label, frame, score_col, floor, am, title) in zip(axes, _panels):
    if frame.empty or score_col not in frame.columns:
        ax.text(0.5, 0.5, f"{label}: no data", ha="center", va="center")
        ax.set_title(title)
        continue
    s = frame[score_col].dropna().to_numpy()
    ax.axvspan(0.0, floor, color=TIER_COLORS["no_match"], alpha=0.18, lw=0)
    ax.axvspan(floor, am, color=TIER_COLORS["human_review"], alpha=0.22, lw=0)
    ax.axvspan(am, 1.0, color=TIER_COLORS["auto_merge"], alpha=0.22, lw=0)
    ax.hist(s, bins=_bins, color="#1F4E79", edgecolor="white", linewidth=0.4)
    ax.axvline(floor, color="#555", linestyle="--", linewidth=0.8)
    ax.axvline(am, color="#555", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.set_ylabel("count (log)")
    ax.set_title(f"{title}  ·  n={len(s):,}")
    ax.text(floor, ax.get_ylim()[1] * 0.6, f" floor={floor:.2f}", fontsize=8, color="#444")
    ax.text(am,    ax.get_ylim()[1] * 0.6, f" AM={am:.2f}",       fontsize=8, color="#444")

fig.suptitle(f"Score distribution — post-rules non_matches pool  ·  {VERSION_TAG}",
             fontsize=13, fontweight="bold", y=1.06)
out_path = FIGURES_DIR / f"score_distribution_3way__{VERSION_TAG}.png"
plt.savefig(out_path)
plt.show()
print(f"Saved: {out_path}")
"""),

    md("""### 11.4 Reject vs review — validates the upstream `classify_non_matches` filter

For every pair in the **enhanced_2 full-pool** scoring artifact, attach the Stage-3 decision (`reject` / `review` / `confirmed`) and overlay their score histograms.

**What this shows:** if the upstream contradiction filter is doing its job, the `reject`-tier pairs should pile up at high FS scores — i.e., FS *would have* promoted many of them to `auto_merge` if Stage 3 hadn't dropped them. A `reject` distribution heavily right-shifted is the validation that the upstream filter prevented a real FP burst. A `reject` distribution overlapping `review` says Stage 3 dropped pairs that Stage 4 would also have handled correctly — i.e., the filter is over-aggressive.
"""),

    code("""# §11.4 — reject vs review score-distribution overlay (enhanced_2 full-pool).
_full = df_enh2_full.copy()
# Tag each pair by its Stage-3 decision.
_full = _full.merge(df_matches11.assign(_dec="confirmed"), on=["PATID_A","PATID_B"], how="left")
_full = _full.merge(df_review.assign(_rev=True), on=["PATID_A","PATID_B"], how="left")
_full = _full.merge(df_reject.assign(_rej=True), on=["PATID_A","PATID_B"], how="left")


def _stage3_decision(row: pd.Series) -> str:
    if row.get("_dec") == "confirmed":
        return "confirmed"
    if row.get("_rej") is True:
        return "reject"
    if row.get("_rev") is True:
        return "review"
    return "unblocked"

_full["stage3"] = _full.apply(_stage3_decision, axis=1)
print("Stage-3 decision counts in enhanced_2 full-pool:")
print(_full["stage3"].value_counts())

fig, ax = plt.subplots(figsize=(11, 4.6))
_bins = np.linspace(0, 1, 51)
_palette = {"confirmed": "#2D7F4B", "review": "#A14B2A", "reject": "#B2182B"}
for decision in ["confirmed", "review", "reject"]:
    s = _full.loc[_full["stage3"] == decision, "score"].dropna().to_numpy()
    if len(s) == 0:
        continue
    ax.hist(s, bins=_bins, density=True, histtype="step", linewidth=2.0,
            color=_palette[decision], label=f"{decision}  (n={len(s):,})")

ax.axvline(_E2_FLOOR, color="#555", linestyle="--", linewidth=0.8)
ax.axvline(_E2_AM, color="#555", linestyle="--", linewidth=0.8)
ax.text(_E2_FLOOR, ax.get_ylim()[1] * 0.95, f" floor={_E2_FLOOR}", fontsize=9, color="#444")
ax.text(_E2_AM,    ax.get_ylim()[1] * 0.95, f" AM={_E2_AM}",       fontsize=9, color="#444")
ax.set_xlabel("enhanced_2 score")
ax.set_ylabel("density")
ax.set_xlim(0, 1)
ax.legend(loc="upper center")
ax.set_title(f"Stage-3 decision vs enhanced_2 score — full candidate pool  ·  {VERSION_TAG}",
             fontsize=13, fontweight="bold")

out_path = FIGURES_DIR / f"stage3_decision_vs_enhanced_2_score__{VERSION_TAG}.png"
plt.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")
"""),

    md("""### 11.5 Per-comparison m/u weights — enhanced_2 (anti-evidence inspection)

For each comparison level in the enhanced_2 model, plot m (P[level fires | match]) vs u (P[level fires | non-match]). Levels where m < u are **anti-evidence** — when they fire, they should pull the score DOWN. The diagnostic check is whether anti-evidence levels (`Household_discount`, `Sex_positive[M↔F]`, `SSN[9-digit mismatch]`) actually satisfy m < u.

**Data dependency:** the enhanced_2 runner embeds `trained_settings` inside `models/artifacts/fs_splink_enhanced_2/diagnostics__<v>.json` (added alongside this notebook section, mirroring baseline + enhanced). If the key is absent — e.g., the diagnostics file was written by an older runner — the cell prints exactly what to re-run.
"""),

    code("""# §11.5 — m/u inspection for enhanced_2 (reads trained_settings from diagnostics).
_diag_jsons = sorted(ARTIFACTS_ENH2.glob("diagnostics__*.json")) if ARTIFACTS_ENH2.exists() else []
# Prefer non-full-pool diagnostics for the m/u inspection (training is the same;
# the non-full-pool variant is the production-shaped run).
_diag_jsons = [p for p in _diag_jsons if not p.stem.endswith("_full_pool")] or _diag_jsons

_model_dict = None
if _diag_jsons:
    _dj = max(_diag_jsons, key=lambda p: p.stat().st_mtime)
    _diag = _json11.loads(_dj.read_text())
    _model_dict = _diag.get("trained_settings")
    if _model_dict is None:
        print(f"§11.5 SKIPPED — `trained_settings` key missing in {_dj.name}.")
        print("Re-run enhanced_2 on the VM with the current runner version; the runner now")
        print("embeds linker.misc.save_model_to_json() into the diagnostics file.")

if _model_dict is None and not _diag_jsons:
    print("§11.5 SKIPPED — no enhanced_2 diagnostics JSON in", ARTIFACTS_ENH2)

    _rows = []
    for comp in _model_dict.get("comparisons", []):
        c_name = comp.get("output_column_name") or comp.get("comparison_description", "?")
        for lvl in comp.get("comparison_levels", []):
            if lvl.get("is_null_level"):
                continue
            _rows.append({
                "comparison": c_name,
                "level": lvl.get("label_for_charts", lvl.get("sql_condition", "?"))[:50],
                "m": lvl.get("m_probability"),
                "u": lvl.get("u_probability"),
            })
    _mu = pd.DataFrame(_rows).dropna(subset=["m", "u"])
    _mu["log2_bf"] = np.log2(_mu["m"] / _mu["u"].clip(lower=1e-9))
    _mu = _mu.sort_values("log2_bf")

    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.32 * len(_mu))))
    _y = np.arange(len(_mu))
    ax.barh(_y - 0.18, _mu["m"], 0.36, color="#2D7F4B", label="m (P[fires | match])")
    ax.barh(_y + 0.18, _mu["u"], 0.36, color="#B2182B", label="u (P[fires | non-match])")
    ax.set_yticks(_y)
    ax.set_yticklabels([f"{r.comparison} — {r.level}" for r in _mu.itertuples()], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(left=1e-6)
    ax.set_xlabel("probability (log scale)")
    ax.legend(loc="lower right")
    ax.set_title(f"enhanced_2 — m vs u per comparison level (sorted by log2 BF)  ·  {VERSION_TAG}",
                 fontsize=12, fontweight="bold")

    out_path = FIGURES_DIR / f"enhanced_2_m_vs_u__{VERSION_TAG}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")
    print()
    print("Anti-evidence levels (m < u) — these should NOT include the on-purpose anti-evidence levels:")
    print(_mu[_mu["m"] < _mu["u"]][["comparison", "level", "m", "u", "log2_bf"]].to_string(index=False))
"""),
]


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]

    # Drop any prior §11 cells (so re-running this script replaces in-place).
    keep = []
    in_section_11 = False
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and src.lstrip().startswith("## 11."):
            in_section_11 = True
            continue
        if in_section_11:
            # Continue dropping until we hit "## Reviewer judgments" (or another top-level §).
            if c["cell_type"] == "markdown" and (
                src.lstrip().startswith("## Reviewer judgments")
                or src.lstrip().startswith("## 12.")
            ):
                in_section_11 = False
                keep.append(c)
                continue
            # Drop §11 code cells.
            if SECTION_MARKER in src or src.lstrip().startswith("### 11.") or "§11" in src[:200]:
                continue
            continue
        keep.append(c)

    # Find insertion point: just before the "## Reviewer judgments" markdown cell.
    insert_at = len(keep)
    for i, c in enumerate(keep):
        src = "".join(c["source"])
        if c["cell_type"] == "markdown" and src.lstrip().startswith("## Reviewer judgments"):
            insert_at = i
            break

    new_cells = keep[:insert_at] + SECTION_11_CELLS + keep[insert_at:]
    nb["cells"] = new_cells
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {NB_PATH} with {len(new_cells)} cells "
          f"({len(SECTION_11_CELLS)} new §11 cells inserted at index {insert_at})")


if __name__ == "__main__":
    main()
