"""FS round-3 decisive experiment — the conflict-veto test.

Research question (from to-do.md, FS round 3): the round-1/round-2 write-up
claimed FS's precision ceiling is structural because "an exact birth date is
decisive on its own (~14 bits)". The critics proposed a fix: hard-veto any
rule-confirmed merge whose records *conflict* on a strong unique identifier
(disjoint phone, conflicting SSN, conflicting email). This script tests whether
such a veto is viable, by measuring — among the rule-confirmed pairs — how often
each conflict signal fires on the FALSE merges (silver-False) we want to catch
vs the TRUE merges (silver-True) we must not break.

Method (the reproduce recipe in to-do.md):
  1. apply_rules to the silver-labelled candidate pairs (real cleaned records),
  2. keep the rule-confirmed pairs, split by silver_label,
  3. compare non-null SSN / Email / Phones_set conflict rates between the
     False (bad-merge) and True (good-merge) subsets.

A veto is only viable if it catches many false merges while wrongly blocking few
true ones. Read-only on data/. Writes a JSON summary to
data/runs/research_fs_conflict_veto.json. No PHI in output (aggregate counts).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.deterministic_rules import _parse_phone_set, apply_rules

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

CLEANED = _ROOT / "data/processed/MDM_Population_cleaned_real_20260620.parquet"
SILVER = _ROOT / "data/raw/silver_labels.csv"
OUT = _ROOT / "data/runs/research_fs_conflict_veto.json"


CHART = _ROOT / "reports/fs/charts/v3_conflict_veto.png"


def _canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _plot(table: dict, nF: int, nT: int, irreducible: int) -> None:
    """Grouped bars: for each conflict signal, % of false merges caught vs % of
    true merges wrongly blocked. A usable veto would be tall-green/short-red; all
    three are the opposite, which is the finding."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigs = ["disjoint_phone", "conflicting_ssn", "conflicting_email"]
    labels = ["disjoint\nphone", "conflicting\nSSN", "conflicting\nemail"]
    caught = [table[s]["catches_false_pct"] * 100 for s in sigs]
    blocked = [table[s]["blocks_true_pct"] * 100 for s in sigs]
    ratios = [table[s]["true_blocked_per_false_caught"] for s in sigs]

    x = np.arange(len(sigs))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bw = 0.38
    b1 = ax.bar(x - bw / 2, caught, bw, label="false merges caught (want HIGH)",
                color="#2e7d32")
    b2 = ax.bar(x + bw / 2, blocked, bw, label="true merges wrongly blocked (want LOW)",
                color="#c62828")
    ax.set_ylabel("% of subset")
    ax.set_title(
        "The conflict-veto fails: every signal blocks more true merges than it "
        "catches false ones\n"
        f"(on the {nF + nT:,} rule-confirmed pairs: {nF:,} false / {nT:,} true)",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 100)
    for rects in (b1, b2):
        for r in rects:
            ax.annotate(f"{r.get_height():.0f}%", (r.get_x() + r.get_width() / 2,
                        r.get_height()), ha="center", va="bottom", fontsize=9)
    for i, ratio in enumerate(ratios):
        ypos = max(caught[i], blocked[i]) + 6
        ax.annotate(f"{ratio:.1f} true lost / false caught", (x[i], min(ypos, 99)),
                    ha="center", va="bottom", fontsize=8, color="#555",
                    style="italic")
    ax.text(0.5, -0.16,
            f"Irreducible core: {irreducible:,} false merges ({irreducible/nF:.0%}) "
            "have NO conflicting strong ID at all — no veto can ever reach them.",
            transform=ax.transAxes, ha="center", fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(CHART, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {CHART.relative_to(_ROOT)}")


def main() -> None:
    df_clean = pd.read_parquet(CLEANED)
    silver = pd.read_csv(SILVER)
    silver["silver_label"] = silver["silver_label"].astype(bool)
    print(f"cleaned records: {len(df_clean):,}  silver pairs: {len(silver):,}")
    print(silver["silver_label"].value_counts().to_dict())

    # 1. Rule-confirm the silver candidate pairs.
    matches = apply_rules(silver[["PATID_A", "PATID_B"]], df_clean)
    print(f"rule-confirmed pairs: {len(matches):,}")

    # 2. Join silver labels onto the rule-confirmed pairs (canonicalize keys).
    silver_map = {
        _canon(a, b): lab
        for a, b, lab in zip(silver.PATID_A, silver.PATID_B, silver.silver_label)
    }
    canon = [_canon(a, b) for a, b in zip(matches.PATID_A, matches.PATID_B)]
    matches = matches.assign(
        silver_label=[silver_map.get(k) for k in canon]
    )
    matches = matches[matches["silver_label"].notna()].copy()
    matches["silver_label"] = matches["silver_label"].astype(bool)

    # 3. Materialize the fields needed for conflict tests on both sides.
    fields = ["SSN_clean", "Email_clean", "Phones_set"]
    attrs = df_clean.set_index("PATID")[fields]
    L = attrs.add_suffix("_L").reindex(matches.PATID_A).reset_index(drop=True)
    R = attrs.add_suffix("_R").reindex(matches.PATID_B).reset_index(drop=True)
    m = pd.concat(
        [matches.reset_index(drop=True)[["silver_label", "match_rule"]], L, R],
        axis=1,
    )

    def _nonnull(s: pd.Series) -> pd.Series:
        return s.notna() & (s.astype(str).str.strip().ne("")) & s.astype(str).str.lower().ne("nan")

    # SSN conflict: both present and unequal.
    ssn_both = _nonnull(m.SSN_clean_L) & _nonnull(m.SSN_clean_R)
    ssn_conflict = ssn_both & (m.SSN_clean_L.astype(str) != m.SSN_clean_R.astype(str))

    # Email conflict: both present and unequal.
    em_both = _nonnull(m.Email_clean_L) & _nonnull(m.Email_clean_R)
    em_conflict = em_both & (m.Email_clean_L.astype(str) != m.Email_clean_R.astype(str))

    # Phone conflict: both have >=1 phone, intersection empty (disjoint).
    pl = m.Phones_set_L.map(_parse_phone_set)
    pr = m.Phones_set_R.map(_parse_phone_set)
    ph_both = (pl.map(len) > 0) & (pr.map(len) > 0)
    ph_disjoint = ph_both & np.array([len(a & b) == 0 for a, b in zip(pl, pr)])

    any_conflict = ssn_conflict | em_conflict | ph_disjoint

    m = m.assign(
        ssn_conflict=ssn_conflict,
        em_conflict=em_conflict,
        ph_disjoint=ph_disjoint,
        any_conflict=any_conflict,
    )

    false_m = m[~m.silver_label]
    true_m = m[m.silver_label]
    nF, nT = len(false_m), len(true_m)
    print(f"\nrule-confirmed split: FALSE(bad merges)={nF:,}  TRUE(good merges)={nT:,}")

    def stat(name: str, fcol: str) -> dict:
        catches = int(false_m[fcol].sum())
        blocks = int(true_m[fcol].sum())
        ratio = (blocks / catches) if catches else None
        print(
            f"  {name:18s} catches {catches:5,}/{nF:,} ({catches/nF:5.1%})  "
            f"wrongly-blocks {blocks:6,}/{nT:,} ({blocks/nT:5.1%})  "
            f"true-per-false {ratio:.2f}" if ratio is not None else f"  {name}: n/a"
        )
        return {
            "catches_false": catches,
            "catches_false_pct": catches / nF if nF else None,
            "blocks_true": blocks,
            "blocks_true_pct": blocks / nT if nT else None,
            "true_blocked_per_false_caught": ratio,
        }

    print("\nconflict-veto confusion matrix (on rule-confirmed pairs):")
    signals = {
        "disjoint_phone": "ph_disjoint",
        "conflicting_ssn": "ssn_conflict",
        "conflicting_email": "em_conflict",
        "any_conflict": "any_conflict",
    }
    table = {k: stat(k, v) for k, v in signals.items()}

    # The irreducible core: false merges with NO conflicting strong ID at all.
    irreducible = int((~false_m.any_conflict).sum())
    print(
        f"\nirreducible core (false merges, no conflicting strong ID at all): "
        f"{irreducible:,}/{nF:,} ({irreducible/nF:.1%})"
    )

    # Per-rule breakdown of the false merges, for context.
    per_rule = (
        false_m.groupby("match_rule").size().sort_values(ascending=False).to_dict()
    )
    print("\nfalse merges by winning rule:")
    for r, c in per_rule.items():
        print(f"  {r:18s} {c:,}")

    _plot(table, nF, nT, irreducible)

    out = {
        "n_silver_pairs": int(len(silver)),
        "n_rule_confirmed": int(len(m)),
        "n_false_merges": nF,
        "n_true_merges": nT,
        "conflict_veto_confusion": table,
        "irreducible_core": {
            "count": irreducible,
            "pct_of_false_merges": irreducible / nF if nF else None,
        },
        "false_merges_by_rule": {k: int(v) for k, v in per_rule.items()},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
