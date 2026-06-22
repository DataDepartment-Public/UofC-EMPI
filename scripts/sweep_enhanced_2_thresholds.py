"""Threshold sweep for fs_splink_enhanced_2.

Phase E2-4 calibration tool. Given a scored-pairs frame and a labels frame,
sweep over the grid:

    auto_merge_threshold ∈ {0.85, 0.90, 0.925, 0.95}
    review_floor         ∈ {0.35, 0.40, 0.50, 0.55}

For each combination compute:

    precision_AM           — fraction of AM-tier predictions that are TRUE positives
    recall_AM              — fraction of TRUE positives that reach AM
    recall_AM_or_HR        — fraction of TRUE positives that reach AM ∪ HR
    fp_count_in_AM         — count of FALSE positives in AM
    n_AM, n_HR, n_NM       — tier sizes

Pareto gates per the plan:
    gate_1: precision_AM ≥ 0.95   (patient-safety floor)
    gate_2: recall_AM_or_HR ≥ 0.66 ON 42-LABEL REAL DATA
            (scaled to 0.80 when running on synthetic, which has 2,000 positives)

USAGE
-----
Local (synthetic test labels — 10,000 pairs):
    python scripts/sweep_enhanced_2_thresholds.py --mode synthetic

VM (real 42 reviewer labels — extracted from the validation notebook):
    python scripts/sweep_enhanced_2_thresholds.py --mode real \\
        --scored models/outputs/fs_splink_enhanced_2__<v>.parquet \\
        --labels-csv /tmp/reviewer_labels_42.csv

VM (silver labels — deterministic-rule confirmations as a held-out
positives-only validation set; Stage 3 catches these pairs and filters
them out of `non_matches`, so the scored frame must come from a
**full candidate-pool** run, NOT the post-rules non_matches scored frame):
    python scripts/sweep_enhanced_2_thresholds.py --mode real \\
        --scored models/outputs/fs_splink_enhanced_2__<v>_full_pool.parquet \\
        --labels-csv data/silver_labels/silver_labels.csv

The labels CSV expected schema:
    PATID_A, PATID_B, label
where `label ∈ {0, 1}` (real-data convention: same → 1, different → 0,
unsure → drop before passing in). All-positive label frames (e.g., silver
labels from deterministic-rule confirmations) are handled gracefully:
`precision_AM` becomes trivially 1.0 and only the recall columns carry
signal — interpret accordingly.

OUTPUT
------
Stdout: full sweep table + Pareto-optimal pick under the gates.
Optional CSV via --output flag for downstream analysis.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

AUTO_MERGE_GRID = (0.85, 0.90, 0.925, 0.95)
REVIEW_FLOOR_GRID = (0.35, 0.40, 0.50, 0.55)


# ── Score-bump helper (replicates FSModel._apply_n_blocks_bump for offline sweep) ─
def _apply_n_blocks_bump(df: pd.DataFrame, threshold: int = 2, max_bits: float = 4.0) -> pd.DataFrame:
    if "n_blocks" not in df.columns:
        return df
    import numpy as np
    bump_bits = (df["n_blocks"] - threshold).clip(lower=0, upper=max_bits)
    if not bool((bump_bits > 0).any()):
        return df
    df = df.copy()
    p_safe = df["match_probability"].clip(lower=1e-12, upper=1 - 1e-12)
    weight = np.log2(p_safe / (1.0 - p_safe))
    df["match_probability"] = 1.0 / (1.0 + np.power(2.0, -(weight + bump_bits)))
    return df


def _classify(p: pd.Series, auto_merge: float, review_floor: float) -> pd.Series:
    """Vectorised threshold classify; matches FSModel.classify semantics."""
    tier = pd.Series("no_match", index=p.index, dtype="object")
    tier = tier.mask(p >= review_floor, "human_review")
    tier = tier.mask(p >= auto_merge, "auto_merge")
    return tier


def _row_metrics(
    p: pd.Series, label: pd.Series, auto_merge: float, review_floor: float,
) -> dict:
    tier = _classify(p, auto_merge, review_floor)
    n_AM = int((tier == "auto_merge").sum())
    n_HR = int((tier == "human_review").sum())
    n_NM = int((tier == "no_match").sum())
    pos_total = int((label == 1).sum())
    tp_AM = int(((tier == "auto_merge") & (label == 1)).sum())
    fp_AM = int(((tier == "auto_merge") & (label == 0)).sum())
    tp_AM_HR = int(((tier.isin(["auto_merge", "human_review"])) & (label == 1)).sum())
    precision_AM = (tp_AM / n_AM) if n_AM > 0 else float("nan")
    recall_AM = (tp_AM / pos_total) if pos_total > 0 else float("nan")
    recall_AM_HR = (tp_AM_HR / pos_total) if pos_total > 0 else float("nan")
    return {
        "auto_merge_threshold": auto_merge,
        "review_floor": review_floor,
        "n_AM": n_AM, "n_HR": n_HR, "n_NM": n_NM,
        "tp_AM": tp_AM, "fp_AM": fp_AM,
        "precision_AM": precision_AM,
        "recall_AM": recall_AM,
        "recall_AM_or_HR": recall_AM_HR,
    }


def sweep(scored: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply n_blocks bump (if present), merge labels, sweep the grid.

    Returns (table, n_labels_missing_from_scored). When silver-labeled pairs
    are excluded by Stage 3 from the scored frame, the inner-merge drops them;
    the missing count is returned for the caller to warn on.
    """
    df = _apply_n_blocks_bump(scored)
    merged = df.merge(labels, on=["PATID_A", "PATID_B"], how="inner")
    n_missing = len(labels) - len(merged)
    rows = [
        _row_metrics(merged["match_probability"], merged["label"], am, rf)
        for am, rf in itertools.product(AUTO_MERGE_GRID, REVIEW_FLOOR_GRID)
        if rf <= am
    ]
    return pd.DataFrame(rows), n_missing


def pick_pareto(table: pd.DataFrame, precision_gate: float, recall_gate: float) -> pd.DataFrame:
    """Filter to feasible points; rank by precision desc, then recall desc."""
    feasible = table[
        (table["precision_AM"] >= precision_gate) &
        (table["recall_AM_or_HR"] >= recall_gate)
    ].copy()
    if feasible.empty:
        return feasible
    return feasible.sort_values(
        ["precision_AM", "recall_AM_or_HR"], ascending=[False, False],
    ).reset_index(drop=True)


# ── Mode handlers ────────────────────────────────────────────────────────────
def _run_synthetic(args) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Train + score on synthetic and return (scored, labels, label-source)."""
    from models.experiments.fs_splink_enhanced_2.fs_enhanced_2 import FSEnhanced2

    synth_dir = _PROJECT_ROOT / "data" / "synthetic"
    df_clean = pd.read_csv(synth_dir / "synthetic_blocking_testing.csv", dtype=str)
    df_clean["BirthDT_clean"] = pd.to_datetime(df_clean["BirthDT_clean"], errors="coerce")
    df_clean["valid_record"] = True

    train = pd.read_csv(synth_dir / "synthetic_train_v3.csv", dtype=str)
    train["label"] = train["label"].astype(int)
    test = pd.read_csv(synth_dir / "synthetic_test_v3.csv", dtype=str)
    test["label"] = test["label"].astype(int)

    cp = test[["PATID_A", "PATID_B"]].copy()
    swap = cp["PATID_A"] > cp["PATID_B"]
    cp.loc[swap, ["PATID_A", "PATID_B"]] = cp.loc[swap, ["PATID_B", "PATID_A"]].values
    cp = cp.drop_duplicates().reset_index(drop=True)
    cp["source_blocks"] = "synthetic_sandbox"
    cp["n_blocks"] = 1

    model = FSEnhanced2(labels_df=train, include_address=False, u_max_pairs=1e4)
    df_model = model.prepare_model_input(df_clean)
    linker = model.build_linker(df_model, cp)
    model.train(linker, df_clean)
    scored = model.predict(linker, cp)
    return scored, test[["PATID_A", "PATID_B", "label"]], "synthetic_test_v3 (10,000 labels)"


def _run_real(args) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    scored = pd.read_parquet(args.scored)
    labels = pd.read_csv(args.labels_csv)
    labels = labels[["PATID_A", "PATID_B", "label"]].copy()
    labels["label"] = labels["label"].astype(int)
    return scored, labels, f"{args.labels_csv} ({len(labels)} labels)"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["synthetic", "real"], required=True)
    parser.add_argument("--scored", type=Path, default=None,
                        help="Scored parquet (real mode only)")
    parser.add_argument("--labels-csv", type=Path, default=None,
                        help="Labels CSV with PATID_A/PATID_B/label (real mode only)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write sweep table to CSV")
    parser.add_argument("--precision-gate", type=float, default=0.95)
    parser.add_argument("--recall-gate", type=float, default=None,
                        help="Default: 0.80 synthetic / 0.66 real")
    args = parser.parse_args()

    if args.mode == "synthetic":
        scored, labels, source_desc = _run_synthetic(args)
        recall_gate = args.recall_gate if args.recall_gate is not None else 0.80
    else:
        if not args.scored or not args.labels_csv:
            parser.error("real mode requires --scored and --labels-csv")
        scored, labels, source_desc = _run_real(args)
        recall_gate = args.recall_gate if args.recall_gate is not None else 0.66

    n_pos = int((labels["label"] == 1).sum())
    n_neg = int((labels["label"] == 0).sum())
    print(f"\nThreshold sweep — source: {source_desc}")
    print(f"  positives: {n_pos}, negatives: {n_neg}")
    print(f"  precision gate: ≥{args.precision_gate:.2f}, "
          f"recall (AM∪HR) gate: ≥{recall_gate:.2f}")
    if n_neg == 0:
        print("  ⚠ all-positive labels detected (e.g., silver labels): "
              "`precision_AM` is trivially 1.0 — only recall columns are meaningful.\n")
    else:
        print()

    table, n_missing = sweep(scored, labels)
    if n_missing > 0:
        print(f"  ⚠ {n_missing}/{len(labels)} labeled pairs were not present in the "
              f"scored frame and were dropped by the inner-merge. If labels come from "
              f"silver (deterministic-rule confirmations), the scored frame must be "
              f"the FULL candidate-pool run, not the post-Stage-3 non_matches subset.\n")
    if args.output:
        table.to_csv(args.output, index=False)

    display = table.copy()
    for c in ("precision_AM", "recall_AM", "recall_AM_or_HR"):
        display[c] = display[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "nan")
    print("Full sweep:")
    print(display.to_string(index=False))

    pareto = pick_pareto(table, args.precision_gate, recall_gate)
    print("\nPareto-feasible (precision desc, recall desc):")
    if pareto.empty:
        print("  ⚠ no combination meets both gates simultaneously.")
    else:
        for c in ("precision_AM", "recall_AM", "recall_AM_or_HR"):
            pareto[c] = pareto[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "nan")
        print(pareto.to_string(index=False))
        top = table.iloc[(
            (table["precision_AM"] >= args.precision_gate) &
            (table["recall_AM_or_HR"] >= recall_gate)
        ).values].sort_values(
            ["precision_AM", "recall_AM_or_HR"], ascending=[False, False],
        ).iloc[0]
        print(
            f"\n→ RECOMMEND: auto_merge={top['auto_merge_threshold']:.3f}, "
            f"review_floor={top['review_floor']:.3f}"
        )


if __name__ == "__main__":
    main()
