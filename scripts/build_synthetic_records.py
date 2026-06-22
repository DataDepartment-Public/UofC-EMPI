"""Build `data/synthetic/synthetic_records_v3.csv` from the paired CSVs.

The paired CSVs (`synthetic_train_v3.csv`, `synthetic_test_v3.csv`) carry
one row per pair with the FS feature columns suffixed `_l` / `_r`. To use
these as a records frame for FS supervised m-training, we need to deconstruct
each pair into two records and union both files — so the labels' PATIDs
(which span both train and test pair sets) resolve against a single records
frame.

Why not reuse `data/synthetic/format_synthetic_to_blocking_testing.ipynb`?
That notebook only deconstructs `synthetic_test_v3.csv`. Records-only-from-test
is correct for blocking-recall testing but wrong for supervised m-training
where the labels live in `synthetic_train_v3.csv` (different PATID space).
This script produces the union output the FS pipeline needs.

USAGE:
    python scripts/build_synthetic_records.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_DIR = _PROJECT_ROOT / "data" / "synthetic"


def deconstruct_paired(df: pd.DataFrame) -> pd.DataFrame:
    """Pair rows (columns suffixed `_l` / `_r`) → records (one row per record).
    Drops duplicate PATIDs (same record appearing on both sides of multiple pairs).
    """
    left_cols = ["PATID_A"] + [c for c in df.columns if c.endswith("_l")]
    right_cols = ["PATID_B"] + [c for c in df.columns if c.endswith("_r")]
    left = df[left_cols].rename(columns=lambda c: c[:-2])
    right = df[right_cols].rename(columns=lambda c: c[:-2])
    return pd.concat([left, right], ignore_index=True).drop_duplicates(subset=["PATID"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=SYNTH_DIR / "synthetic_train_v3.csv")
    parser.add_argument("--test", type=Path, default=SYNTH_DIR / "synthetic_test_v3.csv")
    parser.add_argument("--output", type=Path, default=SYNTH_DIR / "synthetic_records_v3.csv")
    args = parser.parse_args()

    for p in (args.train, args.test):
        if not p.exists():
            print(f"ERROR: input missing: {p}", file=sys.stderr)
            sys.exit(1)

    train = pd.read_csv(args.train, dtype=str)
    test = pd.read_csv(args.test, dtype=str)
    records = pd.concat(
        [deconstruct_paired(train), deconstruct_paired(test)],
        ignore_index=True,
    ).drop_duplicates(subset=["PATID"]).reset_index(drop=True)

    records.to_csv(args.output, index=False)
    print(f"Wrote {len(records):,} unique records → {args.output}")
    print(f"  (train pairs in: {len(train):,}, test pairs in: {len(test):,})")


if __name__ == "__main__":
    main()
