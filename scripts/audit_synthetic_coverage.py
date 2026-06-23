"""Audit the synthetic labeled-pairs CSV for per-comparison-level coverage.

Phase E2-0 advisory step. For each comparison level proposed for the
fs_splink_enhanced_2 registry, compute how many positive and negative
labeled pairs would land in that level. Any level with **<20 positives**
is flagged "RISK_OF_MISCALIBRATION — additional coverage may be needed"
so the model builder can decide whether to add a manual prior, broaden
the synthetic generator, or accept the noisier estimate. Levels are
**not** dropped automatically.

Reads `data/synthetic/synthetic_train_v3.csv` (the larger calibration set,
40k pairs with `label`, `case_type`, `_l` / `_r` paired columns) and writes
`data/synthetic/coverage_report.csv` + a human-readable summary to stdout.

USAGE:
    python scripts/audit_synthetic_coverage.py
    python scripts/audit_synthetic_coverage.py --input data/synthetic/synthetic_train_v3.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.blocking import _dm_primary, _soundex  # noqa: E402

POS_THRESHOLD = 20  # Positives below this flag the level for manual review (not dropped)


# ── Level assignment helpers (pair-wise, vectorised where possible) ──────────
def _safe_str(v):
    return None if pd.isna(v) else str(v)


def _split_email(s: str | None) -> tuple[str | None, str | None]:
    if s is None or "@" not in s:
        return None, None
    local, _, domain = s.partition("@")
    return local or None, domain or None


# ── Per-comparison level deciders ────────────────────────────────────────────
def middle_name_level(l, r):
    l, r = _safe_str(l), _safe_str(r)
    if l is None or r is None:
        return "null_either"
    if l == r:
        return "exact"
    if l[0] == r[0]:
        return "initial_match"
    return "mismatch"


def sex_level(l, r):
    l, r = _safe_str(l), _safe_str(r)
    if l is None or r is None:
        return "null_either"
    if l == r:
        return "exact"
    pair = frozenset({l, r})
    if pair == frozenset({"MALE", "FEMALE"}):
        return "M_vs_F"
    if "OTHER" in pair:
        return "OTHER_either"
    return "other_mismatch"


def phonetic_level(l, r, fn):
    """fn = _dm_primary or _soundex"""
    l, r = _safe_str(l), _safe_str(r)
    if l is None or r is None:
        return "null_either"
    pl, pr = fn(l), fn(r)
    if not pl or not pr:
        return "null_either"
    return "phonetic_equal" if pl == pr else "mismatch"


def dob_swap_level(l, r):
    l, r = _safe_str(l), _safe_str(r)
    if l is None or r is None:
        return "null_either"
    if l == r:
        return "exact"
    try:
        y1, m1, d1 = l.split("-")
        y2, m2, d2 = r.split("-")
    except ValueError:
        return "mismatch"
    if y1 == y2 and m1 == d2 and d1 == m2 and m1 != d1:
        return "month_day_swap"
    if y1 == y2:
        return "same_year"
    return "mismatch"


def zip_base_level(l, r):
    l, r = _safe_str(l), _safe_str(r)
    if l is None or r is None:
        return "null_either"
    if l == r:
        return "exact"
    if len(l) >= 3 and len(r) >= 3 and l[:3] == r[:3]:
        return "prefix3_match"
    return "mismatch"


def full_name_compact_level(l, r):
    l, r = _safe_str(l), _safe_str(r)
    if l is None or r is None:
        return "null_either"
    if l == r:
        return "compact_exact"
    return "mismatch"


def email_local_level(l, r):
    l, r = _safe_str(l), _safe_str(r)
    if l is None or r is None:
        return "null_either"
    if l == r:
        return "exact"
    ll, ld = _split_email(l)
    rl, rd = _split_email(r)
    if ll and rl and ll == rl and ld != rd:
        return "local_only"
    return "mismatch"


# ── Audit driver ─────────────────────────────────────────────────────────────
AUDITS = [
    # (comparison_name, level_fn, columns_l, columns_r)
    ("MiddleNM",       middle_name_level,         "MiddleNM_clean_l",     "MiddleNM_clean_r"),
    ("Sex_positive",   sex_level,                 "SexAtBirthDSC_clean_l","SexAtBirthDSC_clean_r"),
    ("LastNM_DM",      lambda l, r: phonetic_level(l, r, _dm_primary),
                                                  "LastNM_clean_l",       "LastNM_clean_r"),
    ("FirstNM_DM",     lambda l, r: phonetic_level(l, r, _dm_primary),
                                                  "FirstNM_clean_l",      "FirstNM_clean_r"),
    ("LastNM_Soundex", lambda l, r: phonetic_level(l, r, _soundex),
                                                  "LastNM_clean_l",       "LastNM_clean_r"),
    ("DOB_swap",       dob_swap_level,            "BirthDT_clean_l",      "BirthDT_clean_r"),
    ("ZipBase",        zip_base_level,            "ZipCD_clean_base_l",   "ZipCD_clean_base_r"),
    ("FullNameCompact",full_name_compact_level,   "full_name_compact_l",  "full_name_compact_r"),
    ("Email_local",    email_local_level,         "Email_clean_l",        "Email_clean_r"),
]


def run_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = df["label"].astype(int).values
    for comparison, fn, col_l, col_r in AUDITS:
        if col_l not in df or col_r not in df:
            rows.append({"comparison": comparison, "level": "MISSING_COLUMN",
                         "positives": 0, "negatives": 0, "flag": "MISSING"})
            continue
        levels = np.array([fn(l, r) for l, r in zip(df[col_l], df[col_r])])
        for level in np.unique(levels):
            mask = levels == level
            pos = int(((labels == 1) & mask).sum())
            neg = int(((labels == 0) & mask).sum())
            flag = "OK" if pos >= POS_THRESHOLD else "RISK_OF_MISCALIBRATION"
            rows.append({"comparison": comparison, "level": level,
                         "positives": pos, "negatives": neg, "flag": flag})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=_PROJECT_ROOT / "data" / "synthetic" / "synthetic_train_v3.csv")
    parser.add_argument("--output", type=Path,
                        default=_PROJECT_ROOT / "data" / "synthetic" / "coverage_report.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str)
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df):,} labeled pairs from {args.input.name}")
    print(f"  positives: {int((df['label']==1).sum()):,}")
    print(f"  negatives: {int((df['label']==0).sum()):,}")
    print()

    report = run_audit(df)
    report.to_csv(args.output, index=False)
    print(f"Coverage report written to {args.output}")
    print()

    # Pretty-print: per-comparison block with flags
    for comparison in report["comparison"].drop_duplicates():
        sub = report[report["comparison"] == comparison]
        print(f"── {comparison} ──")
        for _, row in sub.iterrows():
            marker = "OK " if row["flag"] == "OK" else ("!! " if row["flag"] == "RISK_OF_MISCALIBRATION" else "?? ")
            print(f"  {marker}{row['level']:<20s} positives={row['positives']:>6d}  "
                  f"negatives={row['negatives']:>6d}  [{row['flag']}]")
        print()

    n_risky = int((report["flag"] == "RISK_OF_MISCALIBRATION").sum())
    print(f"=== Summary: {n_risky} level(s) below the {POS_THRESHOLD}-positive threshold "
          f"(flagged 'RISK_OF_MISCALIBRATION — additional coverage may be needed'; not dropped) ===")


if __name__ == "__main__":
    main()
