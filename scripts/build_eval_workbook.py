"""Run the pipeline on a raw dataset and build a manual-evaluation workbook.

Produces an .xlsx with one tab per deterministic match rule, each holding up to
N sampled matches. Every row shows both records side by side (interleaved
`<field>_A` / `<field>_B`) plus the full match metadata, so a reviewer can judge
each match against every available field.

USAGE:
    python scripts/build_eval_workbook.py
    python scripts/build_eval_workbook.py --input data/raw/MDM_Population.csv \
        --output data/matches/match_eval.xlsx --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.contracts import RULE_NAMES  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

# Fields shown first (most useful for evaluation); remaining columns follow.
_PRIORITY = [
    "FirstNM_clean", "MiddleNM_clean", "LastNM_clean", "SuffixNM_clean",
    "BirthDT_clean", "SexAtBirthDSC_clean",
    "SSN_clean", "last_4_SSN",
    "Email_clean", "Phones_set",
    "AddressLine1_clean", "AddressLine2_clean", "CityNM_clean",
    "StateCD_clean", "ZipCD_clean_base", "ZipCD_clean_ext",
    "FirstNM_raw", "LastNM_raw", "BirthDT_raw", "SSN_raw", "Email_raw",
]
_META = [
    "PATID_A", "PATID_B", "match_rule", "confidence", "rules_fired",
    "is_suspicious", "high_fanout_ssn", "n_blocks", "source_blocks", "cluster_id",
]


def _cellify(v: object) -> object:
    """Make a cell Excel-safe: collapse list/array values to a string."""
    if isinstance(v, (list, tuple, set, frozenset, np.ndarray)):
        return "; ".join(str(x) for x in v)
    return v


def _attr_columns(cleaned: pd.DataFrame) -> list[str]:
    """Cleaned columns to show per side, priority fields first."""
    present = [c for c in _PRIORITY if c in cleaned.columns]
    rest = [c for c in cleaned.columns if c not in present and c != "PATID"]
    return present + rest


def build_rule_frame(
    rule_matches: pd.DataFrame, cleaned: pd.DataFrame, attr_cols: list[str]
) -> pd.DataFrame:
    """Join both records' attributes onto each match, interleaved A/B."""
    attrs = cleaned.set_index("PATID")[attr_cols]
    merged = rule_matches.join(attrs.add_suffix("_A"), on="PATID_A").join(
        attrs.add_suffix("_B"), on="PATID_B"
    )
    interleaved = [f"{c}_{side}" for c in attr_cols for side in ("A", "B")]
    ordered = [c for c in _META if c in merged.columns] + interleaved
    out = merged[ordered].copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(_cellify)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a match-evaluation workbook")
    parser.add_argument("--input", type=Path, default=None, help="Raw CSV/XLSX (default: settings.raw_input)")
    parser.add_argument("--output", type=Path, default=None, help="Output .xlsx path")
    parser.add_argument("--n", type=int, default=100, help="Samples per rule (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42)")
    args = parser.parse_args()

    manifest = run_pipeline(raw_input=args.input)
    root = _PROJECT_ROOT
    cleaned = pd.read_parquet(root / manifest.cleaned.path)
    matches = pd.read_parquet(root / manifest.matches.path)

    output = args.output or (root / "data" / "matches" / f"match_eval_{manifest.run_id}.xlsx")
    output.parent.mkdir(parents=True, exist_ok=True)
    attr_cols = _attr_columns(cleaned)

    summary_rows = []
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for rule in RULE_NAMES:
            rule_matches = matches[matches["match_rule"] == rule]
            n_sample = min(args.n, len(rule_matches))
            sampled = (
                rule_matches.sample(n=n_sample, random_state=args.seed)
                if n_sample
                else rule_matches
            )
            frame = build_rule_frame(sampled, cleaned, attr_cols)
            frame.to_excel(writer, sheet_name=rule[:31], index=False)
            summary_rows.append({
                "match_rule": rule,
                "total_matches": int(len(rule_matches)),
                "sampled": int(n_sample),
                "suspicious_in_sample": int(sampled["is_suspicious"].sum()) if n_sample else 0,
            })

        summary = pd.DataFrame(summary_rows)
        summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Run {manifest.run_id}: {len(matches):,} matches total")
    print(summary.to_string(index=False))
    print(f"\nWorkbook written to {output}")


if __name__ == "__main__":
    main()
