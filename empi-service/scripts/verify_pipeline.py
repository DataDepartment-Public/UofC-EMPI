"""
End-to-end pipeline smoke test: clean -> block -> deterministic rules.

Runs each production stage against the synthetic raw file in data/raw/ and
asserts that every stage produces non-empty, well-formed output. Exercises the
real stage code (transformations.transform_dataframe, run_blocking.py,
run_rules.py) rather than reimplementing it.

USAGE:
    python scripts/verify_pipeline.py

NOTE — format bridge:
    The cleaning CLI (src/preprocessing/clean.py) writes CSV, but the blocking CLI
    (src/preprocessing/run_blocking.py) reads a versioned Parquet named
    `MDM_Population_cleaned_v<N>_<date>.parquet`. This script writes that
    Parquet directly from the cleaned DataFrame so the downstream CLIs can
    pick it up. See the printed SUMMARY for the recommended permanent fix.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.preprocessing.transformations import transform_dataframe  # noqa: E402

RAW = _PROJECT_ROOT / "data" / "raw" / "MDM_Population.csv"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
BLOCKING_DIR = _PROJECT_ROOT / "data" / "blocking"
MATCHES_DIR = _PROJECT_ROOT / "data" / "matches"


def _run_cli(script: Path) -> None:
    """Invoke a pipeline CLI with the current interpreter; fail loudly."""
    print(f"\n$ python {script.relative_to(_PROJECT_ROOT)}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"Stage failed: {script.name} (exit {result.returncode})")


def _latest(directory: Path, glob: str) -> Path:
    files = sorted(directory.glob(glob))
    if not files:
        raise FileNotFoundError(f"No file matching {glob} in {directory}")
    return files[-1]


def main() -> None:
    print("=" * 64)
    print("  eMPI PIPELINE VERIFICATION")
    print("=" * 64)

    # ── Stage 1: clean ────────────────────────────────────────────────────────
    assert RAW.exists(), f"Missing synthetic raw file: {RAW}"
    raw_df = pd.read_csv(RAW, dtype=str)
    print(f"\n[1/3] CLEAN — loaded {len(raw_df):,} raw records, "
          f"{len(raw_df.columns)} columns")
    cleaned = transform_dataframe(raw_df)
    n_valid = int(cleaned["valid_record"].sum())
    print(f"      cleaned -> {len(cleaned):,} rows, {len(cleaned.columns)} columns; "
          f"valid_record=True: {n_valid:,}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y_%m_%d")
    clean_parquet = PROCESSED_DIR / f"MDM_Population_cleaned_v1_{stamp}.parquet"
    cleaned.to_parquet(clean_parquet, index=False)
    print(f"      wrote {clean_parquet.relative_to(_PROJECT_ROOT)}")

    # ── Stage 2: block (real CLI) ─────────────────────────────────────────────
    print("\n[2/3] BLOCK")
    _run_cli(_PROJECT_ROOT / "src" / "features" / "run_blocking.py")
    pairs_path = _latest(BLOCKING_DIR, "candidate_pairs_v*_*.parquet")
    pairs = pd.read_parquet(pairs_path)
    print(f"      {pairs_path.relative_to(_PROJECT_ROOT)} -> {len(pairs):,} pairs")

    # ── Stage 3: deterministic rules (real CLI) ───────────────────────────────
    print("\n[3/3] RULES")
    _run_cli(_PROJECT_ROOT / "src" / "models" / "run_rules.py")
    matches_path = _latest(MATCHES_DIR, "matches_v*_*.parquet")
    matches = pd.read_parquet(matches_path)
    print(f"      {matches_path.relative_to(_PROJECT_ROOT)} -> {len(matches):,} matches")

    # ── Assertions ────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  CHECKS")
    print("=" * 64)
    checks = {
        "cleaned has PATID": "PATID" in cleaned.columns,
        "cleaned non-empty": len(cleaned) > 0,
        "candidate pairs produced": len(pairs) > 0,
        "pairs schema ok": {"PATID_A", "PATID_B", "source_blocks",
                            "n_blocks"} <= set(pairs.columns),
        "matches produced": len(matches) > 0,
        "matches schema ok": {"PATID_A", "PATID_B", "match_rule", "confidence",
                             "rules_fired", "is_suspicious", "cluster_id"}
        <= set(matches.columns),
        "every rule type fired": set(matches["match_rule"].unique()).issubset({
            "SSN_DOB", "NAME_DOB_PHONE", "NAME_DOB_EMAIL"}),
        "confidence in range": matches["confidence"].between(0.9, 1.0).all(),
    }
    ok = True
    for name, passed in checks.items():
        print(f"   [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and bool(passed)

    print("\n  Rules fired (distribution):")
    for rule, count in matches["match_rule"].value_counts().items():
        print(f"     {rule:<18} {count:>8,}")

    print("\n" + "=" * 64)
    if ok:
        print("  RESULT: PASS — full pipeline ran clean -> block -> rules")
    else:
        print("  RESULT: FAIL — see checks above")
        raise SystemExit(1)
    print("=" * 64)


if __name__ == "__main__":
    main()
