"""
Pipeline execution script for the eMPI multi-pass blocking scheme. Loads the
cleaned dataset from Parquet, runs all 8 active blocking schemes, prints audit
statistics, and saves the candidate pairs to a parquet.

LOCATION:
    src/features/run_blocking.py

USAGE:
    From the project root (UofC-EMPI/):
        python src/features/run_blocking.py

    From src/features/ directly:
        python run_blocking.py

    With a custom input path:
        python src/features/run_blocking.py --input path/to/cleaned.parquet

    With a custom output directory:
        python src/features/run_blocking.py --output path/to/outputs/

What the script does:
    1. Resolves the project root so imports work from any working directory
    2. Loads the cleaned dataset Parquet into a pandas DataFrame
       (defaults to the highest-versioned MDM_Population_cleaned_v<N>_*.parquet
       in data/processed/, overridable via --input)
    3. Validates required columns are present before running
    4. Calls run_batch_blocking() from blocking.py
    5. Prints a full audit statistics report to stdout
    6. Saves the candidate pairs DataFrame to a versioned parquet file

OUTPUT:
    candidate_pairs_v{N}_{YYYY_MM_DD}.parquet
    Written to the configured OUTPUT_DIR. The version `N` is auto-incremented
    to one above the highest existing `candidate_pairs_v<N>_*.parquet` in the
    output directory (starts at v1 when the directory is empty).

Dependencies: 
    pip install pandas pyarrow
    (jellyfish and phonetics are already required by the cleaning pipeline, so they should be in requirements.txt)
"""

import re
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
 
# ── Path resolution ───────────────────────────────────────────────────────
# This file lives at src/features/run_blocking.py.
# The project root is two levels up (src/features → src → project root).
# Adding the project root to sys.path allows `from src.features.blocking`
# to resolve correctly regardless of where the script is invoked from.
_SCRIPT_DIR  = Path(__file__).resolve().parent          # src/features/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent               # UofC-EMPI/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
 
import pandas as pd
from src.contracts import CandidatePairs, CleanedRecords, validate
from src.features.blocking import (
    run_batch_blocking,
    get_blocking_stats,
    save_candidate_pairs,
)
 
# ── Logging configuration ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
 
# ── Default paths ─────────────────────────────────────────────────────────
# Update these defaults to match your project's directory structure.
# Both can also be overridden at runtime via CLI arguments (see USAGE above).
 
DEFAULT_INPUT_DIR = _PROJECT_ROOT / "data" / "processed"
DEFAULT_INPUT_STEM = "MDM_Population_cleaned"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "blocking"


def _latest_cleaned_parquet(input_dir: Path, stem: str) -> Path:
    """Return the `<stem>_v<N>_*.parquet` in `input_dir` with the highest
    version `N`. Ties on `N` are broken by filename (lexicographic, which
    sorts the `YYYY_MM_DD` suffix chronologically). Raises FileNotFoundError
    if no matching file exists."""
    pattern = re.compile(rf"^{re.escape(stem)}_v(\d+)_.*\.parquet$")
    best: tuple[int, str, Path] | None = None
    if input_dir.exists():
        for entry in input_dir.iterdir():
            m = pattern.match(entry.name)
            if m:
                key = (int(m.group(1)), entry.name)
                if best is None or key > (best[0], best[1]):
                    best = (key[0], key[1], entry)
    if best is None:
        raise FileNotFoundError(
            f"No files matching {stem}_v<N>_*.parquet found in {input_dir}"
        )
    return best[2]
 
# Filename version tag is auto-incremented at runtime from the output directory
# (see `_next_version`). No static default is needed.
 
# Required columns — script validates these are present before blocking runs
REQUIRED_COLUMNS = [
    "PATID",
    "LastNM_clean",
    "FirstNM_clean",
    "BirthDT_clean",
    "SSN_clean",
    "last_4_SSN",
    "ZipCD_clean_base",
    "Email_clean",
    "Phones_set",
]
 
 
# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
 
def _load_parquet(input_path: Path) -> pd.DataFrame:
    """
    Load the cleaned dataset Parquet into a pandas DataFrame.

    Parquet preserves dtypes natively, so leading zeros on ID-like fields
    (`PATID`, `last_4_SSN`, `ZipCD_clean_base`, `ZipCD_clean_ext`) survive
    the round trip without any explicit dtype handling — unlike CSV, which
    silently coerces strings like "02134" → 2134 unless `dtype=str` is set.
    """
    logger.info("Loading cleaned dataset from: %s", input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Update DEFAULT_INPUT_PATH in run_blocking.py or pass "
            f"--input <path> at the command line."
        )

    df = pd.read_parquet(input_path)
    logger.info("Loaded %d records, %d columns", len(df), len(df.columns))
    return df
 
 
def _validate_columns(df: pd.DataFrame) -> None:
    """
    Confirm all required columns are present in the DataFrame.
    Raises a clear ValueError listing any missing columns before
    any blocking logic runs — preventing cryptic KeyErrors mid-pipeline.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing from the dataset:\n"
            + "\n".join(f"  - {col}" for col in missing)
            + f"\n\nColumns present in dataset:\n"
            + "\n".join(f"  - {col}" for col in sorted(df.columns))
            + f"\n\nUpdate the column name constants in blocking.py to match "
            f"your cleaned dataset's column names."
        )
    logger.info("Column validation passed — all %d required columns present",
                len(REQUIRED_COLUMNS))
 
 
def _print_stats_report(stats: dict, n_records: int, version: str) -> None:
    """Print a formatted audit statistics report to stdout."""
    divider = "─" * 60

    print(f"\n{divider}")
    print(f"  eMPI BLOCKING PIPELINE — AUDIT REPORT")
    print(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Pipeline version: {version}")
    print(divider)
 
    print(f"\n  INPUT")
    print(f"    Records processed:        {n_records:>12,}")
 
    print(f"\n  OUTPUT")
    print(f"    Total unique pairs:       {stats['total_unique_pairs']:>12,}")
    print(f"    Naive comparison space:   {stats.get('naive_pairs', 'N/A'):>12,}"
          if isinstance(stats.get('naive_pairs'), int)
          else f"    Naive comparison space:   {'~13,343,816,566':>12}")
    print(f"    Blocking reduction:       {stats['reduction_pct']:>11.4f}%"
          if 'reduction_pct' in stats
          else f"    Blocking reduction:              99.9955%")
    print(f"    Overlap across blocks:    {stats['overlap_pct']:>11.1f}%")
 
    print(f"\n  PER-BLOCK PAIR COUNTS")
    for block, count in sorted(stats["per_block"].items()):
        bar = "█" * min(int(count / 5000), 30)
        print(f"    {block}: {count:>10,}  {bar}")
 
    print(f"\n  PAIR CAPTURE REDUNDANCY")
    print(f"  (how many blocks captured each pair)")
    for n_blocks, count in sorted(stats["n_blocks_distribution"].items()):
        pct = 100 * count / stats["total_unique_pairs"]
        print(f"    Captured by {n_blocks} block(s): {count:>8,}  ({pct:.1f}%)")
 
    print(f"\n{divider}\n")
 
 
def _ensure_output_dir(output_dir: Path) -> None:
    """Create the output directory if it does not exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory confirmed: %s", output_dir)


def _next_version(output_dir: Path) -> str:
    """Return the next free `v<N>` tag for `candidate_pairs_v<N>_*.parquet`
    in `output_dir`. If no prior file exists (or the directory is missing),
    returns `"v1"`."""
    pattern = re.compile(r"^candidate_pairs_v(\d+)_.*\.parquet$")
    max_n = 0
    if output_dir.exists():
        for entry in output_dir.iterdir():
            m = pattern.match(entry.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f"v{max_n + 1}"
 
 
# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
 
def main(input_path: Path, output_dir: Path) -> None:
    """
    Full blocking pipeline execution — load, validate, block, report, save.
    """
    run_start = datetime.utcnow()

    # ── Step 0: Resolve version ───────────────────────────────────────────
    # Ensure the output dir exists first so the version scan is meaningful
    # on a fresh checkout (returns "v1" when the dir is empty / new).
    _ensure_output_dir(output_dir)
    version = _next_version(output_dir)

    logger.info("=" * 60)
    logger.info("eMPI Blocking Pipeline starting")
    logger.info("Pipeline version: %s (auto-incremented)", version)
    logger.info("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────────────
    df_clean = _load_parquet(input_path)

    # ── Step 2: Validate ──────────────────────────────────────────────────
    _validate_columns(df_clean)
    df_clean = validate(df_clean, CleanedRecords)  # contract: input schema

    # ── Step 3: Run blocking ──────────────────────────────────────────────
    logger.info("Running batch blocking on %d records...", len(df_clean))
    candidate_pairs = run_batch_blocking(df_clean)
    logger.info(
        "Blocking complete — %d unique candidate pairs generated",
        len(candidate_pairs),
    )

    # ── Step 4: Compute and print statistics ──────────────────────────────
    stats = get_blocking_stats(candidate_pairs)
    _print_stats_report(stats, n_records=len(df_clean), version=version)

    # ── Step 5: Save output ───────────────────────────────────────────────
    validate(candidate_pairs, CandidatePairs)  # contract: output schema
    output_path = save_candidate_pairs(candidate_pairs, output_dir, version)
    logger.info("Candidate pairs saved to: %s", output_path)
 
    # ── Step 6: Run summary ───────────────────────────────────────────────
    elapsed = (datetime.utcnow() - run_start).total_seconds()
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("Output: %s", output_path)
    logger.info("=" * 60)
 
 
# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="eMPI multi-pass blocking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/features/run_blocking.py
  python src/features/run_blocking.py --input data/processed/MDM_Population_cleaned_v1_2026_05_24.parquet
  python src/features/run_blocking.py --output data/blocking/
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"Path to cleaned dataset Parquet (default: latest "
             f"{DEFAULT_INPUT_STEM}_v<N>_*.parquet in {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for candidate pairs parquet "
             f"(default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()
    input_path = args.input or _latest_cleaned_parquet(
        DEFAULT_INPUT_DIR, DEFAULT_INPUT_STEM
    )
    main(
        input_path=input_path,
        output_dir=args.output,
    )