"""Entry point for cleaning MDM_Population raw files.

Loads a CSV/XLSX from `data/raw/`, runs the transformations defined in
`docs/Data-Cleaning-Guide.md` via `transformations.transform_dataframe`, and
writes a versioned CSV `<stem>_cleaned_v<N>.csv` into `data/processed/`
(higher than any existing version for that stem).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

PACKAGE_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.data.transformations import transform_dataframe

DEFAULT_RAW_DIR = PACKAGE_ROOT / 'data' / 'raw'
DEFAULT_PROCESSED_DIR = PACKAGE_ROOT / 'data' / 'processed'
SUPPORTED_EXTENSIONS = {'.csv', '.xls', '.xlsx'}


def clean_mdm_population(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all guide-defined transformations to a raw DataFrame."""
    return transform_dataframe(df)


def _next_version(processed_dir: Path, stem: str) -> int:
    """Return the next free `_v<N>` integer for `<stem>_cleaned_v<N>.csv`."""
    pattern = re.compile(rf'^{re.escape(stem)}_cleaned_v(\d+)\.csv$')
    max_n = 0
    if processed_dir.exists():
        for entry in processed_dir.iterdir():
            m = pattern.match(entry.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _load(input_path: Path) -> pd.DataFrame:
    # dtype=str preserves leading zeros on ID-like fields (SSN, ZipCD, PATID).
    # Without it, pandas infers numeric dtypes and "012345678" becomes 12345678
    # before the per-field transformations can apply the left-pad rules from
    # docs/Data-Cleaning-Guide.md.
    suffix = input_path.suffix.lower()
    if suffix in {'.xls', '.xlsx'}:
        return pd.read_excel(input_path, dtype=str)
    return pd.read_csv(input_path, dtype=str)


def load_cleaned_csv(path: str | os.PathLike) -> pd.DataFrame:
    """Read a cleaned CSV produced by this module while preserving leading
    zeros on string-typed ID fields (`PATID`, `last_4_SSN`, `ZipCD_clean_base`,
    `ZipCD_clean_ext`, etc.). Always prefer this over a bare `pd.read_csv`
    on processed files."""
    return pd.read_csv(path, dtype=str)


def clean_from_file(
    input_path: str | os.PathLike,
    processed_dir: Optional[str | os.PathLike] = None,
) -> Tuple[pd.DataFrame, Path]:
    """Load `input_path`, clean it, write the next-version CSV, and return
    (cleaned_df, output_path)."""
    input_path = Path(input_path)
    processed_dir = Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = _load(input_path)
    cleaned = transform_dataframe(df)

    version = _next_version(processed_dir, input_path.stem)
    output_path = processed_dir / f'{input_path.stem}_cleaned_v{version}.csv'
    cleaned.to_csv(output_path, index=False)

    return cleaned, output_path


def process_raw_directory(
    raw_dir: Optional[str | os.PathLike] = None,
    processed_dir: Optional[str | os.PathLike] = None,
) -> list:
    """Clean every supported file in `raw_dir`, write to `processed_dir`."""
    raw_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    processed_dir = Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_DIR

    results = []
    for entry in sorted(raw_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        cleaned, output_path = clean_from_file(entry, processed_dir)
        results.append((entry, output_path, cleaned))
    return results


if __name__ == '__main__':
    args = sys.argv[1:]
    if args:
        in_file = args[0]
        out_dir = args[1] if len(args) >= 2 else None
        cleaned, out_path = clean_from_file(in_file, out_dir)
        print(f'Cleaned {len(cleaned):,} records → {out_path}')
        if 'valid_record' in cleaned.columns:
            print(f"  valid_record=True : {int(cleaned['valid_record'].sum()):,}")
            print(f"  valid_record=False: {int((~cleaned['valid_record']).sum()):,}")
    else:
        results = process_raw_directory()
        print(f'Processed {len(results)} files from {DEFAULT_RAW_DIR} → {DEFAULT_PROCESSED_DIR}')
        for src_path, out_path, cleaned in results:
            n_valid = int(cleaned['valid_record'].sum()) if 'valid_record' in cleaned.columns else 0
            print(f'  {src_path.name}: {len(cleaned):,} rows ({n_valid:,} valid) → {out_path.name}')
