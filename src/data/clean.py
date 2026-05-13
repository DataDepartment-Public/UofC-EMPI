"""
Data cleaning module for MDM_Population dataset.
Provides interface to transformation functions.
"""

import os
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from src.data.transformations import transform_dataframe


def clean_mdm_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and transform MDM_Population dataset for MPI pipeline.
    
    Args:
        df: Raw MDM_Population DataFrame
        
    Returns:
        Transformed DataFrame with cleaned fields and quality indicators
    """
    return transform_dataframe(df)


def clean_from_file(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Load, clean, and optionally save MDM_Population data.
    
    Args:
        input_path: Path to input CSV/Excel file
        output_path: Optional path to save cleaned data
        
    Returns:
        Transformed DataFrame
    """
    # Load data
    if input_path.endswith('.xlsx') or input_path.endswith('.xls'):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)
    
    # Transform
    result = transform_dataframe(df)
    
    # Save if output path provided
    if output_path:
        if output_path.endswith('.xlsx'):
            result.to_excel(output_path, index=False)
        else:
            result.to_csv(output_path, index=False)
    
    return result


def process_raw_directory(raw_dir: Path = None, processed_dir: Path = None) -> list:
    """Clean every supported file in raw_dir and write outputs to processed_dir."""
    raw_dir = Path(raw_dir) if raw_dir is not None else Path(PACKAGE_ROOT) / 'data' / 'raw'
    processed_dir = Path(processed_dir) if processed_dir is not None else Path(PACKAGE_ROOT) / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = {'.csv', '.xls', '.xlsx'}
    processed_files = []

    for input_path in sorted(raw_dir.iterdir()):
        if not input_path.is_file() or input_path.suffix.lower() not in supported_extensions:
            continue

        output_path = processed_dir / input_path.name
        df = clean_from_file(str(input_path), str(output_path))
        processed_files.append((input_path, output_path, df))

    return processed_files


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 1:
        input_file = args[0]
        output_file = args[1] if len(args) >= 2 else None
        df = clean_from_file(input_file, output_file)
        print(f"Processed {len(df)} records")
        print(f"Valid records: {df['ValidRecord'].sum()}")
        print(f"Quality score mean: {df['QUALITY_SCORE'].mean():.2f}")
    else:
        raw_dir = Path(PACKAGE_ROOT) / 'data' / 'raw'
        processed_dir = Path(PACKAGE_ROOT) / 'data' / 'processed'
        processed = process_raw_directory(raw_dir=raw_dir, processed_dir=processed_dir)
        print(f"Processed {len(processed)} files from {raw_dir} to {processed_dir}")
        for input_path, output_path, df in processed:
            print(f"- {input_path.name}: {len(df)} records -> {output_path}")
