"""
Data cleaning module for MDM_Population dataset.
Provides interface to transformation functions.
"""

import pandas as pd
from .transformations import transform_dataframe


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


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) >= 3 else None
        df = clean_from_file(input_file, output_file)
        print(f"Processed {len(df)} records")
        print(f"Valid records: {df['ValidRecord'].sum()}")
        print(f"Quality score mean: {df['QUALITY_SCORE'].mean():.2f}")
