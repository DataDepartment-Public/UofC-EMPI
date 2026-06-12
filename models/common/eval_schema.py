# models/experiments/common/eval_schema.py
from pathlib import Path
import pandas as pd

EVAL_SCHEMA_COLUMNS = ["PATID_A", "PATID_B", "model_name", "score", "predicted_tier"]
VALID_TIERS = {"auto_merge", "human_review", "no_match"}

def validate_evaluation_frame(df: pd.DataFrame) -> None:
    if list(df.columns) != EVAL_SCHEMA_COLUMNS:
        raise ValueError(f"expected {EVAL_SCHEMA_COLUMNS}, got {list(df.columns)}")
    if df["model_name"].nunique() != 1:
        raise ValueError("one output frame must carry exactly one model_name")
    if not df["score"].between(0, 1).all():
        raise ValueError("score must be in [0, 1]")
    if not df["predicted_tier"].isin(VALID_TIERS).all():
        raise ValueError(f"predicted_tier must be one of {VALID_TIERS}")

def save_evaluation_output(df: pd.DataFrame, outputs_dir, data_version: str) -> Path:
    validate_evaluation_frame(df)
    outputs_dir = Path(outputs_dir); outputs_dir.mkdir(parents=True, exist_ok=True)
    path = outputs_dir / f"{df['model_name'].iloc[0]}__{data_version}.parquet"
    df.to_parquet(path, index=False)
    return path