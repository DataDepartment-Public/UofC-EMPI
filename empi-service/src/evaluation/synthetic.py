"""Turn the synthetic *pair* label set into a record population the pipeline can run.

`data/synthetic_data/synthetic_test_v3.csv` is the project's only label source
with **entity-level** ground truth (`entity_id_a` / `entity_id_b`), which makes
it the only one where cluster metrics are exact rather than inferred from a
transitive closure. It is also leakage-free for the ML stages: the Stage-4.25
gate and the Stage-4.5 matcher were trained on gold, never on this.

The catch is its shape — it is a *pair* file carrying already-cleaned `*_l` /
`*_r` attributes, not raw records, so it cannot be fed to
`python -m src.pipeline --input`. This module reconstructs a record frame from
both sides of every pair and hands it to the real pipeline through
`run_pipeline(cleaned_input=...)`, so Stages 2-5 run exactly as production runs
them rather than as a reimplementation.

What that measures and what it doesn't: **Stage 1 (cleaning) is skipped by
construction** — the inputs are already cleaned and the planted corruptions live
at the cleaned level, so re-running the cleaning rules over them would be lossy.
Blocking recall here *is* real, unlike gold/silver, because the positives were
built independently of blocking rather than being a blocking output.

PHI / HIPAA: the synthetic set is fabricated, but this module logs counts only
regardless, matching the rest of the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Settings, settings as default_settings
from src.contracts import RunManifest
from src.preprocessing.clean import write_cleaned

logger = logging.getLogger(__name__)

__all__ = [
    "REQUIRED_COLUMNS",
    "load_synthetic_pairs",
    "synthetic_records",
    "run_synthetic_pipeline",
]

#: Columns the harness cannot work without — `entity_id_*` is the whole point.
REQUIRED_COLUMNS = ("PATID_A", "PATID_B", "entity_id_a", "entity_id_b")

#: Columns stored space-delimited in the CSV that the pipeline expects as
#: collections (see `contracts._is_listlike_or_null`).
_LIST_COLS = ("Phones_set", "full_name_tokens")


def _to_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [tok for tok in str(value).replace(",", " ").split() if tok]


def load_synthetic_pairs(path: str | Path, label_col: str = "label") -> pd.DataFrame:
    """Read the synthetic pair CSV, validating that it carries entity truth."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Synthetic label file not found: {path}")

    syn = pd.read_csv(path, dtype=str, keep_default_na=True, na_values=[""])
    missing = [c for c in REQUIRED_COLUMNS if c not in syn.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing {missing} — this harness needs the "
            "entity-level ground truth that only the v3 synthetic set carries."
        )
    syn[label_col] = syn[label_col].astype(int).astype(bool)
    return syn


def _side(syn: pd.DataFrame, side: str) -> pd.DataFrame:
    """Pull one side (`_l` or `_r`) of every pair into a record frame."""
    suffix = f"_{side}"
    id_col = "PATID_A" if side == "l" else "PATID_B"
    entity_col = "entity_id_a" if side == "l" else "entity_id_b"
    rename = {c: c[: -len(suffix)] for c in syn.columns if c.endswith(suffix)}
    return syn[[id_col, entity_col, *rename]].rename(
        columns={id_col: "PATID", entity_col: "entity_id", **rename}
    )


def synthetic_records(syn: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Reconstruct the record population and its `{PATID: entity_id}` truth.

    Both sides of every pair become records. Each PATID appears on exactly one
    side in the v3 file, but we de-duplicate anyway so this stays correct for a
    future set that reuses a record across pairs.
    """
    records = pd.concat([_side(syn, "l"), _side(syn, "r")], ignore_index=True)
    records = records.drop_duplicates(subset="PATID").reset_index(drop=True)

    truth = {str(p): str(e) for p, e in zip(records["PATID"], records["entity_id"])}
    records = records.drop(columns=["entity_id"])

    for col in _LIST_COLS:
        if col in records.columns:
            records[col] = records[col].map(_to_list)

    # Every synthetic record is well-formed by construction — the corruptions
    # are realistic typos, not the malformed rows Stage 1's validity filter
    # exists to catch. Without this, blocking would drop the whole population.
    records["valid_record"] = True

    logger.info(
        "Synthetic: %d pairs → %d records over %d true entities",
        len(syn), len(records), len(set(truth.values())),
    )
    return records, truth


def run_synthetic_pipeline(
    records: pd.DataFrame,
    run_id: str,
    settings: Settings = default_settings,
) -> RunManifest:
    """Write the reconstructed records as a cleaned frame and run Stages 2-5."""
    from src.pipeline import run_pipeline  # lazy: pulls splink/lightgbm chains

    settings.ensure_dirs()
    cleaned_path = settings.processed_dir / f"{settings.cleaned_stem}_{run_id}.parquet"
    write_cleaned(records, cleaned_path)
    logger.info("Wrote reconstructed cleaned frame → %s", cleaned_path)
    return run_pipeline(cleaned_input=cleaned_path, run_id=run_id, settings=settings)
