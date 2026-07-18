"""Local-mode CLI for incremental record scoring — no FastAPI/uvicorn, no
`empi.db`. Scores records against a population tracked purely in Parquet
files (`ParquetIndexBackend`, `src/api/backends/parquet_backend.py`), for local
dev/notebooks/CI use.

USAGE:
    python -m src.api.ingest.local_score --input record.json
    python -m src.api.ingest.local_score --input batch.json --data-dir data/local_index

`--input` is a JSON file holding one record object or a list of record
objects, using the pipeline's raw column names (`PATID` +
`transformations.RENAME_TO_RAW`: FirstNM, LastNM, BirthDT, SSN, ...) — the
same shape `POST /records/score`'s body takes, just as a file instead of an
HTTP request. `--data-dir` defaults to `settings.local_index_dir`
(`data/local_index/`) and is created if it doesn't exist yet — an empty
directory just means "no population scored into it yet," not an error.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.api.ingest import incremental
from src.api.jobs import new_run_id
from src.api.backends.parquet_backend import ParquetIndexBackend
from src.config import Settings, configure_logging, settings as default_settings

logger = logging.getLogger("eMPI.local_score")


def _load_records(input_path: Path) -> list[dict]:
    payload = json.loads(input_path.read_text())
    records = payload if isinstance(payload, list) else [payload]
    if not records:
        raise ValueError(f"{input_path} contains no records to score.")
    return records


def score_local(
    records: list[dict], settings: Settings, data_dir: Path | None = None
) -> list[dict]:
    """Score `records` against the local Parquet-backed population at
    `data_dir` (default `settings.local_index_dir`), minting a fresh run id.
    The one entry point both `main()` and tests use."""
    backend = ParquetIndexBackend(data_dir or settings.local_index_dir)
    try:
        return incremental.score_records(backend, settings, records, new_run_id())
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="eMPI local-mode incremental scoring")
    parser.add_argument(
        "--input", type=Path, required=True,
        help="JSON file: one record object or a list of record objects "
        "(raw column names — PATID, FirstNM, LastNM, BirthDT, SSN, ...).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Local-index Parquet directory (default: settings.local_index_dir).",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override EMPI_LOG_LEVEL for this run (DEBUG/INFO/WARNING/...).",
    )
    args = parser.parse_args()
    configure_logging(level=args.log_level)

    data_dir = args.data_dir or default_settings.local_index_dir
    records = _load_records(args.input)
    logger.info("Scoring %d record(s) against local index at %s", len(records), data_dir)

    outcomes = score_local(records, default_settings, data_dir=data_dir)

    for outcome in outcomes:
        logger.info(
            "PATID %s -> tier=%s mid=%s match_rule=%s confidence=%s",
            outcome["patid"], outcome["tier"], outcome["mid"],
            outcome["match_rule"], outcome["confidence"],
        )
    print(json.dumps(outcomes, indent=2, default=str))


if __name__ == "__main__":
    main()
