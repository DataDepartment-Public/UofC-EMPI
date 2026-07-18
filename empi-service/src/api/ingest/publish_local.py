"""Local-mode CLI for publishing a completed batch pipeline run — no
FastAPI/uvicorn, no `empi.db`. Materializes one `RunManifest`'s Parquet
output into the local Parquet index (`ParquetIndexBackend`,
`src/api/backends/parquet_backend.py`), for a fully self-contained local/CI batch
workflow with zero SQLite dependency:

    python -m src.pipeline --input data/raw/MDM_Population.csv   # unchanged —
                                                                  # writes Parquet
                                                                  # artifacts + manifest only
    python -m src.api.ingest.publish_local --run-id <id>                # this module —
                                                                  # materializes that
                                                                  # run into data/local_index/

USAGE:
    python -m src.api.ingest.publish_local --run-id 20260714T120000Z
    python -m src.api.ingest.publish_local --run-id 20260714T120000Z --data-dir data/local_index

`--data-dir` defaults to `settings.local_index_dir` (`data/local_index/`) and
is created if it doesn't exist yet. The counterpart for incremental scoring
is `src/api/ingest/local_score.py`; both are `ParquetIndexBackend`-backed and share
the same on-disk index, so a batch publish and a subsequent local score see
the same population.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.api.ingest import publish
from src.api.backends.parquet_backend import ParquetIndexBackend
from src.config import Settings, configure_logging, settings as default_settings

logger = logging.getLogger("eMPI.publish_local")


def publish_local(
    run_id: str, settings: Settings, data_dir: Path | None = None
) -> dict:
    """Publish `run_id`'s manifest into the local Parquet index at `data_dir`
    (default `settings.local_index_dir`). The one entry point both `main()`
    and tests use. Returns `publish_run`'s summary counts dict."""
    backend = ParquetIndexBackend(data_dir or settings.local_index_dir)
    try:
        return publish.publish_run(backend, run_id, settings)
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="eMPI local-mode batch publish (Parquet run -> local Parquet index)"
    )
    parser.add_argument(
        "--run-id", type=str, required=True,
        help="Run id to publish (reads data/runs/run_<run-id>.json).",
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
    logger.info("Publishing run %s into local index at %s", args.run_id, data_dir)

    counts = publish_local(args.run_id, default_settings, data_dir=data_dir)

    logger.info(
        "Published run %s: %d clusters, %d entities, %d members, "
        "%d locked-skipped, %d suggestions, %d review candidates",
        args.run_id, counts["clusters_seen"], counts["entities_upserted"],
        counts["members_upserted"], counts["locked_skipped"],
        counts["suggestions_written"], counts["review_candidates"],
    )
    print(json.dumps(counts, indent=2, default=str))


if __name__ == "__main__":
    main()
