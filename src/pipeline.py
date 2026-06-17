"""End-to-end eMPI pipeline orchestrator — the single production entry point.

Runs the stages **in process**, passing DataFrames stage-to-stage rather than
re-resolving "the latest file in the directory":

    raw  ──►  clean/transform  ──►  blocking  ──►  deterministic rules
                                                      ├─► matches
                                                      └─► non-matches

Because one in-memory cleaned frame feeds both blocking and rules, the lineage
mismatch the standalone CLIs are vulnerable to cannot occur here. Every boundary
is validated against the contracts in `src/contracts.py`, and a single `run_id`
ties all artifacts together via a `RunManifest` written to `data/runs/`.

USAGE:
    python -m src.pipeline                         # uses settings.raw_input
    python -m src.pipeline --input data/raw/MDM_Population.csv
    python -m src.pipeline --run-id 20260603T120000Z   # override the run id

Stages 4 (probabilistic model) and 5 (clustering) are not built yet; the clear
insertion point is marked below.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config.config import Settings, settings as default_settings  # noqa: E402
from src.contracts import (  # noqa: E402
    ArtifactRef,
    CandidatePairs,
    CleanedRecords,
    Matches,
    NonMatches,
    RunManifest,
    assert_patid_coverage,
    validate,
)
from src.data.clean import _load as _read_raw, write_cleaned  # noqa: E402
from src.data.transformations import transform_dataframe  # noqa: E402
from src.features.blocking import run_batch_blocking  # noqa: E402
from src.models.deterministic_rules import (  # noqa: E402
    apply_rules,
    assign_clusters,
    get_match_stats,
    get_non_matches,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eMPI.pipeline")


# ── Small utilities ─────────────────────────────────────────────────────────
def _new_run_id() -> str:
    """A sortable, collision-resistant id for one pipeline run."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _artifact_ref(path: Path, df: pd.DataFrame, root: Path) -> ArtifactRef:
    return ArtifactRef(path=_rel(path, root), rows=len(df), sha256=_file_sha256(path))


# ── Orchestration ────────────────────────────────────────────────────────────
def run_pipeline(
    raw_input: Path | None = None,
    settings: Settings = default_settings,
    run_id: str | None = None,
) -> RunManifest:
    """Run clean → block → rules for one input and return the run manifest."""
    run_id = run_id or _new_run_id()
    raw_input = Path(raw_input) if raw_input is not None else settings.raw_input
    settings.ensure_dirs()
    started = datetime.utcnow()

    logger.info("=" * 64)
    logger.info("eMPI pipeline run %s starting", run_id)
    logger.info("Input: %s", raw_input)
    logger.info("=" * 64)

    if not raw_input.exists():
        raise FileNotFoundError(f"Raw input not found: {raw_input}")

    # ── Stage 1: clean ────────────────────────────────────────────────────
    raw_df = _read_raw(raw_input)
    cleaned = transform_dataframe(raw_df)
    validate(cleaned, CleanedRecords)
    n_valid = int(cleaned["valid_record"].sum())
    logger.info(
        "[1/3] CLEAN — %d raw → %d cleaned rows (%d valid)",
        len(raw_df), len(cleaned), n_valid,
    )
    cleaned_path = settings.processed_dir / f"{settings.cleaned_stem}_{run_id}.parquet"
    write_cleaned(cleaned, cleaned_path)

    # ── Stage 2: block ────────────────────────────────────────────────────
    candidate_pairs = run_batch_blocking(cleaned)
    validate(candidate_pairs, CandidatePairs)
    logger.info("[2/3] BLOCK — %d candidate pairs", len(candidate_pairs))
    pairs_path = settings.blocking_dir / f"candidate_pairs_{run_id}.parquet"
    candidate_pairs.to_parquet(pairs_path, index=False)

    # ── Stage 3: deterministic rules ──────────────────────────────────────
    assert_patid_coverage(candidate_pairs, cleaned)  # guaranteed in-process; guard anyway
    matches = apply_rules(
        candidate_pairs, cleaned, ssn_fanout_threshold=settings.ssn_fanout_threshold
    )
    if not matches.empty:
        clusters = assign_clusters(matches)
        matches = matches.copy()
        matches["cluster_id"] = matches["PATID_A"].map(clusters)
    validate(matches, Matches)
    non_matches = get_non_matches(candidate_pairs, matches)
    validate(non_matches, NonMatches)
    stats = get_match_stats(matches, n_records=len(cleaned))
    logger.info(
        "[3/3] RULES — %d matches, %d non-matches, %d clusters",
        len(matches), len(non_matches), stats.get("n_clusters", 0),
    )
    matches_path = settings.matches_dir / f"matches_{run_id}.parquet"
    matches.to_parquet(matches_path, index=False)
    non_matches_path = settings.non_matches_dir / f"non_matches_{run_id}.parquet"
    non_matches.to_parquet(non_matches_path, index=False)

    # ── Stage 4: probabilistic matching (FS enhanced) ─────────────────────
    # Scores the non_matches that the deterministic rules could not confirm.
    # Lazy import keeps splink/duckdb out of the import path when the model
    # stage is disabled.
    matches_model_path: Path | None = None
    if not non_matches.empty:
        logger.info("[4/5] MODEL — scoring %d non-match pairs with FS enhanced...",
                    len(non_matches))
        from models.experiments.fs_splink_enhanced.fellegi_sunter_enhanced import (
            run_fs_enhanced,
            to_probabilistic_matches,
        )
        from src.contracts import ProbabilisticMatches
        classified = run_fs_enhanced(
            candidate_pairs_path=non_matches_path,
            df_clean=cleaned,
            full_output=True,
        )
        prob_matches = to_probabilistic_matches(classified)
        validate(prob_matches, ProbabilisticMatches)
        matches_model_path = settings.matches_model_dir / f"matches_model_{run_id}.parquet"
        prob_matches.to_parquet(matches_model_path, index=False)
        model_tier_counts = prob_matches["classification_tier"].value_counts().to_dict()
        logger.info("[4/5] MODEL — tier breakdown: %s -> %s",
                    {k: int(v) for k, v in model_tier_counts.items()},
                    _rel(matches_model_path, settings.project_root))
    else:
        logger.info("[4/5] MODEL — skipped (no non-match pairs to score)")

    # ── Stage 5 (clustering): add here, feeding deterministic Matches and
    #    ProbabilisticMatches into a uniform Edges projection for clustering. ──

    # ── Manifest ──────────────────────────────────────────────────────────
    root = settings.project_root
    manifest = RunManifest(
        run_id=run_id,
        created_utc=started.isoformat() + "Z",
        git_sha=_current_git_sha(),
        raw_input=_artifact_ref(raw_input, raw_df, root),
        cleaned=_artifact_ref(cleaned_path, cleaned, root),
        candidate_pairs=_artifact_ref(pairs_path, candidate_pairs, root),
        matches=_artifact_ref(matches_path, matches, root),
        non_matches=_artifact_ref(non_matches_path, non_matches, root),
        counts={
            "raw_rows": len(raw_df),
            "cleaned_rows": len(cleaned),
            "valid_records": n_valid,
            "candidate_pairs": len(candidate_pairs),
            "matches": len(matches),
            "non_matches": len(non_matches),
            "clusters": int(stats.get("n_clusters", 0)),
        },
    )
    manifest_path = settings.runs_dir / f"run_{run_id}.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    elapsed = (datetime.utcnow() - started).total_seconds()
    logger.info("=" * 64)
    logger.info("Run %s complete in %.1fs", run_id, elapsed)
    logger.info("Manifest: %s", _rel(manifest_path, root))
    logger.info("=" * 64)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="eMPI end-to-end pipeline")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Raw CSV/XLSX to process (default: settings.raw_input).",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Override the generated run id (default: UTC timestamp).",
    )
    args = parser.parse_args()
    run_pipeline(raw_input=args.input, run_id=args.run_id)


if __name__ == "__main__":
    main()
