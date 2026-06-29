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

from src.config import (  # noqa: E402
    Settings,
    configure_logging,
    settings as default_settings,
)
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
from src.preprocessing.clean import _load as _read_raw, write_cleaned  # noqa: E402
from src.preprocessing.transformations import transform_dataframe  # noqa: E402
from src.preprocessing.stacked_blocking import run_stacked_blocking  # noqa: E402
from src.models.deterministic_rules import (  # noqa: E402
    AUTO_MERGE_RULES,
    apply_rules,
    assign_clusters,
    classify_non_matches,
    get_match_stats,
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
    configure_logging(settings)
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

    # ── Stage 2: block (stacked: 8-block ∪ q-gram → CNP/ARCS prune) ────────
    candidate_pairs = run_stacked_blocking(cleaned)
    validate(candidate_pairs, CandidatePairs)
    logger.info("[2/3] BLOCK — %d candidate pairs", len(candidate_pairs))
    pairs_path = settings.blocking_dir / f"candidate_pairs_{run_id}.parquet"
    candidate_pairs.to_parquet(pairs_path, index=False)

    # ── Stage 3: deterministic rules ──────────────────────────────────────
    assert_patid_coverage(candidate_pairs, cleaned)  # guaranteed in-process; guard anyway
    confirmed = apply_rules(
        candidate_pairs, cleaned, ssn_fanout_threshold=settings.ssn_fanout_threshold
    )
    # Split confirmed pairs by rule tier: AUTO_MERGE_RULES auto-merge; the rest
    # (NAME_DOB_SEX / NAME_DOB_ADDRESS) are confirmed but routed to review.
    is_auto = confirmed["match_rule"].isin(AUTO_MERGE_RULES)
    matches = confirmed[is_auto].reset_index(drop=True)
    review_confirmed = confirmed[~is_auto].reset_index(drop=True)
    if not matches.empty:
        clusters = assign_clusters(matches)
        matches = matches.copy()
        matches["cluster_id"] = matches["PATID_A"].map(clusters)
    validate(matches, Matches)
    # Three-way split. `confirmed` (both tiers) is removed from the contradiction
    # split so review-tier pairs are never reject-scored; they join review below.
    decided = classify_non_matches(candidate_pairs, confirmed, cleaned)
    pair_cols = list(candidate_pairs.columns)
    non_matches = pd.concat(
        [
            review_confirmed[pair_cols],
            decided[decided["decision"] == "review"][pair_cols],
        ]
    ).reset_index(drop=True)
    rejects = decided[decided["decision"] == "reject"].reset_index(drop=True)
    validate(non_matches, NonMatches)
    stats = get_match_stats(
        matches,
        n_records=len(cleaned),
        decided=decided,
        review_matches=review_confirmed,
    )
    logger.info(
        "[3/3] RULES — %d auto-merge, %d review (%d rule-confirmed), %d reject, "
        "%d clusters",
        len(matches), len(non_matches), len(review_confirmed), len(rejects),
        stats.get("n_clusters", 0),
    )
    matches_path = settings.matches_dir / f"matches_{run_id}.parquet"
    matches.to_parquet(matches_path, index=False)
    non_matches_path = settings.non_matches_dir / f"non_matches_{run_id}.parquet"
    non_matches.to_parquet(non_matches_path, index=False)
    rejects_path = settings.rejects_dir / f"rejects_{run_id}.parquet"
    rejects.to_parquet(rejects_path, index=False)

    # ── Stages 4–5 (model, clustering): add here, feeding the uniform Edges
    #    contract into a terminal clustering step over all confirmed edges. ──

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
        rejects=_artifact_ref(rejects_path, rejects, root),
        counts={
            "raw_rows": len(raw_df),
            "cleaned_rows": len(cleaned),
            "valid_records": n_valid,
            "candidate_pairs": len(candidate_pairs),
            "matches": len(matches),
            "non_matches": len(non_matches),
            "rejects": len(rejects),
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
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override EMPI_LOG_LEVEL for this run (DEBUG/INFO/WARNING/...).",
    )
    args = parser.parse_args()
    configure_logging(level=args.log_level)
    run_pipeline(raw_input=args.input, run_id=args.run_id)


if __name__ == "__main__":
    main()
