"""
Pipeline execution script for the eMPI deterministic matching-rule stage.

Loads the cleaned dataset and the blocking candidate pairs, applies the
deterministic rules from `docs/Deterministic-Rules-Guide.md`, prints an audit
report, and saves the confirmed matches to a versioned parquet file.

LOCATION:
    src/models/run_rules.py

USAGE:
    From the project root (UofC-EMPI/):
        python src/models/run_rules.py

    With explicit inputs:
        python src/models/run_rules.py \
            --clean   data/processed/MDM_Population_cleaned_v1_2026_05_24.parquet \
            --pairs   data/blocking/candidate_pairs_v1_2026_05_24.parquet \
            --output  data/auto_merge/

What the script does:
    1. Resolves the project root so imports work from any working directory
    2. Loads the cleaned dataset (defaults to the highest-versioned
       MDM_Population_cleaned_v<N>_*.parquet in data/processed/)
    3. Loads the candidate pairs (defaults to the highest-versioned
       candidate_pairs_v<N>_*.parquet in data/blocking/)
    4. Applies the deterministic rules via apply_rules()
    5. Prints an audit report and saves the matches to a versioned parquet
    6. Saves the unconfirmed candidate pairs (non-matches) to a versioned
       parquet so they can be picked up by downstream matching processes

OUTPUT:
    matches_v{N}_{YYYY_MM_DD}.parquet written to the output directory. The
    version N is auto-incremented above the highest existing matches file.

    non_matches_v{N}_{YYYY_MM_DD}.parquet written to data/non_matches/. These
    are the candidate pairs that no deterministic rule confirmed — the input to
    the downstream (probabilistic / ML) matching stage. The full blocking
    schema (source_blocks, n_blocks) is preserved for provenance.

Dependencies:
    pip install pandas pyarrow
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

# ── Path resolution ───────────────────────────────────────────────────────────
# This file lives at src/models/run_rules.py; the project root is two levels up.
_SCRIPT_DIR = Path(__file__).resolve().parent  # src/models/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # UofC-EMPI/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.contracts import (  # noqa: E402
    CandidatePairs,
    CleanedRecords,
    Matches,
    NonMatches,
    Rejects,
    TIER_AUTO_MERGE,
    TIER_HUMAN_REVIEW,
    TIER_NO_MATCH,
    assert_patid_coverage,
    validate,
)
from src.config import configure_logging  # noqa: E402
from src.models.clustering import assign_clusters  # noqa: E402
from src.models.deterministic_rules import (  # noqa: E402
    AUTO_MERGE_RULES,
    apply_rules,
    classify_non_matches,
    get_match_stats,
)

logger = logging.getLogger(__name__)

# ── Default paths (sourced from the central config) ──────────────────────────
from src.config import settings  # noqa: E402

DEFAULT_CLEAN_DIR = settings.processed_dir
DEFAULT_CLEAN_STEM = settings.cleaned_stem
DEFAULT_PAIRS_DIR = settings.blocking_dir
DEFAULT_OUTPUT_DIR = settings.auto_merge_dir
DEFAULT_NON_MATCH_DIR = settings.non_matches_dir


def _latest_versioned(input_dir: Path, pattern: re.Pattern[str], label: str) -> Path:
    """Return the file in `input_dir` matching `pattern` with the highest
    captured version number. Ties broken lexicographically (chronological for
    the `YYYY_MM_DD` suffix). Raises FileNotFoundError if none match."""
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
            f"No {label} files matching {pattern.pattern} found in {input_dir}"
        )
    return best[2]


def _next_version(output_dir: Path, stem: str = "matches") -> str:
    """Return the next free `v<N>` tag for `{stem}_v<N>_*.parquet`."""
    pattern = re.compile(rf"^{re.escape(stem)}_v(\d+)_.*\.parquet$")
    max_n = 0
    if output_dir.exists():
        for entry in output_dir.iterdir():
            m = pattern.match(entry.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f"v{max_n + 1}"


def _print_report(stats: dict, version: str) -> None:
    """Print a formatted audit report to stdout."""
    divider = "─" * 60
    print(f"\n{divider}")
    print("  eMPI DETERMINISTIC RULES — AUDIT REPORT")
    print(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Pipeline version: {version}")
    print(divider)

    if not stats:
        print("\n  No confirmed matches produced.\n")
        print(f"{divider}\n")
        return

    print("\n  MATCH DISTRIBUTION (auto-merge)")
    total = stats["total_matches"]
    for rule, count in stats["match_distribution"].items():
        pct = 100 * count / total
        bar = "█" * min(int(count / 1000), 30)
        print(f"    {rule:<18} {count:>10,}  ({pct:5.1f}%)  {bar}")
    print(f"    {'Total':<18} {total:>10,}")

    review_dist = stats.get("review_match_distribution")
    if review_dist:
        print("\n  REVIEW-TIER RULES (confirmed but routed to review, not merged)")
        for rule, count in review_dist.items():
            bar = "█" * min(int(count / 1000), 30)
            print(f"    {rule:<18} {count:>10,}  {bar}")
        print(f"    {'Total':<18} {sum(review_dist.values()):>10,}")

    dist = stats.get("decision_distribution")
    if dist:
        print("\n  THREE-WAY ROUTING")
        print(f"    {'auto-merge':<18} {dist[TIER_AUTO_MERGE]:>10,}")
        print(f"    {'review':<18} {dist[TIER_HUMAN_REVIEW]:>10,}")
        print(f"    {'reject':<18} {dist[TIER_NO_MATCH]:>10,}")

    print("\n  COVERAGE")
    print(f"    Patients matched:         {stats['patients_matched']:>12,}")
    if stats.get("total_patients"):
        print(f"    Total patients:           {stats['total_patients']:>12,}")
        print(f"    Coverage rate:            {stats['coverage_rate']:>11.2f}%")

    print("\n  QUALITY")
    print(f"    Average confidence:       {stats['avg_confidence']:>12.4f}")
    print(
        f"    Suspicious matches:       {stats['suspicious_matches']:>12,}"
        f"  ({stats['suspicious_rate']:.2f}%)"
    )
    if stats.get("high_fanout_ssn_matches"):
        print(
            f"    High-fanout SSN matches:  {stats['high_fanout_ssn_matches']:>12,}"
            "  (shared/fraudulent SSN — review)"
        )

    print("\n  CLUSTERS")
    print(f"    Distinct clusters:        {stats['n_clusters']:>12,}")
    print(f"    Max cluster size:         {stats['max_cluster_size']:>12,}")

    print(f"\n{divider}\n")


def main(
    clean_path: Path,
    pairs_path: Path,
    output_dir: Path,
    non_match_dir: Path,
) -> None:
    """Full rules pipeline: load, apply, report, save matches + non-matches."""
    run_start = datetime.utcnow()

    output_dir.mkdir(parents=True, exist_ok=True)
    non_match_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(output_dir)

    logger.info("=" * 60)
    logger.info("eMPI Deterministic Rules Pipeline starting (version %s)", version)
    logger.info("=" * 60)

    logger.info("Loading cleaned dataset: %s", clean_path)
    df_clean = pd.read_parquet(clean_path)
    df_clean = validate(df_clean, CleanedRecords)  # contract: cleaned input
    logger.info("Loaded %d cleaned records", len(df_clean))

    logger.info("Loading candidate pairs: %s", pairs_path)
    candidate_pairs = pd.read_parquet(pairs_path)
    candidate_pairs = validate(candidate_pairs, CandidatePairs)  # contract: pairs input
    logger.info("Loaded %d candidate pairs", len(candidate_pairs))

    # Lineage guard: the cleaned frame and the pairs must be from the same run.
    assert_patid_coverage(candidate_pairs, df_clean)

    confirmed = apply_rules(candidate_pairs, df_clean)

    # Split confirmed pairs by rule tier: AUTO_MERGE_RULES auto-merge; any
    # review-tier rule (none defined today — see REVIEW_RULES) would be
    # confirmed but routed to review instead.
    is_auto = confirmed["match_rule"].isin(AUTO_MERGE_RULES)
    matches = confirmed[is_auto].reset_index(drop=True)
    review_confirmed = confirmed[~is_auto].reset_index(drop=True)

    # Three-way split of the pairs no AUTO-MERGE rule confirmed. Passing the full
    # `confirmed` set keeps review-tier pairs out of the contradiction split (they
    # join review explicitly below) so they are never reject-scored.
    decided = classify_non_matches(candidate_pairs, confirmed, df_clean)

    stats = get_match_stats(
        matches,
        n_records=len(df_clean),
        decided=decided,
        review_matches=review_confirmed,
    )
    _print_report(stats, version)

    # Attach cluster ids so downstream consumers get linked-entity groupings.
    if not matches.empty:
        clusters = assign_clusters(matches)
        matches = matches.copy()
        matches["cluster_id"] = matches["PATID_A"].map(clusters)

    date_tag = datetime.utcnow().strftime("%Y_%m_%d")

    output_path = output_dir / f"matches_{version}_{date_tag}.parquet"
    validate(matches, Matches)  # contract: matches output (empty frames skipped)
    matches.to_parquet(output_path, index=False)
    logger.info("Matches saved to %s (%d rows)", output_path, len(matches))

    rejects = decided[decided["decision"] == TIER_NO_MATCH].reset_index(drop=True)
    validate(rejects, Rejects)  # contract: rejects output
    pair_cols = list(candidate_pairs.columns)

    # Review output = review-tier rule confirmations + unconfirmed-but-uncertain.
    non_matches = pd.concat(
        [
            review_confirmed[pair_cols],
            decided[decided["decision"] == TIER_HUMAN_REVIEW][pair_cols],
        ]
    ).reset_index(drop=True)
    validate(non_matches, NonMatches)  # contract: non-matches (review) output
    nm_version = _next_version(non_match_dir, stem="non_matches")
    non_match_path = non_match_dir / f"non_matches_{nm_version}_{date_tag}.parquet"
    non_matches.to_parquet(non_match_path, index=False)
    logger.info(
        "Review pairs saved to %s (%d rows) for downstream matching.",
        non_match_path,
        len(non_matches),
    )

    settings.no_match_dir.mkdir(parents=True, exist_ok=True)
    rj_version = _next_version(settings.no_match_dir, stem="rejects")
    reject_path = settings.no_match_dir / f"rejects_{rj_version}_{date_tag}.parquet"
    rejects.to_parquet(reject_path, index=False)
    logger.info(
        "Rejected (confident non-match) pairs saved to %s (%d rows); not routed "
        "downstream.",
        reject_path,
        len(rejects),
    )

    elapsed = (datetime.utcnow() - run_start).total_seconds()
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("Matches:     %s", output_path)
    logger.info("Review:      %s", non_match_path)
    logger.info("Rejects:     %s", reject_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="eMPI deterministic matching-rule pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clean",
        type=Path,
        default=None,
        help=f"Cleaned dataset parquet (default: latest "
        f"{DEFAULT_CLEAN_STEM}_v<N>_*.parquet in {DEFAULT_CLEAN_DIR})",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help=f"Candidate pairs parquet (default: latest "
        f"candidate_pairs_v<N>_*.parquet in {DEFAULT_PAIRS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for matches parquet (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--non-match-output",
        type=Path,
        default=DEFAULT_NON_MATCH_DIR,
        help=f"Output directory for the non-matches parquet routed to "
        f"downstream matching (default: {DEFAULT_NON_MATCH_DIR})",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override EMPI_LOG_LEVEL for this run (DEBUG/INFO/WARNING/...).",
    )

    args = parser.parse_args()
    configure_logging(level=args.log_level)

    clean_path = args.clean or _latest_versioned(
        DEFAULT_CLEAN_DIR,
        re.compile(rf"^{re.escape(DEFAULT_CLEAN_STEM)}_v(\d+)_.*\.parquet$"),
        "cleaned dataset",
    )
    pairs_path = args.pairs or _latest_versioned(
        DEFAULT_PAIRS_DIR,
        re.compile(r"^candidate_pairs_v(\d+)_.*\.parquet$"),
        "candidate pairs",
    )

    main(
        clean_path=clean_path,
        pairs_path=pairs_path,
        output_dir=args.output,
        non_match_dir=args.non_match_output,
    )
