"""
run_u_sweep.py — sweep the FS Splink baseline across u_max_pairs values to
resolve the "u not trained" warnings on rare high-precision levels (SSN exact,
Email exact / username, Phones_array intersect >= 2) reported at the default
u_max_pairs=1e6.

*** RUN THIS ON THE VM ONLY. NEVER in the sandbox / off the VM. ***

The 1e6 random pair sample is ~7.5e-5 of the 13.3B-pair population, which is
why rare exact-identifier matches go unobserved during u-estimation. Sweeping
1e6 -> 1e7 -> 1e8 lets us see (i) which levels become "trained" (move from
default-u fallback to a sampled estimate), (ii) how the per-level m/u shifts,
(iii) whether the tier distribution and the boundary-band membership change.

Each sweep value drives one full call to ``run_fs_baseline`` (same code path
as ``run_real_baseline.py``), so this is end-to-end retraining per value;
predict is fast but EM + u-sampling dominates wall time. Expect ~linear-ish
scaling in the u-sampling phase: 1e7 is ~10x base cost, 1e8 ~100x.

Outputs (one set per u value, plus two side-by-side diff CSVs):
    models/outputs/fs_splink_baseline__<data-version>__u<exp>.parquet
        Rich (full_output=True) frame for the run; carries match_probability,
        classification_tier, source_blocks, n_blocks, gamma_* levels, retained
        identifier columns.
    models/artifacts/fs_splink_baseline/diagnostics__<data-version>__u<exp>.json
        Non-PHI diagnostics bundle.
    models/artifacts/fs_splink_baseline/u_sweep_summary__<data-version>.csv
        One row per (u_max_pairs) with tier counts, score quantiles, and
        boundary-band counts.
    models/artifacts/fs_splink_baseline/u_sweep_mu__<data-version>.csv
        One row per (u_max_pairs, comparison, level) with trained m/u so the
        weight-inversion / untrained-level changes are visible at a glance.

Usage (from project root, with the project venv activated):
    python models/experiments/fs_splink_baseline/run_u_sweep.py

    # Custom sweep list:
    python models/experiments/fs_splink_baseline/run_u_sweep.py \\
        --u-max-pairs-list 1e6,1e7

    # Pin inputs (otherwise auto-resolved to the newest parquets on disk):
    python models/experiments/fs_splink_baseline/run_u_sweep.py \\
        --cleaned-index data/processed/MDM_Population_cleaned_v3_2026_06_11.parquet \\
        --candidate-pairs src/features/outputs/blocking/candidate_pairs_v4_2026_06_11.parquet
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

# Project root = three levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from models.experiments.fs_splink_baseline import fellegi_sunter_baseline as fs
from models.common.versioning import latest_versioned, version_tag_from_filename

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "models" / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts" / fs.MODEL_NAME

CLEANED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_GLOB = "MDM_Population_cleaned_v*_*.parquet"
PAIRS_DIR = PROJECT_ROOT / "src" / "features" / "outputs" / "blocking"
PAIRS_GLOB = "candidate_pairs_v*_*.parquet"

DEFAULT_U_LIST = [1e6, 1e7, 1e8]

# Boundary bands used to characterize threshold-edge populations. These match
# the ranges the validation notebook's Section 5.2 sampling work targets, so
# the row counts here line up with what the manual review will inspect.
BOUNDARY_BANDS = [
    ("review_floor_band", 0.45, 0.55),
    ("auto_merge_band", 0.85, 0.95),
]


def _u_tag(u: float) -> str:
    """Compact tag like 'u1e6' for filenames."""
    if u <= 0:
        return f"u{u:g}"
    exp = math.log10(u)
    if abs(exp - round(exp)) < 1e-9:
        return f"u1e{int(round(exp))}"
    return f"u{u:g}"


def _parse_u_list(raw: str) -> list[float]:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError(f"--u-max-pairs-list parsed to empty list from {raw!r}")
    return out


def _summarize_run(
    u: float,
    df_full: pd.DataFrame,
    diagnostics: dict,
) -> tuple[dict, list[dict]]:
    """
    Return (summary_row, mu_rows) for one sweep value.

    summary_row: tier counts + score quantiles + boundary-band counts +
                 list of comparison levels still missing u-training.
    mu_rows:     per-(comparison, level) m/u rows, ready to write to CSV.
    """
    p = df_full["match_probability"]
    tier_counts = df_full["classification_tier"].value_counts().to_dict()

    summary: dict = {
        "u_max_pairs": u,
        "u_tag": _u_tag(u),
        "n_pairs": int(len(df_full)),
        "tier_auto_merge": int(tier_counts.get("auto_merge", 0)),
        "tier_human_review": int(tier_counts.get("human_review", 0)),
        "tier_no_match": int(tier_counts.get("no_match", 0)),
        "score_min": float(p.min()),
        "score_p25": float(p.quantile(0.25)),
        "score_median": float(p.median()),
        "score_p75": float(p.quantile(0.75)),
        "score_max": float(p.max()),
    }
    for label, lo, hi in BOUNDARY_BANDS:
        summary[f"{label}_count"] = int(((p >= lo) & (p < hi)).sum())

    mu_rows: list[dict] = []
    untrained: list[str] = []
    for r in diagnostics.get("parameter_records", []) or []:
        comparison = r.get("comparison_name")
        level = r.get("label_for_charts")
        m = r.get("m_probability")
        u_prob = r.get("u_probability")
        mu_rows.append(
            {
                "u_max_pairs": u,
                "u_tag": _u_tag(u),
                "comparison": comparison,
                "level": level,
                "m_probability": m,
                "u_probability": u_prob,
                "m_minus_u": (
                    (m - u_prob) if isinstance(m, (int, float)) and isinstance(u_prob, (int, float)) else None
                ),
                "weight_inverted": (
                    bool(isinstance(m, (int, float)) and isinstance(u_prob, (int, float)) and u_prob > 0 and m < u_prob)
                ),
            }
        )
        # Splink emits the level with u_probability=None (or sometimes the
        # default-fallback marker) when u-sampling never observed it.
        if u_prob is None:
            untrained.append(f"{comparison}/{level}")

    summary["untrained_levels_count"] = len(untrained)
    summary["untrained_levels"] = "; ".join(untrained) if untrained else ""

    return summary, mu_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cleaned-index", type=Path, default=None,
        help=(
            "Path to the cleaned patient index parquet. Defaults to the "
            f"highest-versioned file matching {CLEANED_GLOB} in {CLEANED_DIR}."
        ),
    )
    p.add_argument(
        "--candidate-pairs", type=Path, default=None,
        help=(
            "Path to the candidate-pairs parquet from blocking.py. Defaults to "
            f"the highest-versioned file matching {PAIRS_GLOB} in {PAIRS_DIR}."
        ),
    )
    p.add_argument(
        "--data-version", default=None,
        help=(
            "Tag used in output filenames, e.g. v4_2026_06_11. Defaults to the "
            "version tag parsed from the resolved candidate-pairs filename."
        ),
    )
    p.add_argument(
        "--u-max-pairs-list", default=",".join(f"{v:g}" for v in DEFAULT_U_LIST),
        help="Comma-separated u_max_pairs values to sweep (default 1e6,1e7,1e8).",
    )
    p.add_argument(
        "--auto-merge-threshold", type=float, default=fs.DEFAULT_AUTO_MERGE_THRESHOLD,
    )
    p.add_argument(
        "--review-floor", type=float, default=fs.DEFAULT_REVIEW_FLOOR,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cleaned_index is None:
        args.cleaned_index = latest_versioned(CLEANED_DIR, CLEANED_GLOB)
        logger.info("Auto-resolved --cleaned-index -> %s", args.cleaned_index)
    if args.candidate_pairs is None:
        args.candidate_pairs = latest_versioned(PAIRS_DIR, PAIRS_GLOB)
        logger.info("Auto-resolved --candidate-pairs -> %s", args.candidate_pairs)
    if args.data_version is None:
        args.data_version = version_tag_from_filename(args.candidate_pairs)
        logger.info("Auto-resolved --data-version -> %s", args.data_version)

    u_list = _parse_u_list(args.u_max_pairs_list)
    logger.info("Sweeping u_max_pairs over: %s", u_list)

    logger.info("Loading cleaned patient index from %s", args.cleaned_index)
    df_clean = pd.read_parquet(args.cleaned_index)
    logger.info("Loaded %d records", len(df_clean))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    mu_rows_all: list[dict] = []

    for u in u_list:
        tag = _u_tag(u)
        logger.info(
            "=== sweep run: u_max_pairs=%.0e (%s) ===", u, tag,
        )
        df_full, diagnostics = fs.run_fs_baseline(
            args.candidate_pairs,
            df_clean,
            auto_merge_threshold=args.auto_merge_threshold,
            review_floor=args.review_floor,
            u_max_pairs=u,
            full_output=True,
            return_diagnostics=True,
        )

        out_path = OUTPUTS_DIR / f"{fs.MODEL_NAME}__{args.data_version}__{tag}.parquet"
        df_full.to_parquet(out_path, index=False)
        logger.info("Wrote %d scored pairs -> %s", len(df_full), out_path)

        diag_path = ARTIFACTS_DIR / f"diagnostics__{args.data_version}__{tag}.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2, default=str)
        logger.info("Wrote diagnostics -> %s", diag_path)

        summary_row, mu_rows = _summarize_run(u, df_full, diagnostics)
        summary_rows.append(summary_row)
        mu_rows_all.extend(mu_rows)
        logger.info(
            "Tier breakdown @ %s: auto_merge=%d  human_review=%d  no_match=%d  "
            "(untrained_levels=%d)",
            tag, summary_row["tier_auto_merge"], summary_row["tier_human_review"],
            summary_row["tier_no_match"], summary_row["untrained_levels_count"],
        )

    summary_csv = ARTIFACTS_DIR / f"u_sweep_summary__{args.data_version}.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    logger.info("Wrote sweep summary -> %s", summary_csv)

    mu_csv = ARTIFACTS_DIR / f"u_sweep_mu__{args.data_version}.csv"
    with open(mu_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mu_rows_all[0].keys()))
        w.writeheader()
        w.writerows(mu_rows_all)
    logger.info("Wrote sweep m/u table -> %s", mu_csv)

    # Aggregate-only printout — no PHI / identifiers.
    logger.info("Sweep complete. %d runs.", len(summary_rows))
    for row in summary_rows:
        logger.info(
            "  %s: tier=(am=%d, hr=%d, nm=%d)  untrained_levels=%d  "
            "review_floor_band=%d  auto_merge_band=%d",
            row["u_tag"], row["tier_auto_merge"], row["tier_human_review"],
            row["tier_no_match"], row["untrained_levels_count"],
            row["review_floor_band_count"], row["auto_merge_band_count"],
        )


if __name__ == "__main__":
    main()
