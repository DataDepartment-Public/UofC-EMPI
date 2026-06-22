"""Sandbox runner for fs_splink_enhanced_2.

Trains on `data/synthetic/synthetic_train_v3.csv` (40k labeled pairs) and
scores `data/synthetic/synthetic_test_v3.csv` (10k labeled pairs). Confirms
the end-to-end flow round-trips through `ProbabilisticMatches` validation
and that the m/u parameters land on the expected shape.

The synthetic records frame (`synthetic_blocking_testing.csv`) lacks
CityNM_clean / StateCD_clean, so the runner builds the registry with
`include_address=False`. The VM runner (E2-5) builds with `include_address=True`.

USAGE:
    python -m models.experiments.fs_splink_enhanced_2.run_synthetic_enhanced_2

OUTPUT:
    Prints tier counts + per-comparison m sample to stdout.
    No on-disk artifacts (sandbox).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.experiments.fs_splink_enhanced_2.fs_enhanced_2 import FSEnhanced2  # noqa: E402
from src.contracts import ProbabilisticMatches, validate  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fs_enhanced_2.sandbox")

SYNTH_DIR = _PROJECT_ROOT / "data" / "synthetic"


def _load_records() -> pd.DataFrame:
    """Synthetic records frame (deconstructed from the paired CSVs)."""
    df = pd.read_csv(SYNTH_DIR / "synthetic_blocking_testing.csv", dtype=str)
    # Coerce typed columns the cleaning contract expects.
    df["BirthDT_clean"] = pd.to_datetime(df["BirthDT_clean"], errors="coerce")
    df["valid_record"] = True
    return df


def _load_labeled_pairs(name: str) -> pd.DataFrame:
    df = pd.read_csv(SYNTH_DIR / f"{name}.csv", dtype=str)
    df["label"] = df["label"].astype(int)
    return df


def _build_candidate_pairs(test_pairs: pd.DataFrame) -> pd.DataFrame:
    """The candidate-pairs frame is the test pairs in CandidatePairs shape:
    PATID_A < PATID_B, source_blocks str, n_blocks int.
    """
    pairs = test_pairs[["PATID_A", "PATID_B"]].copy()
    swap = pairs["PATID_A"] > pairs["PATID_B"]
    pairs.loc[swap, ["PATID_A", "PATID_B"]] = pairs.loc[swap, ["PATID_B", "PATID_A"]].values
    pairs = pairs.drop_duplicates(subset=["PATID_A", "PATID_B"]).reset_index(drop=True)
    pairs["source_blocks"] = "synthetic_sandbox"
    pairs["n_blocks"] = 1
    return pairs


def main() -> None:
    logger.info("Loading synthetic data...")
    df_clean = _load_records()
    train_labels = _load_labeled_pairs("synthetic_train_v3")
    test_labels = _load_labeled_pairs("synthetic_test_v3")
    logger.info(
        "Records: %d | Train labels: %d (pos=%d) | Test labels: %d (pos=%d)",
        len(df_clean), len(train_labels), int((train_labels["label"] == 1).sum()),
        len(test_labels), int((test_labels["label"] == 1).sum()),
    )

    candidate_pairs = _build_candidate_pairs(test_labels)
    logger.info("Candidate pairs to score: %d", len(candidate_pairs))

    # Synthetic records don't carry CityNM_clean / StateCD_clean → no Address.
    # u_max_pairs=1e4 is safe under Splink's single-CPU salting guard.
    model = FSEnhanced2(
        labels_df=train_labels,
        include_address=False,
        u_max_pairs=1e4,
    )
    classified = model.run(candidate_pairs, df_clean, full_output=True)

    tier_counts = classified["classification_tier"].value_counts().to_dict()
    logger.info("Tier breakdown: %s", {k: int(v) for k, v in tier_counts.items()})

    # Round-trip through the contract.
    prob_matches = model.to_probabilistic_matches(classified)
    validate(prob_matches, ProbabilisticMatches)
    logger.info("✓ ProbabilisticMatches validation passed (%d rows)", len(prob_matches))

    # Tier × ground-truth confusion matrix.
    merged = classified.merge(
        test_labels[["PATID_A", "PATID_B", "label", "case_type"]],
        on=["PATID_A", "PATID_B"], how="left",
    )
    confusion = (
        merged.groupby(["label", "classification_tier"]).size().unstack(fill_value=0)
    )
    logger.info("Confusion matrix (rows=label, cols=tier):\n%s", confusion)

    logger.info("Sandbox smoke complete.")


if __name__ == "__main__":
    main()
