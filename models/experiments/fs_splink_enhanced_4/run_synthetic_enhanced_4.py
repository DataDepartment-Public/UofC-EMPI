"""Sandbox runner for fs_splink_enhanced_4.

Smoke test on synthetic data (real silver labels are VM-only): trains on
`data/synthetic/synthetic_train_v3.csv` and scores
`data/synthetic/synthetic_test_v3.csv`. Confirms the end-to-end flow round-trips
through `ProbabilisticMatches` validation, the lambda prior seeds without error,
the test-split confusion matrix is sane, and the corroboration gate fires (or
gracefully no-ops when signals are absent).

USAGE:
    python -m models.experiments.fs_splink_enhanced_4.run_synthetic_enhanced_4

OUTPUT:
    Prints tier counts + test-split confusion matrix + gate-demotion count to
    stdout. No on-disk artifacts (sandbox).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.experiments.fs_splink_enhanced_4.fs_enhanced_4 import FSEnhanced4  # noqa: E402
from models.experiments.fs_splink_enhanced_4.corroboration_gate import (  # noqa: E402
    CORROBORATION_COL,
    DEMOTED_REASON,
)
from src.contracts import ProbabilisticMatches, validate  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fs_enhanced_4.sandbox")

SYNTH_DIR = _PROJECT_ROOT / "data" / "synthetic"


def _load_records() -> pd.DataFrame:
    df = pd.read_csv(SYNTH_DIR / "synthetic_records_v3.csv", dtype=str)
    df["BirthDT_clean"] = pd.to_datetime(df["BirthDT_clean"], errors="coerce")
    df["valid_record"] = True
    return df


def _load_labeled_pairs(name: str) -> pd.DataFrame:
    df = pd.read_csv(SYNTH_DIR / f"{name}.csv", dtype=str)
    df["label"] = df["label"].astype(int)
    return df


def _build_candidate_pairs(test_pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = test_pairs[["PATID_A", "PATID_B"]].copy()
    swap = pairs["PATID_A"] > pairs["PATID_B"]
    pairs.loc[swap, ["PATID_A", "PATID_B"]] = pairs.loc[
        swap, ["PATID_B", "PATID_A"]
    ].values
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

    # Synthetic records carry AddressLine1_clean / SexAtBirthDSC_clean, so the
    # full comparison registry + all prior rules apply. u_max_pairs=1e4 is safe
    # under Splink's single-CPU salting guard.
    # model.run() sets _gate_df_clean internally before calling classify(), so
    # the corroboration gate is armed without any manual attribute setting here.
    model = FSEnhanced4(
        labels_df=train_labels,
        include_address=True,
        u_max_pairs=1e4,
    )
    classified = model.run(candidate_pairs, df_clean, full_output=True)

    tier_counts = classified["classification_tier"].value_counts().to_dict()
    logger.info("Tier breakdown: %s", {k: int(v) for k, v in tier_counts.items()})

    prob_matches = model.to_probabilistic_matches(classified)
    validate(prob_matches, ProbabilisticMatches)
    logger.info("ProbabilisticMatches validation passed (%d rows)", len(prob_matches))

    merged = classified.merge(
        test_labels[["PATID_A", "PATID_B", "label"]],
        on=["PATID_A", "PATID_B"], how="left",
    )
    confusion = (
        merged.groupby(["label", "classification_tier"]).size().unstack(fill_value=0)
    )
    logger.info("Confusion matrix (rows=label, cols=tier):\n%s", confusion)

    # Corroboration gate demotion count (aggregate, non-PHI).
    if CORROBORATION_COL in classified.columns:
        gate_demotions = int((classified[CORROBORATION_COL] == DEMOTED_REASON).sum())
    else:
        gate_demotions = 0
    logger.info(
        "Corroboration gate demotions (synthetic pool): %d", gate_demotions
    )

    logger.info("Sandbox smoke complete.")


if __name__ == "__main__":
    main()
