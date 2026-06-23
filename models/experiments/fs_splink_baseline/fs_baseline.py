"""FSBaseline — thin FSModel subclass wrapping the baseline FS experiment.

Mirrors the behavior of `fellegi_sunter_baseline.py` using the shared OO
scaffold from `models/common/fs_base.py`.  The comparison vector, EM training
sessions, and classification thresholds are identical to the existing module;
only the delivery mechanism changes.

Phase E3-1: this file is **additive only**.  The old `fellegi_sunter_baseline.py`
module remains in place and continues to be used by runners and tests until
Phase E3-2 swaps the wiring.
"""

from __future__ import annotations

import logging

import pandas as pd
from splink import SettingsCreator

from models.common.fs_base import ClassificationConfig, EMTraining, FSModel
from models.experiments.fs_splink_baseline.comparisons import (
    BASELINE_REGISTRY,
    COL_DOB_STR,
    COL_PATID,
    COL_PHONES_ARRAY,
    COL_SSN_LAST4,
    COL_ZIP,
    EM_BLOCKING_RULES,
    PRIOR_RULES,
)
from src.preprocessing.blocking import (
    _COL_PHONES_SET,
    _compute_derived_columns,
)

# Required columns that must be present after derivation — same list as
# `fellegi_sunter_baseline.REQUIRED_MODEL_COLUMNS`.
REQUIRED_MODEL_COLUMNS = [COL_PATID, COL_DOB_STR, COL_SSN_LAST4, COL_ZIP]


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if derivation didn't run / inputs are malformed."""
    missing = [c for c in REQUIRED_MODEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "prepare_model_input is missing required columns "
            f"{missing}. This usually means _compute_derived_columns() was not "
            "run upstream or the cleaned input is malformed. Required: "
            f"{REQUIRED_MODEL_COLUMNS}."
        )

logger = logging.getLogger(__name__)

MODEL_NAME = "fs_splink_baseline"

CANDIDATE_PAIRS_TABLE = "candidate_pairs"
CP_PATID_A = "PATID_A"
CP_PATID_B = "PATID_B"

# Single-equi candidate-pairs blocking rule (identical to the functional module).
# DuckDB decorrelates this into a HASH_JOIN on (PATID_A, PATID_B) — no OR branch.
CANDIDATE_PAIRS_BLOCKING_RULE = f"""
EXISTS (
    SELECT 1 FROM {CANDIDATE_PAIRS_TABLE} cp
    WHERE cp.{CP_PATID_A} = l.{COL_PATID} AND cp.{CP_PATID_B} = r.{COL_PATID}
)
"""

# Inert placeholder columns added to df_model so Splink's blocking-rule column
# harvesting binds under retain_matching_columns=True.
_PAIR_SHIM_COLUMNS = (CP_PATID_A, CP_PATID_B)


class FSBaseline(FSModel):
    """Baseline Fellegi-Sunter model as a thin FSModel subclass.

    7 comparisons: FirstNM, LastNM, BirthDT, SSN, Email, Phones, ZIP.
    Trained with 3 EM sessions + match-prevalence prior (recall=0.80).
    Classification thresholds: auto_merge=0.90, review_floor=0.50.
    """

    model_name = MODEL_NAME
    registry = BASELINE_REGISTRY
    classification_config = ClassificationConfig(
        auto_merge_threshold=0.90,
        review_floor=0.50,
    )
    candidate_pairs_table_name = CANDIDATE_PAIRS_TABLE
    unique_id_column = COL_PATID

    def __init__(
        self,
        u_max_pairs: float = 1e6,
        seed: int = 42,
        auto_merge_threshold: float | None = None,
        review_floor: float | None = None,
    ):
        self.training = EMTraining(
            em_blocking_rules=EM_BLOCKING_RULES,
            prior_rules=PRIOR_RULES,
            recall=0.80,
            u_max_pairs=u_max_pairs,
            seed=seed,
        )
        if auto_merge_threshold is not None or review_floor is not None:
            base = self.__class__.classification_config
            self.classification_config = ClassificationConfig(
                auto_merge_threshold=(
                    auto_merge_threshold
                    if auto_merge_threshold is not None
                    else base.auto_merge_threshold
                ),
                review_floor=(
                    review_floor
                    if review_floor is not None
                    else base.review_floor
                ),
                n_blocks_bump_threshold=base.n_blocks_bump_threshold,
                n_blocks_bump_max_bits=base.n_blocks_bump_max_bits,
            )

    # ── Subclass hooks ────────────────────────────────────────────────────────
    def prepare_model_input(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Compute derived columns + add candidate-pairs shim columns.

        Steps (mirrors `fellegi_sunter_baseline.prepare_model_input`):
          1. Re-derive transient blocking columns via `_compute_derived_columns`.
          2. Build `Phones_array` (sorted list) from the `_phones_parsed` set.
          3. Add inert PATID_A / PATID_B shim columns so the blocking-rule
             binding succeeds under `retain_matching_columns=True`.
        """
        df = _compute_derived_columns(df_clean)

        # _phones_parsed (set) → Phones_array (sorted list); empty set → [].
        df[COL_PHONES_ARRAY] = df[_COL_PHONES_SET].apply(
            lambda s: sorted(s) if isinstance(s, set) and s else []
        )

        for shim in _PAIR_SHIM_COLUMNS:
            if shim not in df.columns:
                df[shim] = pd.NA

        _validate_required_columns(df)
        logger.info(
            "FSBaseline.prepare_model_input: %d records, %d columns",
            len(df), df.shape[1],
        )
        return df

    def build_settings(self) -> dict:
        """Assemble the full Splink settings dict from the registry + boilerplate."""
        settings = SettingsCreator(
            link_type="dedupe_only",
            unique_id_column_name=COL_PATID,
            blocking_rules_to_generate_predictions=[CANDIDATE_PAIRS_BLOCKING_RULE],
            comparisons=[],
            retain_intermediate_calculation_columns=True,
            retain_matching_columns=True,
        ).get_settings("duckdb").as_dict()
        settings["comparisons"] = self.registry.build_all()
        return settings

    def predict(
        self,
        linker,
        candidate_pairs_df=None,
    ) -> pd.DataFrame:
        """Score pairs and drop the inert PATID_A / PATID_B shim passthrough
        columns (PATID_A_l/r, PATID_B_l/r) that Splink emits because
        retain_matching_columns=True binds on the shim column names from
        the candidate-pairs blocking rule."""
        df = super().predict(linker, candidate_pairs_df)
        junk = [
            f"{c}{suf}"
            for c in _PAIR_SHIM_COLUMNS
            for suf in ("_l", "_r")
        ]
        return df.drop(columns=[c for c in junk if c in df.columns])


__all__ = ["FSBaseline", "MODEL_NAME"]
