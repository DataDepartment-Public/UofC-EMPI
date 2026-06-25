"""FSEnhanced3 — the fourth Fellegi-Sunter experiment.

A deliberately simple, highly interpretable model composed over the shared OO
base in `models/common/fs_base.py`:

    FSEnhanced3 = FSModel(
        registry            = ENHANCED_3_REGISTRY (7 two-level comparisons),
        training            = SupervisedTraining(
                                  labels_df   = silver-label TRAIN split,
                                  prior_rules = PRIOR_RULES (lambda seed),
                              ),
        classification_config = ClassificationConfig(auto_merge=0.95, review_floor=0.40),
    )

Design differences from `fs_splink_enhanced_2`:

- **Two-level comparisons only.** Each of the seven fields (FirstNM, LastNM,
  BirthDT, SSN, Email, Phones, Address) is a single Exact / All-other
  distinction (plus a standard null no-evidence level). No JW bands, no ±1-day
  DOB, no SSN last-4, no household anti-evidence. The point is a match-weight
  table a reviewer can read off one Bayes factor per field.
- **m supervised from the real-cohort silver labels** (not synthetic), via
  `estimate_m_from_pairwise_labels` on the TRAIN split.
- **lambda seeded from the deterministic rules** via
  `estimate_probability_two_random_records_match` (PRIOR_RULES). u is still
  random-sampled on the cohort.

Training sequence (owned by SupervisedTraining):
    a. estimate_probability_two_random_records_match(PRIOR_RULES, recall)  → lambda
    b. estimate_m_from_pairwise_labels(train split)                        → m
    c. estimate_u_using_random_sampling                                    → u

Public API mirrors the other FS runners:

    run_fs_enhanced_3(candidate_pairs_path, df_clean, labels_df, ...)
        -> 5-col evaluation frame (or rich classified frame if full_output=True;
           or a (frame, linker) tuple when return_linker=True)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from splink import SettingsCreator

from models.common.fs_base import (
    ClassificationConfig,
    ComparisonRegistry,
    FSModel,
    SupervisedTraining,
)
from models.experiments.fs_splink_enhanced_3.comparisons import (
    COL_PATID,
    COL_PHONES_ARRAY,
    DEFAULT_PRIOR_RECALL,
    PRIOR_RULES,
    build_registry,
)

# Reuse blocking's derivation + column constants (single source of truth for the
# shim columns the candidate-pairs blocking rule needs).
from src.preprocessing.blocking import (
    _COL_PHONES_SET,
    _compute_derived_columns,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "fs_splink_enhanced_3"

CANDIDATE_PAIRS_TABLE = "candidate_pairs"
CP_PATID_A = "PATID_A"
CP_PATID_B = "PATID_B"

# Single-equi candidate-pairs blocking rule (DuckDB decorrelates to a HASH_JOIN
# keyed on (PATID_A, PATID_B)) — same pattern as enhanced / enhanced_2.
CANDIDATE_PAIRS_BLOCKING_RULE = f"""
EXISTS (
    SELECT 1 FROM {CANDIDATE_PAIRS_TABLE} cp
    WHERE cp.{CP_PATID_A} = l.{COL_PATID} AND cp.{CP_PATID_B} = r.{COL_PATID}
)
"""

# Inert placeholder columns so Splink's blocking-rule column harvesting binds.
_PAIR_SHIM_COLUMNS = (CP_PATID_A, CP_PATID_B)


class FSEnhanced3(FSModel):
    model_name = MODEL_NAME
    candidate_pairs_table_name = CANDIDATE_PAIRS_TABLE
    unique_id_column = COL_PATID

    def __init__(
        self,
        labels_df: pd.DataFrame,
        label_col: str = "label",
        include_address: bool = True,
        classification_config: ClassificationConfig | None = None,
        u_max_pairs: float = 1e6,
        seed: int = 42,
        labels_records_df: pd.DataFrame | None = None,
        prior_rules: list[str] | None = None,
        prior_recall: float = DEFAULT_PRIOR_RECALL,
    ):
        """
        Parameters
        ----------
        labels_df : pd.DataFrame
            The silver-label TRAIN split: PATID_A / PATID_B + a binary
            `label_col`. Only positives (label==1) are used for m-training.
        prior_rules : optional
            Deterministic match rules (SQL) seeding lambda. Defaults to the
            module's PRIOR_RULES; pass an explicit list (or `[]`) to override.
        prior_recall : float
            Assumed recall of `prior_rules` over all true matches.
        labels_records_df : optional
            Records frame whose PATIDs the labels reference, for split-training.
            Silver labels reference real-cohort PATIDs present in `df_clean`, so
            this is normally None (single-linker training).
        """
        self.include_address = include_address
        self.registry: ComparisonRegistry = build_registry(
            include_address=include_address
        )
        self.classification_config = classification_config or ClassificationConfig()
        self.training = SupervisedTraining(
            labels_df=labels_df,
            label_col=label_col,
            u_max_pairs=u_max_pairs,
            seed=seed,
            unique_id_column=COL_PATID,
            labels_records_df=labels_records_df,
            prior_rules=PRIOR_RULES if prior_rules is None else prior_rules,
            prior_recall=prior_recall,
        )

    # ── Subclass hooks ────────────────────────────────────────────────────────
    def prepare_model_input(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Compute the derived columns Splink reads (`_dob_str`, `Phones_array`,
        …) and add the candidate-pairs shim columns. Reuses blocking's
        `_compute_derived_columns` so derivations stay single-sourced."""
        df = _compute_derived_columns(df_clean)
        df[COL_PHONES_ARRAY] = df[_COL_PHONES_SET].apply(
            lambda s: sorted(s) if isinstance(s, set) and s else []
        )
        for shim in _PAIR_SHIM_COLUMNS:
            if shim not in df.columns:
                df[shim] = pd.NA
        return df

    def build_settings(self) -> dict:
        settings = (
            SettingsCreator(
                link_type="dedupe_only",
                unique_id_column_name=COL_PATID,
                blocking_rules_to_generate_predictions=[CANDIDATE_PAIRS_BLOCKING_RULE],
                comparisons=[],  # placeholder; replaced by the registry below
                retain_intermediate_calculation_columns=True,
                retain_matching_columns=True,
            )
            .get_settings("duckdb")
            .as_dict()
        )
        settings["comparisons"] = self.registry.build_all()
        return settings


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration shim (mirrors run_fs_enhanced_2 signature)
# ═══════════════════════════════════════════════════════════════════════════════
def run_fs_enhanced_3(
    candidate_pairs_path: str | Path,
    df_clean: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_col: str = "label",
    include_address: bool = True,
    full_output: bool = False,
    u_max_pairs: float = 1e6,
    classification_config: ClassificationConfig | None = None,
    return_linker: bool = False,
):
    """End-to-end entry point: candidate pairs + cleaned index + train labels → scored pairs.

    Returns the 5-col evaluation frame by default; the rich classified frame when
    `full_output=True`; and a `(frame, linker)` tuple when `return_linker=True`
    (for the validation notebook's Splink charts).
    """
    candidate_pairs = pd.read_parquet(candidate_pairs_path)
    model = FSEnhanced3(
        labels_df=labels_df,
        label_col=label_col,
        include_address=include_address,
        classification_config=classification_config,
        u_max_pairs=u_max_pairs,
    )
    return model.run(
        candidate_pairs, df_clean, full_output=full_output, return_linker=return_linker
    )


__all__ = [
    "FSEnhanced3",
    "MODEL_NAME",
    "run_fs_enhanced_3",
]
