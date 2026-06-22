"""FSEnhanced2 — the third Fellegi-Sunter experiment.

Composition over the shared OO base in `models/common/fs_base.py`:

    FSEnhanced2 = FSModel(
        registry            = ENHANCED_2_REGISTRY (or build_registry(include_address=False)),
        training            = SupervisedTraining(labels_df=synthetic_train_v3),
        classification_config = ClassificationConfig(auto_merge=0.95, review_floor=0.40),
    )

Design differences from `fs_splink_enhanced`:

- **No vetoes.** Veto logic was removed from the FS scoring stage (a separate
  workstream will reimplement it in the deterministic-rules layer). The base
  `to_probabilistic_matches` projection therefore omits `veto_reason`.
- **Supervised m, not EM.** m-probabilities come from
  `data/synthetic/synthetic_train_v3.csv` (40k labeled pairs) via
  `linker.training.estimate_m_from_pairwise_labels`. u is still random-sampled
  on the real cohort.
- **No manual priors.** The locked `ADDRESS_MU` / `PHONES_MU` overrides from
  the enhanced module are gone — supervised m makes them redundant.
- **Four new comparisons + new levels** declared in `comparisons.py`:
  MiddleNM, Sex_positive, LastNM_Phonetic, FirstNM_Phonetic, plus BirthDT
  month-day-swap and LastNM full_name_compact levels.

Public API mirrors fs_splink_enhanced for runner compatibility:

    run_fs_enhanced_2(candidate_pairs_path, df_clean, labels_df, ...)
        -> 5-col evaluation frame (or rich classified frame if full_output=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from splink import SettingsCreator

from models.common.fs_base import (
    ClassificationConfig,
    ComparisonRegistry,
    FSModel,
    SupervisedTraining,
)
from models.experiments.fs_splink_enhanced_2.comparisons import (
    COL_PATID,
    COL_PHONES_ARRAY,
    build_registry,
)

# Reuse blocking's derivation + column constants (single source of truth for
# the shim columns the candidate-pairs blocking rule needs).
from src.features.blocking import (
    _COL_DOB_STR,
    _COL_PHONES_SET,
    _compute_derived_columns,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "fs_splink_enhanced_2"

CANDIDATE_PAIRS_TABLE = "candidate_pairs"
CP_PATID_A = "PATID_A"
CP_PATID_B = "PATID_B"

# Same single-equi candidate-pairs blocking-rule pattern used by fs_splink_enhanced
# (DuckDB decorrelates this into a HASH_JOIN keyed on (PATID_A, PATID_B)).
CANDIDATE_PAIRS_BLOCKING_RULE = f"""
EXISTS (
    SELECT 1 FROM {CANDIDATE_PAIRS_TABLE} cp
    WHERE cp.{CP_PATID_A} = l.{COL_PATID} AND cp.{CP_PATID_B} = r.{COL_PATID}
)
"""

# Inert placeholder columns added to df_model so Splink's blocking-rule column
# harvesting binds (see enhanced module's _PAIR_SHIM_COLUMNS rationale).
_PAIR_SHIM_COLUMNS = (CP_PATID_A, CP_PATID_B)


class FSEnhanced2(FSModel):
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
    ):
        """
        Parameters
        ----------
        labels_records_df : optional
            Records frame whose PATIDs the labels (PATID_A / PATID_B) reference.
            When supplied, m-training runs on a separate "auxiliary" linker
            bound to this frame and the trained m values are transferred to the
            production linker for scoring. Required whenever the production
            records frame (the `df_clean` passed to `run()`) does NOT contain
            the labels' PATIDs — e.g., training on synthetic labels and scoring
            on the real cohort. Pass
            `pd.read_csv("data/synthetic/synthetic_blocking_testing.csv", ...)`
            for the standard synthetic-train-v3 label set. See SupervisedTraining
            docstring for the split-training rationale.
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
        )

    # ── Subclass hooks ────────────────────────────────────────────────────────
    def prepare_model_input(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Compute the derived columns Splink reads and add the candidate-pairs
        shim columns. `_dm_LastNM` / `_dm_FirstNM` are already on the cleaned
        frame (Phase E2-2); blocking's helper reuses them when present."""
        df = _compute_derived_columns(df_clean)
        # Phones_array: list form for cl.ArrayIntersectAtSizes.
        df[COL_PHONES_ARRAY] = df[_COL_PHONES_SET].apply(
            lambda s: sorted(s) if isinstance(s, set) and s else []
        )
        for shim in _PAIR_SHIM_COLUMNS:
            if shim not in df.columns:
                df[shim] = pd.NA
        return df

    def build_settings(self) -> dict:
        settings = SettingsCreator(
            link_type="dedupe_only",
            unique_id_column_name=COL_PATID,
            blocking_rules_to_generate_predictions=[CANDIDATE_PAIRS_BLOCKING_RULE],
            # Placeholder comparison list — replaced below with the registry's
            # dict-form output. SettingsCreator needs *something* in the list to
            # construct, but we want the registry to be authoritative.
            comparisons=[],
            retain_intermediate_calculation_columns=True,
            retain_matching_columns=True,
        ).get_settings("duckdb").as_dict()
        settings["comparisons"] = self.registry.build_all()
        return settings


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration shim (mirrors fs_splink_enhanced.run_fs_enhanced signature)
# ═══════════════════════════════════════════════════════════════════════════════
def run_fs_enhanced_2(
    candidate_pairs_path: str | Path,
    df_clean: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_col: str = "label",
    include_address: bool = True,
    full_output: bool = False,
    u_max_pairs: float = 1e6,
    classification_config: ClassificationConfig | None = None,
) -> pd.DataFrame:
    """End-to-end entry point: candidate pairs + cleaned index + labels → scored pairs.

    By default returns the standardized five-column evaluation frame
    (PATID_A, PATID_B, model_name, score, predicted_tier). Set `full_output=True`
    to receive the rich classified frame for `to_probabilistic_matches` projection.
    """
    candidate_pairs = pd.read_parquet(candidate_pairs_path)
    model = FSEnhanced2(
        labels_df=labels_df,
        label_col=label_col,
        include_address=include_address,
        classification_config=classification_config,
        u_max_pairs=u_max_pairs,
    )
    return model.run(candidate_pairs, df_clean, full_output=full_output)


__all__ = [
    "FSEnhanced2",
    "MODEL_NAME",
    "run_fs_enhanced_2",
]
