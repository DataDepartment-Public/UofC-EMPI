"""FSEnhanced — FSModel subclass for the enhanced Fellegi-Sunter experiment.

This module re-implements the behavior of ``fellegi_sunter_enhanced.py`` using
the shared OO scaffold from ``models/common/fs_base.py``.  The comparison
vector, EM training sessions, classification thresholds, veto layer, and
manual-prior locking are all preserved.  Only the delivery mechanism changes.

Key differences from ``fs_splink_enhanced_2``:

- **EMTraining**, not SupervisedTraining — 3 EM sessions + prevalence prior.
- **Veto layer** — ``apply_vetoes`` from ``deterministic_vetoes.py`` fires via
  ``FSEnhanced.classify()`` override.  ``df_clean`` is stashed on ``self`` in
  ``prepare_model_input()`` so ``classify()`` can stay single-argument (matching
  the ``FSModel.classify`` signature).
- **Manual priors** — ``apply_manual_priors`` from ``manual_priors.py`` is
  called inside ``FSEnhanced.build_settings()`` to lock Address + Phones m/u
  in the settings dict before the Splink Linker is constructed (identical
  wiring to the old ``fellegi_sunter_enhanced.build_settings``).

PHI / HIPAA: this module logs aggregate counts only.  No PATIDs, names, SSNs,
DOBs, or addresses appear in any log line.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from splink import SettingsCreator

from models.common.fs_base import ClassificationConfig, EMTraining, FSModel
from models.experiments.fs_splink_enhanced.comparisons import (
    COL_PATID,
    COL_PHONES_ARRAY,
    EM_BLOCKING_RULES,
    PRIOR_RULES,
    build_registry,
)
from models.experiments.fs_splink_enhanced.deterministic_vetoes import (
    VETO_REASON_COL,
    apply_vetoes,
)
from models.experiments.fs_splink_enhanced.manual_priors import apply_manual_priors
from src.preprocessing.blocking import (
    _COL_PHONES_SET,
    _compute_derived_columns,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "fs_splink_enhanced"

CANDIDATE_PAIRS_TABLE = "candidate_pairs"
CP_PATID_A = "PATID_A"
CP_PATID_B = "PATID_B"

# Single-equi candidate-pairs blocking rule (DuckDB decorrelates to HASH_JOIN).
CANDIDATE_PAIRS_BLOCKING_RULE = f"""
EXISTS (
    SELECT 1 FROM {CANDIDATE_PAIRS_TABLE} cp
    WHERE cp.{CP_PATID_A} = l.{COL_PATID} AND cp.{CP_PATID_B} = r.{COL_PATID}
)
"""

# Inert placeholder columns added to df_model so Splink's blocking-rule column
# harvesting binds under retain_matching_columns=True.
_PAIR_SHIM_COLUMNS = (CP_PATID_A, CP_PATID_B)

# Required columns that must be present after _compute_derived_columns().
REQUIRED_MODEL_COLUMNS = [COL_PATID, "_dob_str", "last_4_SSN", "ZipCD_clean_base"]

# ProbabilisticMatches column order (includes veto_reason, unlike FSModel base).
PROBABILISTIC_MATCHES_COLUMNS = (
    "PATID_A", "PATID_B", "match_source", "score", "match_weight",
    "classification_tier", "veto_reason", "source_blocks", "n_blocks",
)


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


class FSEnhanced(FSModel):
    """Enhanced Fellegi-Sunter model as a thin FSModel subclass.

    9 comparisons (include_address=True): FirstNM, LastNM, BirthDT, SSN, Email,
    Phones, ZIP, Household_discount, Address.  With include_address=False: 7
    (first seven only).

    E4 extra levels baked into comparisons.py builders: JW<0.5 mismatch levels
    on FirstNM and LastNM, full-9 SSN mismatch level on SSN.

    Veto layer: after scoring, ``apply_vetoes(df_predictions, df_clean)`` is
    called; vetoed pairs are forced to ``classification_tier = "no_match"``
    regardless of their probabilistic score.

    Manual priors: ``apply_manual_priors(settings_dict)`` is applied inside
    ``build_settings()`` to lock Address + Phones m/u before Linker construction
    so EM cannot overwrite them.

    Classification thresholds: auto_merge=0.95, review_floor=0.40 (E5 values).
    """

    model_name = MODEL_NAME
    classification_config = ClassificationConfig(
        auto_merge_threshold=0.95,
        review_floor=0.40,
        n_blocks_bump_threshold=2,
        n_blocks_bump_max_bits=4.0,
    )
    candidate_pairs_table_name = CANDIDATE_PAIRS_TABLE
    unique_id_column = COL_PATID

    # _df_clean is set in prepare_model_input() so classify() can access it
    # without changing the FSModel.classify() single-argument signature.
    _df_clean: pd.DataFrame | None

    def __init__(
        self,
        include_address: bool = True,
        u_max_pairs: float = 1e6,
        seed: int = 42,
        auto_merge_threshold: float | None = None,
        review_floor: float | None = None,
    ):
        """
        Parameters
        ----------
        include_address : bool, default True
            Include the Household_discount and Address comparisons.  Set False
            for the synthetic sandbox fixture that predates address columns.
        u_max_pairs : float, default 1e6
            Splink u-sampling budget.  On a single-CPU host use 1e4.
        seed : int, default 42
            Reproducible u-sampling seed.
        auto_merge_threshold, review_floor : float | None
            Override the E5 class-level defaults.  Both or neither must be
            overridden to keep the constraint review_floor <= auto_merge_threshold.
        """
        self.include_address = include_address
        self.registry = build_registry(include_address=include_address)
        self.training = EMTraining(
            em_blocking_rules=EM_BLOCKING_RULES,
            prior_rules=PRIOR_RULES,
            recall=0.80,
            u_max_pairs=u_max_pairs,
            seed=seed,
        )
        self._df_clean = None

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

        Also stashes ``df_clean`` on ``self._df_clean`` so that the veto layer
        in ``classify()`` can look up identifier attributes without changing the
        ``FSModel.classify(df_predictions)`` single-argument signature.

        Steps (mirrors ``fellegi_sunter_enhanced.prepare_model_input``):
          1. Re-derive transient blocking columns via ``_compute_derived_columns``.
          2. Build ``Phones_array`` (sorted list) from the ``_phones_parsed`` set.
          3. Add inert PATID_A / PATID_B shim columns.
          4. Validate that required columns are present.
        """
        # Stash the original cleaned frame (not the derived copy) so apply_vetoes
        # can reindex on the original PATID values.
        self._df_clean = df_clean

        df = _compute_derived_columns(df_clean)
        df[COL_PHONES_ARRAY] = df[_COL_PHONES_SET].apply(
            lambda s: sorted(s) if isinstance(s, set) and s else []
        )
        for shim in _PAIR_SHIM_COLUMNS:
            if shim not in df.columns:
                df[shim] = pd.NA

        _validate_required_columns(df)
        logger.info(
            "FSEnhanced.prepare_model_input: %d records, %d columns",
            len(df), df.shape[1],
        )
        return df

    def build_settings(self) -> dict:
        """Assemble the full Splink settings dict.

        Builds the dict from the registry + boilerplate, then applies
        ``apply_manual_priors`` to lock Address + Phones m/u before the
        Linker is constructed.  This mirrors the existing
        ``fellegi_sunter_enhanced.build_settings`` behaviour exactly.
        """
        settings = SettingsCreator(
            link_type="dedupe_only",
            unique_id_column_name=COL_PATID,
            blocking_rules_to_generate_predictions=[CANDIDATE_PAIRS_BLOCKING_RULE],
            comparisons=[],
            retain_intermediate_calculation_columns=True,
            retain_matching_columns=True,
        ).get_settings("duckdb").as_dict()
        settings["comparisons"] = self.registry.build_all()
        # Lock candidate-pool-aware m/u on Address + Phones (and E4 mismatch
        # levels) so EM cannot overwrite them with random-pair-biased estimates.
        apply_manual_priors(settings)
        return settings

    def predict(self, linker, candidate_pairs_df=None) -> pd.DataFrame:
        """Score pairs and drop the inert PATID_A / PATID_B shim passthrough
        columns (PATID_A_l/r, PATID_B_l/r) that Splink emits because
        retain_matching_columns=True harvests the shim column names from the
        candidate-pairs blocking rule."""
        df = super().predict(linker, candidate_pairs_df)
        junk = [
            f"{c}{suf}"
            for c in _PAIR_SHIM_COLUMNS
            for suf in ("_l", "_r")
        ]
        return df.drop(columns=[c for c in junk if c in df.columns])

    def classify(self, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """n_blocks bump + threshold classification + deterministic veto override.

        Extends ``FSModel.classify`` with the veto layer:
          1. Apply n_blocks bump in log-odds space (base class).
          2. Assign tiers by threshold (base class).
          3. Call ``apply_vetoes(df, self._df_clean)`` to annotate each pair
             with the first-firing deterministic rule (or None).
          4. Force ``classification_tier = "no_match"`` for any vetoed pair.

        ``self._df_clean`` is set by ``prepare_model_input()`` which always runs
        before ``classify()`` in the ``FSModel.run()`` orchestration.
        """
        out = super().classify(df_predictions)

        if self._df_clean is not None:
            out = apply_vetoes(out, self._df_clean)
            vetoed = out[VETO_REASON_COL].notna()
            if vetoed.any():
                out = out.copy()
                out.loc[vetoed, "classification_tier"] = "no_match"
                logger.info(
                    "FSEnhanced.classify: %d pair(s) forced to no_match by veto",
                    int(vetoed.sum()),
                )
        else:
            logger.warning(
                "FSEnhanced.classify: _df_clean is None — veto layer skipped. "
                "Call prepare_model_input() before classify() or use run()."
            )

        return out

    def run(
        self,
        candidate_pairs_df: pd.DataFrame,
        df_clean: pd.DataFrame,
        full_output: bool = False,
    ) -> pd.DataFrame:
        """Delegate to FSModel.run(), then clear the _df_clean stash.

        The stash is needed only during the classify() call inside run().
        Clearing it in a finally block prevents stale state from leaking if
        a caller later invokes prepare_model_input() with one frame and
        run() with a different one.
        """
        try:
            return super().run(candidate_pairs_df, df_clean, full_output=full_output)
        finally:
            self._df_clean = None

    def to_probabilistic_matches(self, df_classified: pd.DataFrame) -> pd.DataFrame:
        """Project to ProbabilisticMatches contract including veto_reason.

        The base class omits veto_reason (it is Optional in the Pandera contract).
        This override adds it back so the Stage 4 on-disk artifact is complete.
        """
        required = {
            "PATID_A", "PATID_B", "match_probability", "match_weight",
            "classification_tier",
        }
        missing = required - set(df_classified.columns)
        if missing:
            raise ValueError(
                f"to_probabilistic_matches is missing columns {sorted(missing)}; "
                "expected the output of classify() with full_output=True."
            )
        n = len(df_classified)
        out = pd.DataFrame({
            "PATID_A": df_classified["PATID_A"].values,
            "PATID_B": df_classified["PATID_B"].values,
            "match_source": ["model"] * n,
            "score": df_classified["match_probability"].values,
            "match_weight": df_classified["match_weight"].values,
            "classification_tier": df_classified["classification_tier"].values,
            "veto_reason": (
                df_classified[VETO_REASON_COL].values
                if VETO_REASON_COL in df_classified.columns
                else [None] * n
            ),
            "source_blocks": (
                df_classified["source_blocks"].values
                if "source_blocks" in df_classified.columns
                else [None] * n          # str column → None is correct
            ),
            "n_blocks": (
                df_classified["n_blocks"].values
                if "n_blocks" in df_classified.columns
                else [np.nan] * n        # int/float column → NaN, not None
            ),
        })
        return out[list(PROBABILISTIC_MATCHES_COLUMNS)]


__all__ = ["FSEnhanced", "MODEL_NAME", "PROBABILISTIC_MATCHES_COLUMNS"]
