"""FSMatcher — the production Fellegi-Sunter matcher (Stage 4).

`FSMatcher(FSModel)` lifts the enhanced_3 model (7 two-level comparisons,
supervised-m, λ seeded from the deterministic rules) and adds the **serving
path** the research base lacks: load a pre-trained Splink model artifact and
score without retraining.

Two lifecycles:

* **Training** (offline, `fs-train` CLI) — construct with `labels_df`, call
  `run(..., full_output=True, return_linker=True)`; persist the trained linker's
  settings via `linker.misc.save_model_to_json()`.
* **Serving** (the pipeline) — construct with no labels, `load_settings(path)`,
  then `score(candidate_pairs, df_clean, settings)` → a classified frame. From
  that, `to_probabilistic_matches()` (full audit) and `to_fs_features()` (the
  candidate-filtered GBT parquet) are projected.

The FS output does **not** feed clustering — it is a candidate + feature
generator for the downstream GBT.

PHI / HIPAA: logs aggregate counts only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.models.fs_matcher.base import (
    ClassificationConfig,
    ComparisonRegistry,
    FSModel,
    SupervisedTraining,
    _register_candidate_pairs,
)
from src.models.fs_matcher.comparisons import (
    COL_PATID,
    COL_PHONES_ARRAY,
    DEFAULT_PRIOR_RECALL,
    PRIOR_RULES,
    build_registry,
)

# Reuse blocking's derivation + column constants (single source of truth for the
# derived columns Splink reads: `_dob_str`, `_phones_parsed`, …).
from src.preprocessing.blocking import (
    _COL_PHONES_SET,
    _compute_derived_columns,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "fs_matcher"

CANDIDATE_PAIRS_TABLE = "candidate_pairs"
CP_PATID_A = "PATID_A"
CP_PATID_B = "PATID_B"

# Single-equi candidate-pairs blocking rule (DuckDB decorrelates to a HASH_JOIN
# keyed on (PATID_A, PATID_B)) — Splink scores only the pre-blocked pairs.
CANDIDATE_PAIRS_BLOCKING_RULE = f"""
EXISTS (
    SELECT 1 FROM {CANDIDATE_PAIRS_TABLE} cp
    WHERE cp.{CP_PATID_A} = l.{COL_PATID} AND cp.{CP_PATID_B} = r.{COL_PATID}
)
"""

# Inert placeholder columns so Splink's blocking-rule column harvesting binds.
_PAIR_SHIM_COLUMNS = (CP_PATID_A, CP_PATID_B)


def classification_config_from_settings(settings) -> ClassificationConfig:
    """Build a `ClassificationConfig` from the shared `Settings` cutoffs."""
    return ClassificationConfig(
        auto_merge_threshold=settings.fs_auto_merge_threshold,
        review_floor=settings.fs_review_floor,
    )


class FSMatcher(FSModel):
    model_name = MODEL_NAME
    candidate_pairs_table_name = CANDIDATE_PAIRS_TABLE
    unique_id_column = COL_PATID

    def __init__(
        self,
        labels_df: pd.DataFrame | None = None,
        label_col: str = "silver_label",
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
        labels_df : optional
            Silver-label TRAIN split (PATID_A / PATID_B + a binary `label_col`).
            Required for TRAINING; pass None for SERVING (loading a pre-trained
            model), where `self.training` stays None and is never invoked.
        classification_config : optional
            Thresholds. Defaults to `ClassificationConfig()` (0.95 / 0.40); the
            pipeline/CLI pass `classification_config_from_settings(settings)`.
        """
        self.include_address = include_address
        self.registry: ComparisonRegistry = build_registry(include_address=include_address)
        self.classification_config = classification_config or ClassificationConfig()
        if labels_df is not None:
            self.training: SupervisedTraining | None = SupervisedTraining(
                labels_df=labels_df,
                label_col=label_col,
                u_max_pairs=u_max_pairs,
                seed=seed,
                unique_id_column=COL_PATID,
                labels_records_df=labels_records_df,
                prior_rules=PRIOR_RULES if prior_rules is None else prior_rules,
                prior_recall=prior_recall,
            )
        else:
            self.training = None  # serving only — score() never trains

    # ── FSModel hooks ─────────────────────────────────────────────────────────
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
        from splink import SettingsCreator
        settings = (
            SettingsCreator(
                link_type="dedupe_only",
                unique_id_column_name=COL_PATID,
                blocking_rules_to_generate_predictions=[CANDIDATE_PAIRS_BLOCKING_RULE],
                comparisons=[],  # placeholder; replaced by the registry below
                retain_intermediate_calculation_columns=True,  # -> gamma_/bf_ cols
                retain_matching_columns=True,
            )
            .get_settings("duckdb")
            .as_dict()
        )
        settings["comparisons"] = self.registry.build_all()
        return settings

    # ── Serving path (no training) ────────────────────────────────────────────
    @staticmethod
    def load_settings(path: str | Path) -> dict:
        """Load a trained Splink model JSON (written by `save_model_to_json`)."""
        with open(path) as fh:
            return json.load(fh)

    def score(
        self,
        candidate_pairs_df: pd.DataFrame,
        df_clean: pd.DataFrame,
        trained_settings: dict,
    ) -> pd.DataFrame:
        """Score `candidate_pairs_df` through an already-trained model.

        Builds a DuckDB linker directly from `trained_settings` (m/u/λ already
        estimated) over the prepared records and the candidate pairs, then
        `predict()` + `classify()` — **no training**. Returns the rich classified
        frame (with `gamma_/bf_` feature columns and `classification_tier`).
        """
        from splink import DuckDBAPI, Linker

        df_model = self.prepare_model_input(df_clean)
        db_api = DuckDBAPI()
        _register_candidate_pairs(db_api, candidate_pairs_df, self.candidate_pairs_table_name)
        linker = Linker(df_model, trained_settings, db_api=db_api)
        predictions = self.predict(linker, candidate_pairs_df)
        classified = self.classify(predictions)
        logger.info("FSMatcher: scored %d candidate pairs (serving)", len(classified))
        return classified

    def score_with_model_path(
        self,
        candidate_pairs_df: pd.DataFrame,
        df_clean: pd.DataFrame,
        model_path: str | Path,
    ) -> pd.DataFrame:
        """Convenience: `load_settings(model_path)` then `score(...)`."""
        return self.score(candidate_pairs_df, df_clean, self.load_settings(model_path))

    # ── GBT feature projection ────────────────────────────────────────────────
    def to_fs_features(
        self,
        df_classified: pd.DataFrame,
        labels_df: pd.DataFrame | None = None,
        label_col: str = "label",
        candidates_only: bool = True,
    ) -> pd.DataFrame:
        """Project the classified frame to the GBT feature parquet.

        Emits the base columns (`PATID_A/B`, `match_probability`, `match_weight`,
        `classification_tier`) plus every per-field `gamma_<field>` / `bf_<field>`
        feature column Splink produced. When `labels_df` is given, joins a `label`
        column (0/1) on the pair key. When `candidates_only` (the pipeline
        default), keeps only rows at/above the review floor — the "candidates" the
        FS module surfaces for the GBT.
        """
        required = {"PATID_A", "PATID_B", "match_probability", "match_weight",
                    "classification_tier"}
        missing = required - set(df_classified.columns)
        if missing:
            raise ValueError(
                f"to_fs_features is missing columns {sorted(missing)}; expected "
                "the output of score()/classify()."
            )
        feature_cols = [
            c for c in df_classified.columns
            if c.startswith("gamma_") or c.startswith("bf_")
        ]
        cols = ["PATID_A", "PATID_B", "match_probability", "match_weight",
                "classification_tier", *feature_cols]
        out = df_classified[cols].copy().reset_index(drop=True)

        if labels_df is not None:
            lbl = labels_df.rename(columns={label_col: "label"})[
                ["PATID_A", "PATID_B", "label"]
            ].copy()
            lbl["label"] = lbl["label"].astype(float)
            out = out.merge(lbl, on=["PATID_A", "PATID_B"], how="left")

        if candidates_only:
            floor = self.classification_config.review_floor
            before = len(out)
            out = out[out["match_probability"] >= floor].reset_index(drop=True)
            logger.info(
                "to_fs_features: kept %d/%d candidate pairs (match_probability >= %.2f)",
                len(out), before, floor,
            )
        return out


__all__ = [
    "FSMatcher",
    "MODEL_NAME",
    "classification_config_from_settings",
]
