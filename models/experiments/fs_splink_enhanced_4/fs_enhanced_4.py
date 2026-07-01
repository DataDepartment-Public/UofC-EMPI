"""FSEnhanced4 — the fifth Fellegi-Sunter experiment.

Enhanced_4 keeps enhanced_3's methodology (supervised m from the real-cohort
silver labels, lambda seeded from the deterministic rules, u from random
sampling) and adds three structural precision levers, composed over the shared
OO base in `models/common/fs_base.py`:

    FSEnhanced4 = FSModel(
        registry            = ENHANCED_4_REGISTRY (multi-level comparisons with
                              explicit conflict-vs-missing mismatch levels),
        training            = SupervisedTraining(
                                  labels_df   = silver-label TRAIN split,
                                  prior_rules = PRIOR_RULES (lambda seed),
                              ),
        classification_config = ClassificationConfig(auto_merge=0.95, review_floor=0.40),
    )

Design differences from `fs_splink_enhanced_3`:

- **Multi-level comparisons.** Each identity field carries explicit hard-conflict
  mismatch levels (SSN 9-digit mismatch, JW<0.5 name conflict, …) so a
  contradicting identifier contributes real negative weight, instead of falling
  into an uninformative "else". See `comparisons.py`.
- **Corroboration gate in `classify()`.** After tier assignment, an
  `auto_merge` pair is demoted to `human_review` unless it carries corroborating
  evidence beyond DOB + name (a person-unique SSN/Email agreement, or ≥2
  household signals). See `corroboration_gate.py`.
- **Grounded SSN / Email exact u.** The two exact levels whose random-sampling u
  is untrained in enhanced_3 (they never co-occur in random pairs) are grounded
  to `1 / n_distinct` before training and pinned via `fix_u_probability=True`,
  so their Bayes factor is not left at a Splink default.

m stays **purely supervised** from the real-cohort silver labels — no manual
m-locking (unlike `fs_splink_enhanced.manual_priors`); only u is grounded.

Public API mirrors the other FS runners:

    run_fs_enhanced_4(candidate_pairs_path, df_clean, labels_df, ...)
        -> 5-col evaluation frame (or rich classified frame if full_output=True;
           or a (frame, linker) tuple when return_linker=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from splink import SettingsCreator

from src import contracts
from models.common.fs_base import (
    ClassificationConfig,
    ComparisonRegistry,
    FSModel,
    SupervisedTraining,
)
from models.experiments.fs_splink_enhanced_4.comparisons import (
    COL_PATID,
    COL_PHONES_ARRAY,
    DEFAULT_PRIOR_RECALL,
    PRIOR_RULES,
    build_registry,
)
from models.experiments.fs_splink_enhanced_4.corroboration_gate import (
    CORROBORATION_COL,
    DEMOTED_REASON,
    apply_corroboration_gate,
)

# Reuse blocking's derivation + column constants (single source of truth for the
# shim columns the candidate-pairs blocking rule needs).
from src.preprocessing.blocking import (
    _COL_PHONES_SET,
    _compute_derived_columns,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "fs_splink_enhanced_4"

CANDIDATE_PAIRS_TABLE = "candidate_pairs"
CP_PATID_A = "PATID_A"
CP_PATID_B = "PATID_B"

# Single-equi candidate-pairs blocking rule (DuckDB decorrelates to a HASH_JOIN
# keyed on (PATID_A, PATID_B)) — same pattern as enhanced / enhanced_3.
CANDIDATE_PAIRS_BLOCKING_RULE = f"""
EXISTS (
    SELECT 1 FROM {CANDIDATE_PAIRS_TABLE} cp
    WHERE cp.{CP_PATID_A} = l.{COL_PATID} AND cp.{CP_PATID_B} = r.{COL_PATID}
)
"""

# Inert placeholder columns so Splink's blocking-rule column harvesting binds.
_PAIR_SHIM_COLUMNS = (CP_PATID_A, CP_PATID_B)

# Exact levels whose random-sampling u is untrained (they never co-occur in
# random pairs), so u is grounded to 1/n_distinct before training. Keyed by the
# comparison's output_column_name → (attribute column, fallback u).
_GROUNDED_U_TARGETS: dict[str, tuple[str, float]] = {
    "SSN": (contracts.SSN, 1e-6),
    "Email": (contracts.EMAIL, 5e-4),
}


class FSEnhanced4(FSModel):
    model_name = MODEL_NAME
    candidate_pairs_table_name = CANDIDATE_PAIRS_TABLE
    unique_id_column = COL_PATID

    # _df_clean is set in prepare_model_input() so build_settings() (u-grounding)
    # can access it without changing the FSModel single-argument hook signatures.
    # _gate_df_clean is the AUTHORITATIVE production frame for the corroboration
    # gate, set in run(): the split-training path re-invokes prepare_model_input
    # on the auxiliary labels_records_df (fs_base._build_m_training_linker), which
    # would otherwise clobber _df_clean and make the gate reindex production pairs
    # against the wrong frame (silently demoting every auto_merge pair).
    _df_clean: pd.DataFrame | None
    _gate_df_clean: pd.DataFrame | None

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
        self._df_clean = None
        self._gate_df_clean = None
        self._grounded_u: dict[str, float] = {}

    # ── Subclass hooks ────────────────────────────────────────────────────────
    def prepare_model_input(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Compute the derived columns Splink reads (`_dob_str`, `Phones_array`,
        …) and add the candidate-pairs shim columns. Reuses blocking's
        `_compute_derived_columns` so derivations stay single-sourced.

        Also stashes the ORIGINAL cleaned frame on ``self._df_clean`` so that
        (a) ``build_settings()`` can ground the SSN/Email exact u from the
        cohort's distinct-value counts and (b) the corroboration gate in
        ``classify()`` can reindex on the original PATID values.
        """
        # Stash the original cleaned frame (not the derived copy) so the
        # corroboration gate reindexes on the original PATIDs, mirroring
        # FSEnhanced.
        self._df_clean = df_clean

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
        self._ground_untrained_u(settings)
        return settings

    def _ground_untrained_u(self, settings: dict) -> None:
        """Ground the SSN / Email *exact* levels' u to 1/n_distinct (in place).

        These two exact levels never co-occur in random pairs, so
        `estimate_u_using_random_sampling` leaves their u untrained (Splink
        default). Grounding u to `1 / n_distinct(col)` — the probability two
        random records collide on that identifier — and pinning it via
        `fix_u_probability=True` (honoured by random-sampling u estimation, the
        same mechanism FSEnhanced.manual_priors relies on) gives each a
        well-founded Bayes factor. m stays supervised (no fix_m_probability).
        """
        self._grounded_u = {}
        comparisons = settings.get("comparisons", [])

        for out_name, (col, fallback_u) in _GROUNDED_U_TARGETS.items():
            comp = next(
                (
                    c
                    for c in comparisons
                    if c.get("output_column_name") == out_name
                ),
                None,
            )
            if comp is None:
                logger.warning(
                    "_ground_untrained_u: comparison %r not found (rename "
                    "guard) — skipping u-grounding for it.",
                    out_name,
                )
                continue

            target_label = f"Exact match on {col}"
            level = next(
                (
                    lvl
                    for lvl in comp.get("comparison_levels", [])
                    if lvl.get("label_for_charts") == target_label
                ),
                None,
            )
            if level is None:
                logger.warning(
                    "_ground_untrained_u: exact level %r not found on "
                    "comparison %r (rename guard) — skipping.",
                    target_label,
                    out_name,
                )
                continue

            u = fallback_u
            if self._df_clean is not None and col in self._df_clean.columns:
                n = int(self._df_clean[col].nunique(dropna=True))
                u = (1.0 / n) if n > 0 else fallback_u
            else:
                logger.warning(
                    "_ground_untrained_u: column %r unavailable on df_clean — "
                    "using fallback u for comparison %r.",
                    col,
                    out_name,
                )

            level["u_probability"] = u
            level["fix_u_probability"] = True
            self._grounded_u[out_name] = u
            logger.info(
                "_ground_untrained_u: grounded %s exact-level u = %.3g (pinned)",
                out_name,
                u,
            )

    def audit_untrained_u(self, linker: Any) -> list[dict]:
        """Report every NON-null comparison level left with no u after training.

        Reads the trained model JSON and returns a list of
        ``{"comparison": ..., "level": ...}`` for each non-null level whose
        `u_probability` is None/missing. Rare levels that never appear in random
        u-sampling silently keep a Splink default; this surfaces them (labels
        only, no PHI) so a human can decide whether to ground them too.
        """
        model = linker.misc.save_model_to_json()
        untrained: list[dict] = []
        for comp in model.get("comparisons", []):
            comp_name = comp.get("output_column_name")
            for lvl in comp.get("comparison_levels", []):
                if lvl.get("is_null_level", False):
                    continue
                if lvl.get("u_probability") is None:
                    untrained.append(
                        {
                            "comparison": comp_name,
                            "level": lvl.get("label_for_charts"),
                        }
                    )
        if untrained:
            logger.warning(
                "audit_untrained_u: %d non-null level(s) have no u after "
                "training: %s",
                len(untrained),
                untrained,
            )
        return untrained

    def classify(self, df_predictions: pd.DataFrame) -> pd.DataFrame:
        """n_blocks bump + threshold classification + corroboration gate.

        Extends ``FSModel.classify`` with the corroboration gate:
          1. n_blocks bump + threshold tiers (base class).
          2. ``apply_corroboration_gate(out, self._df_clean)`` demotes
             uncorroborated ``auto_merge`` pairs to ``human_review`` (never to
             ``no_match``, never promotes), annotating the ``corroboration``
             column.

        The gate frame is ``self._gate_df_clean`` (the authoritative production
        frame set by ``run()``); it falls back to ``self._df_clean`` (set by
        ``prepare_model_input()``) for direct ``classify()`` callers. Preferring
        ``_gate_df_clean`` keeps the gate correct under split-training, where the
        auxiliary ``prepare_model_input`` call clobbers ``_df_clean``.
        """
        out = super().classify(df_predictions)

        gate_frame = (
            self._gate_df_clean if self._gate_df_clean is not None else self._df_clean
        )
        if gate_frame is not None:
            out = apply_corroboration_gate(out, gate_frame)
            demoted = out[CORROBORATION_COL] == DEMOTED_REASON
            logger.info(
                "FSEnhanced4.classify: %d pair(s) demoted to human_review by "
                "the corroboration gate",
                int(demoted.sum()),
            )
        else:
            # Stable output schema even when the gate is skipped.
            out[CORROBORATION_COL] = None
            logger.warning(
                "FSEnhanced4.classify: no cleaned frame available — corroboration "
                "gate skipped. Call prepare_model_input() before classify() or "
                "use run()."
            )

        return out

    def run(
        self,
        candidate_pairs_df: pd.DataFrame,
        df_clean: pd.DataFrame,
        full_output: bool = False,
        return_linker: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, Any]:
        """Delegate to FSModel.run(), then clear the _df_clean stash.

        The stash is needed only during the build_settings()/classify() calls
        inside run(). Clearing it in a finally block prevents stale state from
        leaking if a caller later invokes prepare_model_input() with one frame
        and run() with a different one.

        ``_gate_df_clean`` captures the authoritative production frame here so the
        corroboration gate is not misled by the split-training path's auxiliary
        ``prepare_model_input`` call (which reassigns ``_df_clean``).
        """
        self._gate_df_clean = df_clean
        try:
            return super().run(
                candidate_pairs_df,
                df_clean,
                full_output=full_output,
                return_linker=return_linker,
            )
        finally:
            self._df_clean = None
            self._gate_df_clean = None


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration shim (mirrors run_fs_enhanced_3 signature)
# ═══════════════════════════════════════════════════════════════════════════════
def run_fs_enhanced_4(
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
    model = FSEnhanced4(
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
    "FSEnhanced4",
    "MODEL_NAME",
    "run_fs_enhanced_4",
]
