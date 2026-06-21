"""Shared object-oriented base for Fellegi-Sunter Splink models in this repo.

This module is consumed by `models/experiments/fs_splink_enhanced_2/` and is
the abstraction baseline + enhanced will eventually be refactored onto in a
separate workstream. Three concepts live here:

* `ComparisonSpec` / `ComparisonRegistry` — declarative, ordered, mutable-by-copy
  collection of Splink comparisons. A concrete FS model assembles its registry
  once at class load and downstream code can build new registries via
  `with_added` / `with_removed` / `with_replaced` without mutating the original.

* `TrainingStrategy` (ABC) + concrete `EMTraining` / `SupervisedTraining` — the
  m/u estimation procedure. EM trains m via random-pair sampling + EM sessions
  on the blocked candidate pool; SupervisedTraining trains m directly from a
  labeled-pairs frame (e.g., `data/synthetic/synthetic_train_v3.csv`). u is
  estimated by random sampling on the real cohort in both cases — synthetic
  negatives cannot represent blocking-induced household co-occurrence.

* `FSModel` (ABC) — owns Splink-linker boilerplate, the n_blocks log-odds bump,
  threshold classification, and the two output projections (5-col eval schema,
  ProbabilisticMatches contract). Subclasses configure `model_name`, `registry`,
  `training`, and `classification_config`, and implement `prepare_model_input`
  + `build_settings` (the project-specific derivations and the final settings
  dict assembly).

PHI / HIPAA: this module logs aggregate counts only — no PATIDs, names, SSNs,
DOBs, or addresses. Subclasses must mirror this discipline.
"""

from __future__ import annotations

import logging
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ComparisonSpec / ComparisonRegistry
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ComparisonSpec:
    """Declarative spec for one Splink Comparison.

    `builder` is a zero-arg callable that returns the comparison's dict form
    (the shape Splink Linker accepts inside `settings["comparisons"]`). It is
    invoked lazily by `ComparisonRegistry.build_all()` so the registry can be
    constructed at module-import time without paying the Splink import cost.
    """

    name: str
    builder: Callable[[], dict]
    notes: str = ""


class ComparisonRegistry:
    """Ordered, immutable-by-copy collection of ComparisonSpec.

    Mutation methods (`with_added`, `with_removed`, `with_replaced`) return
    *new* registries. The original is never modified, so a subclass's
    class-level registry constant is safe under concurrent runs and tests.
    """

    def __init__(self, specs: Sequence[ComparisonSpec]):
        names = [s.name for s in specs]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"ComparisonRegistry: duplicate comparison names {duplicates}"
            )
        self._specs: tuple[ComparisonSpec, ...] = tuple(specs)

    def __iter__(self):
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self.names()

    def names(self) -> list[str]:
        return [s.name for s in self._specs]

    def get(self, name: str) -> ComparisonSpec:
        for s in self._specs:
            if s.name == name:
                return s
        raise KeyError(f"ComparisonRegistry: {name!r} not in registry (have: {self.names()})")

    def build_all(self) -> list[dict]:
        return [s.builder() for s in self._specs]

    def with_added(
        self, spec: ComparisonSpec, *, position: int | None = None
    ) -> "ComparisonRegistry":
        new = list(self._specs)
        if position is None:
            new.append(spec)
        else:
            new.insert(position, spec)
        return ComparisonRegistry(new)

    def with_removed(self, name: str) -> "ComparisonRegistry":
        if name not in self.names():
            raise KeyError(
                f"ComparisonRegistry: {name!r} not in registry (have: {self.names()})"
            )
        return ComparisonRegistry([s for s in self._specs if s.name != name])

    def with_replaced(self, name: str, replacement: ComparisonSpec) -> "ComparisonRegistry":
        if name not in self.names():
            raise KeyError(
                f"ComparisonRegistry: {name!r} not in registry (have: {self.names()})"
            )
        return ComparisonRegistry(
            [replacement if s.name == name else s for s in self._specs]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Training strategies
# ═══════════════════════════════════════════════════════════════════════════════
class TrainingStrategy(ABC):
    """Strategy interface for m/u estimation. Implementations train the Splink
    linker in place; they do not return a new linker."""

    @abstractmethod
    def train(self, linker: Any, df_clean: pd.DataFrame | None = None) -> None:
        """Train `linker` in place. `df_clean` is provided for strategies that
        need the underlying cleaned records (e.g., custom u-sampling)."""


class EMTraining(TrainingStrategy):
    """EM-based m-estimation: random u-sampling + N EM sessions over named
    blocking rules + (optional) match-prevalence prior.

    Used by `fs_splink_baseline` and `fs_splink_enhanced`. Refactoring those
    modules onto this base is deferred to a follow-up workstream.
    """

    def __init__(
        self,
        em_blocking_rules: Sequence[str],
        prior_rules: Sequence[str] | None = None,
        recall: float = 0.80,
        u_max_pairs: float = 1e6,
        seed: int = 42,
    ):
        self.em_blocking_rules = list(em_blocking_rules)
        self.prior_rules = list(prior_rules) if prior_rules else None
        self.recall = recall
        self.u_max_pairs = u_max_pairs
        self.seed = seed

    def train(self, linker: Any, df_clean: pd.DataFrame | None = None) -> None:
        _estimate_u_with_guard(linker, self.u_max_pairs, self.seed)
        for rule in self.em_blocking_rules:
            linker.training.estimate_parameters_using_expectation_maximisation(rule)
        if self.prior_rules:
            linker.training.estimate_probability_two_random_records_match(
                deterministic_matching_rules=list(self.prior_rules),
                recall=self.recall,
            )
        _warn_on_weight_inversions(linker)


class SupervisedTraining(TrainingStrategy):
    """Supervised m-estimation from a labeled-pairs frame.

    `labels_df` is a pandas DataFrame whose rows are pre-paired records — one
    row per pair, with the FS feature columns suffixed `_l` / `_r` and a
    binary `label_col` indicating ground truth. u-estimation is still done
    via random sampling on the real cohort (synthetic negatives don't reflect
    blocking-induced household co-occurrence).

    See `data/synthetic/synthetic_train_v3.csv` for the expected schema and
    `data/synthetic/coverage_report.csv` for the per-comparison-level audit.
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        label_col: str = "label",
        u_max_pairs: float = 1e6,
        seed: int = 42,
        labels_table_name: str = "synthetic_labels",
    ):
        if label_col not in labels_df.columns:
            raise ValueError(
                f"SupervisedTraining: label column {label_col!r} not in "
                f"labels_df (have: {sorted(labels_df.columns)[:10]}...)"
            )
        self.labels_df = labels_df
        self.label_col = label_col
        self.u_max_pairs = u_max_pairs
        self.seed = seed
        self.labels_table_name = labels_table_name

    def train(self, linker: Any, df_clean: pd.DataFrame | None = None) -> None:
        _estimate_u_with_guard(linker, self.u_max_pairs, self.seed)
        # Splink 4.x: register the labels table, then call
        # estimate_m_from_pairwise_labels. Exact API binding is verified in E2-3
        # against the installed splink==4.0.16; this base intentionally relies
        # on the public `training` namespace only.
        linker.table_management.register_labels_table(
            self.labels_df, self.label_col
        )
        linker.training.estimate_m_from_pairwise_labels(self.label_col)
        _warn_on_weight_inversions(linker)


# ═══════════════════════════════════════════════════════════════════════════════
# Classification config
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ClassificationConfig:
    auto_merge_threshold: float = 0.95
    review_floor: float = 0.40
    n_blocks_bump_threshold: int = 2
    n_blocks_bump_max_bits: float = 4.0

    def __post_init__(self):
        if not 0.0 <= self.review_floor <= self.auto_merge_threshold <= 1.0:
            raise ValueError(
                f"ClassificationConfig: need 0 <= review_floor "
                f"({self.review_floor}) <= auto_merge_threshold "
                f"({self.auto_merge_threshold}) <= 1."
            )
        if self.n_blocks_bump_max_bits < 0:
            raise ValueError(
                f"ClassificationConfig: n_blocks_bump_max_bits must be >= 0 "
                f"(got {self.n_blocks_bump_max_bits})"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# FSModel base
# ═══════════════════════════════════════════════════════════════════════════════
class FSModel(ABC):
    """Abstract base for one FS experiment.

    Subclass responsibilities (class attributes or instance attributes set in
    __init__):
        model_name             — self-identifying tag for the eval-schema output
        registry               — ComparisonRegistry assembled from ComparisonSpecs
        classification_config  — ClassificationConfig (thresholds + n_blocks bump)
        training               — TrainingStrategy instance

    Subclass methods to implement:
        prepare_model_input(df_clean)  — project-specific derived columns
        build_settings()               — assemble the full Splink settings dict
                                         from the registry + linker config

    Base provides: build_linker, train, predict, classify (with n_blocks bump),
    to_evaluation_schema, to_probabilistic_matches, run.
    """

    model_name: str = "fs_model"
    registry: ComparisonRegistry
    classification_config: ClassificationConfig
    training: TrainingStrategy

    candidate_pairs_table_name: str = "candidate_pairs"
    unique_id_column: str = "PATID"

    eval_schema_columns: tuple[str, ...] = (
        "PATID_A", "PATID_B", "model_name", "score", "predicted_tier",
    )
    probabilistic_matches_columns: tuple[str, ...] = (
        "PATID_A", "PATID_B", "match_source", "score", "match_weight",
        "classification_tier", "source_blocks", "n_blocks",
    )

    # ── Subclass hooks ────────────────────────────────────────────────────────
    @abstractmethod
    def prepare_model_input(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Return a model-ready frame: derived columns + Splink shim columns."""

    @abstractmethod
    def build_settings(self) -> dict:
        """Return the full Splink settings dict (link_type, blocking rules,
        comparisons from self.registry, retain flags)."""

    # ── Base methods ──────────────────────────────────────────────────────────
    def build_linker(
        self,
        df_model: pd.DataFrame,
        candidate_pairs_df: pd.DataFrame,
        db_api: Any = None,
    ) -> Any:
        from splink import DuckDBAPI, Linker
        db_api = db_api or DuckDBAPI()
        _register_candidate_pairs(
            db_api, candidate_pairs_df, self.candidate_pairs_table_name
        )
        linker = Linker(df_model, self.build_settings(), db_api=db_api)
        logger.info(
            "FSModel[%s]: linker built (%d records, %d candidate pairs)",
            self.model_name, len(df_model), len(candidate_pairs_df),
        )
        return linker

    def train(self, linker: Any, df_clean: pd.DataFrame | None = None) -> Any:
        self.training.train(linker, df_clean)
        logger.info("FSModel[%s]: training complete", self.model_name)
        return linker

    def predict(
        self,
        linker: Any,
        candidate_pairs_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        df = linker.inference.predict().as_pandas_dataframe()
        # Canonical PATID_A / PATID_B from Splink's _l / _r columns.
        l_col, r_col = f"{self.unique_id_column}_l", f"{self.unique_id_column}_r"
        if l_col in df.columns and r_col in df.columns:
            a = df[[l_col, r_col]].min(axis=1)
            b = df[[l_col, r_col]].max(axis=1)
            df.insert(0, "PATID_A", a)
            df.insert(1, "PATID_B", b)
        if candidate_pairs_df is not None:
            meta_cols = [
                c for c in ("source_blocks", "n_blocks")
                if c in candidate_pairs_df.columns
            ]
            if meta_cols:
                df = df.merge(
                    candidate_pairs_df[["PATID_A", "PATID_B", *meta_cols]],
                    on=["PATID_A", "PATID_B"], how="left",
                )
        logger.info("FSModel[%s]: scored %d pairs", self.model_name, len(df))
        return df

    def classify(self, df_predictions: pd.DataFrame) -> pd.DataFrame:
        cfg = self.classification_config
        out = df_predictions.copy()
        out = self._apply_n_blocks_bump(
            out, cfg.n_blocks_bump_threshold, cfg.n_blocks_bump_max_bits,
        )
        p = out["match_probability"]
        tier = pd.Series("no_match", index=out.index, dtype="object")
        tier = tier.mask(p >= cfg.review_floor, "human_review")
        tier = tier.mask(p >= cfg.auto_merge_threshold, "auto_merge")
        out["classification_tier"] = tier
        counts = {k: int(v) for k, v in out["classification_tier"].value_counts().items()}
        logger.info("FSModel[%s]: classify %s", self.model_name, counts)
        return out

    @staticmethod
    def _apply_n_blocks_bump(
        df: pd.DataFrame, threshold: int, max_bits: float,
    ) -> pd.DataFrame:
        """Apply +1 bit / block above `threshold` (capped at `max_bits`) in
        log-odds space, then back to probability. Operates on `match_probability`
        and keeps `match_weight` in sync when present. No-op if `n_blocks` is
        absent or no pair exceeds the threshold. Returns a possibly-copied frame.
        """
        if "n_blocks" not in df.columns:
            return df
        bump_bits = (df["n_blocks"] - threshold).clip(lower=0, upper=max_bits)
        if not bool((bump_bits > 0).any()):
            return df
        df = df.copy()
        p_safe = df["match_probability"].clip(lower=1e-12, upper=1 - 1e-12)
        weight = np.log2(p_safe / (1.0 - p_safe))
        weight_bumped = weight + bump_bits
        df["match_probability"] = 1.0 / (1.0 + np.power(2.0, -weight_bumped))
        if "match_weight" in df.columns:
            df["match_weight"] = weight_bumped
        n_bumped = int((bump_bits > 0).sum())
        logger.info(
            "n_blocks bump applied to %d/%d pairs (max bump %.1f bits)",
            n_bumped, len(df), float(bump_bits.max()),
        )
        return df

    def to_evaluation_schema(self, df_classified: pd.DataFrame) -> pd.DataFrame:
        """Project to the cross-model 5-col evaluation contract."""
        required = {"PATID_A", "PATID_B", "match_probability", "classification_tier"}
        missing = required - set(df_classified.columns)
        if missing:
            raise ValueError(
                f"to_evaluation_schema is missing columns {sorted(missing)}; "
                "expected the output of classify()."
            )
        out = pd.DataFrame({
            "PATID_A": df_classified["PATID_A"].values,
            "PATID_B": df_classified["PATID_B"].values,
            "model_name": self.model_name,
            "score": df_classified["match_probability"].values,
            "predicted_tier": df_classified["classification_tier"].values,
        })
        return out[list(self.eval_schema_columns)]

    def to_probabilistic_matches(self, df_classified: pd.DataFrame) -> pd.DataFrame:
        """Project to the ProbabilisticMatches contract.

        Default projection omits `veto_reason`; `veto_reason` is optional in the
        Pandera contract (made optional in Phase E2-1), so a model that does
        produce a veto column can subclass and override this method to include it.
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
            "source_blocks": (
                df_classified["source_blocks"].values
                if "source_blocks" in df_classified.columns
                else [None] * n
            ),
            "n_blocks": (
                df_classified["n_blocks"].values
                if "n_blocks" in df_classified.columns
                else [None] * n
            ),
        })
        return out[list(self.probabilistic_matches_columns)]

    def run(
        self,
        candidate_pairs_df: pd.DataFrame,
        df_clean: pd.DataFrame,
        full_output: bool = False,
    ) -> pd.DataFrame:
        """Train + predict + classify in one call.

        Returns the 5-col eval schema by default; set `full_output=True` for the
        rich classified frame (used by `to_probabilistic_matches` callers).
        """
        df_model = self.prepare_model_input(df_clean)
        linker = self.build_linker(df_model, candidate_pairs_df)
        self.train(linker, df_clean)
        predictions = self.predict(linker, candidate_pairs_df)
        classified = self.classify(predictions)
        return classified if full_output else self.to_evaluation_schema(classified)


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level helpers (lifted from fs_splink_enhanced.fellegi_sunter_enhanced)
# ═══════════════════════════════════════════════════════════════════════════════
def _register_candidate_pairs(
    db_api: Any, candidate_pairs_df: pd.DataFrame, table_name: str,
) -> None:
    """Register candidate_pairs as a queryable table in the DuckDB connection."""
    con = getattr(db_api, "_con", None)
    if con is None:
        raise RuntimeError(
            "Could not access the DuckDB connection on DuckDBAPI (_con). "
            "Splink internals may have changed; verify against the installed "
            "version."
        )
    con.register(table_name, candidate_pairs_df)


def _estimate_u_with_guard(linker: Any, max_pairs: float, seed: int) -> None:
    """Run u-estimation, converting Splink's cryptic single-CPU salting error
    into an actionable RuntimeError."""
    try:
        linker.training.estimate_u_using_random_sampling(
            max_pairs=max_pairs, seed=seed,
        )
    except ValueError as exc:
        if "Salting partitions" in str(exc) and multiprocessing.cpu_count() <= 1:
            raise RuntimeError(
                "u-estimation failed because this host reports a single CPU and "
                f"u_max_pairs={max_pairs:g} (>1e4) triggers Splink's salted DuckDB "
                "sampling path, which needs cpu_count() > 1. Run on a multi-core "
                "allocation, or pass u_max_pairs=1e4."
            ) from exc
        raise


def _warn_on_weight_inversions(linker: Any) -> None:
    """Log a warning for any comparison level where m < u (weight sign inverts)."""
    try:
        records = linker._settings_obj._parameters_as_detailed_records
    except AttributeError as exc:
        logger.warning(
            "Weight-inversion diagnostic skipped — Splink private API "
            "_parameters_as_detailed_records not accessible: %s", exc,
        )
        return
    for r in records:
        m, u = r.get("m_probability"), r.get("u_probability")
        if isinstance(m, (int, float)) and isinstance(u, (int, float)) and u > 0 and m < u:
            logger.warning(
                "Weight inversion (m < u) on comparison '%s' level '%s' "
                "(m=%.4g, u=%.4g) — consider a literature-informed m override.",
                r.get("comparison_name"), r.get("label_for_charts"), m, u,
            )


__all__ = [
    "ClassificationConfig",
    "ComparisonRegistry",
    "ComparisonSpec",
    "EMTraining",
    "FSModel",
    "SupervisedTraining",
    "TrainingStrategy",
]
