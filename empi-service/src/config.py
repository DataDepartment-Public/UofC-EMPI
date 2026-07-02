"""Centralized configuration and logging setup for the eMPI pipeline.

This module is the single source of truth for runtime configuration. It holds:

* :class:`Settings` — a pydantic-settings object replacing the paths and
  thresholds that were once hardcoded across stage modules. Every value can be
  overridden by an ``EMPI_``-prefixed environment variable (or ``.env`` entry)::

      EMPI_GOVERNANCE_THRESHOLD=1000 python -m src.pipeline

* :func:`configure_logging` — one entry point that configures the **root**
  logger from ``Settings`` (level and optional file output), replacing the
  per-module ``logging.basicConfig`` blocks the stage CLIs used to repeat.

Import the shared instances::

    from src.config import settings, configure_logging
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/config.py -> project root is one parent up from src/.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA = _PROJECT_ROOT / "data"
_MODELS = _PROJECT_ROOT / "models"


class Settings(BaseSettings):
    """Pipeline configuration. Defaults assume the standard repo layout."""

    model_config = SettingsConfigDict(
        env_prefix="EMPI_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = _PROJECT_ROOT

    # ── Stage directories ───────────────────────────────────────────────────
    raw_dir: Path = _DATA / "raw"
    processed_dir: Path = _DATA / "processed"
    blocking_dir: Path = _DATA / "blocking"
    matches_dir: Path = _DATA / "matches"
    matches_model_dir: Path = _DATA / "matches_model"
    non_matches_dir: Path = _DATA / "non_matches"
    rejects_dir: Path = _DATA / "rejects"
    clusters_dir: Path = _DATA / "clusters"
    runs_dir: Path = _DATA / "runs"
    log_dir: Path = _PROJECT_ROOT / "logs"

    # ── API service (src/api/) ──────────────────────────────────────────────
    db_path: Path = Field(
        default=_DATA / "empi.db",
        description="SQLite file holding the resolved output (entities, "
        "membership, audit log) the API serves and reviewers edit.",
    )
    api_cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed browser origins for the future Next.js front-end "
        "(see docs/Application-Architecture.md).",
    )
    reviewer_header: str = Field(
        default="X-Reviewer-Id",
        description="Header the API trusts for reviewer identity on /audit/* "
        "(set by a front-end BFF; never client-asserted in a real deploy).",
    )
    records_page_size: int = Field(
        default=50, description="Default page size for GET /records."
    )

    # ── Defaults ────────────────────────────────────────────────────────────
    raw_input: Path = _DATA / "raw" / "MDM_Population.csv"
    cleaned_stem: str = "MDM_Population_cleaned"

    # ── Blocking governance ─────────────────────────────────────────────────
    governance_threshold: int = Field(
        default=500,
        description="Per-key record cap; blocks larger than this are capped.",
    )

    # ── Stacked blocker: q-gram pass (src/preprocessing/qgram_blocking.py) ────
    # The 8-block scheme is unioned with a typo-tolerant char-n-gram cosine pass
    # restricted to a coarse DOB block, then pruned by meta-blocking (below).
    # Validated recommended config: fulldob block, char (2,4)-grams, cosine >=0.30.
    qgram_block_kind: str = Field(
        default="fulldob",
        description="Coarse block the q-gram cosine is restricted to within "
        "('fulldob' = exact birth date, or 'birthyear').",
    )
    qgram_threshold: float = Field(
        default=0.30,
        description="Minimum char-n-gram cosine similarity for a q-gram "
        "candidate pair. Pending gold-label confirmation.",
    )
    qgram_ngram_min: int = Field(default=2, description="Min char n-gram size.")
    qgram_ngram_max: int = Field(default=4, description="Max char n-gram size.")
    qgram_min_df: int = Field(
        default=2,
        description="TfidfVectorizer min_df — drop n-grams rarer than this.",
    )

    # ── Stacked blocker: meta-blocking prune (src/preprocessing/meta_blocking.py) ─
    cnp_top_k: int = Field(
        default=10,
        description="Cardinality Node Pruning: keep an edge if it is in the "
        "top-k ARCS-weighted edges of either endpoint.",
    )
    cnp_qgram_only_weight: float = Field(
        default=0.5,
        description="ARCS weight assigned to q-gram-only edges (no 8-block "
        "source) when meta-blocking ranks them.",
    )

    # ── Rules ───────────────────────────────────────────────────────────────
    ssn_fanout_threshold: int = Field(
        default=4,
        description="Flag SSN_DOB matches whose shared SSN is carried by at "
        "least this many distinct patients (likely shared/fraudulent SSN).",
    )

    # ── Stage 4: Fellegi-Sunter matcher (src/models/fs_matcher/) ─────────────
    # The FS module is a candidate + feature generator for the downstream GBT.
    # It loads a pre-trained model artifact (no per-run training) and scores the
    # rules' `non_matches` pool. Its output does NOT feed clustering.
    fs_model_dir: Path = Field(
        default=_MODELS / "fs",
        description="Directory of trained FS model artifacts (Splink JSON + "
        "meta sidecars) and the 'active' pointer. Gitignored — populated by "
        "`python -m src.models.fs_matcher.train` on the VM.",
    )
    fs_active_model: Path | None = Field(
        default=None,
        description="Explicit active FS model JSON to serve. When None, the "
        "registry resolves the active pointer (or latest) in fs_model_dir.",
    )
    fs_output_dir: Path = Field(
        default=_DATA / "FS_output",
        description="Output dir for the GBT feature parquet — the pipeline's "
        "per-run candidate set and the train CLI's labeled training set. Gitignored.",
    )
    fs_auto_merge_threshold: float = Field(
        default=0.95,
        description="Match-probability at/above which a scored pair is tiered "
        "'auto_merge'. Informational only — the FS stage routes nothing; it "
        "labels tiers for the GBT/audit.",
    )
    fs_review_floor: float = Field(
        default=0.40,
        description="Match-probability at/above which a pair is tiered "
        "'human_review'. Doubles as the CANDIDATE CUTOFF: only pairs at/above "
        "this land in the FSFeatures GBT parquet.",
    )
    fs_deploy_gate_margin: float = Field(
        default=0.02,
        description="A retrained FS model may only be promoted to 'active' if "
        "its held-out precision/recall are within this margin of the current "
        "active model's (guards against deploying a degraded retrain).",
    )

    # ── Logging (see configure_logging below) ────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Root log level: DEBUG / INFO / WARNING / ERROR / CRITICAL.",
    )
    log_to_file: bool = Field(
        default=False,
        description="When true, also write logs to a file under log_dir.",
    )
    log_file: str | None = Field(
        default=None,
        description="Log filename under log_dir; default empi_<UTC-date>.log.",
    )
    log_format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="logging.Formatter format string for all handlers.",
    )
    log_datefmt: str = Field(default="%Y-%m-%d %H:%M:%S")

    # ── Evaluation report (src/evaluation/run_eval.py) ───────────────────────
    deploy_blocking_method: str = Field(
        default="loose",
        description="Wide-candidate strategy for the blocking-recall eval "
        "('loose' or 'sample').",
    )
    deploy_blocking_sample: int = Field(
        default=500,
        description="Sample size when deploy_blocking_method='sample' (O(S^2)).",
    )
    deploy_seed: int = Field(
        default=7, description="Seed for the evaluation-report runs."
    )

    # ── Evaluation / adjudication (src/evaluation/rule_eval.py) ──────────────
    adj_sample_size: int = Field(
        default=300,
        description="Matches sampled per rule when adjudicating precision.",
    )
    adj_first_strong: float = Field(
        default=0.92,
        description="First-name similarity treated as the same name outright.",
    )
    adj_first_ok: float = Field(
        default=0.85,
        description="First-name similarity acceptable when a field corroborates.",
    )
    adj_last_ok: float = Field(
        default=0.80,
        description="Last-name similarity acceptable (maiden-name tolerance).",
    )
    adj_first_different: float = Field(
        default=0.70,
        description="Below this first-name similarity, names are distinct people.",
    )

    def ensure_dirs(self) -> None:
        """Create every output directory this run will write to."""
        for d in (
            self.raw_dir,
            self.processed_dir,
            self.blocking_dir,
            self.matches_dir,
            self.non_matches_dir,
            self.rejects_dir,
            self.clusters_dir,
            self.runs_dir,
            self.matches_model_dir,
            self.fs_model_dir,
            self.fs_output_dir,
            self.db_path.parent,
        ):
            d.mkdir(parents=True, exist_ok=True)


#: Shared, import-once settings instance.
settings = Settings()


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════
#: Module-level guard so importing several CLIs in one process configures once.
_LOGGING_CONFIGURED = False

#: Third-party loggers that are noisy at DEBUG and rarely useful to operators.
_NOISY_LIBRARIES = ("matplotlib", "PIL", "fontTools")


def configure_logging(
    settings: Settings = settings,
    *,
    level: str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure root logging from ``settings`` (and optional ``level`` override).

    Call once at process entry (each CLI ``main`` and ``run_pipeline``). It
    configures the **root** logger, so every module's
    ``logging.getLogger(__name__)`` inherits the level and handlers. Idempotent:
    repeated calls do not stack duplicate handlers.

    Args:
        settings: Configuration source for level, file output, and format.
        level: Explicit level name (e.g. ``"DEBUG"``) overriding
            ``settings.log_level`` — typically wired to a ``--log-level`` flag.
        force: Reconfigure even if a previous call already ran (replaces the
            existing handlers). Useful in tests.

    Returns:
        The configured root logger.

    PHI: this only sets formatting and handlers; it never emits record fields
    itself. Call sites stay responsible for logging aggregates only — see the
    convention documented in ``src/preprocessing/blocking.py``.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return logging.getLogger()

    level_name = (level or settings.log_level).upper()
    numeric = logging.getLevelNamesMapping().get(level_name, logging.INFO)

    root = logging.getLogger()
    # Drop any handlers an import-time basicConfig (or a prior call) installed so
    # records are not emitted twice.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(settings.log_format, datefmt=settings.log_datefmt)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if settings.log_to_file:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        filename = settings.log_file or (
            f"empi_{datetime.now(timezone.utc):%Y%m%d}.log"
        )
        file_handler = logging.FileHandler(
            settings.log_dir / filename, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(numeric)
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(max(numeric, logging.WARNING))

    _LOGGING_CONFIGURED = True
    logging.getLogger(__name__).debug(
        "Logging configured: level=%s file=%s", level_name, settings.log_to_file
    )
    return root


__all__ = ["Settings", "settings", "configure_logging"]
