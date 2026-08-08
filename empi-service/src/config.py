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
    auto_merge_dir: Path = _DATA / "auto_merge"
    non_matches_dir: Path = _DATA / "non_matches"
    no_match_dir: Path = _DATA / "no_match"
    clusters_dir: Path = _DATA / "clusters"
    runs_dir: Path = _DATA / "runs"
    evaluations_dir: Path = Field(
        default=_DATA / "evaluations",
        description="Stored end-to-end evaluation reports (one JSON + TXT per "
        "session/label-source/holdout). Kept out of runs_dir because those are "
        "pipeline RunManifests — immutable per-run lineage — while these are "
        "measurements ABOUT runs and accumulate on their own timeline.",
    )
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
    index_backend: str = Field(
        default="sqlite",
        description="Storage backend for incremental scoring (POST "
        "/records/score, src/api/incremental.py): 'sqlite' (empi.db, the "
        "live service), 'parquet' (local_index_dir, no DB required — see "
        "src/api/local_score.py), or 'postgres' (Azure Database for "
        "PostgreSQL — see postgres_* settings below and "
        "src/api/backends/postgres_backend.py). Swap point: "
        "src/api/backends/index_backend.py.",
    )
    local_index_dir: Path = Field(
        default=_DATA / "local_index",
        description="Parquet files for the 'parquet' index_backend "
        "(block_key/cleaned_attrs/entity/entity_member/review_candidate/"
        "entity_suggestion) — local-mode incremental scoring, no empi.db.",
    )
    postgres_host: str | None = Field(
        default=None,
        description="Postgres server hostname for the 'postgres' index_backend "
        "(e.g. Azure Database for PostgreSQL's *.postgres.database.azure.com "
        "FQDN — see terraform/postgres.tf). Required when index_backend='postgres'.",
    )
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(
        default="empi",
        description="Database name for the 'postgres' index_backend.",
    )
    postgres_user: str | None = Field(
        default=None,
        description="Postgres user for the 'postgres' index_backend — this "
        "app's own Azure AD identity (its App Service name), registered as "
        "the server's AAD administrator by terraform/postgres.tf. There is "
        "deliberately no postgres_password setting: "
        "src/api/backends/postgres_backend.py authenticates with a token "
        "from DefaultAzureCredential, never a stored secret.",
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
        default=_DATA / "fs_output",
        description="Output dir for both the ProbabilisticMatches audit frame "
        "and the GBT feature parquet — the pipeline's per-run candidate set "
        "and the train CLI's labeled training set. Gitignored.",
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
    fs_feeds_clustering: bool = Field(
        default=False,
        description="When True, the FS matcher's auto_merge-tier "
        "ClassificationResults union into Stage 5 clustering edges alongside "
        "the deterministic rules' matches. Default OFF preserves today's "
        "behavior (FS is audit-only) until a trained FS model is "
        "deliberately promoted for production auto-merge.",
    )

    # ── Stage 4.25: ML non-match gate (src/models/nonmatch_gate/) ────────────
    # The pipeline's non-match gate: a LightGBM classifier over the same 12
    # features the ML matcher uses, trained on the WHOLE candidate pool to
    # separate confident non-matches from plausible pairs (match ∪ ambiguous).
    # It decides which of the rules' `non_matches` survive to the ML matcher —
    # the job the FS matcher used to do. Skipped (pool passes through
    # ungated, or falls back to the FS gate) when no active model resolves.
    gate_model_dir: Path = Field(
        default=_MODELS / "nonmatch_gate",
        description="Directory of trained non-match gate artifacts "
        "(nonmatch_gate_<ts>.pkl + meta sidecars) and the 'active' pointer. "
        "Gitignored — populated by the notebook's export cell "
        "(notebooks/ml_model/confident_nonmatch/).",
    )
    gate_active_model: Path | None = Field(
        default=None,
        description="Explicit active gate artifact to serve. When None, the "
        "registry resolves the active pointer (or latest) in gate_model_dir.",
    )
    gate_output_dir: Path = Field(
        default=_DATA / "gate_output",
        description="Output dir for the gate's per-run ClassificationResults "
        "audit frame. Gitignored.",
    )
    gate_threshold: float = Field(
        default=0.30,
        description="P(plausible) at/above which a pair PASSES the gate and "
        "reaches the ML matcher. Below it the pair is a confident non-match "
        "and is discarded. 0.30 is the notebook's operating point "
        "(plausible recall 0.999 on held-out test).",
    )
    gate_deploy_gate_margin: float = Field(
        default=0.02,
        description="A retrained gate model may only be promoted to 'active' "
        "if its held-out plausible precision/recall are within this margin of "
        "the current active model's.",
    )
    gate_supersedes_fs: bool = Field(
        default=True,
        description="When True (the default), the ML non-match gate decides "
        "the ML matcher's input pool. Set False to fall back to the legacy FS "
        "gate (FS `no_match` tier discarded) even when a gate model is active.",
    )

    # ── Stage 4.5: pluggable ML matcher (src/models/ml_matcher/) ─────────────
    # Bring-your-own-model / bring-your-own-features candidate + feature
    # generator, structurally parallel to the FS matcher: loads a pre-trained
    # model artifact (no per-run training) and scores the rules' `non_matches`
    # pool, optionally enriched with the FS matcher's FSFeatures. Skipped
    # entirely when no active model is resolvable, same as Stage 4.
    ml_model_dir: Path = Field(
        default=_MODELS / "ml",
        description="Directory of trained ML model artifacts and the "
        "'active' pointer. Gitignored — populated by "
        "`python -m src.models.ml_matcher.train` once a real training "
        "implementation is plugged in.",
    )
    ml_active_model: Path | None = Field(
        default=None,
        description="Explicit active ML model artifact to serve. When None, "
        "the registry resolves the active pointer (or latest) in "
        "ml_model_dir.",
    )
    ml_output_dir: Path = Field(
        default=_DATA / "ml_output",
        description="Output dir for both the ML matcher's full "
        "ClassificationResults audit frame and its feature parquet — the "
        "pipeline's per-run candidate set and the train CLI's labeled "
        "training set. Gitignored.",
    )
    ml_auto_merge_threshold: float = Field(
        default=0.70,
        description="Match-probability at/above which a scored pair is tiered "
        "'auto_merge'; below it, 'human_review'. Stage 4.5 is a 2-tier "
        "classifier with no floor — discarding confident non-matches is the "
        "Stage-4.25 gate's job, and it is the only stage that records its "
        "drops. Since ml_feeds_clustering is on, this bounds merge precision: "
        "tune it against a held-out precision target, not a round number.",
    )
    ml_deploy_gate_margin: float = Field(
        default=0.02,
        description="A retrained ML model may only be promoted to 'active' "
        "if its held-out precision/recall are within this margin of the "
        "current active model's (guards against deploying a degraded "
        "retrain).",
    )
    ml_feeds_clustering: bool = Field(
        default=True,
        description="When True, the ML matcher's auto_merge-tier "
        "ClassificationResults union into Stage 5 clustering edges alongside "
        "the deterministic rules' matches. Independent of fs_feeds_clustering. "
        "Default ON: Stage 4.5 is a decision stage, not an audit stage — a "
        "model whose verdict reaches nothing cannot raise recall. Two "
        "consequences to know: (1) merge precision is now bounded by the "
        "served model's, so tune ml_auto_merge_threshold against a held-out "
        "precision target rather than leaving it at the training notebook's "
        "operating point; (2) the served model was fit on gold, so the "
        "end-to-end headline is no longer leakage-free at --holdout none — "
        "use --holdout strict. Set EMPI_ML_FEEDS_CLUSTERING=false to restore "
        "deterministic-edges-only clustering.",
    )

    # ── Per-pair SHAP explanations (src/models/explanations.py) ──────────────
    # The gate and the ML matcher each emit a per-pair contribution frame
    # alongside their scores, served read-only by GET /explanations/... .
    # Computed at score time (not on demand) so an explanation always
    # describes the model and feature vector that produced the recorded
    # decision — see the module docstring.
    explanations_enabled: bool = Field(
        default=True,
        description="Emit per-pair SHAP explanations from the non-match gate "
        "and the ML matcher. Exact TreeSHAP via LightGBM's pred_contrib; adds "
        "roughly 20us/pair/model to a run. Turn off to skip the artifacts "
        "entirely (the explanation endpoint then 404s for that run).",
    )
    explanation_top_n: int = Field(
        default=8,
        description="How many features the explanation endpoint suggests "
        "showing in a waterfall (a hint the UI may ignore; every feature is "
        "always returned, ranked by |contribution|).",
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
            self.auto_merge_dir,
            self.non_matches_dir,
            self.no_match_dir,
            self.clusters_dir,
            self.runs_dir,
            self.evaluations_dir,
            self.fs_model_dir,
            self.fs_output_dir,
            self.ml_model_dir,
            self.ml_output_dir,
            self.gate_model_dir,
            self.gate_output_dir,
            self.db_path.parent,
            self.local_index_dir,
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
