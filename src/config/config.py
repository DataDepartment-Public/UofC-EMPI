"""Centralized, validated configuration for the eMPI pipeline.

Replaces the paths and thresholds that were hardcoded across stage modules with
a single pydantic-settings object. Every value can be overridden by an
``EMPI_``-prefixed environment variable (or `.env` entry), e.g.::

    EMPI_GOVERNANCE_THRESHOLD=1000 python -m src.pipeline

Import the shared instance: ``from src.config.config import settings``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/config/config.py -> project root is two parents up from src/.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA = _PROJECT_ROOT / "data"


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
    non_matches_dir: Path = _DATA / "non_matches"
    rejects_dir: Path = _DATA / "rejects"
    runs_dir: Path = _DATA / "runs"

    # ── Defaults ────────────────────────────────────────────────────────────
    raw_input: Path = _DATA / "raw" / "MDM_Population.csv"
    cleaned_stem: str = "MDM_Population_cleaned"

    # ── Blocking governance ─────────────────────────────────────────────────
    governance_threshold: int = Field(
        default=500,
        description="Per-key record cap; blocks larger than this are capped.",
    )

    # ── Rules ───────────────────────────────────────────────────────────────
    ssn_fanout_threshold: int = Field(
        default=4,
        description="Flag SSN_DOB matches whose shared SSN is carried by at "
        "least this many distinct patients (likely shared/fraudulent SSN).",
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
            self.processed_dir,
            self.blocking_dir,
            self.matches_dir,
            self.non_matches_dir,
            self.rejects_dir,
            self.runs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


#: Shared, import-once settings instance.
settings = Settings()
