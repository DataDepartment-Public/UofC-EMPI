"""Executable data contracts for the eMPI pipeline.

This module is the single source of truth for the **shape of the data at every
stage boundary** — the runtime counterpart to `docs/Data-Contract.md`. Stages
import their column names and validate their inputs/outputs here instead of
re-declaring `COL_*` constants, so a rename happens in exactly one place and a
contract violation fails *loudly at the boundary* rather than silently
producing NaN-suppressed matches deep inside the rule engine.

Two kinds of contract live here:

* **pandera `DataFrameModel`s** — the bulk frames that move between stages
  (cleaned records, candidate pairs, matches, …). Validation is vectorized over
  whole columns, so it is cheap to run on every read and write.
* **pydantic models** — the scalar/structured artifacts: the run manifest that
  binds stage outputs into one lineage, and (see `src/config/config.py`) the
  settings object.

Authority is still the producing code; if a producer changes a column, update
the matching model here in the same change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DateTime, Series
from pydantic import BaseModel, Field

# ── Canonical column names (import these instead of re-declaring COL_* ) ───────
PATID = "PATID"
PATID_A = "PATID_A"
PATID_B = "PATID_B"

# Cleaned attribute columns the downstream stages depend on.
FIRST_NM = "FirstNM_clean"
LAST_NM = "LastNM_clean"
BIRTH_DT = "BirthDT_clean"
SSN = "SSN_clean"
SSN_LAST4 = "last_4_SSN"
EMAIL = "Email_clean"
ZIP_BASE = "ZipCD_clean_base"
ADDRESS1 = "AddressLine1_clean"
SEX = "SexAtBirthDSC_clean"
PHONES = "Phones_set"
VALID_RECORD = "valid_record"

#: Columns a consumer of the cleaned dataset may rely on being present + typed.
CLEANED_REQUIRED_COLUMNS: tuple[str, ...] = (
    PATID, FIRST_NM, LAST_NM, BIRTH_DT, SSN, SSN_LAST4,
    EMAIL, ZIP_BASE, ADDRESS1, SEX, PHONES, VALID_RECORD,
)

#: Deterministic rule names — keep in sync with deterministic_rules.RULES.
RULE_NAMES: tuple[str, ...] = (
    "SSN_DOB", "NAME_DOB_EMAIL",
    "NAME_DOB_PHONE", "NAME_DOB_SEX", "NAME_DOB_ADDRESS",
)

#: Auto-merge-tier rule names — keep in sync with
#: deterministic_rules.AUTO_MERGE_RULES. NAME_DOB_SEX / NAME_DOB_ADDRESS are
#: review-tier and must never appear in the (auto-merge) Matches artifact.
AUTO_MERGE_RULE_NAMES: tuple[str, ...] = (
    "SSN_DOB", "NAME_DOB_EMAIL", "NAME_DOB_PHONE",
)

SEX_VALUES: tuple[str, ...] = ("MALE", "FEMALE", "OTHER")
MATCH_SOURCES: tuple[str, ...] = ("deterministic", "model")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _is_listlike_or_null(value: object) -> bool:
    """True if a cell is a collection of phones/tokens or a null.

    Parquet round-trips the pipeline's list-valued columns as ``np.ndarray``;
    in memory they are lists; legacy CSV produced ``set``. All are valid.
    """
    if value is None or isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        return True
    return isinstance(value, float) and bool(pd.isna(value))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Cleaned dataset
# ═══════════════════════════════════════════════════════════════════════════════
class CleanedRecords(pa.DataFrameModel):
    """Output of the clean/transform stage (consumed by blocking and rules).

    `strict=False`: the cleaned frame carries many passthrough columns (every
    ``<field>_raw``, plus derived artifacts) that downstream stages ignore. We
    validate only the columns the pipeline depends on; extras are allowed.
    """

    # coerce is intentionally OFF for the string columns: coercing a nullable
    # str column would turn real NaNs into the literal "nan" and silently
    # corrupt the frame. The producer already emits correct str/NaN; we only
    # coerce the datetime/bool columns, where coercion is safe and NA-preserving.
    PATID: Series[str] = pa.Field(nullable=False)
    FirstNM_clean: Series[str] = pa.Field(nullable=True)
    LastNM_clean: Series[str] = pa.Field(nullable=True)
    BirthDT_clean: Series[DateTime] = pa.Field(nullable=True, coerce=True)
    SSN_clean: Series[str] = pa.Field(nullable=True, str_matches=r"^\d{9}$")
    last_4_SSN: Series[str] = pa.Field(nullable=True, str_matches=r"^\d{4}$")
    Email_clean: Series[str] = pa.Field(nullable=True)
    ZipCD_clean_base: Series[str] = pa.Field(nullable=True, str_matches=r"^\d{5}$")
    AddressLine1_clean: Series[str] = pa.Field(nullable=True)
    SexAtBirthDSC_clean: Series[str] = pa.Field(nullable=True, isin=list(SEX_VALUES))
    Phones_set: Series[object] = pa.Field(nullable=True, coerce=True)
    valid_record: Series[bool] = pa.Field(nullable=False, coerce=True)

    class Config:
        strict = False
        coerce = False

    @pa.dataframe_check
    def _phones_listlike(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        if PHONES not in df.columns:
            return True
        return bool(df[PHONES].map(_is_listlike_or_null).all())


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Candidate pairs
# ═══════════════════════════════════════════════════════════════════════════════
class CandidatePairs(pa.DataFrameModel):
    """Output of blocking. `strict=True`: the schema is exact and closed."""

    PATID_A: Series[str] = pa.Field(nullable=False, coerce=True)
    PATID_B: Series[str] = pa.Field(nullable=False, coerce=True)
    source_blocks: Series[str] = pa.Field(nullable=False, coerce=True)
    n_blocks: Series[int] = pa.Field(nullable=False, coerce=True, ge=1)

    class Config:
        strict = True
        coerce = True

    @pa.dataframe_check
    def _canonical_order(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        return bool((df[PATID_A] < df[PATID_B]).all())


class NonMatches(CandidatePairs):
    """Candidate pairs no deterministic rule confirmed — identical schema to
    `CandidatePairs`, kept as a distinct boundary for clarity and provenance."""


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Deterministic matches
# ═══════════════════════════════════════════════════════════════════════════════
class Matches(pa.DataFrameModel):
    """Confirmed deterministic matches (validated after `cluster_id` is stamped).

    Validate empty match frames with `allow_empty=True` (the default in
    `validate`) — a run can legitimately confirm zero pairs.
    """

    PATID_A: Series[str] = pa.Field(nullable=False, coerce=True)
    PATID_B: Series[str] = pa.Field(nullable=False, coerce=True)
    match_rule: Series[str] = pa.Field(
        nullable=False, coerce=True, isin=list(AUTO_MERGE_RULE_NAMES)
    )
    confidence: Series[float] = pa.Field(nullable=False, coerce=True, ge=0.985, le=1.0)
    rules_fired: Series[str] = pa.Field(nullable=False, coerce=True)
    is_suspicious: Series[bool] = pa.Field(nullable=False, coerce=True)
    high_fanout_ssn: Series[bool] = pa.Field(nullable=False, coerce=True)
    cluster_id: Series[int] = pa.Field(nullable=False, coerce=True, ge=0)
    source_blocks: Series[str] = pa.Field(nullable=False, coerce=True)
    n_blocks: Series[int] = pa.Field(nullable=False, coerce=True, ge=1)

    class Config:
        strict = True
        coerce = True

    @pa.dataframe_check
    def _canonical_order(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        return bool((df[PATID_A] < df[PATID_B]).all())


# ═══════════════════════════════════════════════════════════════════════════════
# Stages 4–5 — PROPOSED target contracts (no stage emits these yet)
# ═══════════════════════════════════════════════════════════════════════════════
class Edges(pa.DataFrameModel):
    """Uniform edge schema emitted by BOTH the rules and the model stages, so
    clustering consumes one concatenated frame. PROPOSED — see Data-Contract.md.
    """

    PATID_A: Series[str] = pa.Field(nullable=False, coerce=True)
    PATID_B: Series[str] = pa.Field(nullable=False, coerce=True)
    confidence: Series[float] = pa.Field(nullable=False, coerce=True, ge=0.0, le=1.0)
    match_source: Series[str] = pa.Field(nullable=False, coerce=True, isin=list(MATCH_SOURCES))
    evidence: Series[str] = pa.Field(nullable=True, coerce=True)

    class Config:
        strict = True
        coerce = True

    @pa.dataframe_check
    def _canonical_order(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        return bool((df[PATID_A] < df[PATID_B]).all())


class ClusterAssignments(pa.DataFrameModel):
    """Terminal clustering output: one row per record, including singletons.
    PROPOSED — see Data-Contract.md."""

    PATID: Series[str] = pa.Field(nullable=False, coerce=True, unique=True)
    cluster_id: Series[int] = pa.Field(nullable=False, coerce=True, ge=0)

    class Config:
        strict = True
        coerce = True


# ═══════════════════════════════════════════════════════════════════════════════
# Validation entry points
# ═══════════════════════════════════════════════════════════════════════════════
def validate(
    df: pd.DataFrame,
    schema: type[pa.DataFrameModel],
    *,
    lazy: bool = True,
    allow_empty: bool = True,
) -> pd.DataFrame:
    """Validate `df` against `schema`, returning the (coerced) frame.

    Raises ``pandera.errors.SchemaErrors`` listing *all* violations when
    `lazy=True`. Empty frames skip validation when `allow_empty` (a degenerate
    but legal state, e.g. a run that confirms zero matches).
    """
    if allow_empty and len(df) == 0:
        return df
    return schema.validate(df, lazy=lazy)


def assert_patid_coverage(
    pairs: pd.DataFrame, clean: pd.DataFrame, *, patid_col: str = PATID
) -> None:
    """Fail loudly if any PATID referenced by `pairs` is absent from `clean`.

    This is the cross-frame invariant a single-frame schema cannot express. It
    catches the lineage bug where the rules stage loads a cleaned dataset that
    does not correspond to the candidate pairs (e.g. a re-versioned clean run),
    which would otherwise silently NaN-suppress matches.
    """
    known = set(clean[patid_col])
    referenced = set(pairs[PATID_A]) | set(pairs[PATID_B])
    missing = referenced - known
    if missing:
        sample = sorted(str(m) for m in missing)[:5]
        raise ValueError(
            f"Lineage mismatch: {len(missing):,} PATID(s) referenced by candidate "
            f"pairs are absent from the cleaned dataset (e.g. {sample}). The pairs "
            f"and cleaned frame are from different cleaning runs."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Run manifest — binds one run's stage outputs into a single lineage record
# ═══════════════════════════════════════════════════════════════════════════════
class ArtifactRef(BaseModel):
    """A persisted stage artifact, with enough metadata to verify lineage."""

    path: str
    rows: int
    sha256: str


class RunManifest(BaseModel):
    """One pipeline run. Serialized to ``data/runs/run_<run_id>.json`` so every
    downstream artifact can be traced to the exact inputs that produced it —
    replacing fragile "latest version in the directory" resolution."""

    run_id: str
    created_utc: str
    git_sha: str | None = None
    raw_input: ArtifactRef
    cleaned: ArtifactRef
    candidate_pairs: ArtifactRef
    matches: ArtifactRef
    non_matches: ArtifactRef
    rejects: ArtifactRef | None = None
    clusters: ArtifactRef | None = None
    counts: dict[str, int] = Field(default_factory=dict)


__all__ = [
    "ArtifactRef",
    "CandidatePairs",
    "CleanedRecords",
    "ClusterAssignments",
    "Edges",
    "Matches",
    "NonMatches",
    "RunManifest",
    "RULE_NAMES",
    "CLEANED_REQUIRED_COLUMNS",
    "assert_patid_coverage",
    "validate",
]
