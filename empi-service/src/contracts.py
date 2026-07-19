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

from typing import Optional

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

#: Reject-rule names — keep in sync with deterministic_rules.REJECT_RULES.
REJECT_RULE_NAMES: tuple[str, ...] = ("STRONG_ID_CONFLICT",)

SEX_VALUES: tuple[str, ...] = ("MALE", "FEMALE", "OTHER")
MATCH_SOURCES: tuple[str, ...] = ("deterministic", "model", "ml")

#: The three-way pair-classification vocabulary shared by every classifier
#: stage (deterministic rules, the FS matcher, and the ML matcher) — see
#: `src.models.base.PairClassifier`. A single source of truth so "auto_merge"
#: / "human_review" / "no_match" is never re-typed as a bare string literal.
TIER_AUTO_MERGE: str = "auto_merge"
TIER_HUMAN_REVIEW: str = "human_review"
TIER_NO_MATCH: str = "no_match"
CLASSIFICATION_TIERS: tuple[str, ...] = (TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH)


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


class Rejects(CandidatePairs):
    """Unconfirmed pairs dropped for >=min_contradictions strong-identifier
    conflicts (`deterministic_rules.classify_non_matches`). Terminal — nothing
    downstream reads this back; kept for audit/compliance only.

    `reject_rule` is non-nullable (unlike in the intermediate `decided` frame,
    where it is null on `human_review` rows): every row that survives the
    `decision == TIER_NO_MATCH` filter already fired the reject rule, so it
    is populated for 100% of rows in this artifact.
    """

    n_contradictions: Series[int] = pa.Field(nullable=False, coerce=True, ge=0)
    decision: Series[str] = pa.Field(nullable=False, coerce=True, isin=[TIER_NO_MATCH])
    reject_rule: Series[str] = pa.Field(
        nullable=False, coerce=True, isin=list(REJECT_RULE_NAMES)
    )


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
# Stage 4 — Fellegi-Sunter probabilistic scoring (candidate/feature generator)
# ═══════════════════════════════════════════════════════════════════════════════
class ProbabilisticMatches(pa.DataFrameModel):
    """Full audit record of the FS module's scoring of the `non_matches` pool.

    Produced by `src/models/fs_matcher/` (`FSMatcher.to_probabilistic_matches`).
    Written per run to `data/fs_output/matches_model_<run_id>.parquet` for
    auditability/review — it is **not** unioned into clustering (clustering stays
    on the deterministic auto-merge edges only). Carries every scored pair with
    its tier, so it includes `no_match`-tier rows too.

    `veto_reason` is **optional**: the enhanced_3-derived production matcher does
    not apply a veto layer, so it omits the column entirely; a producer that does
    emit vetoes keeps it as a nullable string. Both pass validation.
    """

    PATID_A: Series[str] = pa.Field(nullable=False, coerce=True)
    PATID_B: Series[str] = pa.Field(nullable=False, coerce=True)
    match_source: Series[str] = pa.Field(nullable=False, coerce=True, isin=["model"])
    score: Series[float] = pa.Field(nullable=False, coerce=True, ge=0.0, le=1.0)
    match_weight: Series[float] = pa.Field(nullable=False, coerce=True)
    classification_tier: Series[str] = pa.Field(
        nullable=False, coerce=True, isin=list(CLASSIFICATION_TIERS),
    )
    veto_reason: Optional[Series[str]] = pa.Field(nullable=True, coerce=True)
    source_blocks: Series[str] = pa.Field(nullable=True, coerce=True)
    n_blocks: Series[int] = pa.Field(nullable=True, coerce=True, ge=1)

    class Config:
        strict = True
        coerce = True

    @pa.dataframe_check
    def _canonical_order(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        return bool((df[PATID_A] < df[PATID_B]).all())


#: Required base columns of the FSFeatures GBT parquet (the per-field
#: gamma_<field> / bf_<field> feature columns are dynamic — see
#: validate_fs_features). `label` is present only on the training feature set.
FS_FEATURES_REQUIRED_COLUMNS: tuple[str, ...] = (
    PATID_A, PATID_B, "match_probability", "match_weight", "classification_tier",
)


class FSFeatures(pa.DataFrameModel):
    """Per-pair feature parquet the FS module emits for the downstream GBT.

    Produced by `FSMatcher.to_fs_features`. The pipeline writes the
    candidate-filtered set (``match_probability >= review_floor``) per run to
    `data/fs_output/fs_features_<run_id>.parquet`; the `fs-train` CLI writes a
    labeled full set for GBT training. In addition to the base columns below,
    the frame carries one `gamma_<field>` (level index) and one `bf_<field>`
    (Bayes-factor bits) column per Splink comparison — validated by presence via
    `validate_fs_features` since their exact names depend on each comparison's
    output column. `strict=False` so those feature columns are allowed extras.

    `label` (0/1) is present on the training set and absent (or null) when
    scoring — hence Optional + nullable.
    """

    PATID_A: Series[str] = pa.Field(nullable=False, coerce=True)
    PATID_B: Series[str] = pa.Field(nullable=False, coerce=True)
    match_probability: Series[float] = pa.Field(nullable=False, coerce=True, ge=0.0, le=1.0)
    match_weight: Series[float] = pa.Field(nullable=False, coerce=True)
    classification_tier: Series[str] = pa.Field(
        nullable=False, coerce=True, isin=list(CLASSIFICATION_TIERS),
    )
    label: Optional[Series[float]] = pa.Field(nullable=True, coerce=True, isin=[0.0, 1.0])

    class Config:
        strict = False  # gamma_<field> / bf_<field> feature columns are allowed extras
        coerce = True

    @pa.dataframe_check
    def _canonical_order(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        return bool((df[PATID_A] < df[PATID_B]).all())


class MLFeatures(pa.DataFrameModel):
    """Per-pair feature parquet the ML matcher emits (`src/models/ml_matcher/`).

    Unlike `FSFeatures` (whose `gamma_*`/`bf_*` columns are Splink-comparison-
    derived but still a known naming pattern), the ML matcher is bring-your-
    own-features — a custom `FeatureBuilder` decides its own column names and
    count entirely. Only the pair key is validated here; use
    `validate_ml_features` for the presence check analogous to
    `validate_fs_features`.
    """

    PATID_A: Series[str] = pa.Field(nullable=False, coerce=True)
    PATID_B: Series[str] = pa.Field(nullable=False, coerce=True)

    class Config:
        strict = False  # arbitrary BYOF feature columns are allowed extras
        coerce = True

    @pa.dataframe_check
    def _canonical_order(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        return bool((df[PATID_A] < df[PATID_B]).all())


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-model classifier output (deterministic rules, FS matcher, ML matcher)
# ═══════════════════════════════════════════════════════════════════════════════
class ClassificationResults(pa.DataFrameModel):
    """Uniform per-pair output every `src.models.base.PairClassifier.run()`
    returns — deterministic rules, the FS matcher, and the ML matcher all
    project to this same 5-column shape. Formalizes the shape
    `FSModel.to_evaluation_schema()` already used
    (`src/models/fs_matcher/base.py`) into a real contract; FS needs no
    changes to satisfy it.

    `score` is nullable: the deterministic-rules adapter has no continuous
    confidence for contradiction-based `no_match`/`human_review` rows (only a
    discrete contradiction count) — FS and the ML matcher always populate it.
    """

    PATID_A: Series[str] = pa.Field(nullable=False, coerce=True)
    PATID_B: Series[str] = pa.Field(nullable=False, coerce=True)
    model_name: Series[str] = pa.Field(nullable=False, coerce=True)
    score: Series[float] = pa.Field(nullable=True, coerce=True, ge=0.0, le=1.0)
    predicted_tier: Series[str] = pa.Field(
        nullable=False, coerce=True, isin=list(CLASSIFICATION_TIERS),
    )

    class Config:
        strict = True
        coerce = True

    @pa.dataframe_check
    def _canonical_order(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        return bool((df[PATID_A] < df[PATID_B]).all())


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Clustering + Edges (optional union of classifier stages)
# ═══════════════════════════════════════════════════════════════════════════════
class Edges(pa.DataFrameModel):
    """A uniform edge schema letting clustering optionally consume one
    concatenated frame of deterministic + probabilistic/ML edges.

    Produced by `src.models.base.to_edges()`, projecting the `auto_merge`-tier
    rows of any `ClassificationResults` frame. Deterministic rules' matches
    always feed clustering; the FS matcher's and ML matcher's `auto_merge`
    edges union in here too only when `settings.fs_feeds_clustering` /
    `settings.ml_feeds_clustering` is on (both default `False`, preserving
    clustering-on-deterministic-edges-only as the out-of-the-box behavior).
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
    """Terminal clustering output: one row per valid record, including
    singletons. Implemented and in production (`src/models/clustering.py`,
    Stage 5 of `src/pipeline.py`) — clusters the deterministic auto-merge
    edges only; see Data-Contract.md."""

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


def validate_fs_features(df: pd.DataFrame, *, allow_empty: bool = True) -> pd.DataFrame:
    """Validate the FSFeatures GBT parquet.

    Beyond the pandera `FSFeatures` base-column check, this asserts the frame
    carries at least one `gamma_<field>` and one `bf_<field>` feature column
    (their exact names are Splink-comparison-dependent, so they can't be
    enumerated in the strict schema). Returns the (coerced) frame.
    """
    if allow_empty and len(df) == 0:
        return df
    missing_base = [c for c in FS_FEATURES_REQUIRED_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"FSFeatures is missing required base column(s) {missing_base}; "
            f"have {sorted(df.columns)[:12]}..."
        )
    has_gamma = any(c.startswith("gamma_") for c in df.columns)
    has_bf = any(c.startswith("bf_") for c in df.columns)
    if not (has_gamma and has_bf):
        raise ValueError(
            "FSFeatures carries no per-field feature columns: expected at least "
            "one 'gamma_<field>' and one 'bf_<field>' column (found "
            f"gamma={has_gamma}, bf={has_bf}). These are the GBT's inputs."
        )
    return validate(df, FSFeatures, allow_empty=allow_empty)


#: Required base columns of the MLFeatures parquet — just the pair key, since
#: BYOF feature-column names/count are a custom FeatureBuilder's decision.
ML_FEATURES_REQUIRED_COLUMNS: tuple[str, ...] = (PATID_A, PATID_B)


def validate_ml_features(df: pd.DataFrame, *, allow_empty: bool = True) -> pd.DataFrame:
    """Validate the ML matcher's MLFeatures parquet.

    Looser than `validate_fs_features`: BYOF means feature-column names are
    unknowable ahead of time, so this only checks the pair key is present and
    defers to the pandera `MLFeatures` schema for everything else. Returns
    the (coerced) frame.
    """
    if allow_empty and len(df) == 0:
        return df
    missing_base = [c for c in ML_FEATURES_REQUIRED_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"MLFeatures is missing required base column(s) {missing_base}; "
            f"have {sorted(df.columns)[:12]}..."
        )
    return validate(df, MLFeatures, allow_empty=allow_empty)


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
    review_evidence: ArtifactRef | None = None
    # Stage-4 FS artifacts (present only when an active FS model scored the run):
    matches_model: ArtifactRef | None = None  # ProbabilisticMatches audit frame
    fs_features: ArtifactRef | None = None     # FSFeatures GBT candidate parquet
    # Stage-4.5 ML artifacts (present only when an active ML model scored the run):
    matches_ml: ArtifactRef | None = None     # ClassificationResults audit frame
    ml_features: ArtifactRef | None = None    # MLFeatures candidate parquet
    counts: dict[str, int] = Field(default_factory=dict)


__all__ = [
    "ArtifactRef",
    "CandidatePairs",
    "ClassificationResults",
    "CleanedRecords",
    "ClusterAssignments",
    "Edges",
    "FSFeatures",
    "MLFeatures",
    "Matches",
    "NonMatches",
    "ProbabilisticMatches",
    "Rejects",
    "RunManifest",
    "RULE_NAMES",
    "AUTO_MERGE_RULE_NAMES",
    "REJECT_RULE_NAMES",
    "FS_FEATURES_REQUIRED_COLUMNS",
    "ML_FEATURES_REQUIRED_COLUMNS",
    "CLEANED_REQUIRED_COLUMNS",
    "TIER_AUTO_MERGE",
    "TIER_HUMAN_REVIEW",
    "TIER_NO_MATCH",
    "CLASSIFICATION_TIERS",
    "MATCH_SOURCES",
    "assert_patid_coverage",
    "validate",
    "validate_fs_features",
    "validate_ml_features",
]
