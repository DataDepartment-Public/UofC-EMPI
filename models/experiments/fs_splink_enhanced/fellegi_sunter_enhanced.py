"""
fellegi_sunter_enhanced.py — Fellegi-Sunter probabilistic record linkage baseline
for the AllianceChicago eMPI platform (Phase 5).

Consumes the pre-computed candidate pairs from `blocking.py` / `run_blocking.py`
and scores each pair with a graded, multi-field Splink comparison model, then
maps the resulting match probabilities to a three-tier classification
(auto_merge / human_review / no_match).

PHASE
-----
Phase A (calibration & debugging build) per the implementation plan, Section 0.
`retain_intermediate_calculation_columns` and `retain_matching_columns` are
True throughout for auditability; output-size trimming and model-artifact
persistence are deferred to Phase B.

ENVIRONMENT
-----------
    pip install "splink>=4.0.16,<5.0.0" --break-system-packages
Resolved/verified against: splink==4.0.16, duckdb==1.5.3 (DuckDB bundled).

DEPENDENCIES
------------
    splink>=4.0.16,<5.0.0
    blocking.py (same package) — for _compute_derived_columns() and COL_* constants

HIPAA NOTE
----------
This module never writes PHI to logs (aggregate counts only), mirroring
blocking.py. The scored output retains intermediate columns for calibration;
when porting to Phase B, set both retain_* flags False to drop raw identifier
values from the output.

PUBLIC API
----------
    load_candidate_pairs(path)                       -> pd.DataFrame
    prepare_model_input(df_clean)                    -> pd.DataFrame
    build_settings()                                 -> SettingsCreator
    build_linker(df_model, candidate_pairs_df)       -> Linker
    train_model(linker)                              -> Linker
    predict_pairs(linker, candidate_pairs_df)        -> pd.DataFrame
    classify_pairs(df_predictions, ...)              -> pd.DataFrame
    to_evaluation_schema(df_classified)              -> pd.DataFrame
    get_model_diagnostics(linker)                    -> dict
    run_fs_enhanced(candidate_pairs_path, df_clean)  -> pd.DataFrame

OUTPUT CONTRACT
---------------
run_fs_enhanced() returns the standardized five-column evaluation frame by
default — PATID_A, PATID_B, model_name, score, predicted_tier — so this model
is poolable/comparable with the other approaches under experimentation/.
`score` is the match probability in [0, 1]. Pass full_output=True for the rich
Splink frame used in this model's own Phase A calibration.
"""

from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
from typing import Any

import pandas as pd

from splink import DuckDBAPI, Linker, SettingsCreator
import splink.comparison_library as cl
import splink.comparison_level_library as cll

# Reuse blocking.py as the single source of truth for derived columns + names.
# NOTE: path matches the conftest.py convention (PROJECT_ROOT on sys.path).
# If blocking.py lives elsewhere in your tree, update this single import.
from src.features.blocking import (
    _compute_derived_columns,
    COL_PATID,
    COL_FIRST_NM,
    COL_LAST_NM,
    COL_SSN,
    COL_SSN_LAST4,
    COL_ZIP,
    COL_EMAIL,
    _COL_DOB_STR,
    _COL_PHONES_SET,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column constants (aligned with blocking.py; new columns this module adds)
# ---------------------------------------------------------------------------
COL_PHONES_ARRAY = "Phones_array"  # DuckDB LIST built from _phones_parsed

# Candidate-pairs (blocking output) schema
CP_PATID_A = "PATID_A"
CP_PATID_B = "PATID_B"
CP_SOURCE_BLOCKS = "source_blocks"
CP_N_BLOCKS = "n_blocks"

CANDIDATE_PAIRS_TABLE = "candidate_pairs"

# Required columns prepare_model_input() must see *after* derivation, i.e. the
# evidence the FS settings depend on. Missing any => _compute_derived_columns()
# was not run upstream, or the cleaned parquet is malformed.
REQUIRED_MODEL_COLUMNS = [COL_PATID, _COL_DOB_STR, COL_SSN_LAST4, COL_ZIP]

# Default classification thresholds (plan Section 8 starting points; calibrate
# against real score distributions before locking these down).
DEFAULT_AUTO_MERGE_THRESHOLD = 0.90
DEFAULT_REVIEW_FLOOR = 0.50

# ---------------------------------------------------------------------------
# Standardized evaluation output contract.
#
# This module is one of several competing approaches in the experimentation
# phase. To be poolable and comparable across models, run_fs_enhanced() returns
# exactly these five columns (in this order) by default. The rich Splink frame
# (gamma_* comparison levels, match_weight, retained identifier columns) remains
# available via run_fs_enhanced(..., full_output=True) for this model's own
# Phase A calibration/debugging, but is NOT the evaluation-facing contract.
#
#   PATID_A         canonical lower PATID  (string sort, matches blocking.py)
#   PATID_B         canonical higher PATID
#   model_name      self-identifying tag so pooled rows know their origin
#   score           match_probability in [0, 1] (probability, not log-odds weight)
#   predicted_tier  auto_merge / human_review / no_match
# ---------------------------------------------------------------------------
MODEL_NAME = "fs_splink_enhanced"
EVAL_SCHEMA_COLUMNS = ["PATID_A", "PATID_B", "model_name", "score", "predicted_tier"]

# ---------------------------------------------------------------------------
# Candidate-pairs blocking rule (plan Section 5).
#
# Restricts Splink's comparison to exactly the pre-computed candidate pairs.
#
# IMPORTANT — single equi-condition only, NO OR (Section 5.2 performance
# finding, post-deployment): an EXISTS subquery with `(A AND B) OR (B AND A)`
# prevents DuckDB from decorrelating the subquery into a hash join. DuckDB
# instead builds the FULL l x r self-join cross product (13.3B pairs at
# 163K records) and then nested-loop-joins that against candidate_pairs —
# computationally infeasible (BLOCKWISE_NL_JOIN in EXPLAIN).
#
# Dropping the OR lets DuckDB decorrelate to a single HASH_JOIN keyed on
# (PATID_A, PATID_B) = (l.PATID, r.PATID) — verified via EXPLAIN. This relies
# on blocking.py's canonical pair ordering (Python string sort: PATID_A <
# PATID_B) matching Splink's dedupe_only ordering (`l.PATID < r.PATID`, also
# a DuckDB string comparison) — both are byte-wise lexicographic on PATID, so
# they agree. blocking.py's run_batch_blocking() canonicalizes every pair this
# way, so every candidate pair satisfies PATID_A < PATID_B = l.PATID < r.PATID,
# and the second (reversed) EXISTS branch was always dead code.
#
# IMPORTANT (see _PAIR_SHIM_COLUMNS below): because this rule references the
# external `candidate_pairs` columns PATID_A / PATID_B, Splink's
# retain_matching_columns logic harvests those tokens and tries to SELECT them
# from the patient table. We add them as inert placeholder columns so that
# binding succeeds; predict_pairs() drops the resulting junk columns.
# ---------------------------------------------------------------------------
CANDIDATE_PAIRS_BLOCKING_RULE = f"""
EXISTS (
    SELECT 1 FROM {CANDIDATE_PAIRS_TABLE} cp
    WHERE cp.{CP_PATID_A} = l.{COL_PATID} AND cp.{CP_PATID_B} = r.{COL_PATID}
)
"""

# Inert placeholder columns added to df_model purely so Splink's blocking-rule
# column harvesting binds (see CANDIDATE_PAIRS_BLOCKING_RULE). They carry no
# information and are dropped from the prediction output.
_PAIR_SHIM_COLUMNS = [CP_PATID_A, CP_PATID_B]


# ===========================================================================
# 1. Inputs
# ===========================================================================
def load_candidate_pairs(path: str | Path) -> pd.DataFrame:
    """
    Load the candidate-pairs parquet produced by blocking.run_batch_blocking().

    Returns a DataFrame with columns PATID_A, PATID_B, source_blocks, n_blocks.
    Raises a clear error if the file is missing or the schema is unexpected.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate-pairs file not found: {path}")
    df = pd.read_parquet(path)
    missing = {CP_PATID_A, CP_PATID_B} - set(df.columns)
    if missing:
        raise ValueError(
            f"Candidate-pairs file {path} is missing required columns: {sorted(missing)}"
        )
    logger.info("Loaded %d candidate pairs from %s", len(df), path)
    return df


def prepare_model_input(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the cleaned patient index into the frame the Splink linker consumes.

    Steps:
      1. Re-derive the transient columns blocking.py computes but does not
         persist (`_dob_str`, `_birth_year`, `_dm_LastNM`, `_sx_LastNM`,
         `_sx_FirstNM`, `_phones_parsed`, `_fn_prefix3`) via the idempotent
         `_compute_derived_columns()` import.
      2. Build `Phones_array` (a DuckDB-friendly sorted LIST) from the
         `_phones_parsed` set — `cl.ArrayIntersectAtSizes` needs a LIST, not a set.
      3. Add inert PATID_A / PATID_B placeholder columns (see
         `_PAIR_SHIM_COLUMNS`) so the candidate-pairs blocking rule binds under
         `retain_matching_columns=True`.
      4. Validate that the columns the FS settings depend on are present.

    Parameters
    ----------
    df_clean : pd.DataFrame
        The cleaned patient index (same frame passed to build_blocking_index()).
        Must NOT already contain the derived columns — they are regenerated here.

    Returns
    -------
    pd.DataFrame
        Model-ready frame with derived columns, Phones_array, and shim columns.
    """
    df = _compute_derived_columns(df_clean)  # returns a copy; caller's frame is not mutated

    # _phones_parsed (set) -> Phones_array (sorted list); empty set -> [].
    # Guard with isinstance: pandas treats NaN as truthy, so `if s else []` would
    # send a NaN into sorted() if _parse_phone_set's set-output invariant ever drifts.
    df[COL_PHONES_ARRAY] = df[_COL_PHONES_SET].apply(
        lambda s: sorted(s) if isinstance(s, set) and s else []
    )

    # Inert placeholders so Splink's retain_matching_columns harvesting binds.
    for shim in _PAIR_SHIM_COLUMNS:
        if shim not in df.columns:
            df[shim] = pd.NA

    _validate_required_columns(df)
    logger.info(
        "prepare_model_input: %d records, %d columns ready for linker",
        len(df), df.shape[1],
    )
    return df


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


# ===========================================================================
# 2. Settings (plan Section 4)
# ===========================================================================
def build_settings() -> SettingsCreator:
    """
    Build the Splink settings: dedupe_only, candidate-pairs blocking rule, and
    the graded multi-field comparison vector with TF adjustments on names/email.

    Notes
    -----
    * `source_blocks` / `n_blocks` are pair-level columns living on the
      candidate_pairs table, NOT record-level columns on df_model, so they are
      intentionally NOT in `additional_columns_to_retain` (that harvests from
      the record table and would raise a binder error). predict_pairs() merges
      them back onto the output instead.
    * `_dob_str` is "YYYY-MM-DD"; verified directly compatible with
      `input_is_string=True` (Splink emits TRY_STRPTIME(..., '%Y-%m-%d')) — no
      explicit `datetime_format` needed.
    """
    return SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name=COL_PATID,
        blocking_rules_to_generate_predictions=[CANDIDATE_PAIRS_BLOCKING_RULE],
        comparisons=[
            # FirstNM: tightened JW thresholds + TF.
            # Plan R1: the previous JW >= 0.7 fuzzy level had m < u on the real
            # cohort (Phase A diagnostic) because "kinda starts the same" carries
            # no marginal evidence past the Soundex blocking key. Tightening to
            # [0.92, 0.85] makes the fuzzy bands correspond to "typo / nickname"
            # rather than "loosely similar."
            cl.NameComparison(
                COL_FIRST_NM,
                jaro_winkler_thresholds=[0.92, 0.85],
            ).configure(term_frequency_adjustments=True),
            # LastNM: default thresholds (this level was not flagged in Phase A).
            cl.NameComparison(COL_LAST_NM).configure(term_frequency_adjustments=True),
            # DOB: exact / ±1 day / ±1 month / else.
            # Plan R1: the ±1-year level had m << u (anti-evidence) — drop it.
            # ±1 day and ±1 month still cover transposition errors.
            cl.DateOfBirthComparison(
                _COL_DOB_STR,
                input_is_string=True,
                datetime_metrics=["day", "month"],
                datetime_thresholds=[1, 1],
            ),
            # SSN: graded — full exact > last-4 match > else. (Unchanged from Phase A.)
            cl.CustomComparison(
                output_column_name="SSN",
                comparison_levels=[
                    cll.NullLevel(COL_SSN),
                    cll.ExactMatchLevel(COL_SSN),
                    cll.CustomLevel(
                        sql_condition=(
                            f"{COL_SSN_LAST4}_l IS NOT NULL "
                            f"AND {COL_SSN_LAST4}_r IS NOT NULL "
                            f"AND {COL_SSN_LAST4}_l = {COL_SSN_LAST4}_r"
                        ),
                        label_for_charts="Last 4 digits match",
                    ),
                    cll.ElseLevel(),
                ],
            ),
            # Email: custom hierarchy that drops the JW>0.88-username fuzzy
            # band (had m < u on the real cohort).
            # Levels: null / exact email / exact username (different domain) / else.
            cl.CustomComparison(
                output_column_name="Email",
                comparison_levels=[
                    cll.NullLevel(COL_EMAIL),
                    cll.ExactMatchLevel(COL_EMAIL, term_frequency_adjustments=True),
                    cll.CustomLevel(
                        sql_condition=(
                            f"split_part({COL_EMAIL}_l, '@', 1) "
                            f"= split_part({COL_EMAIL}_r, '@', 1) "
                            f"AND {COL_EMAIL}_l IS NOT NULL "
                            f"AND {COL_EMAIL}_r IS NOT NULL"
                        ),
                        label_for_charts="Exact username (different domain)",
                    ),
                    cll.ElseLevel(),
                ],
            ),
            # Phone: array intersection over the consolidated phone set.
            cl.ArrayIntersectAtSizes(COL_PHONES_ARRAY, [2, 1]),
            # ZIP: null / 5-digit exact / first 3 digits match / else.
            # Plan R1 originally dropped the 3-digit-prefix level on the theory
            # that its m≈u meant "no signal." The 2026-06-15 VM rerun showed
            # this was wrong for THIS cohort: ~40% of candidate pairs share a
            # regional 3-digit ZIP, so removing the prefix level cascades those
            # pairs into Else (m=0.48 / u=0.99 → -1.04 bits anti-evidence) and
            # crushes scores for true matches with partial-ZIP overlap. The
            # 3-digit prefix level carries near-zero weight (which is what we
            # want) and prevents the negative-evidence cascade.
            cl.CustomComparison(
                output_column_name="ZIP",
                comparison_levels=[
                    cll.NullLevel(COL_ZIP),
                    cll.ExactMatchLevel(COL_ZIP),
                    cll.CustomLevel(
                        sql_condition=f"left({COL_ZIP}_l, 3) = left({COL_ZIP}_r, 3)",
                        label_for_charts="First 3 digits match",
                    ),
                    cll.ElseLevel(),
                ],
            ),
            # NOTE: a brand-new Address comparison was introduced and then
            # reverted on 2026-06-15. EM trained large positive weights on the
            # Address levels using u-estimates from random pairs, but candidate
            # pairs are heavily preselected by the blocking layer for likely-
            # same-household pairs (families, namesakes, roommates), so Address
            # agreement is far more common in the candidate pool than in random
            # pairs. The trained weights over-rewarded address-sharing pairs,
            # pushing ~30K borderline non-matches into human_review. Until we
            # have ground-truth labels (from the §9 sampling work) to calibrate
            # Address m/u directly, the comparison stays out. The R2 plan
            # remains the target for a follow-up phase.
        ],
        # Phase A: keep full audit detail. Phase B trims both to reduce size.
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
    )


# ===========================================================================
# 3. Linker (plan Section 5 integration)
# ===========================================================================
def build_linker(
    df_model: pd.DataFrame,
    candidate_pairs_df: pd.DataFrame,
    db_api: DuckDBAPI | None = None,
) -> Linker:
    """
    Construct the Splink Linker with the candidate_pairs table registered in the
    same DuckDB connection the blocking rule queries.
    """
    db_api = db_api or DuckDBAPI()
    _register_candidate_pairs(db_api, candidate_pairs_df)
    linker = Linker(df_model, build_settings(), db_api=db_api)
    logger.info(
        "Linker built: %d records, %d candidate pairs registered",
        len(df_model), len(candidate_pairs_df),
    )
    return linker


def _register_candidate_pairs(db_api: DuckDBAPI, candidate_pairs_df: pd.DataFrame) -> None:
    """Register candidate_pairs as a queryable table in the DuckDB connection."""
    con = getattr(db_api, "_con", None)
    if con is None:  # defensive: API shape changed
        raise RuntimeError(
            "Could not access the DuckDB connection on DuckDBAPI (_con). "
            "Splink internals may have changed; verify against the installed version."
        )
    con.register(CANDIDATE_PAIRS_TABLE, candidate_pairs_df)


# ===========================================================================
# 4. Training (plan Section 6)
# ===========================================================================
def train_model(linker: Linker, u_max_pairs: float = 1e6, seed: int = 42) -> Linker:
    """
    Train the model in place: global u-estimation, three complementary EM
    sessions (plan Section 6.2), and the match-prevalence prior (Section 6.3).

    Parameters
    ----------
    u_max_pairs : float
        Random-sampling budget for u-estimation. Plan default is 1e6. NOTE: on a
        single-CPU host, Splink's DuckDB u-sampling path requires cpu_count()>1
        when max_pairs > 1e4 (it salts with cpu_count() partitions); see
        _estimate_u_with_guard().
    seed : int
        Seed for reproducible u-sampling (regression-test stability).
    """
    _estimate_u_with_guard(linker, u_max_pairs, seed)

    # --- Three EM sessions with complementary blocking rules (Section 6.2) ---
    # Session 1 — SSN-exact anchor (high precision, ~21% coverage).
    linker.training.estimate_parameters_using_expectation_maximisation(
        f"l.{COL_SSN} = r.{COL_SSN} AND l.{COL_SSN} IS NOT NULL"
    )
    # Session 2 — Email-exact anchor (~32% coverage).
    linker.training.estimate_parameters_using_expectation_maximisation(
        f"l.{COL_EMAIL} = r.{COL_EMAIL} AND l.{COL_EMAIL} IS NOT NULL"
    )
    # Session 3 — Soundex(FN)+Soundex(LN)+BirthYear (broad ~99% coverage).
    linker.training.estimate_parameters_using_expectation_maximisation(
        "l._sx_FirstNM = r._sx_FirstNM AND l._sx_LastNM = r._sx_LastNM "
        "AND l._birth_year = r._birth_year"
    )

    # --- Match-prevalence prior (Section 6.3, Option B: recall=0.80) ---
    linker.training.estimate_probability_two_random_records_match(
        deterministic_matching_rules=[
            f"l.{COL_SSN} = r.{COL_SSN} AND l.{COL_SSN} IS NOT NULL",
            f"l.{COL_EMAIL} = r.{COL_EMAIL} AND l.{COL_EMAIL} IS NOT NULL",
            "l._dm_LastNM = r._dm_LastNM AND l._dob_str = r._dob_str "
            "AND l._dm_LastNM IS NOT NULL AND l._dob_str IS NOT NULL",
        ],
        recall=0.80,
    )

    _warn_on_weight_inversions(linker)
    logger.info("train_model: training complete (u + 3 EM sessions + prevalence prior)")
    return linker


def _estimate_u_with_guard(linker: Linker, max_pairs: float, seed: int) -> None:
    """
    Run u-estimation, converting Splink's cryptic single-CPU salting error into
    an actionable message.

    Splink salts the DuckDB u-sampling join with multiprocessing.cpu_count()
    partitions when max_pairs > 1e4; on a 1-CPU host that becomes 1 partition
    and Splink raises "Salting partitions must be specified and > 1".
    """
    try:
        linker.training.estimate_u_using_random_sampling(max_pairs=max_pairs, seed=seed)
    except ValueError as exc:
        if "Salting partitions" in str(exc) and multiprocessing.cpu_count() <= 1:
            raise RuntimeError(
                "u-estimation failed because this host reports a single CPU and "
                f"u_max_pairs={max_pairs:g} (>1e4) triggers Splink's salted DuckDB "
                "sampling path, which needs cpu_count() > 1. Run on a multi-core "
                "allocation, or call train_model(linker, u_max_pairs=1e4)."
            ) from exc
        raise


def _warn_on_weight_inversions(linker: Linker) -> None:
    """
    Flag any comparison level where m < u (weight sign inverts) — typically a
    sign EM was under-determined on a sparse field (plan Section 6.4). Logs a
    warning so the team can apply a literature-informed m override if needed.
    """
    try:
        records = linker._settings_obj._parameters_as_detailed_records
    except AttributeError as exc:
        # Splink private API changed. Diagnostics must never break training,
        # but we want the silent-pass condition to be observable in the logs.
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
                "(m=%.4g, u=%.4g) — consider a literature-informed m override "
                "(plan Section 6.4).",
                r.get("comparison_name"), r.get("label_for_charts"), m, u,
            )


# ===========================================================================
# 5. Prediction & classification (plan Sections 7-8)
# ===========================================================================
def predict_pairs(linker: Linker, candidate_pairs_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Score the candidate pairs and return a tidy prediction frame.

    Post-processing:
      * Maps Splink's PATID_l / PATID_r to canonical PATID_A / PATID_B.
      * Drops the inert shim passthrough columns (PATID_A_l/r, PATID_B_l/r).
      * Re-attaches pair-level source_blocks / n_blocks from candidate_pairs_df
        (which could not ride through additional_columns_to_retain).
    """
    df = linker.inference.predict().as_pandas_dataframe()

    # Drop the inert shim passthrough columns.
    junk = [f"{c}{suf}" for c in _PAIR_SHIM_COLUMNS for suf in ("_l", "_r")]
    df = df.drop(columns=[c for c in junk if c in df.columns])

    # Canonical PATID_A / PATID_B from Splink's _l / _r.
    a = df[["PATID_l", "PATID_r"]].min(axis=1)
    b = df[["PATID_l", "PATID_r"]].max(axis=1)
    df.insert(0, CP_PATID_A, a)
    df.insert(1, CP_PATID_B, b)

    # Re-attach pair-level metadata if available.
    if candidate_pairs_df is not None:
        meta_cols = [c for c in (CP_SOURCE_BLOCKS, CP_N_BLOCKS) if c in candidate_pairs_df.columns]
        if meta_cols:
            df = df.merge(
                candidate_pairs_df[[CP_PATID_A, CP_PATID_B, *meta_cols]],
                on=[CP_PATID_A, CP_PATID_B], how="left",
            )

    logger.info("predict_pairs: scored %d pairs", len(df))
    return df


def classify_pairs(
    df_predictions: pd.DataFrame,
    auto_merge_threshold: float = DEFAULT_AUTO_MERGE_THRESHOLD,
    review_floor: float = DEFAULT_REVIEW_FLOOR,
) -> pd.DataFrame:
    """
    Add a `classification_tier` column. Pure post-processing — no retraining —
    so thresholds can be tuned cheaply and unit-tested in isolation.

    Tier boundaries (plan Section 8):
        match_probability >= auto_merge_threshold        -> "auto_merge"
        review_floor <= match_probability < auto_merge   -> "human_review"
        match_probability < review_floor                 -> "no_match"
    """
    if not 0.0 <= review_floor <= auto_merge_threshold <= 1.0:
        raise ValueError(
            f"Need 0 <= review_floor ({review_floor}) <= auto_merge_threshold "
            f"({auto_merge_threshold}) <= 1."
        )
    p = df_predictions["match_probability"]
    tier = pd.Series("no_match", index=df_predictions.index, dtype="object")
    tier = tier.mask(p >= review_floor, "human_review")
    tier = tier.mask(p >= auto_merge_threshold, "auto_merge")
    out = df_predictions.copy()
    out["classification_tier"] = tier
    logger.info(
        "classify_pairs: %s",
        {k: int(v) for k, v in out["classification_tier"].value_counts().items()},
    )
    return out


def to_evaluation_schema(df_classified: pd.DataFrame) -> pd.DataFrame:
    """
    Project the rich classified frame down to the standardized five-column
    evaluation contract so this model's output can be pooled and compared with
    the other approaches in the experimentation phase.

    Mapping:
        PATID_A, PATID_B    <- canonical IDs already on the classified frame
        model_name          <- MODEL_NAME constant (self-identifying tag)
        score               <- match_probability (probability in [0, 1])
        predicted_tier      <- classification_tier

    Parameters
    ----------
    df_classified : pd.DataFrame
        Output of classify_pairs() (must carry PATID_A, PATID_B,
        match_probability, classification_tier).

    Returns
    -------
    pd.DataFrame
        Exactly EVAL_SCHEMA_COLUMNS, in order.
    """
    required = {CP_PATID_A, CP_PATID_B, "match_probability", "classification_tier"}
    missing = required - set(df_classified.columns)
    if missing:
        raise ValueError(
            f"to_evaluation_schema is missing columns {sorted(missing)}; "
            "expected the output of classify_pairs()."
        )
    out = pd.DataFrame(
        {
            "PATID_A": df_classified[CP_PATID_A].values,
            "PATID_B": df_classified[CP_PATID_B].values,
            "model_name": MODEL_NAME,
            "score": df_classified["match_probability"].values,
            "predicted_tier": df_classified["classification_tier"].values,
        }
    )
    return out[EVAL_SCHEMA_COLUMNS]


# ===========================================================================
# 6. Diagnostics
# ===========================================================================
def get_model_diagnostics(linker: Linker) -> dict[str, Any]:
    """
    Return a non-PHI diagnostics bundle: trained settings (m/u + thresholds),
    per-level detailed parameter records, and the per-EM-session estimate
    records used for the cross-session comparison (plan Section 6.2 diagnostic).
    """
    diagnostics: dict[str, Any] = {}
    try:
        diagnostics["trained_settings"] = linker.misc.save_model_to_json()
    except AttributeError as exc:  # pragma: no cover - defensive
        logger.warning("Splink save_model_to_json missing: %s", exc)
        diagnostics["trained_settings_error"] = str(exc)
    try:
        diagnostics["parameter_records"] = linker._settings_obj._parameters_as_detailed_records
    except AttributeError as exc:  # pragma: no cover - defensive
        logger.warning(
            "Splink _parameters_as_detailed_records missing: %s", exc,
        )
        diagnostics["parameter_records_error"] = str(exc)
    try:
        diagnostics["per_session_estimates"] = linker._settings_obj._parameter_estimates_as_records
    except AttributeError as exc:  # pragma: no cover - defensive
        logger.warning(
            "Splink _parameter_estimates_as_records missing: %s", exc,
        )
    diagnostics["probability_two_random_records_match"] = getattr(
        linker._settings_obj, "_probability_two_random_records_match", None
    )
    return diagnostics


# ===========================================================================
# 7. Orchestration
# ===========================================================================
def run_fs_enhanced(
    candidate_pairs_path: str | Path,
    df_clean: pd.DataFrame,
    auto_merge_threshold: float = DEFAULT_AUTO_MERGE_THRESHOLD,
    review_floor: float = DEFAULT_REVIEW_FLOOR,
    u_max_pairs: float = 1e6,
    full_output: bool = False,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    End-to-end entry point: candidate pairs + cleaned index -> scored pairs.

    Output contract
    ---------------
    By default returns the standardized five-column evaluation frame
    (EVAL_SCHEMA_COLUMNS: PATID_A, PATID_B, model_name, score, predicted_tier)
    so this model is directly comparable with the other approaches in the
    experimentation phase. `score` is the match probability in [0, 1].

    Set ``full_output=True`` to instead receive the rich classified frame
    (canonical IDs + all retained Splink gamma_* / match_weight / identifier
    columns + classification_tier) for this model's own Phase A calibration.

    Set ``return_diagnostics=True`` to also receive the non-PHI diagnostics
    bundle (trained m/u parameters, per-EM-session estimates) as the second
    element of a (frame, diagnostics) tuple. This composes with either output
    shape.
    """
    candidate_pairs = load_candidate_pairs(candidate_pairs_path)
    df_model = prepare_model_input(df_clean)
    linker = build_linker(df_model, candidate_pairs)
    train_model(linker, u_max_pairs=u_max_pairs)
    predictions = predict_pairs(linker, candidate_pairs)
    classified = classify_pairs(predictions, auto_merge_threshold, review_floor)

    result = classified if full_output else to_evaluation_schema(classified)
    if return_diagnostics:
        return result, get_model_diagnostics(linker)
    return result