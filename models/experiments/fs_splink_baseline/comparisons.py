"""Comparison registry for fs_splink_baseline.

Each comparison is declared as a `ComparisonSpec(name, builder)` where each
`builder` is a lazy callable that returns a Splink comparison dict (via
`_comparison_to_dict`).  Splink itself is imported eagerly at module level
because `_comparison_to_dict` calls `SettingsCreator` to materialise the dict.

Comparisons in declared order (matches the existing `build_settings()` output):

    FirstNM  — JW thresholds [0.92, 0.85] + TF
    LastNM   — default JW thresholds + TF
    BirthDT  — exact / ±1 day / ±1 month / else
    SSN      — null / exact / last-4 / else  (4-level)
    Email    — null / exact / same-username-diff-domain / else
    Phones   — array intersect at [2, 1]
    ZIP      — null / exact / 3-digit prefix / else

EM training rules and match-prevalence prior rules are also exported here
so `FSBaseline` can wire them into `EMTraining` without re-declaring them.
"""

from __future__ import annotations

import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import SettingsCreator

from models.common.fs_base import ComparisonRegistry, ComparisonSpec

# ── Column names (aligned with src.preprocessing.blocking COL_* constants) ──
COL_PATID = "PATID"
COL_FIRST_NM = "FirstNM_clean"
COL_LAST_NM = "LastNM_clean"
COL_SSN = "SSN_clean"
COL_SSN_LAST4 = "last_4_SSN"
COL_EMAIL = "Email_clean"
COL_ZIP = "ZipCD_clean_base"
COL_DOB_STR = "_dob_str"
COL_PHONES_ARRAY = "Phones_array"


# ─── Helper: extract one comparison dict from a SettingsCreator wrapper ──────
def _comparison_to_dict(creator) -> dict:
    """Run a one-comparison SettingsCreator and return the comparison dict."""
    s = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name=COL_PATID,
        comparisons=[creator],
    ).get_settings("duckdb").as_dict()
    return s["comparisons"][0]


# ═══════════════════════════════════════════════════════════════════════════════
# Builders — one per comparison, mirroring build_settings() in fellegi_sunter_baseline.py
# ═══════════════════════════════════════════════════════════════════════════════
def _build_first_nm() -> dict:
    """FirstNM: tightened JW thresholds [0.92, 0.85] + TF."""
    return _comparison_to_dict(
        cl.NameComparison(
            COL_FIRST_NM,
            jaro_winkler_thresholds=[0.92, 0.85],
        ).configure(term_frequency_adjustments=True)
    )


def _build_last_nm() -> dict:
    """LastNM: default JW thresholds + TF."""
    return _comparison_to_dict(
        cl.NameComparison(COL_LAST_NM).configure(term_frequency_adjustments=True)
    )


def _build_birth_dt() -> dict:
    """BirthDT: exact / ±1 day / ±1 month / else.

    The ±1-year level was removed (had m << u anti-evidence on the real cohort).
    """
    return _comparison_to_dict(
        cl.DateOfBirthComparison(
            COL_DOB_STR,
            input_is_string=True,
            datetime_metrics=["day", "month"],
            datetime_thresholds=[1, 1],
        )
    )


def _build_ssn() -> dict:
    """SSN: null / exact / last-4 / else (4-level)."""
    return _comparison_to_dict(
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
        )
    )


def _build_email() -> dict:
    """Email: null / exact / same-username-different-domain / else."""
    return _comparison_to_dict(
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
        )
    )


def _build_phones() -> dict:
    """Phones: array intersection at sizes [2, 1]."""
    return _comparison_to_dict(cl.ArrayIntersectAtSizes(COL_PHONES_ARRAY, [2, 1]))


def _build_zip() -> dict:
    """ZIP: null / 5-digit exact / 3-digit prefix / else."""
    return _comparison_to_dict(
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
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════
BASELINE_REGISTRY: ComparisonRegistry = ComparisonRegistry([
    ComparisonSpec("FirstNM", _build_first_nm),
    ComparisonSpec("LastNM",  _build_last_nm),
    ComparisonSpec("BirthDT", _build_birth_dt),
    ComparisonSpec("SSN",     _build_ssn),
    ComparisonSpec("Email",   _build_email),
    ComparisonSpec("Phones",  _build_phones),
    ComparisonSpec("ZIP",     _build_zip),
])


# ═══════════════════════════════════════════════════════════════════════════════
# EM training rules (mirrors train_model() in fellegi_sunter_baseline.py)
# ═══════════════════════════════════════════════════════════════════════════════
# Session 1 — SSN-exact anchor (~21% coverage).
SSN_BLOCK: str = (
    f"l.{COL_SSN} = r.{COL_SSN} AND l.{COL_SSN} IS NOT NULL"
)

# Session 2 — Email-exact anchor (~32% coverage).
EMAIL_BLOCK: str = (
    f"l.{COL_EMAIL} = r.{COL_EMAIL} AND l.{COL_EMAIL} IS NOT NULL"
)

# Session 3 — Soundex(FN)+Soundex(LN)+BirthYear (~99% coverage).
SOUNDEX_BLOCK: str = (
    "l._sx_FirstNM = r._sx_FirstNM AND l._sx_LastNM = r._sx_LastNM "
    "AND l._birth_year = r._birth_year"
)

# EM blocking rules — three sessions in train_model() order.
EM_BLOCKING_RULES: list[str] = [SSN_BLOCK, EMAIL_BLOCK, SOUNDEX_BLOCK]

# Prior rule: Double-Metaphone(LastNM) + exact DOB anchor.
DM_LAST_DOB_BLOCK: str = (
    "l._dm_LastNM = r._dm_LastNM AND l._dob_str = r._dob_str "
    "AND l._dm_LastNM IS NOT NULL AND l._dob_str IS NOT NULL"
)

# Match-prevalence prior rules (recall=0.80 in train_model()).
PRIOR_RULES: list[str] = [SSN_BLOCK, EMAIL_BLOCK, DM_LAST_DOB_BLOCK]
