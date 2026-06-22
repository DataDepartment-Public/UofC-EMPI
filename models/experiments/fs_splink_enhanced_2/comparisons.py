"""Comparison registry for fs_splink_enhanced_2.

Each comparison is declared as a `ComparisonSpec(name, builder)` so the
`ENHANCED_2_REGISTRY` lives at module-import time without paying the Splink
import cost. Builders return the Splink comparison dict directly.

Comparisons in declared order:

    Carried over from fs_splink_enhanced (verbatim shape; m is supervised here
    rather than EM-trained):
        FirstNM    — JW thresholds + TF + JW<0.5 mismatch
        LastNM     — TF + JW<0.5 mismatch + full_name_compact level (NEW)
        BirthDT    — exact / ±1 day / ±1 month / month-day-swap (NEW) / else
        SSN        — null / exact / last4 / 5-9 conflict / else
        Email      — null / exact / local-part-only / else
        Phones     — array intersect at [2, 1]
        ZIP        — null / exact / 3-prefix / else

    New comparisons added in Phase E2-3:
        MiddleNM         — null / exact / initial-match / mismatch
        Sex_positive     — null / exact / OTHER-either / M↔F mismatch
                           (m for M↔F is supervised; if synthetic has 0 positives
                           there, Splink reports m=0 — that's the desired
                           strong-negative weight, no manual clamp needed)
        LastNM_Phonetic  — null / DM-equal / DM-mismatch (reads `_dm_LastNM`)
        FirstNM_Phonetic — null / DM-equal / DM-mismatch (reads `_dm_FirstNM`)
        Household_discount — composite anti-evidence; gated on include_address

    The Address comparison and Household_discount composite are gated on the
    `include_address` flag (off for the synthetic sandbox, which doesn't
    carry CityNM_clean / StateCD_clean).
"""

from __future__ import annotations

import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import SettingsCreator

from models.common.fs_base import ComparisonRegistry, ComparisonSpec

# ── Column names ─────────────────────────────────────────────────────────────
COL_PATID = "PATID"
COL_FIRST_NM = "FirstNM_clean"
COL_LAST_NM = "LastNM_clean"
COL_MIDDLE_NM = "MiddleNM_clean"
COL_SSN = "SSN_clean"
COL_SSN_LAST4 = "last_4_SSN"
COL_EMAIL = "Email_clean"
COL_ZIP = "ZipCD_clean_base"
COL_SEX = "SexAtBirthDSC_clean"
COL_DOB_STR = "_dob_str"
COL_PHONES_ARRAY = "Phones_array"
COL_FULL_NAME_COMPACT = "full_name_compact"
COL_DM_LAST = "_dm_LastNM"
COL_DM_FIRST = "_dm_FirstNM"
COL_ADDRESS1 = "AddressLine1_clean"
COL_CITY = "CityNM_clean"
COL_STATE = "StateCD_clean"


# ─── Helper: extract one comparison dict from a SettingsCreator wrapper ──────
def _comparison_to_dict(creator) -> dict:
    """Run a one-comparison SettingsCreator and return the comparison dict."""
    s = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name=COL_PATID,
        comparisons=[creator],
    ).get_settings("duckdb").as_dict()
    return s["comparisons"][0]


def _insert_level_before_else(comp_dict: dict, new_level: dict) -> dict:
    """Insert `new_level` immediately before the trailing ElseLevel."""
    comp_dict["comparison_levels"].insert(-1, new_level)
    return comp_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Builders — one per comparison
# ═══════════════════════════════════════════════════════════════════════════════
def _build_first_nm() -> dict:
    d = _comparison_to_dict(
        cl.NameComparison(
            COL_FIRST_NM, jaro_winkler_thresholds=[0.92, 0.85],
        ).configure(term_frequency_adjustments=True)
    )
    return _insert_level_before_else(d, {
        "sql_condition": (
            f"jaro_winkler_similarity({COL_FIRST_NM}_l, {COL_FIRST_NM}_r) < 0.5 "
            f"AND {COL_FIRST_NM}_l IS NOT NULL AND {COL_FIRST_NM}_r IS NOT NULL"
        ),
        "label_for_charts": f"Jaro-Winkler distance of {COL_FIRST_NM} < 0.5",
    })


def _build_last_nm() -> dict:
    d = _comparison_to_dict(
        cl.NameComparison(COL_LAST_NM).configure(term_frequency_adjustments=True)
    )
    # JW<0.5 mismatch level (anti-evidence band).
    _insert_level_before_else(d, {
        "sql_condition": (
            f"jaro_winkler_similarity({COL_LAST_NM}_l, {COL_LAST_NM}_r) < 0.5 "
            f"AND {COL_LAST_NM}_l IS NOT NULL AND {COL_LAST_NM}_r IS NOT NULL"
        ),
        "label_for_charts": f"Jaro-Winkler distance of {COL_LAST_NM} < 0.5",
    })
    # full_name_compact equality — catches hyphenation / spacing variance
    # (e.g. MARTINEZ-CASTILLO ↔ MARTINEZCASTILLO) that JW alone misses. Placed
    # near the top so it can outscore JW≥0.92 when both surnames are present.
    d["comparison_levels"].insert(2, {
        "sql_condition": (
            f"{COL_FULL_NAME_COMPACT}_l = {COL_FULL_NAME_COMPACT}_r "
            f"AND {COL_FULL_NAME_COMPACT}_l IS NOT NULL"
        ),
        "label_for_charts": "full_name_compact exact match",
    })
    return d


def _build_birth_dt() -> dict:
    d = _comparison_to_dict(
        cl.DateOfBirthComparison(
            COL_DOB_STR,
            input_is_string=True,
            datetime_metrics=["day", "month"],
            datetime_thresholds=[1, 1],
        )
    )
    # Month-day-swap level: same year, day/month transposed. Diagnostic in
    # the audit (67 positives / 1 negative on synthetic train).
    return _insert_level_before_else(d, {
        "sql_condition": (
            f"substr({COL_DOB_STR}_l, 1, 4) = substr({COL_DOB_STR}_r, 1, 4) "
            f"AND substr({COL_DOB_STR}_l, 6, 2) = substr({COL_DOB_STR}_r, 9, 2) "
            f"AND substr({COL_DOB_STR}_l, 9, 2) = substr({COL_DOB_STR}_r, 6, 2) "
            f"AND substr({COL_DOB_STR}_l, 6, 2) <> substr({COL_DOB_STR}_l, 9, 2) "
            f"AND {COL_DOB_STR}_l IS NOT NULL AND {COL_DOB_STR}_r IS NOT NULL"
        ),
        "label_for_charts": "DOB month-day swap (same year)",
    })


def _build_ssn() -> dict:
    d = _comparison_to_dict(
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
    return _insert_level_before_else(d, {
        "sql_condition": (
            f"{COL_SSN}_l IS NOT NULL AND {COL_SSN}_r IS NOT NULL "
            f"AND {COL_SSN}_l != {COL_SSN}_r"
        ),
        "label_for_charts": "Both populated, full 9-digit mismatch",
    })


def _build_email() -> dict:
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
    return _comparison_to_dict(cl.ArrayIntersectAtSizes(COL_PHONES_ARRAY, [2, 1]))


def _build_zip() -> dict:
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


def _build_middle_nm() -> dict:
    """NEW (E2-3): MiddleNM agreement — exact / initial / mismatch / null."""
    return _comparison_to_dict(
        cl.CustomComparison(
            output_column_name="MiddleNM",
            comparison_levels=[
                cll.NullLevel(COL_MIDDLE_NM),
                cll.ExactMatchLevel(COL_MIDDLE_NM),
                cll.CustomLevel(
                    sql_condition=(
                        f"left({COL_MIDDLE_NM}_l, 1) = left({COL_MIDDLE_NM}_r, 1) "
                        f"AND {COL_MIDDLE_NM}_l IS NOT NULL "
                        f"AND {COL_MIDDLE_NM}_r IS NOT NULL"
                    ),
                    label_for_charts="First-initial match",
                ),
                cll.ElseLevel(),
            ],
        )
    )


def _build_sex_positive() -> dict:
    """NEW (E2-3): Sex as positive + anti-evidence comparison.

    M↔F mismatch is a strong negative (in the audit: 0 positives / 6,702
    negatives on synthetic — Splink will set m≈0 here, giving the desired
    strong-negative weight). OTHER-either preserves trans/nonbinary records
    by keeping the level less penalizing than M↔F.
    """
    return _comparison_to_dict(
        cl.CustomComparison(
            output_column_name="Sex_positive",
            comparison_levels=[
                cll.NullLevel(COL_SEX),
                cll.ExactMatchLevel(COL_SEX),
                cll.CustomLevel(
                    sql_condition=(
                        f"({COL_SEX}_l = 'OTHER' OR {COL_SEX}_r = 'OTHER') "
                        f"AND {COL_SEX}_l IS NOT NULL AND {COL_SEX}_r IS NOT NULL "
                        f"AND {COL_SEX}_l != {COL_SEX}_r"
                    ),
                    label_for_charts="OTHER on either side, different",
                ),
                cll.CustomLevel(
                    sql_condition=(
                        f"(({COL_SEX}_l = 'MALE' AND {COL_SEX}_r = 'FEMALE') "
                        f"OR ({COL_SEX}_l = 'FEMALE' AND {COL_SEX}_r = 'MALE'))"
                    ),
                    label_for_charts="MALE ↔ FEMALE mismatch",
                ),
                cll.ElseLevel(),
            ],
        )
    )


def _build_last_nm_phonetic() -> dict:
    """NEW (E2-3): LastNM Double-Metaphone equality. Reads `_dm_LastNM`
    persisted by the cleaning stage (Phase E2-2)."""
    return _comparison_to_dict(
        cl.CustomComparison(
            output_column_name="LastNM_Phonetic",
            comparison_levels=[
                cll.NullLevel(COL_DM_LAST),
                cll.ExactMatchLevel(COL_DM_LAST),
                cll.ElseLevel(),
            ],
        )
    )


def _build_first_nm_phonetic() -> dict:
    """NEW (E2-3): FirstNM Double-Metaphone equality. Reads `_dm_FirstNM`."""
    return _comparison_to_dict(
        cl.CustomComparison(
            output_column_name="FirstNM_Phonetic",
            comparison_levels=[
                cll.NullLevel(COL_DM_FIRST),
                cll.ExactMatchLevel(COL_DM_FIRST),
                cll.ElseLevel(),
            ],
        )
    )


def _build_household_discount() -> dict:
    """Composite anti-evidence — shared address OR phone, but name+DOB don't
    agree. Targets the family-same-household FP class. Gated on
    include_address."""
    return {
        "output_column_name": "Household_discount",
        "comparison_levels": [
            {
                "sql_condition": (
                    f"{COL_FIRST_NM}_l IS NULL OR {COL_FIRST_NM}_r IS NULL "
                    f"OR {COL_DOB_STR}_l IS NULL OR {COL_DOB_STR}_r IS NULL"
                ),
                "label_for_charts": "Household_discount is NULL",
                "is_null_level": True,
            },
            {
                "sql_condition": (
                    "("
                    f"  ({COL_ADDRESS1}_l = {COL_ADDRESS1}_r "
                    f"   AND {COL_ADDRESS1}_l IS NOT NULL)"
                    "  OR "
                    "  ("
                    f"     {COL_PHONES_ARRAY}_l IS NOT NULL "
                    f"     AND {COL_PHONES_ARRAY}_r IS NOT NULL "
                    f"     AND len(list_intersect({COL_PHONES_ARRAY}_l, {COL_PHONES_ARRAY}_r)) >= 1"
                    "  )"
                    ") "
                    f"AND jaro_winkler_similarity({COL_FIRST_NM}_l, {COL_FIRST_NM}_r) < 0.7 "
                    f"AND {COL_DOB_STR}_l != {COL_DOB_STR}_r"
                ),
                "label_for_charts": "Household indicator without identity match",
            },
            {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
        ],
    }


def _build_address() -> dict:
    """4-level Address comparison from the enhanced module. Gated on
    include_address."""
    return {
        "output_column_name": "Address",
        "comparison_levels": [
            {
                "sql_condition": (
                    f"{COL_ADDRESS1}_l IS NULL OR {COL_ADDRESS1}_r IS NULL"
                ),
                "label_for_charts": f"{COL_ADDRESS1} is NULL",
                "is_null_level": True,
            },
            {
                "sql_condition": f"{COL_ADDRESS1}_l = {COL_ADDRESS1}_r",
                "label_for_charts": f"Exact match on {COL_ADDRESS1}",
            },
            {
                "sql_condition": (
                    f"{COL_CITY}_l = {COL_CITY}_r "
                    f"AND {COL_STATE}_l = {COL_STATE}_r "
                    f"AND {COL_ZIP}_l = {COL_ZIP}_r "
                    f"AND {COL_CITY}_l IS NOT NULL "
                    f"AND {COL_STATE}_l IS NOT NULL "
                    f"AND {COL_ZIP}_l IS NOT NULL"
                ),
                "label_for_charts": "Same City + State + Zip",
            },
            {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Registry assembly
# ═══════════════════════════════════════════════════════════════════════════════
_BASE_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec("FirstNM",          _build_first_nm),
    ComparisonSpec("LastNM",           _build_last_nm),
    ComparisonSpec("BirthDT",          _build_birth_dt),
    ComparisonSpec("SSN",              _build_ssn),
    ComparisonSpec("Email",            _build_email),
    ComparisonSpec("Phones",           _build_phones),
    ComparisonSpec("ZIP",              _build_zip),
    ComparisonSpec("MiddleNM",         _build_middle_nm),
    ComparisonSpec("Sex_positive",     _build_sex_positive),
    ComparisonSpec("LastNM_Phonetic",  _build_last_nm_phonetic),
    ComparisonSpec("FirstNM_Phonetic", _build_first_nm_phonetic),
)

_ADDRESS_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec("Household_discount", _build_household_discount),
    ComparisonSpec("Address",            _build_address),
)


def build_registry(include_address: bool = True) -> ComparisonRegistry:
    """Return the enhanced_2 ComparisonRegistry.

    `include_address=False` drops the Address comparison and Household_discount
    composite — used by the synthetic sandbox (no CityNM_clean / StateCD_clean).
    """
    specs = _BASE_SPECS + (_ADDRESS_SPECS if include_address else ())
    return ComparisonRegistry(specs)


#: Default registry for production runs (with Address).
ENHANCED_2_REGISTRY: ComparisonRegistry = build_registry(include_address=True)
