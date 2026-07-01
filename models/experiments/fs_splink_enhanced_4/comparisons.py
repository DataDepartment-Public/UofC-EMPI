"""Comparison registry for fs_splink_enhanced_4.

Enhanced_4 extends enhanced_3's interpretable two-level design by adding
**explicit conflict-vs-missing mismatch levels** to each identity field.
Enhanced_3's weakness was that a conflicting SSN, email, address, or phone fell
into the uninformative "else" level and contributed almost no negative weight —
only BirthDT disagreement bit hard (−8.93 bits). Enhanced_4 fixes this
structurally:

    For every field:
        null                   → no evidence (zero weight), exact as enhanced_3
        exact / near-match     → positive evidence (Bayes factor > 1)
        explicit hard-conflict → negative evidence (Bayes factor < 1, m ≈ 0)
        else                   → weak residual

The corroboration gate (requiring ≥2 independent fields before auto-merge) and
the Household_discount composite address the precision side; m stays purely
supervised from the real-cohort silver labels, exactly as enhanced_3.

Ten comparisons in declared order:
    BASE (also present in the synthetic sandbox):
        FirstNM    — null / exact(TF) / JW≥0.92 / JW≥0.88 / JW<0.5 conflict / else
        LastNM     — null / exact(TF) / JW≥0.95 / full_name_compact / JW≥0.88 /
                     DM-phonetic / JW<0.5 conflict / else
        BirthDT    — null / exact / month-day-swap / ±1 day / ±1 month /
                     ±1 year / else
        SSN        — null / exact-9 / last-4 match / full-9 conflict / else
        Email      — null / exact(TF) / same-username / JW≥0.88 / else
        Phones     — null+empty / ≥2 shared / ≥1 shared / else
        ZIP        — null / 5-digit exact / 3-prefix / else

    ADDRESS-GATED (requires CityNM_clean / StateCD_clean — off for sandbox):
        State              — null / exact / else
        Household_discount — composite anti-evidence
        Address            — null / exact street / Same City+State+Zip / else

`PRIOR_RULES` and `DEFAULT_PRIOR_RECALL` are copied verbatim from
fs_splink_enhanced_3 so the lambda-seeding training procedure is identical.
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
COL_DOB_STR = "_dob_str"
COL_SSN = "SSN_clean"
COL_SSN_LAST4 = "last_4_SSN"
COL_EMAIL = "Email_clean"
COL_ZIP = "ZipCD_clean_base"
COL_SEX = "SexAtBirthDSC_clean"
COL_PHONES_ARRAY = "Phones_array"
COL_FULL_NAME_COMPACT = "full_name_compact"
COL_DM_LAST = "_dm_LastNM"
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
    """FirstNM: null / exact(TF) / JW≥0.92 / JW≥0.88 / JW<0.5 conflict / else."""
    d = _comparison_to_dict(
        cl.NameComparison(
            COL_FIRST_NM, jaro_winkler_thresholds=[0.92, 0.88],
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
    """LastNM: null / exact(TF) / JW bands / full_name_compact / DM-phonetic /
    JW<0.5 conflict / else.

    Level order (first match wins):
      1. null
      2. exact (TF)        — JW produces this level at the top
      3. full_name_compact — catches hyphenation/spacing (inserted at index 2)
      4. JW≥0.95
      5. JW≥0.88
      6. DM-phonetic equal — same-sound different-spelling after JW bands
      7. JW<0.5 hard-conflict
      8. else
    """
    d = _comparison_to_dict(
        cl.NameComparison(COL_LAST_NM, jaro_winkler_thresholds=[0.95, 0.88]).configure(
            term_frequency_adjustments=True
        )
    )
    # Insert JW<0.5 hard-conflict before else (position -1).
    _insert_level_before_else(d, {
        "sql_condition": (
            f"jaro_winkler_similarity({COL_LAST_NM}_l, {COL_LAST_NM}_r) < 0.5 "
            f"AND {COL_LAST_NM}_l IS NOT NULL AND {COL_LAST_NM}_r IS NOT NULL"
        ),
        "label_for_charts": f"Jaro-Winkler distance of {COL_LAST_NM} < 0.5",
    })
    # Insert DM-phonetic BEFORE the JW<0.5 level (now at -2) so JW bands win
    # first, phonetic catches same-sound/different-spelling, then hard-conflict.
    d["comparison_levels"].insert(-2, {
        "sql_condition": (
            f"{COL_DM_LAST}_l = {COL_DM_LAST}_r "
            f"AND {COL_DM_LAST}_l IS NOT NULL"
        ),
        "label_for_charts": "LastNM Double-Metaphone match",
    })
    # full_name_compact equality — catches hyphenation/spacing variance
    # (e.g. MARTINEZ-CASTILLO ↔ MARTINEZCASTILLO). Placed near the top at
    # index 2 so it can score above JW bands when both surnames are present.
    d["comparison_levels"].insert(2, {
        "sql_condition": (
            f"{COL_FULL_NAME_COMPACT}_l = {COL_FULL_NAME_COMPACT}_r "
            f"AND {COL_FULL_NAME_COMPACT}_l IS NOT NULL"
        ),
        "label_for_charts": "full_name_compact exact match",
    })
    return d


def _build_birth_dt() -> dict:
    """BirthDT: null / exact / month-day-swap / ±1 day / ±1 month / ±1 year / else.

    The month-day-swap level is inserted right after `exact` (index 2), NOT before
    `else`: because we enable the `year` metric, the `<= 1 year` band would otherwise
    absorb every same-year transposition (year-diff 0) before the swap level could
    fire, making it dead. A strict transposition has Damerau-Levenshtein distance 2
    and both day and month differing, so it is not falsely caught by the auto DL<=1
    or the +/-1 day/month bands that follow it.
    """
    d = _comparison_to_dict(
        cl.DateOfBirthComparison(
            COL_DOB_STR,
            input_is_string=True,
            datetime_metrics=["day", "month", "year"],
            datetime_thresholds=[1, 1, 1],
        )
    )
    # Insert after null(0) + exact(1) so it precedes the +/-1 year band.
    d["comparison_levels"].insert(2, {
        "sql_condition": (
            f"substr({COL_DOB_STR}_l, 1, 4) = substr({COL_DOB_STR}_r, 1, 4) "
            f"AND substr({COL_DOB_STR}_l, 6, 2) = substr({COL_DOB_STR}_r, 9, 2) "
            f"AND substr({COL_DOB_STR}_l, 9, 2) = substr({COL_DOB_STR}_r, 6, 2) "
            f"AND substr({COL_DOB_STR}_l, 6, 2) <> substr({COL_DOB_STR}_l, 9, 2) "
            f"AND {COL_DOB_STR}_l IS NOT NULL AND {COL_DOB_STR}_r IS NOT NULL"
        ),
        "label_for_charts": "DOB month-day swap (same year)",
    })
    return d


def _build_ssn() -> dict:
    """SSN: null / exact-9 / last-4 match / full-9 mismatch / else."""
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
    """Email: null / exact(TF) / same-username-diff-domain / JW≥0.88 / else."""
    d = _comparison_to_dict(
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
    # JW≥0.88 on full email — catches typos/aliases before else.
    return _insert_level_before_else(d, {
        "sql_condition": (
            f"jaro_winkler_similarity({COL_EMAIL}_l, {COL_EMAIL}_r) >= 0.88 "
            f"AND {COL_EMAIL}_l IS NOT NULL AND {COL_EMAIL}_r IS NOT NULL"
        ),
        "label_for_charts": "Jaro-Winkler on full email >= 0.88",
    })


def _build_phones() -> dict:
    """Phones: null+empty / ≥2 shared / ≥1 shared / else."""
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


def _build_state() -> dict:
    """State: null / exact / else."""
    return _comparison_to_dict(cl.ExactMatch(COL_STATE))


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
    """Address: null / exact street / Same City+State+Zip / else. Gated on
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
    ComparisonSpec("FirstNM",  _build_first_nm),
    ComparisonSpec("LastNM",   _build_last_nm),
    ComparisonSpec("BirthDT",  _build_birth_dt),
    ComparisonSpec("SSN",      _build_ssn),
    ComparisonSpec("Email",    _build_email),
    ComparisonSpec("Phones",   _build_phones),
    ComparisonSpec("ZIP",      _build_zip),
)

_ADDRESS_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec("State",              _build_state),
    ComparisonSpec("Household_discount", _build_household_discount),
    ComparisonSpec("Address",            _build_address),
)


def build_registry(include_address: bool = True) -> ComparisonRegistry:
    """Return the enhanced_4 ComparisonRegistry.

    `include_address=False` drops State, Household_discount, and Address —
    used by the synthetic sandbox which lacks StateCD_clean / CityNM_clean.
    """
    specs = _BASE_SPECS + (_ADDRESS_SPECS if include_address else ())
    return ComparisonRegistry(specs)


#: Default registry for production runs (all 10 comparisons).
ENHANCED_4_REGISTRY: ComparisonRegistry = build_registry(include_address=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Match-prevalence prior rules (lambda seeding)
# ═══════════════════════════════════════════════════════════════════════════════
# SQL translations of src/models/deterministic_rules.py::RULES. Used by
# estimate_probability_two_random_records_match to seed lambda (the overall
# probability two random records match), assuming these high-precision rules
# capture DEFAULT_PRIOR_RECALL of all true matches. Column references use the
# derived `_dob_str` / `Phones_array` columns that prepare_model_input materialises.
_SSN_DOB: str = (
    f"l.{COL_SSN} = r.{COL_SSN} AND l.{COL_DOB_STR} = r.{COL_DOB_STR} "
    f"AND l.{COL_SSN} IS NOT NULL"
)
_NAME_DOB = (
    f"l.{COL_FIRST_NM} = r.{COL_FIRST_NM} AND l.{COL_LAST_NM} = r.{COL_LAST_NM} "
    f"AND l.{COL_DOB_STR} = r.{COL_DOB_STR}"
)
_NAME_DOB_EMAIL: str = (
    f"{_NAME_DOB} AND l.{COL_EMAIL} = r.{COL_EMAIL} AND l.{COL_EMAIL} IS NOT NULL"
)
_NAME_DOB_PHONE: str = (
    f"{_NAME_DOB} "
    f"AND len(list_intersect(l.{COL_PHONES_ARRAY}, r.{COL_PHONES_ARRAY})) >= 1"
)
_NAME_DOB_SEX: str = (
    f"{_NAME_DOB} AND l.{COL_SEX} = r.{COL_SEX} AND l.{COL_SEX} IS NOT NULL"
)
_NAME_DOB_ADDRESS: str = (
    f"{_NAME_DOB} AND l.{COL_ADDRESS1} = r.{COL_ADDRESS1} "
    f"AND l.{COL_ADDRESS1} IS NOT NULL"
)

#: Deterministic rules (SQL form) that seed the match-prevalence prior.
PRIOR_RULES: list[str] = [
    _SSN_DOB,
    _NAME_DOB_EMAIL,
    _NAME_DOB_PHONE,
    _NAME_DOB_SEX,
    _NAME_DOB_ADDRESS,
]

#: Assumed recall of the deterministic rules over all true matches, for lambda.
DEFAULT_PRIOR_RECALL: float = 0.9
