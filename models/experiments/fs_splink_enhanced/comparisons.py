"""Comparison registry for fs_splink_enhanced.

Each comparison is declared as a ``ComparisonSpec(name, builder)`` where each
``builder`` is a lazy callable that returns a Splink comparison dict.

The enhanced module extends the 7-comparison baseline with:
  - E4 explicit JW<0.5 mismatch levels on FirstNM and LastNM
  - E4 explicit SSN full-9 mismatch level
  - E4 Household_discount composite comparison (gated on include_address)
  - E3 Address comparison (gated on include_address)

Two registries are available:

  _BASE_REGISTRY  — 8 comparisons (7 baseline + Household_discount excluded;
                    same 7 baseline comparisons with E4 extra levels on
                    FirstNM / LastNM / SSN, plus Household_discount only when
                    address is included via build_registry).

  build_registry(include_address)  — factory returning the appropriate registry.
    include_address=True  → 9 comparisons (7 base + Household_discount + Address)
    include_address=False → 7 comparisons (7 base comparisons only)

EM training rules and match-prevalence prior rules are also exported here so
``FSEnhanced`` can wire them into ``EMTraining`` without re-declaring them.
"""

from __future__ import annotations

import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import SettingsCreator

from models.common.fs_base import ComparisonRegistry, ComparisonSpec

# ── Column names ──────────────────────────────────────────────────────────────
COL_PATID        = "PATID"
COL_FIRST_NM     = "FirstNM_clean"
COL_LAST_NM      = "LastNM_clean"
COL_SSN          = "SSN_clean"
COL_SSN_LAST4    = "last_4_SSN"
COL_EMAIL        = "Email_clean"
COL_ZIP          = "ZipCD_clean_base"
COL_DOB_STR      = "_dob_str"
COL_PHONES_ARRAY = "Phones_array"
COL_ADDRESS1     = "AddressLine1_clean"
COL_CITY         = "CityNM_clean"
COL_STATE        = "StateCD_clean"


# ── Helper: materialise one comparison into dict form ─────────────────────────
def _comparison_to_dict(creator) -> dict:
    """Run a single-comparison SettingsCreator and return the comparison dict."""
    s = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name=COL_PATID,
        comparisons=[creator],
    ).get_settings("duckdb").as_dict()
    return s["comparisons"][0]


# ── _insert_level_before_else ─────────────────────────────────────────────────
def _insert_level_before_else(
    comp_dict: dict,
    new_level: dict,
) -> None:
    """Insert ``new_level`` immediately before the ElseLevel of a comparison dict.

    Splink's NameComparison / CustomComparison emit dicts whose last level is
    conventionally the catch-all ElseLevel.  We add the new level at position -1
    so it evaluates before Else but after the more specific earlier levels.
    Mutates ``comp_dict`` in place.
    """
    comp_dict["comparison_levels"].insert(-1, new_level)


# ═══════════════════════════════════════════════════════════════════════════════
# Builders — one per comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _build_first_nm() -> dict:
    """FirstNM: tightened JW thresholds [0.92, 0.85] + TF + E4 JW<0.5 level."""
    comp = _comparison_to_dict(
        cl.NameComparison(
            COL_FIRST_NM,
            jaro_winkler_thresholds=[0.92, 0.85],
        ).configure(term_frequency_adjustments=True)
    )
    # E4: explicit JW<0.5 mismatch level inserted before ElseLevel.
    _insert_level_before_else(
        comp,
        {
            "sql_condition": (
                f"jaro_winkler_similarity({COL_FIRST_NM}_l, {COL_FIRST_NM}_r) < 0.5 "
                f"AND {COL_FIRST_NM}_l IS NOT NULL AND {COL_FIRST_NM}_r IS NOT NULL"
            ),
            "label_for_charts": f"Jaro-Winkler distance of {COL_FIRST_NM} < 0.5",
        },
    )
    return comp


def _build_last_nm() -> dict:
    """LastNM: default JW thresholds + TF + E4 JW<0.5 level."""
    comp = _comparison_to_dict(
        cl.NameComparison(COL_LAST_NM).configure(term_frequency_adjustments=True)
    )
    # E4: explicit JW<0.5 mismatch level inserted before ElseLevel.
    _insert_level_before_else(
        comp,
        {
            "sql_condition": (
                f"jaro_winkler_similarity({COL_LAST_NM}_l, {COL_LAST_NM}_r) < 0.5 "
                f"AND {COL_LAST_NM}_l IS NOT NULL AND {COL_LAST_NM}_r IS NOT NULL"
            ),
            "label_for_charts": f"Jaro-Winkler distance of {COL_LAST_NM} < 0.5",
        },
    )
    return comp


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
    """SSN: null / exact / last-4 / (E4) full-9 mismatch / else."""
    comp = _comparison_to_dict(
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
    # E4: explicit SSN full-9 mismatch level inserted before ElseLevel.
    _insert_level_before_else(
        comp,
        {
            "sql_condition": (
                f"{COL_SSN}_l IS NOT NULL AND {COL_SSN}_r IS NOT NULL "
                f"AND {COL_SSN}_l != {COL_SSN}_r"
            ),
            "label_for_charts": "Both populated, full 9-digit mismatch",
        },
    )
    return comp


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


def _build_household_discount() -> dict:
    """Household_discount: composite anti-evidence comparison (E4).

    Fires when the candidate pair carries a clear household indicator (shared
    address OR shared phone) but the identity fields disagree (different first
    name + different DOB).  This is the negative-interaction signal that
    per-field comparisons cannot express.

    The comparison references AddressLine1_clean; it is only included in
    registries when include_address=True (see build_registry).
    """
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
    """Address: null / exact street / same city+state+zip / else (E3)."""
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
# Registries
# ═══════════════════════════════════════════════════════════════════════════════

# Base 7-comparison registry (no address-gated comparisons).
# The E4 JW<0.5 / SSN-mismatch levels are baked into the builders above.
_BASE_REGISTRY: ComparisonRegistry = ComparisonRegistry([
    ComparisonSpec("FirstNM",  _build_first_nm),
    ComparisonSpec("LastNM",   _build_last_nm),
    ComparisonSpec("BirthDT",  _build_birth_dt),
    ComparisonSpec("SSN",      _build_ssn),
    ComparisonSpec("Email",    _build_email),
    ComparisonSpec("Phones",   _build_phones),
    ComparisonSpec("ZIP",      _build_zip),
])

# Address-gated specs (both need AddressLine1_clean on df_model).
_HOUSEHOLD_DISCOUNT_SPEC = ComparisonSpec("Household_discount", _build_household_discount)
_ADDRESS_SPEC = ComparisonSpec("Address", _build_address)


def build_registry(include_address: bool = True) -> ComparisonRegistry:
    """Return the enhanced comparison registry.

    Parameters
    ----------
    include_address : bool, default True
        When True, append Household_discount and Address comparisons (both
        require ``AddressLine1_clean`` on df_model).  Set False for the
        synthetic sandbox fixture that predates address columns.

    Returns
    -------
    ComparisonRegistry
        9 comparisons when include_address=True (7 base + Household_discount +
        Address); 7 comparisons when include_address=False.
    """
    if not include_address:
        return _BASE_REGISTRY
    return _BASE_REGISTRY.with_added(_HOUSEHOLD_DISCOUNT_SPEC).with_added(_ADDRESS_SPEC)


# Convenience alias: the full 9-comparison registry (address included).
ENHANCED_REGISTRY: ComparisonRegistry = build_registry(include_address=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EM training rules (mirrors train_model() in fellegi_sunter_enhanced.py)
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

# EM blocking rules — three sessions in order.
EM_BLOCKING_RULES: list[str] = [SSN_BLOCK, EMAIL_BLOCK, SOUNDEX_BLOCK]

# Prior rule: Double-Metaphone(LastNM) + exact DOB anchor.
DM_LAST_DOB_BLOCK: str = (
    "l._dm_LastNM = r._dm_LastNM AND l._dob_str = r._dob_str "
    "AND l._dm_LastNM IS NOT NULL AND l._dob_str IS NOT NULL"
)

# Match-prevalence prior rules (recall=0.80).
PRIOR_RULES: list[str] = [SSN_BLOCK, EMAIL_BLOCK, DM_LAST_DOB_BLOCK]
