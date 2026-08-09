"""Comparison registry for the production FS matcher (lifted from fs_enhanced_3).

The production model is a **deliberately simple, interpretable** Fellegi-Sunter
variant. Each field collapses to a single substantive distinction:

    Exact match  vs  All other

so the derived match-weight table reads off one Bayes factor per field. A
standard Splink **null level** precedes each pair of substantive levels — null is
a *no-evidence* level (it contributes zero weight), not a third comparison
outcome, so a record with a missing field is neither rewarded nor penalised, and
only a *populated-and-equal* vs *populated-and-different* split carries evidence.

Seven comparisons in declared order — FirstNM, LastNM, BirthDT, SSN, Email,
Phones, Address. Term-frequency adjustments are enabled on the high-cardinality
identity fields (FirstNM, LastNM, Email).

`PRIOR_RULES` translate the pipeline's deterministic match rules
(`src/models/deterministic_rules.py::RULES`) into Splink `l.<col> = r.<col>` SQL
predicates. They seed the overall match-prevalence prior (lambda) via
`estimate_probability_two_random_records_match` — they do NOT select training
pairs; m comes from the silver labels.

**Structure frozen** — this is the shipped model; changing the comparison levels
changes precision/recall and must be a deliberate, re-validated model revision.
"""

from __future__ import annotations

import splink.comparison_level_library as cll  # noqa: F401  (kept for parity/future levels)
import splink.comparison_library as cl
from splink import SettingsCreator

from src.models.fs_matcher.base import ComparisonRegistry, ComparisonSpec

# ── Column names ─────────────────────────────────────────────────────────────
COL_PATID = "PATID"
COL_FIRST_NM = "FirstNM_clean"
COL_LAST_NM = "LastNM_clean"
COL_DOB_STR = "_dob_str"
COL_SSN = "SSN_clean"
COL_EMAIL = "Email_clean"
COL_SEX = "SexAtBirthDSC_clean"
COL_PHONES_ARRAY = "Phones_array"
COL_ADDRESS1 = "AddressLine1_clean"


# ─── Helper: extract one comparison dict from a SettingsCreator wrapper ──────
def _comparison_to_dict(creator) -> dict:
    """Run a one-comparison SettingsCreator and return the comparison dict."""
    s = (
        SettingsCreator(
            link_type="dedupe_only",
            unique_id_column_name=COL_PATID,
            comparisons=[creator],
        )
        .get_settings("duckdb")
        .as_dict()
    )
    return s["comparisons"][0]


# ═══════════════════════════════════════════════════════════════════════════════
# Builders — one per comparison (each: null / exact / all-other)
# ═══════════════════════════════════════════════════════════════════════════════
def _build_first_nm() -> dict:
    """FirstNM: null / exact (TF-adjusted) / all-other."""
    return _comparison_to_dict(
        cl.ExactMatch(COL_FIRST_NM).configure(term_frequency_adjustments=True)
    )


def _build_last_nm() -> dict:
    """LastNM: null / exact (TF-adjusted) / all-other."""
    return _comparison_to_dict(
        cl.ExactMatch(COL_LAST_NM).configure(term_frequency_adjustments=True)
    )


def _build_birth_dt() -> dict:
    """BirthDT: null / exact / all-other. Reads the derived `_dob_str`
    (YYYY-MM-DD) so the comparison is a plain string equality."""
    return _comparison_to_dict(cl.ExactMatch(COL_DOB_STR))


def _build_ssn() -> dict:
    """SSN: null / exact (full 9-digit) / all-other."""
    return _comparison_to_dict(cl.ExactMatch(COL_SSN))


def _build_email() -> dict:
    """Email: null / exact (TF-adjusted) / all-other."""
    return _comparison_to_dict(
        cl.ExactMatch(COL_EMAIL).configure(term_frequency_adjustments=True)
    )


def _build_phones() -> dict:
    """Phones: null/empty / shared-phone / all-other.

    Phones are multi-valued (a sorted list per record), so the "exact match"
    notion is *sharing at least one phone number* — an array intersection of
    size >= 1. An empty or null list on either side is the no-evidence (null)
    level rather than a mismatch, mirroring every other field.
    """
    return {
        "output_column_name": "Phones",
        "comparison_levels": [
            {
                "sql_condition": (
                    f"{COL_PHONES_ARRAY}_l IS NULL OR {COL_PHONES_ARRAY}_r IS NULL "
                    f"OR len({COL_PHONES_ARRAY}_l) = 0 "
                    f"OR len({COL_PHONES_ARRAY}_r) = 0"
                ),
                "label_for_charts": "Phones null/empty",
                "is_null_level": True,
            },
            {
                "sql_condition": (
                    f"len(list_intersect({COL_PHONES_ARRAY}_l, {COL_PHONES_ARRAY}_r)) >= 1"
                ),
                "label_for_charts": "Shared phone (intersection >= 1)",
            },
            {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
        ],
    }


def _build_address() -> dict:
    """Address: null / exact street (AddressLine1) / all-other."""
    return _comparison_to_dict(cl.ExactMatch(COL_ADDRESS1))


# ═══════════════════════════════════════════════════════════════════════════════
# Registry assembly
# ═══════════════════════════════════════════════════════════════════════════════
_BASE_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec("FirstNM", _build_first_nm),
    ComparisonSpec("LastNM", _build_last_nm),
    ComparisonSpec("BirthDT", _build_birth_dt),
    ComparisonSpec("SSN", _build_ssn),
    ComparisonSpec("Email", _build_email),
    ComparisonSpec("Phones", _build_phones),
)

_ADDRESS_SPEC = ComparisonSpec("Address", _build_address)


def build_registry(include_address: bool = True) -> ComparisonRegistry:
    """Return the production ComparisonRegistry.

    `include_address=False` drops the Address comparison (only needed if a
    records frame ever lacks `AddressLine1_clean`; the real cohort carries it,
    so the default includes it).
    """
    specs = _BASE_SPECS + ((_ADDRESS_SPEC,) if include_address else ())
    return ComparisonRegistry(specs)


#: Default registry for production runs (all 7 comparisons).
FS_MATCHER_REGISTRY: ComparisonRegistry = build_registry(include_address=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Match-prevalence prior rules (lambda seeding)
# ═══════════════════════════════════════════════════════════════════════════════
# SQL translations of src/models/deterministic_rules.py::RULES. Used by
# estimate_probability_two_random_records_match to seed lambda, assuming these
# high-precision rules capture DEFAULT_PRIOR_RECALL of all true matches. Column
# references use the derived `_dob_str` / `Phones_array` columns that
# prepare_model_input materialises.
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
