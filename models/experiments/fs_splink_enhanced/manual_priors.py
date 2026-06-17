"""
manual_priors.py — Candidate-pool-aware m/u priors for Address + Phones.

CONTEXT
-------
EM-trained Splink m/u values for Address and Phones agreement come out
miscalibrated on the AllianceChicago candidate pool. EM samples random pairs
to estimate u (prob[level | not a match]), but our candidate pool is heavily
preselected by the blocking stage for likely household members (families,
roommates, namesakes). Those pairs share address and phone numbers far more
often than two truly random patients would, so the EM-trained u under-counts
the within-pool prevalence of agreement → Splink rewards Address-exact and
Phones-intersect with large positive weights → the human_review band gets
flooded with family/household pairs the model wrongly thinks are matches.

This module supplies manually-set, candidate-pool-aware m/u values for the
levels of those two comparisons. The values come from the user's 42-pair
manual-review sample (see `notebooks/fellegi_sunter/fellegi_sunter_validation.ipynb`
§9 "Reviewer judgments"). They are LOCKED via Splink's
`fix_m_probability` / `fix_u_probability` settings so the EM training pass
will not overwrite them.

REFRESH
-------
Refine these priors after each new round of labeled review pairs. Section E3
of the build plan notes that after E2 (vetoes) some labeled pairs may be
absorbed by veto rejection and the surviving distribution may shift. Update
the dicts below — DO NOT change apply_manual_priors's contract.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Address — 4 levels. Level labels MUST match the `label_for_charts` of the
# corresponding Splink CustomLevels in fellegi_sunter_enhanced.build_settings.
#
# Direction summary (log2(m/u) in bits):
#   Exact match              +0.28  ≈ zero weight (the "matters in a vacuum"
#                                    advantage is washed out by candidate-pool
#                                    preselection — household pairs share addr)
#   Same City+State+Zip       0.00  zero weight (geographic agreement alone
#                                    is uninformative in this regional cohort)
#   All other comparisons    -2.32  ANTI-EVIDENCE: address truly disagreeing
#                                    in a preselected candidate pool is a
#                                    strong negative signal
ADDRESS_MU: dict[str, tuple[float, float]] = {
    "Exact match on AddressLine1_clean": (0.67, 0.55),
    "Same City + State + Zip":           (0.30, 0.30),
    "All other comparisons":             (0.03, 0.15),
}


# Phones — recalibrate the existing ArrayIntersectAtSizes levels. The level
# labels below match Splink's default labels for cl.ArrayIntersectAtSizes
# (verified against the baseline diagnostics JSON).
#
# Baseline EM gave Phones intersect ≥1 a +18 bit positive weight — wildly
# over-credited because household members are nearly perfectly correlated
# with shared phones in the candidate pool. The recalibrated values below
# preserve a strong positive bonus for ≥2-phone overlap (rare even within
# households) while pushing the ≥1-phone level near zero.
PHONES_MU: dict[str, tuple[float, float]] = {
    "Array intersection size >= 2": (0.40, 0.05),  # +3.0 bits
    "Array intersection size >= 1": (0.50, 0.55),  # ~0 bits
    "All other comparisons":        (0.10, 0.40),  # -2.0 bits
}


# E4 — Explicit name-mismatch levels (inserted between JW>=0.85 and Else by
# build_settings). The baseline NameComparison's "All other comparisons"
# bucket lumped "typo-similar" and "completely different name" together; this
# splits the truly-different cases out and assigns strong anti-evidence.
FIRSTNM_JW_LT_05_MU: dict[str, tuple[float, float]] = {
    # ~3 bits anti-evidence: two records with very dissimilar first names are
    # almost never the same person in a clinical context.
    "Jaro-Winkler distance of FirstNM_clean < 0.5": (0.005, 0.30),
}

LASTNM_JW_LT_05_MU: dict[str, tuple[float, float]] = {
    # ~4-5 bits anti-evidence: last names in our candidate pool already cluster
    # (regional cohort + blocking on phonetic last name), so a JW<0.5 last
    # name is a stronger signal than for first names.
    "Jaro-Winkler distance of LastNM_clean < 0.5": (0.005, 0.15),
}


# E4 — Explicit SSN 5-9 conflict level (defense in depth alongside the
# ssn_conflict veto). Both SSNs populated, full 9-digit mismatch but the
# pair is still being scored (e.g., the veto layer is bypassed in some
# diagnostic mode or fails to load on a partial-schema dataset). Strong
# anti-evidence so the FS score itself reflects the conflict.
SSN_FULL_MISMATCH_MU: dict[str, tuple[float, float]] = {
    "Both populated, full 9-digit mismatch": (0.01, 0.05),  # ~-2.3 bits
}


# E4 — Household-discount composite. Fires when the candidate pair carries a
# clear household indicator (shared address OR shared phone) but the identity
# fields disagree (different first name + different DOB). Captures the
# negative-interaction pattern the per-field comparisons cannot express:
# "I see the household signature without the same-person signature." Strong
# anti-evidence; this is the single biggest expected mover for the
# family-same-household false-positive class (19/42 of the labeled sample).
HOUSEHOLD_DISCOUNT_MU: dict[str, tuple[float, float]] = {
    # m ~0.05  — real same-person pairs almost never look like this
    # u ~0.45  — ~half of our candidate-pool non-matches ARE this pattern
    # m/u ~0.11 -> ~-3.2 bits anti-evidence
    "Household indicator without identity match": (0.05, 0.45),
}


# Map: comparison output_column_name -> prior dict. Drives apply_manual_priors.
# Multiple comparisons (FirstNM_clean, LastNM_clean, SSN) carry only a partial
# label-set in their prior dicts — apply_manual_priors locks only the labels
# that match, leaving EM-trained values alone on the rest. That's intentional:
# the explicit-mismatch levels are the only ones we're locking; the
# "JW>=0.92 / JW>=0.85 / exact" levels still benefit from EM training.
_TARGET_COMPARISONS = {
    "Address":           ADDRESS_MU,
    "Phones_array":      PHONES_MU,
    "FirstNM_clean":     FIRSTNM_JW_LT_05_MU,
    "LastNM_clean":      LASTNM_JW_LT_05_MU,
    "SSN":               SSN_FULL_MISMATCH_MU,
    "Household_discount": HOUSEHOLD_DISCOUNT_MU,
}


def apply_manual_priors(
    settings_dict: dict,
    *,
    address_mu: Optional[dict[str, tuple[float, float]]] = None,
    phones_mu: Optional[dict[str, tuple[float, float]]] = None,
) -> dict:
    """Mutate `settings_dict` in place to lock m/u on Address + Phones levels.

    For each target comparison level, sets:
        m_probability        <- prior m̂
        u_probability        <- prior û
        fix_m_probability    <- True  (EM will respect this as a starting value
                                       AND hold it fixed across iterations)
        fix_u_probability    <- True

    The function is a NO-OP for any target comparison not present in the
    settings (e.g., when the synthetic fixture lacks address columns and
    build_settings(include_address=False) was called). This mirrors the
    graceful-skip pattern used by deterministic_vetoes.apply_vetoes.

    Parameters
    ----------
    settings_dict : dict
        Splink settings dict. Typically produced by
        SettingsCreator(...).get_settings("duckdb").as_dict().
    address_mu, phones_mu : optional override dicts
        Override the module-level priors for one-off experiments (e.g., when
        refining values from a new labeled sample). Defaults use the values
        defined above in this module.

    Returns
    -------
    dict
        The same settings_dict (returned for chaining convenience). Mutation
        happens in place.
    """
    # Start from the module-level map (covers all 6 target comparisons), then
    # override individual entries from kwargs for ad-hoc experiments.
    priors_by_ocn: dict[str, dict[str, tuple[float, float]]] = dict(_TARGET_COMPARISONS)
    if address_mu is not None:
        priors_by_ocn["Address"] = address_mu
    if phones_mu is not None:
        priors_by_ocn["Phones_array"] = phones_mu

    comparisons = settings_dict.get("comparisons", [])
    n_locked = 0
    for comp in comparisons:
        ocn = comp.get("output_column_name")
        priors = priors_by_ocn.get(ocn)
        if priors is None:
            continue
        applied_in_this_comp = 0
        for level in comp.get("comparison_levels", []):
            label = level.get("label_for_charts")
            if label in priors:
                m, u = priors[label]
                level["m_probability"] = m
                level["u_probability"] = u
                level["fix_m_probability"] = True
                level["fix_u_probability"] = True
                applied_in_this_comp += 1
                n_locked += 1
        if applied_in_this_comp == 0:
            # Comparison present but none of the labels matched — likely a
            # rename mismatch. Log loudly because the comparison will then
            # be EM-trained with random-pair-biased u, the exact bug we're
            # trying to fix.
            logger.warning(
                "apply_manual_priors: %s present but no level labels matched "
                "the prior dict %s — m/u will be EM-trained (random-pair-biased).",
                ocn, sorted(priors.keys()),
            )
        else:
            logger.info(
                "apply_manual_priors: locked %d/%d levels on comparison %s",
                applied_in_this_comp, len(priors), ocn,
            )

    if n_locked == 0:
        logger.warning(
            "apply_manual_priors: no levels locked. Either no Address/Phones "
            "comparison is in the settings, or all label_for_charts mismatch."
        )
    return settings_dict
