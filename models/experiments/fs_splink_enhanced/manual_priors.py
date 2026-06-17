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


# Map: comparison output_column_name -> prior dict. Drives apply_manual_priors.
_TARGET_COMPARISONS = {
    "Address":      ADDRESS_MU,
    "Phones_array": PHONES_MU,
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
    priors_by_ocn = {
        "Address":      address_mu if address_mu is not None else ADDRESS_MU,
        "Phones_array": phones_mu  if phones_mu  is not None else PHONES_MU,
    }

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
