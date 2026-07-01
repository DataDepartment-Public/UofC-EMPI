"""
corroboration_gate.py — Precision lever for the enhanced_4 Fellegi-Sunter matcher.

Applied AFTER tier classification. A pair that the FS score placed in the
`auto_merge` tier may STAY there only if it carries real corroborating evidence
beyond the DOB + name agreement that (per the enhanced_3 post-mortem) alone
clears the auto-merge threshold. Pairs lacking corroboration are **demoted to
`human_review`** (never to `no_match`, and never promoted). The demotion reason
is recorded in the `corroboration` column.

TIERED CORROBORATION RULE
-------------------------
Identifiers are split into two classes by how uniquely they name a *person*:

    person-unique  — SSN, Email        (one live human per value; enough alone)
    household      — Phone, Address     (SHARED among cohabitants; only in pairs)

For each current-`auto_merge` pair:

    ssn_agree      = both SSN_clean populated AND equal
    email_agree    = both Email_clean populated AND equal
    phone_share    = both phone sets non-empty AND |set_A ∩ set_B| >= 1
    address_agree  = both AddressLine1_clean populated AND equal

    person_unique   = ssn_agree OR email_agree
    household_count = int(phone_share) + int(address_agree)     # 0, 1, or 2
    keep            = person_unique OR (household_count >= 2)

WHY HOUSEHOLD IDENTIFIERS ONLY COUNT IN PAIRS
---------------------------------------------
Twins and cohabitants legitimately SHARE a phone number and a mailing address.
A pair with matching DOB + one name + a shared phone (or a shared address) is
exactly the twin / same-household false-positive that enhanced_3 leaked at
high confidence. Requiring BOTH household signals together (or one truly
person-unique signal) closes that hole without penalising a genuine duplicate
that happens to also agree on SSN or email.

Degraded-signal note: if either household column is unavailable on df_clean, its
signal is all-False, so household_count caps at 1 and the household keep-branch
becomes unreachable — pairs must then satisfy the person-unique branch (SSN/Email)
to stay auto_merge. This is fail-closed (more demotions to human_review, never a
wrongly-kept pair) and matches the "never falsely keep" intent.

RELATIONSHIP TO THE FS SCORE AND Household_discount
---------------------------------------------------
This gate is deterministic and orthogonal to the probabilistic score: the FS
match weight and the `Household_discount` comparison already *down-weight*
shared-household evidence inside the score, but a strong DOB+name agreement can
still push a twin over the auto-merge line. The gate is the hard backstop that
demotes those survivors to human review rather than auto-confirming them. It
does not touch pairs the score already placed in `human_review` / `no_match`.

HIPAA NOTE
----------
No identifier values are ever logged. Aggregate integer counts only.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.contracts import (
    PATID,
    SSN,
    EMAIL,
    ADDRESS1,
    PHONES,
)
from src.preprocessing.blocking import _parse_phone_set

logger = logging.getLogger(__name__)

# Output column carrying the demotion reason (str) or None.
CORROBORATION_COL = "corroboration"

# The single reason string stamped on demoted pairs.
DEMOTED_REASON = "demoted_no_corroboration"

# Per-signal required-column map. Each corroboration signal needs a specific
# attribute column on df_clean; a missing column degrades that ONE signal to
# all-False (with a warning) instead of failing the whole call — mirroring
# apply_vetoes. Production cleaned parquet carries every column.
_SIGNAL_REQUIRED_COLS: dict[str, tuple[str, ...]] = {
    "ssn_agree": (SSN,),
    "email_agree": (EMAIL,),
    "phone_share": (PHONES,),
    "address_agree": (ADDRESS1,),
}


def apply_corroboration_gate(
    df_classified: pd.DataFrame, df_clean: pd.DataFrame
) -> pd.DataFrame:
    """Demote uncorroborated `auto_merge` pairs to `human_review`.

    Parameters
    ----------
    df_classified : pd.DataFrame
        Tier-classified pairs — must carry PATID_A, PATID_B and
        `classification_tier`. All other columns are preserved unchanged.
    df_clean : pd.DataFrame
        The cleaned dataset the pairs were generated from. Corroborating
        columns consulted: SSN_clean, Email_clean, Phones_set,
        AddressLine1_clean. Each is optional (graceful degradation); PATID is
        required as the join key.

    Returns
    -------
    pd.DataFrame
        Copy of df_classified with one new column `corroboration: str | None`
        (None everywhere except demoted pairs, which get DEMOTED_REASON) and
        `classification_tier` updated to "human_review" on demoted pairs.
        df_classified and df_clean are NOT mutated.
    """
    if df_classified.empty:
        out = df_classified.copy()
        out[CORROBORATION_COL] = pd.Series([], dtype="object")
        return out

    if PATID not in df_clean.columns:
        raise ValueError(
            f"apply_corroboration_gate: df_clean is missing the required join "
            f"key {PATID!r}."
        )

    out = df_classified.copy()

    # Which signals are usable given the columns present on df_clean.
    usable_signals: dict[str, tuple[str, ...]] = {}
    for signal, cols in _SIGNAL_REQUIRED_COLS.items():
        if all(c in df_clean.columns for c in cols):
            usable_signals[signal] = cols
        else:
            missing = [c for c in cols if c not in df_clean.columns]
            logger.warning(
                "apply_corroboration_gate: signal %s unavailable — df_clean "
                "missing columns %s (treated as all-False)",
                signal,
                missing,
            )

    # Edge case: with NO corroborating columns at all the gate cannot confirm
    # anything and would wrongly demote every auto_merge pair. No-op instead:
    # return the frame with corroboration=None and tiers untouched.
    if not usable_signals:
        logger.warning(
            "apply_corroboration_gate: no corroborating columns available "
            "(%s) — gate skipped, no pairs demoted.",
            sorted({c for cols in _SIGNAL_REQUIRED_COLS.values() for c in cols}),
        )
        out[CORROBORATION_COL] = pd.Series([None] * len(out), dtype="object", index=out.index)
        return out

    # Side-by-side attribute join, keyed on PATID. Only pull the columns the
    # usable signals actually need. drop_duplicates is defensive against any
    # duplicate PATIDs on the cleaned frame (there should be none).
    needed_cols: set[str] = set()
    for cols in usable_signals.values():
        needed_cols.update(cols)

    attr_frame = (
        df_clean[[PATID, *sorted(needed_cols)]]
        .drop_duplicates(subset=[PATID])
        .set_index(PATID)
    )
    side_a = attr_frame.reindex(out["PATID_A"].values).reset_index(drop=True)
    side_b = attr_frame.reindex(out["PATID_B"].values).reset_index(drop=True)
    side_a.columns = [f"{c}_A" for c in side_a.columns]
    side_b.columns = [f"{c}_B" for c in side_b.columns]
    paired = pd.concat(
        [out.reset_index(drop=True), side_a, side_b], axis=1
    )

    n = len(paired)
    false_mask = pd.Series([False] * n, index=paired.index)

    # --- Per-signal boolean agreement masks --------------------------------
    # ssn_agree — both populated AND equal.
    if "ssn_agree" in usable_signals:
        ssn_agree = (
            paired[f"{SSN}_A"].notna()
            & paired[f"{SSN}_B"].notna()
            & (paired[f"{SSN}_A"] == paired[f"{SSN}_B"])
        )
    else:
        ssn_agree = false_mask

    # email_agree — both populated AND equal.
    if "email_agree" in usable_signals:
        email_agree = (
            paired[f"{EMAIL}_A"].notna()
            & paired[f"{EMAIL}_B"].notna()
            & (paired[f"{EMAIL}_A"] == paired[f"{EMAIL}_B"])
        )
    else:
        email_agree = false_mask

    # phone_share — both phone sets non-empty AND intersect.
    if "phone_share" in usable_signals:
        sets_a = paired[f"{PHONES}_A"].apply(_parse_phone_set)
        sets_b = paired[f"{PHONES}_B"].apply(_parse_phone_set)
        phone_share = pd.Series(
            [
                bool(a) and bool(b) and len(a & b) >= 1
                for a, b in zip(sets_a, sets_b)
            ],
            index=paired.index,
        )
    else:
        phone_share = false_mask

    # address_agree — both populated AND equal.
    if "address_agree" in usable_signals:
        address_agree = (
            paired[f"{ADDRESS1}_A"].notna()
            & paired[f"{ADDRESS1}_B"].notna()
            & (paired[f"{ADDRESS1}_A"] == paired[f"{ADDRESS1}_B"])
        )
    else:
        address_agree = false_mask

    # --- Tiered keep rule ---------------------------------------------------
    person_unique = ssn_agree | email_agree
    household_count = phone_share.astype(int) + address_agree.astype(int)
    keep = person_unique | (household_count >= 2)

    is_auto_merge = paired["classification_tier"] == "auto_merge"
    demote = is_auto_merge & ~keep

    corroboration = pd.Series([None] * n, dtype="object", index=paired.index)
    corroboration.loc[demote] = DEMOTED_REASON

    # Apply demotions to the tier column.
    new_tier = paired["classification_tier"].copy()
    new_tier.loc[demote] = "human_review"

    # --- Aggregate logging only — no identifier values ---------------------
    signal_counts = {
        "ssn_agree": int((ssn_agree & is_auto_merge).sum()),
        "email_agree": int((email_agree & is_auto_merge).sum()),
        "phone_share": int((phone_share & is_auto_merge).sum()),
        "address_agree": int((address_agree & is_auto_merge).sum()),
    }
    n_auto = int(is_auto_merge.sum())
    n_demoted = int(demote.sum())
    logger.info(
        "apply_corroboration_gate: %d/%d auto_merge pairs demoted to review; "
        "per-signal agreement counts (auto_merge only): %s",
        n_demoted,
        n_auto,
        signal_counts,
    )

    out[CORROBORATION_COL] = corroboration.values
    out["classification_tier"] = new_tier.values
    return out
