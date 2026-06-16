"""
deterministic_vetoes.py — Patient-safety hard-rule rejection layer for the
enhanced Fellegi-Sunter matcher.

Applied AFTER FS scoring but BEFORE tier classification. Any pair whose
identifiers hit one of the four conflicts below is forced to tier=no_match
regardless of probabilistic score, with `veto_reason` recording the first
rule that fired (precedence order, top to bottom):

    ssn_conflict          — Both SSN_clean populated, full 9-digit mismatch
    dob_year_gap          — Both BirthDT_clean populated, |year diff| >= 5
    gender_conflict       — Both SexAtBirthDSC_clean populated, strictly
                            MALE <-> FEMALE (MALE<->OTHER and FEMALE<->OTHER
                            do NOT veto, to preserve merge eligibility for
                            transgender / nonbinary patients whose recorded
                            sex may legitimately differ across source EHRs)
    dob_and_ssn4_conflict — Both DOB AND last_4_SSN populated and BOTH mismatch

RELATIONSHIP TO DEVELOP'S DETERMINISTIC-RULES STAGE
---------------------------------------------------
These vetoes are NEGATIVE (reject an FS-scored match on hard conflict).
Develop's `src/models/deterministic_rules.py` rules are POSITIVE (confirm a
match on multi-field agreement). They are complementary and run at different
pipeline positions:

    deterministic rules (Stage 3) --> confirm + flag is_suspicious
                                  --> non-confirmed pairs flow downstream

    FS scoring (Stage 4) --> apply_vetoes (this module)
                         --> classify_pairs (threshold + force no_match on veto)

Develop's `is_suspicious` flag KEEPS the confirmed match and tags it for
review. These vetoes REJECT the FS pair outright. Both signals can coexist
on the eventual union of edges in a future Stage 5 clustering step.

HIPAA NOTE
----------
No identifier values are ever logged. Aggregate per-rule firing counts only.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.contracts import (
    PATID,
    SSN,
    SSN_LAST4,
    BIRTH_DT,
    SEX,
)

logger = logging.getLogger(__name__)

# Output column name carrying the first-fired rule (or null for no veto).
VETO_REASON_COL = "veto_reason"

# Precedence: when multiple rules would fire on the same pair, the first
# in this tuple wins. Ordered by signal strength / clinical-evidence value.
VETO_PRECEDENCE: tuple[str, ...] = (
    "ssn_conflict",
    "dob_year_gap",
    "gender_conflict",
    "dob_and_ssn4_conflict",
)

# Year-gap threshold for the dob_year_gap veto. A 5-year gap is wider than
# any realistic data-entry typo (transposed digits / off-by-one years are
# all under 2); anything beyond this is structurally a different person.
DOB_YEAR_GAP_THRESHOLD = 5


def apply_vetoes(df_predictions: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Annotate each scored pair with the first-firing deterministic veto.

    Parameters
    ----------
    df_predictions : pd.DataFrame
        Output of predict_pairs() — must carry PATID_A and PATID_B columns.
        Any other columns are preserved unchanged.
    df_clean : pd.DataFrame
        The cleaned dataset the pairs were generated from. Required columns:
        PATID, SSN_clean, last_4_SSN, BirthDT_clean, SexAtBirthDSC_clean.

    Returns
    -------
    pd.DataFrame
        Copy of df_predictions with one new column `veto_reason: str | None`.
        Non-null entries name the first-firing rule per VETO_PRECEDENCE.
        df_predictions and df_clean are NOT mutated.
    """
    if df_predictions.empty:
        out = df_predictions.copy()
        out[VETO_REASON_COL] = pd.Series([], dtype="object")
        return out

    required_clean = {PATID, SSN, SSN_LAST4, BIRTH_DT, SEX}
    missing = required_clean - set(df_clean.columns)
    if missing:
        raise ValueError(
            f"apply_vetoes: df_clean is missing required columns {sorted(missing)}. "
            "These must be present on the cleaned parquet per src/contracts.py."
        )

    # Side-by-side attribute join. df_clean is the index; PATID_A and PATID_B
    # each look up the same attribute frame. drop_duplicates is defensive in
    # case the cleaned frame ever contains duplicate PATIDs (it shouldn't).
    attr_cols = [SSN, SSN_LAST4, BIRTH_DT, SEX]
    attr_frame = (
        df_clean[[PATID, *attr_cols]]
        .drop_duplicates(subset=[PATID])
        .set_index(PATID)
    )
    side_a = attr_frame.reindex(df_predictions["PATID_A"].values).reset_index(drop=True)
    side_b = attr_frame.reindex(df_predictions["PATID_B"].values).reset_index(drop=True)
    side_a.columns = [f"{c}_A" for c in side_a.columns]
    side_b.columns = [f"{c}_B" for c in side_b.columns]
    paired = pd.concat(
        [df_predictions.reset_index(drop=True), side_a, side_b], axis=1
    )

    # --- Per-rule masks -----------------------------------------------------
    masks: dict[str, pd.Series] = {}

    # ssn_conflict — both populated, full-9 mismatch.
    masks["ssn_conflict"] = (
        paired[f"{SSN}_A"].notna()
        & paired[f"{SSN}_B"].notna()
        & (paired[f"{SSN}_A"] != paired[f"{SSN}_B"])
    )

    # dob_year_gap — both populated, |year_A - year_B| >= threshold.
    year_a = pd.to_datetime(paired[f"{BIRTH_DT}_A"], errors="coerce").dt.year
    year_b = pd.to_datetime(paired[f"{BIRTH_DT}_B"], errors="coerce").dt.year
    masks["dob_year_gap"] = (
        year_a.notna()
        & year_b.notna()
        & ((year_a - year_b).abs() >= DOB_YEAR_GAP_THRESHOLD)
    )

    # gender_conflict — strict MALE <-> FEMALE only; OTHER never vetoes.
    sex_a = paired[f"{SEX}_A"]
    sex_b = paired[f"{SEX}_B"]
    masks["gender_conflict"] = sex_a.notna() & sex_b.notna() & (
        ((sex_a == "MALE") & (sex_b == "FEMALE"))
        | ((sex_a == "FEMALE") & (sex_b == "MALE"))
    )

    # dob_and_ssn4_conflict — both DOB AND last-4-SSN populated and mismatch.
    dob_diff = (
        paired[f"{BIRTH_DT}_A"].notna()
        & paired[f"{BIRTH_DT}_B"].notna()
        & (paired[f"{BIRTH_DT}_A"] != paired[f"{BIRTH_DT}_B"])
    )
    ssn4_diff = (
        paired[f"{SSN_LAST4}_A"].notna()
        & paired[f"{SSN_LAST4}_B"].notna()
        & (paired[f"{SSN_LAST4}_A"] != paired[f"{SSN_LAST4}_B"])
    )
    masks["dob_and_ssn4_conflict"] = dob_diff & ssn4_diff

    # --- Resolve precedence: first rule to fire per pair wins. -------------
    veto_reason = pd.Series([None] * len(paired), dtype="object", index=paired.index)
    for rule in VETO_PRECEDENCE:
        unassigned = veto_reason.isna()
        veto_reason.loc[unassigned & masks[rule]] = rule

    # Aggregate logging only — no identifier values.
    fire_counts = {rule: int(masks[rule].sum()) for rule in VETO_PRECEDENCE}
    n_with_veto = int(veto_reason.notna().sum())
    logger.info(
        "apply_vetoes: %d/%d pairs vetoed (per-rule firings before precedence: %s)",
        n_with_veto,
        len(paired),
        fire_counts,
    )

    out = df_predictions.copy()
    out[VETO_REASON_COL] = veto_reason.values
    return out
