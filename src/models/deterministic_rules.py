"""
Deterministic matching-rule engine for the AllianceChicago eMPI pipeline.

This module consumes the candidate-pair output of the blocking stage
(`src/features/blocking.py`) and applies the exact-match rules documented in
`docs/Deterministic-Rules-Guide.md` to decide which candidate pairs are
confirmed deterministic matches.

PIPELINE POSITION:
    raw → src/data/clean.py → cleaned dataset (PATID + *_clean fields)
        → src/features/blocking.py → candidate pairs (PATID_A, PATID_B, ...)
        → THIS MODULE → confirmed matches (+ rule, confidence, clusters)

INPUTS:
    candidate_pairs : pd.DataFrame
        Blocking output. Required columns: PATID_A, PATID_B. Optional
        passthrough columns (source_blocks, n_blocks) are preserved.
    df_clean : pd.DataFrame
        The cleaned dataset the pairs were generated from. Indexed/keyed by
        PATID and carrying the `*_clean` attribute columns plus `Phones_set`.

PUBLIC API:
    apply_rules(candidate_pairs, df_clean)      -> pd.DataFrame (confirmed matches)
    get_non_matches(candidate_pairs, matches)   -> pd.DataFrame (downstream input)
    assign_clusters(matches)                    -> dict[str, int] (PATID -> cluster)
    get_match_stats(matches, n_records)         -> dict         (audit report)

OUTPUT SCHEMA (matches DataFrame):
    PATID_A        str   — canonical first PATID (carried from blocking)
    PATID_B        str   — canonical second PATID
    match_rule     str   — highest-confidence rule that fired for the pair
    confidence     float — confidence of `match_rule`
    rules_fired    str   — pipe-delimited list of every rule that fired
    is_suspicious  bool  — DOB/last-name/SSN disagreement flag (see guide)
    high_fanout_ssn bool — pair's shared SSN is carried by >= threshold patients
                           (likely shared/fraudulent SSN — flag for review)
    source_blocks  str   — passed through from blocking (if present)
    n_blocks       int   — passed through from blocking (if present)

HIPAA NOTE:
    No PHI is written to logs — only aggregate counts. The rule metadata gives
    full audit traceability without exposing field-level values.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.config import settings

logger = logging.getLogger(__name__)

# ── Column-name constants (must match the cleaning + blocking pipelines) ──────
COL_PATID = "PATID"
COL_FIRST_NM = "FirstNM_clean"
COL_LAST_NM = "LastNM_clean"
COL_BIRTH_DT = "BirthDT_clean"
COL_SSN = "SSN_clean"
COL_EMAIL = "Email_clean"
COL_ADDRESS = "AddressLine1_clean"
COL_SEX = "SexAtBirthDSC_clean"
COL_PHONES = "Phones_set"

# A confirmed match whose shared SSN is carried by at least this many distinct
# patients is flagged `high_fanout_ssn` for clerical review (likely shared or
# fraudulent SSN rather than true duplicates). Sourced from central config
# (EMPI_SSN_FANOUT_THRESHOLD); still overridable per-call via apply_rules().
DEFAULT_SSN_FANOUT_THRESHOLD = settings.ssn_fanout_threshold

# Attribute columns pulled from df_clean onto each side of a pair. Phones are
# handled separately (set intersection), so Phones_set is not in this list.
_ATTR_COLS = (
    COL_FIRST_NM,
    COL_LAST_NM,
    COL_BIRTH_DT,
    COL_SSN,
    COL_EMAIL,
    COL_ADDRESS,
    COL_SEX,
)


# ── Rule definitions ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MatchRule:
    """A deterministic match rule.

    `fields` names the per-pair agreement predicates (keys of the agreement
    map built in `_build_agreement`) that must all be True for the rule to
    fire. Confidence values are taken verbatim from the guide.
    """

    name: str
    confidence: float
    fields: tuple[str, ...]


# Ordered by confidence (descending) — the order the winning rule is resolved in.
#
# EMAIL_EXACT (email-only) was REMOVED. On the real AllianceChicago data it ran
# at ~63-80% precision: shared family/clinic inboxes linked parents to children,
# siblings, and unrelated patients (see docs/Deterministic-Rules-Guide.md
# "Evaluation"). Email is only trustworthy when corroborated by name + DOB, which
# is exactly NAME_DOB_EMAIL. Bare email agreement now flows to the downstream
# probabilistic stage as a non-match rather than being auto-confirmed here.
RULES: tuple[MatchRule, ...] = (
    MatchRule("EXACT_SSN", 1.000, ("ssn",)),
    MatchRule("NAME_DOB_EMAIL", 0.990, ("first", "last", "dob", "email")),
    MatchRule("NAME_DOB_PHONE", 0.985, ("first", "last", "dob", "phone")),
    MatchRule("NAME_DOB_SEX", 0.980, ("first", "last", "dob", "sex")),
    MatchRule("NAME_DOB_ADDRESS", 0.970, ("first", "last", "dob", "address")),
)


# ── Phone-set parsing (kept in sync with src/features/blocking._parse_phone_set)
def _parse_phone_set(value) -> frozenset[str]:
    """Parse the pipeline's serialized `Phones_set` into a frozenset of strings.

    Mirrors the parser in `src/features/blocking.py` but is duplicated here so
    this module stays dependency-light (no jellyfish/phonetics import).
    """
    if isinstance(value, (set, frozenset, list, tuple, np.ndarray)):
        # Parquet round-trips the pipeline's Phones_set list as an np.ndarray.
        return frozenset(str(p).strip() for p in value if str(p).strip())
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return frozenset()
    if not isinstance(value, str):
        return frozenset()

    value = value.strip()
    if value in ("", "nan", "None", "set()", "{}", "[]"):
        return frozenset()

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (set, frozenset, list, tuple)):
            return frozenset(str(p).strip() for p in parsed if str(p).strip())
    except (ValueError, SyntaxError, TypeError):
        pass

    cleaned = value.strip("{}[]").replace("'", "").replace('"', "")
    return frozenset(p.strip() for p in cleaned.split(",") if p.strip())


# ── Internals ─────────────────────────────────────────────────────────────────
def _materialize_pairs(
    candidate_pairs: pd.DataFrame, df_clean: pd.DataFrame
) -> pd.DataFrame:
    """Join the available `*_clean` attributes onto both sides of each pair.

    Returns `candidate_pairs` with `<col>_L` / `<col>_R` columns appended for
    every attribute column present in `df_clean`. Missing attribute columns are
    simply skipped — the rules that depend on them will never fire.
    """
    if COL_PATID not in df_clean.columns:
        raise ValueError(f"df_clean must contain a '{COL_PATID}' column.")

    present = [c for c in _ATTR_COLS if c in df_clean.columns]
    missing = [c for c in _ATTR_COLS if c not in df_clean.columns]
    if missing:
        logger.warning(
            "df_clean missing %d attribute column(s); rules needing them "
            "cannot fire: %s",
            len(missing),
            ", ".join(missing),
        )

    attrs = df_clean[[COL_PATID, *present]]
    if attrs[COL_PATID].duplicated().any():
        n_dupes = int(attrs[COL_PATID].duplicated().sum())
        logger.warning(
            "df_clean has %d duplicate PATID(s); keeping first occurrence.",
            n_dupes,
        )
        attrs = attrs.drop_duplicates(subset=COL_PATID, keep="first")

    attrs = attrs.set_index(COL_PATID)
    out = candidate_pairs.join(attrs.add_suffix("_L"), on="PATID_A")
    out = out.join(attrs.add_suffix("_R"), on="PATID_B")
    return out


def _agree(frame: pd.DataFrame, col: str) -> pd.Series:
    """Element-wise equality on `<col>_L` vs `<col>_R`.

    `NaN == anything` is False in pandas, so this also enforces the
    "both sides non-null" requirement the guide states for every rule.
    """
    left = f"{col}_L"
    right = f"{col}_R"
    if left not in frame.columns or right not in frame.columns:
        return pd.Series(False, index=frame.index)
    return (frame[left] == frame[right]).fillna(False)


def _phone_agreement(
    candidate_pairs: pd.DataFrame, df_clean: pd.DataFrame
) -> pd.Series:
    """True where the two records share at least one cleaned phone number."""
    if COL_PHONES not in df_clean.columns:
        return pd.Series(False, index=candidate_pairs.index)

    phone_map = {
        patid: _parse_phone_set(val)
        for patid, val in zip(df_clean[COL_PATID], df_clean[COL_PHONES])
    }
    empty: frozenset[str] = frozenset()
    agree = [
        bool(phone_map.get(a, empty) & phone_map.get(b, empty))
        for a, b in zip(candidate_pairs["PATID_A"], candidate_pairs["PATID_B"])
    ]
    return pd.Series(agree, index=candidate_pairs.index)


def _build_agreement(
    frame: pd.DataFrame, phone_agree: pd.Series
) -> dict[str, pd.Series]:
    """Per-pair boolean agreement series keyed by the names used in `RULES`."""
    return {
        "first": _agree(frame, COL_FIRST_NM),
        "last": _agree(frame, COL_LAST_NM),
        "dob": _agree(frame, COL_BIRTH_DT),
        "ssn": _agree(frame, COL_SSN),
        "email": _agree(frame, COL_EMAIL),
        "address": _agree(frame, COL_ADDRESS),
        "sex": _agree(frame, COL_SEX),
        "phone": phone_agree,
    }


def _ssn_fanout_map(df_clean: pd.DataFrame) -> dict[str, int]:
    """Map each non-null SSN to the number of distinct patients carrying it.

    A valid SSN shared by many distinct identities is almost always a shared or
    fraudulent number (a family member's SSN entered, or fraud) rather than true
    duplicates — used by `_high_fanout_ssn_flag`.
    """
    if COL_SSN not in df_clean.columns:
        return {}
    sub = df_clean[[COL_PATID, COL_SSN]].dropna(subset=[COL_SSN])
    return sub.groupby(COL_SSN)[COL_PATID].nunique().to_dict()


def _high_fanout_ssn_flag(
    frame: pd.DataFrame, fanout: dict[str, int], threshold: int
) -> pd.Series:
    """True where the pair's shared SSN is carried by >= `threshold` patients.

    Only fires when both sides present the same SSN (the EXACT_SSN agreement
    condition); pairs confirmed by other rules without SSN agreement are False.
    """
    left, right = f"{COL_SSN}_L", f"{COL_SSN}_R"
    if left not in frame.columns or right not in frame.columns or not fanout:
        return pd.Series(False, index=frame.index)
    same_ssn = frame[left].notna() & frame[right].notna() & (frame[left] == frame[right])
    counts = frame[left].map(fanout).fillna(0)
    return same_ssn & (counts >= threshold)


def _suspicious_flag(frame: pd.DataFrame) -> pd.Series:
    """Replicate the guide's suspicious-match definition.

    A confirmed pair is suspicious when DOB differs, last name differs, or both
    SSNs are present but differ. "Differs" means both sides are non-null and
    unequal (a null on either side is not evidence of disagreement).
    """

    def _disagree(col: str) -> pd.Series:
        left, right = f"{col}_L", f"{col}_R"
        if left not in frame.columns or right not in frame.columns:
            return pd.Series(False, index=frame.index)
        both_present = frame[left].notna() & frame[right].notna()
        return both_present & (frame[left] != frame[right])

    return _disagree(COL_BIRTH_DT) | _disagree(COL_LAST_NM) | _disagree(COL_SSN)


# ── Public API ────────────────────────────────────────────────────────────────
def apply_rules(
    candidate_pairs: pd.DataFrame,
    df_clean: pd.DataFrame,
    ssn_fanout_threshold: int = DEFAULT_SSN_FANOUT_THRESHOLD,
) -> pd.DataFrame:
    """Apply every deterministic rule to the candidate pairs.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Blocking output. Must contain `PATID_A` and `PATID_B`. Any
        `source_blocks` / `n_blocks` columns are passed through.
    df_clean : pd.DataFrame
        The cleaned dataset, keyed by `PATID`, carrying the `*_clean`
        attribute columns and `Phones_set`.

    Returns
    -------
    pd.DataFrame
        One row per *confirmed* pair (at least one rule fired) with the output
        schema documented at module level. Pairs that no rule confirmed are
        dropped.
    """
    required = {"PATID_A", "PATID_B"}
    missing = required - set(candidate_pairs.columns)
    if missing:
        raise ValueError(
            f"candidate_pairs missing required column(s): {sorted(missing)}"
        )

    if candidate_pairs.empty:
        logger.info("No candidate pairs supplied — nothing to evaluate.")
        return _empty_matches(candidate_pairs)

    logger.info(
        "Applying %d deterministic rules to %d candidate pairs...",
        len(RULES),
        len(candidate_pairs),
    )

    frame = _materialize_pairs(candidate_pairs, df_clean)
    phone_agree = _phone_agreement(candidate_pairs, df_clean)
    agreement = _build_agreement(frame, phone_agree)

    # Each rule fires where all of its agreement predicates hold.
    rule_masks: dict[str, pd.Series] = {}
    for rule in RULES:
        mask = pd.Series(True, index=frame.index)
        for fld in rule.fields:
            mask &= agreement[fld]
        rule_masks[rule.name] = mask

    fired = pd.DataFrame(rule_masks, index=frame.index)
    any_fired = fired.any(axis=1)

    n_confirmed = int(any_fired.sum())
    logger.info(
        "Rule evaluation complete: %d/%d pairs confirmed as deterministic "
        "matches.",
        n_confirmed,
        len(frame),
    )
    if n_confirmed == 0:
        return _empty_matches(candidate_pairs)

    confirmed = frame[any_fired].copy()
    fired = fired[any_fired]

    # Winning rule = highest-confidence rule that fired (RULES is sorted desc).
    winning = pd.Series(pd.NA, index=confirmed.index, dtype="object")
    confidence = pd.Series(np.nan, index=confirmed.index, dtype="float64")
    for rule in RULES:
        take = fired[rule.name] & winning.isna()
        winning[take] = rule.name
        confidence[take] = rule.confidence

    rule_cols = list(fired.columns)
    rules_fired = fired.apply(
        lambda row: "|".join(c for c in rule_cols if row[c]), axis=1
    )

    out = pd.DataFrame(index=confirmed.index)
    out["PATID_A"] = confirmed["PATID_A"]
    out["PATID_B"] = confirmed["PATID_B"]
    out["match_rule"] = winning
    out["confidence"] = confidence
    out["rules_fired"] = rules_fired
    out["is_suspicious"] = _suspicious_flag(confirmed)
    out["high_fanout_ssn"] = _high_fanout_ssn_flag(
        confirmed, _ssn_fanout_map(df_clean), ssn_fanout_threshold
    )
    for passthrough in ("source_blocks", "n_blocks"):
        if passthrough in confirmed.columns:
            out[passthrough] = confirmed[passthrough]

    return out.reset_index(drop=True)


def get_non_matches(
    candidate_pairs: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """Return the candidate pairs that no deterministic rule confirmed.

    These are the pairs that fall through the deterministic stage and become
    the input to the downstream (probabilistic / ML) matching processes. The
    original blocking schema is preserved verbatim so downstream consumers keep
    the full candidate provenance (`source_blocks`, `n_blocks`, ...).

    Identity is the canonical `(PATID_A, PATID_B)` tuple, which `apply_rules`
    carries through unchanged from the blocking output, so this is an exact set
    difference: every confirmed match is removed and everything else is kept.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Blocking output. Must contain `PATID_A` and `PATID_B`.
    matches : pd.DataFrame
        Output of `apply_rules` (the confirmed deterministic matches).

    Returns
    -------
    pd.DataFrame
        The subset of `candidate_pairs` (same columns) whose `(PATID_A,
        PATID_B)` does not appear in `matches`, with a reset index.
    """
    required = {"PATID_A", "PATID_B"}
    missing = required - set(candidate_pairs.columns)
    if missing:
        raise ValueError(
            f"candidate_pairs missing required column(s): {sorted(missing)}"
        )

    if candidate_pairs.empty:
        return candidate_pairs.copy()
    if matches.empty:
        return candidate_pairs.reset_index(drop=True).copy()

    matched_keys = set(zip(matches["PATID_A"], matches["PATID_B"]))
    keep = [
        (a, b) not in matched_keys
        for a, b in zip(candidate_pairs["PATID_A"], candidate_pairs["PATID_B"])
    ]
    non_matches = candidate_pairs[pd.Series(keep, index=candidate_pairs.index)]
    logger.info(
        "%d/%d candidate pairs were not confirmed; routing to downstream "
        "matching.",
        len(non_matches),
        len(candidate_pairs),
    )
    return non_matches.reset_index(drop=True)


def assign_clusters(matches: pd.DataFrame) -> dict[str, int]:
    """Group confirmed matches into connected-component clusters.

    Treats every confirmed pair as an undirected edge and runs union-find to
    assign each PATID a cluster id. PATIDs not appearing in `matches` are not
    included (singletons). Cluster ids are deterministic: the smallest PATID in
    a component (by sort order) seeds its id ordering.

    Returns
    -------
    dict[str, int]
        PATID -> integer cluster id (ids are contiguous starting at 0).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Keep the lexicographically smaller root for deterministic output.
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        parent[hi] = lo

    for a, b in zip(matches["PATID_A"], matches["PATID_B"]):
        union(a, b)

    roots = sorted({find(p) for p in parent})
    root_to_id = {root: i for i, root in enumerate(roots)}
    return {patid: root_to_id[find(patid)] for patid in parent}


def get_match_stats(matches: pd.DataFrame, n_records: int | None = None) -> dict:
    """Compute the audit statistics described in the guide's Results Summary.

    Parameters
    ----------
    matches : pd.DataFrame
        Output of `apply_rules`.
    n_records : int, optional
        Total patient count in the source dataset, used for the coverage rate.
        When omitted, coverage is computed against the number of matched
        patients only (a lower bound) and `coverage_rate` is reported as None.

    Returns
    -------
    dict
        Match distribution, coverage, quality indicators, and cluster stats.
    """
    if matches.empty:
        return {}

    rule_counts = matches["match_rule"].value_counts().to_dict()
    total = len(matches)

    matched_patids = pd.concat([matches["PATID_A"], matches["PATID_B"]]).unique()
    n_matched = len(matched_patids)

    clusters = assign_clusters(matches)
    if clusters:
        sizes = pd.Series(list(clusters.values())).value_counts()
        max_cluster = int(sizes.max())
        n_clusters = int(sizes.size)
    else:
        max_cluster = 0
        n_clusters = 0

    coverage_rate = (
        round(100 * n_matched / n_records, 2)
        if n_records and n_records > 0
        else None
    )

    return {
        "total_matches": total,
        "match_distribution": dict(
            sorted(rule_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "avg_confidence": round(float(matches["confidence"].mean()), 4),
        "suspicious_matches": int(matches["is_suspicious"].sum()),
        "suspicious_rate": round(
            100 * float(matches["is_suspicious"].mean()), 2
        ),
        "high_fanout_ssn_matches": (
            int(matches["high_fanout_ssn"].sum())
            if "high_fanout_ssn" in matches.columns
            else 0
        ),
        "patients_matched": n_matched,
        "total_patients": n_records,
        "coverage_rate": coverage_rate,
        "n_clusters": n_clusters,
        "max_cluster_size": max_cluster,
    }


def _empty_matches(candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    """Return an empty matches frame with the full output schema."""
    cols = [
        "PATID_A",
        "PATID_B",
        "match_rule",
        "confidence",
        "rules_fired",
        "is_suspicious",
        "high_fanout_ssn",
    ]
    for passthrough in ("source_blocks", "n_blocks"):
        if passthrough in candidate_pairs.columns:
            cols.append(passthrough)
    return pd.DataFrame(columns=cols)
