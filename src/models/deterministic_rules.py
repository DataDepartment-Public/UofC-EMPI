"""
Deterministic matching-rule engine for the AllianceChicago eMPI pipeline.

This module consumes the candidate-pair output of the blocking stage
(`src/preprocessing/blocking.py`) and applies the exact-match rules documented in
`docs/Deterministic-Rules-Guide.md` to decide which candidate pairs are
confirmed deterministic matches.

PIPELINE POSITION:
    raw → src/preprocessing/clean.py → cleaned dataset (PATID + *_clean fields)
        → src/preprocessing/blocking.py → candidate pairs (PATID_A, PATID_B, ...)
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
    get_non_matches(candidate_pairs, matches)   -> pd.DataFrame (unconfirmed pairs)
    classify_non_matches(pairs, matches, clean) -> pd.DataFrame (+ reject/review)
    assign_clusters(matches)                    -> dict[str, int] (PATID -> cluster)
    get_match_stats(matches, n_records)         -> dict         (audit report)

THREE-WAY DECISION:
    Each candidate pair lands in one of three buckets:
      * match  — a rule confirmed it (`apply_rules`)               -> auto-merge
      * reject — a confident non-match (>= 3 strong contradictions) -> dropped
      * review — genuinely uncertain                               -> probabilistic
    `classify_non_matches` assigns reject/review to the pairs no rule confirmed.

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
from functools import lru_cache

import jellyfish
import numpy as np
import pandas as pd

from src.config import settings

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

# Fuzzy name agreement — names count as agreeing when exact, or within one typo.
# Applied ONLY to the first/last predicates the match rules consume (never to the
# strict disagreement count used for throw-out). Closes the documented recall gap
# on single-character name typos (e.g. MEAGAN vs MEGAN). Thresholds are
# calibratable; current values mirror the conventions in RULES.md.
NAME_JW_THRESHOLD = 0.92  # Jaro-Winkler similarity at/above which names agree
NAME_LEV_MAX = 1  # max Damerau-Levenshtein distance at/below which names agree

# Three-way decision for pairs no rule confirmed. A pair is auto-rejected (a
# confident non-match, dropped from the pipeline) only when at least this many
# strong identifiers STRICTLY disagree; fewer routes to review (the downstream
# probabilistic stage) instead.
#
# Calibrated on real run `real_20260620`: adjudicating reject samples by
# contradiction count gave a false-reject rate (truly-same pairs wrongly dropped)
# of ~10% at 2 contradictions but 0% at 3 and 4. So the threshold is 3 — two
# strong-identifier conflicts is not yet decisive (a pair can carry two
# independent typos), three is. Pairs with exactly two contradictions route to
# review, where the probabilistic stage can still reject them, rather than being
# discarded outright.
DEFAULT_REJECT_MIN_CONTRADICTIONS = 3

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

# Strong identifiers whose strict (exact, both-present) disagreement counts as a
# contradiction when deciding whether an unconfirmed pair is a confident
# non-match. Name typos are intentionally counted strictly here (fuzzy matching
# is only for *confirming* a pair), but a lone disagreement never rejects — see
# DEFAULT_REJECT_MIN_CONTRADICTIONS.
_CONTRADICTION_COLS = (COL_SSN, COL_FIRST_NM, COL_LAST_NM, COL_BIRTH_DT)


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
# SSN_DOB (SSN + date-of-birth) REPLACES the former bare EXACT_SSN rule. SSN
# agreement now requires an independent DOB corroborator before a pair is
# confirmed: an SSN match with a disagreeing or missing DOB no longer auto-
# confirms here and instead flows to the downstream probabilistic stage. This
# trades a small amount of recall on single-source SSN matches (pairs whose only
# evidence was a shared SSN) for protection against typo'd / shared / placeholder
# SSNs that name+DOB cannot vouch for.
#
# EMAIL_EXACT (email-only) was REMOVED for the same reason. On the real
# AllianceChicago data it ran at ~63-80% precision: shared family/clinic inboxes
# linked parents to children, siblings, and unrelated patients (see
# docs/Deterministic-Rules-Guide.md "Evaluation"). Email is only trustworthy when
# corroborated by name + DOB, which is exactly NAME_DOB_EMAIL.
RULES: tuple[MatchRule, ...] = (
    MatchRule("SSN_DOB", 1.000, ("ssn", "dob")),
    MatchRule("NAME_DOB_EMAIL", 0.990, ("first", "last", "dob", "email")),
    MatchRule("NAME_DOB_PHONE", 0.985, ("first", "last", "dob", "phone")),
    MatchRule("NAME_DOB_SEX", 0.980, ("first", "last", "dob", "sex")),
    MatchRule("NAME_DOB_ADDRESS", 0.970, ("first", "last", "dob", "address")),
)


# ── Phone-set parsing (kept in sync with src/preprocessing/blocking._parse_phone_set)
def _parse_phone_set(value) -> frozenset[str]:
    """Parse the pipeline's serialized `Phones_set` into a frozenset of strings.

    Mirrors the parser in `src/preprocessing/blocking.py` but is duplicated here so
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


@lru_cache(maxsize=200_000)
def _names_match_cached(sa: str, sb: str) -> bool:
    """Memoized fuzzy comparison for two non-null name strings.

    Wide evaluation candidate sets repeat the same name pairs many times, so
    caching the (pure) Jaro-Winkler / Damerau-Levenshtein computation keyed on
    the string pair avoids recomputing it per row.
    """
    if sa == sb:
        return True
    if jellyfish.jaro_winkler_similarity(sa, sb) >= NAME_JW_THRESHOLD:
        return True
    return jellyfish.damerau_levenshtein_distance(sa, sb) <= NAME_LEV_MAX


def _names_match(a: object, b: object) -> bool:
    """True if two name strings are equal or within one typo.

    Typo tolerance is Jaro-Winkler >= NAME_JW_THRESHOLD OR Damerau-Levenshtein
    <= NAME_LEV_MAX. A null on either side is never a match.
    """
    if pd.isna(a) or pd.isna(b):
        return False
    return _names_match_cached(str(a), str(b))


def _name_agree(frame: pd.DataFrame, col: str) -> pd.Series:
    """Fuzzy element-wise name agreement on `<col>_L` vs `<col>_R`.

    Like `_agree`, but tolerant of a single-character typo (see `_names_match`).
    Exact matches short-circuit, so the per-row string comparison only runs on
    the pairs that are not already exactly equal.
    """
    left = f"{col}_L"
    right = f"{col}_R"
    if left not in frame.columns or right not in frame.columns:
        return pd.Series(False, index=frame.index)
    exact = (frame[left] == frame[right]).fillna(False).to_numpy()
    lvals = frame[left].to_numpy()
    rvals = frame[right].to_numpy()
    result = exact.copy()
    for i in range(len(result)):  # only the non-exact rows need the fuzzy check
        if not exact[i]:
            result[i] = _names_match(lvals[i], rvals[i])
    return pd.Series(result, index=frame.index)


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
    """Per-pair boolean agreement series keyed by the names used in `RULES`.

    First/last name agreement is fuzzy (single-typo tolerant); all other
    predicates require exact equality.
    """
    return {
        "first": _name_agree(frame, COL_FIRST_NM),
        "last": _name_agree(frame, COL_LAST_NM),
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

    Only fires when both sides present the same SSN (the SSN-agreement condition
    shared by SSN_DOB); pairs confirmed by other rules without SSN agreement are
    False.
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


def _count_contradictions(frame: pd.DataFrame) -> pd.Series:
    """Count strong identifiers that strictly disagree on a materialized pair.

    A field contributes 1 when both sides are present and unequal (exact
    comparison — name typos count here; fuzzy matching is only for confirming a
    pair, never for rejecting one). Counted fields are `_CONTRADICTION_COLS`.
    """
    total = pd.Series(0, index=frame.index, dtype="int64")
    for col in _CONTRADICTION_COLS:
        left, right = f"{col}_L", f"{col}_R"
        if left not in frame.columns or right not in frame.columns:
            continue
        both_present = frame[left].notna() & frame[right].notna()
        total = total + (both_present & (frame[left] != frame[right])).astype("int64")
    return total


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


def classify_non_matches(
    candidate_pairs: pd.DataFrame,
    matches: pd.DataFrame,
    df_clean: pd.DataFrame,
    min_contradictions: int = DEFAULT_REJECT_MIN_CONTRADICTIONS,
) -> pd.DataFrame:
    """Split the unconfirmed candidate pairs into `reject` and `review`.

    Completes the three-way decision the deterministic stage emits:

        * confirmed by a rule           -> `apply_rules` (the match set)
        * `reject`  — confident non-match  -> dropped from the pipeline
        * `review`  — genuinely uncertain  -> downstream probabilistic stage

    A pair is `reject` only when at least `min_contradictions` strong identifiers
    strictly disagree (`_count_contradictions`); otherwise it is `review`. A
    single disagreement is never enough to reject — when in doubt we keep the
    pair for the probabilistic stage rather than discard a possible true match.

    Parameters
    ----------
    candidate_pairs, matches
        As in `get_non_matches`.
    df_clean : pd.DataFrame
        The cleaned dataset, keyed by `PATID`, used to materialize the
        identifier values the contradiction count compares.
    min_contradictions : int
        Reject threshold (default `DEFAULT_REJECT_MIN_CONTRADICTIONS`).

    Returns
    -------
    pd.DataFrame
        The `get_non_matches` frame plus two columns: `n_contradictions` (int)
        and `decision` (`"reject"` / `"review"`).
    """
    non_matches = get_non_matches(candidate_pairs, matches)
    if non_matches.empty:
        non_matches = non_matches.copy()
        non_matches["n_contradictions"] = pd.Series(dtype="int64")
        non_matches["decision"] = pd.Series(dtype="object")
        return non_matches

    frame = _materialize_pairs(non_matches, df_clean)
    n_contradictions = _count_contradictions(frame)
    decision = np.where(
        n_contradictions >= min_contradictions, "reject", "review"
    )

    out = non_matches.copy()
    out["n_contradictions"] = n_contradictions.to_numpy()
    out["decision"] = decision
    n_reject = int((out["decision"] == "reject").sum())
    logger.info(
        "Three-way split of %d unconfirmed pairs: %d reject (>=%d "
        "contradictions, dropped), %d review (-> probabilistic stage).",
        len(out),
        n_reject,
        min_contradictions,
        len(out) - n_reject,
    )
    return out


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


def get_match_stats(
    matches: pd.DataFrame,
    n_records: int | None = None,
    decided: pd.DataFrame | None = None,
) -> dict:
    """Compute the audit statistics described in the guide's Results Summary.

    Parameters
    ----------
    matches : pd.DataFrame
        Output of `apply_rules`.
    n_records : int, optional
        Total patient count in the source dataset, used for the coverage rate.
        When omitted, coverage is computed against the number of matched
        patients only (a lower bound) and `coverage_rate` is reported as None.
    decided : pd.DataFrame, optional
        Output of `classify_non_matches`. When given, the result includes a
        `decision_distribution` with the three-way `match` / `review` / `reject`
        counts.

    Returns
    -------
    dict
        Match distribution, coverage, quality indicators, cluster stats, and
        (when `decided` is supplied) the three-way decision distribution.
    """
    if matches.empty:
        return {}

    decision_distribution = None
    if decided is not None:
        dvc = decided["decision"].value_counts() if not decided.empty else {}
        decision_distribution = {
            "match": len(matches),
            "review": int(dvc.get("review", 0)),
            "reject": int(dvc.get("reject", 0)),
        }

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
        **(
            {"decision_distribution": decision_distribution}
            if decision_distribution is not None
            else {}
        ),
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
