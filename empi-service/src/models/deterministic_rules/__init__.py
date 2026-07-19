"""
Deterministic matching-rule engine for the AllianceChicago eMPI pipeline.

This package consumes the candidate-pair output of the blocking stage
(`src/preprocessing/blocking.py`) and applies the exact-match rules documented in
`docs/Deterministic-Rules-Guide.md` to decide which candidate pairs are
confirmed deterministic matches.

PIPELINE POSITION:
    raw → src/preprocessing/clean.py → cleaned dataset (PATID + *_clean fields)
        → src/preprocessing/blocking.py → candidate pairs (PATID_A, PATID_B, ...)
        → THIS PACKAGE → confirmed matches (+ rule, confidence, clusters)

INPUTS:
    candidate_pairs : pd.DataFrame
        Blocking output. Required columns: PATID_A, PATID_B. Optional
        passthrough columns (source_blocks, n_blocks) are preserved.
    df_clean : pd.DataFrame
        The cleaned dataset the pairs were generated from. Indexed/keyed by
        PATID and carrying the `*_clean` attribute columns plus `Phones_set`.

PUBLIC API:
    apply_rules(candidate_pairs, df_clean)      -> pd.DataFrame (confirmed pairs)
    get_non_matches(candidate_pairs, matches)   -> pd.DataFrame (unconfirmed pairs)
    classify_non_matches(pairs, matches, clean) -> pd.DataFrame (+ no_match/human_review)
    get_match_stats(matches, n_records)         -> dict         (audit report)

    Connected-component clustering lives in `src/models/clustering.py`
    (stage 5 of the pipeline) — see `assign_clusters` / `build_cluster_assignments`
    there. `get_match_stats` imports it for the cluster-size stats below.

    `DeterministicRulesClassifier` adapts this functional API to
    `src.models.base.PairClassifier` — the same shared interface the FS
    matcher and the ML matcher satisfy.

SUBMODULES (mirrors the shape of `src/models/fs_matcher/` and
`src/models/ml_matcher/` — a thin package with focused submodules):
    rules.py       — MatchRule definitions, tier membership (RULES,
                     AUTO_MERGE_RULES, REVIEW_RULES), and the
                     confirm-or-fall-through decision (apply_rules,
                     get_non_matches).
    classifier.py  — the reject/review three-way decision
                     (classify_non_matches), the audit report
                     (get_match_stats), and DeterministicRulesClassifier.
    Every name below is re-exported from one of the two, so
    `from src.models.deterministic_rules import X` call sites are
    unaffected by this internal split.

THREE-WAY DECISION:
    Each candidate pair lands in one of three buckets — the same vocabulary
    used by every classifier stage (`src.contracts.CLASSIFICATION_TIERS`):
      * auto_merge   — confirmed by an AUTO-MERGE-tier rule (`apply_rules`)
      * no_match     — a confident non-match (>= 3 strong contradictions), dropped
      * human_review — confirmed only by a REVIEW-tier rule, OR unconfirmed
                       with < 3 contradictions -> probabilistic stage
    `apply_rules` returns every pair any rule confirmed (both tiers, with the
    winning `match_rule`); the caller routes them by tier via `AUTO_MERGE_RULES` /
    `REVIEW_RULES`. `classify_non_matches` assigns no_match/human_review to the
    pairs no rule confirmed. NAME_DOB_SEX and NAME_DOB_ADDRESS are REVIEW-tier
    (~65% / ~67% silver precision) — they fire and keep provenance but never
    auto-merge on their own. See docs/Deterministic-Rules-Guide.md.

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

from src.models.deterministic_rules.rules import (
    COL_ADDRESS,
    COL_BIRTH_DT,
    COL_EMAIL,
    COL_FIRST_NM,
    COL_LAST_NM,
    COL_PATID,
    COL_PHONES,
    COL_SEX,
    COL_SSN,
    DEFAULT_SSN_FANOUT_THRESHOLD,
    NAME_JW_THRESHOLD,
    NAME_LEV_MAX,
    AUTO_MERGE_RULES,
    REVIEW_RULES,
    RULES,
    MatchRule,
    _parse_phone_set,
    apply_rules,
    get_non_matches,
)
from src.models.deterministic_rules.classifier import (
    DEFAULT_REJECT_MIN_CONTRADICTIONS,
    REJECT_RULES,
    REJECT_RULE_NAMES,
    DeterministicRulesClassifier,
    RejectRule,
    classify_non_matches,
    get_match_stats,
)

__all__ = [
    "COL_PATID", "COL_FIRST_NM", "COL_LAST_NM", "COL_BIRTH_DT", "COL_SSN",
    "COL_EMAIL", "COL_ADDRESS", "COL_SEX", "COL_PHONES",
    "DEFAULT_SSN_FANOUT_THRESHOLD", "DEFAULT_REJECT_MIN_CONTRADICTIONS",
    "NAME_JW_THRESHOLD", "NAME_LEV_MAX",
    "MatchRule", "RejectRule",
    "RULES", "AUTO_MERGE_RULES", "REVIEW_RULES", "REJECT_RULES", "REJECT_RULE_NAMES",
    "apply_rules", "get_non_matches", "classify_non_matches", "get_match_stats",
    "DeterministicRulesClassifier",
    "_parse_phone_set",
]
