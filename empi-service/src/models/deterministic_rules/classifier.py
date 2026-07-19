"""Three-way decision layer + `PairClassifier` adapter for deterministic rules.

Completes the three-way decision `rules.py` starts (confirm-or-fall-through
via `apply_rules`): `classify_non_matches` splits the unconfirmed pairs into
`no_match`/`human_review`, `get_match_stats` computes the audit report, and
`DeterministicRulesClassifier` adapts the whole functional API to
`src.models.base.PairClassifier` — the same shared interface the FS matcher
and the ML matcher satisfy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.contracts import TIER_AUTO_MERGE, TIER_HUMAN_REVIEW, TIER_NO_MATCH
from src.models.clustering import assign_clusters
from src.models.deterministic_rules.rules import (
    AUTO_MERGE_RULES,
    COL_BIRTH_DT,
    COL_FIRST_NM,
    COL_LAST_NM,
    COL_SSN,
    DEFAULT_SSN_FANOUT_THRESHOLD,
    _materialize_pairs,
    apply_rules,
    get_non_matches,
)

logger = logging.getLogger(__name__)

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

# Strong identifiers whose strict (exact, both-present) disagreement counts as a
# contradiction when deciding whether an unconfirmed pair is a confident
# non-match. Name typos are intentionally counted strictly here (fuzzy matching
# is only for *confirming* a pair), but a lone disagreement never rejects — see
# DEFAULT_REJECT_MIN_CONTRADICTIONS.
_CONTRADICTION_COLS = (COL_SSN, COL_FIRST_NM, COL_LAST_NM, COL_BIRTH_DT)


@dataclass(frozen=True)
class RejectRule:
    """A deterministic *reject* rule — the third tier, symmetric with `MatchRule`.

    Where a `MatchRule` fires on field *agreement*, a `RejectRule` fires on field
    *disagreement*: an unconfirmed pair is a confident non-match when at least
    `min_contradictions` of its strong-identifier `fields` strictly disagree (both
    sides present and unequal — name typos count strictly here; fuzzy matching is
    only ever used to *confirm* a pair). A pair that fires a reject rule is dropped
    from the pipeline; one that does not routes to review.
    """

    name: str
    min_contradictions: int
    fields: tuple[str, ...]


# The reject tier. Today there is exactly one rule — the calibrated
# strong-identifier-conflict rule. Its threshold of 3 was calibrated on real run
# `real_20260620` (false-reject rate ~10% at 2 contradictions, 0% at 3-4), so two
# conflicts route to review and only three confidently reject. NOTE: a *single*
# strong-ID conflict is deliberately NOT a reject rule — the Fellegi-Sunter
# conflict-veto analysis showed true duplicates conflict on SSN/email/phone nearly
# as often as false merges (churn + data-entry error), so a single-conflict veto
# destroys more true matches than it saves. Only the multi-field threshold is safe.
REJECT_RULES: tuple[RejectRule, ...] = (
    RejectRule(
        "STRONG_ID_CONFLICT", DEFAULT_REJECT_MIN_CONTRADICTIONS, _CONTRADICTION_COLS
    ),
)

#: Reject-rule names, derived from REJECT_RULES so it never drifts. Mirrors
#: contracts.REJECT_RULE_NAMES (kept manually in sync there, same convention as
#: contracts.AUTO_MERGE_RULE_NAMES).
REJECT_RULE_NAMES: frozenset[str] = frozenset(r.name for r in REJECT_RULES)


def _count_contradictions(
    frame: pd.DataFrame, fields: tuple[str, ...] = _CONTRADICTION_COLS
) -> pd.Series:
    """Count strong identifiers that strictly disagree on a materialized pair.

    A field contributes 1 when both sides are present and unequal (exact
    comparison — name typos count here; fuzzy matching is only for confirming a
    pair, never for rejecting one). Counted fields default to `_CONTRADICTION_COLS`
    (the `STRONG_ID_CONFLICT` reject rule's fields).
    """
    total = pd.Series(0, index=frame.index, dtype="int64")
    for col in fields:
        left, right = f"{col}_L", f"{col}_R"
        if left not in frame.columns or right not in frame.columns:
            continue
        both_present = frame[left].notna() & frame[right].notna()
        total = total + (both_present & (frame[left] != frame[right])).astype("int64")
    return total


# ── Public API ────────────────────────────────────────────────────────────────
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

    The reject decision is driven by the `STRONG_ID_CONFLICT` reject rule
    (`REJECT_RULES[0]`): a pair is `reject` only when at least `min_contradictions`
    of the rule's strong-identifier `fields` strictly disagree
    (`_count_contradictions`); otherwise it is `review`. A single disagreement is
    never enough to reject — when in doubt we keep the pair for the probabilistic
    stage rather than discard a possible true match.

    Parameters
    ----------
    candidate_pairs, matches
        As in `get_non_matches`.
    df_clean : pd.DataFrame
        The cleaned dataset, keyed by `PATID`, used to materialize the
        identifier values the contradiction count compares.
    min_contradictions : int
        Reject threshold, overriding the reject rule's own threshold (default
        `DEFAULT_REJECT_MIN_CONTRADICTIONS`, which matches it).

    Returns
    -------
    pd.DataFrame
        The `get_non_matches` frame plus three columns: `n_contradictions` (int),
        `decision` (`TIER_NO_MATCH` / `TIER_HUMAN_REVIEW`), and `reject_rule`
        (the firing reject rule's name on rejected rows, else NA).
    """
    reject_rule = REJECT_RULES[0]
    non_matches = get_non_matches(candidate_pairs, matches)
    if non_matches.empty:
        non_matches = non_matches.copy()
        non_matches["n_contradictions"] = pd.Series(dtype="int64")
        non_matches["decision"] = pd.Series(dtype="object")
        non_matches["reject_rule"] = pd.Series(dtype="object")
        return non_matches

    frame = _materialize_pairs(non_matches, df_clean)
    n_contradictions = _count_contradictions(frame, reject_rule.fields)
    is_reject = n_contradictions >= min_contradictions
    decision = np.where(is_reject, TIER_NO_MATCH, TIER_HUMAN_REVIEW)

    out = non_matches.copy()
    out["n_contradictions"] = n_contradictions.to_numpy()
    out["decision"] = decision
    out["reject_rule"] = np.where(is_reject, reject_rule.name, pd.NA)
    n_reject = int((out["decision"] == TIER_NO_MATCH).sum())
    logger.info(
        "Three-way split of %d unconfirmed pairs: %d reject (>=%d "
        "contradictions, dropped), %d review (-> probabilistic stage).",
        len(out),
        n_reject,
        min_contradictions,
        len(out) - n_reject,
    )
    return out


def get_match_stats(
    matches: pd.DataFrame,
    n_records: int | None = None,
    decided: pd.DataFrame | None = None,
    review_matches: pd.DataFrame | None = None,
) -> dict:
    """Compute the audit statistics described in the guide's Results Summary.

    Parameters
    ----------
    matches : pd.DataFrame
        The AUTO-MERGE matches (`apply_rules` output filtered to
        `AUTO_MERGE_RULES`). All distribution / coverage / cluster stats are
        computed over this auto-merge set.
    n_records : int, optional
        Total patient count in the source dataset, used for the coverage rate.
        When omitted, coverage is computed against the number of matched
        patients only (a lower bound) and `coverage_rate` is reported as None.
    decided : pd.DataFrame, optional
        Output of `classify_non_matches`. When given, the result includes a
        `decision_distribution` with the three-way `match` / `review` / `reject`
        counts. Review-tier rule confirmations (`review_matches`) are counted in
        the `review` bucket, not `match`.
    review_matches : pd.DataFrame, optional
        The REVIEW-tier rule confirmations (`apply_rules` output filtered to
        `REVIEW_RULES`). When given, adds a `review_match_distribution` (per-rule
        counts) and rolls these pairs into the `review` decision count.

    Returns
    -------
    dict
        Match distribution, coverage, quality indicators, cluster stats, and
        (when `decided` is supplied) the three-way decision distribution.
    """
    if matches.empty:
        return {}

    n_review_confirmed = 0 if review_matches is None else len(review_matches)
    decision_distribution = None
    if decided is not None:
        dvc = decided["decision"].value_counts() if not decided.empty else {}
        decision_distribution = {
            TIER_AUTO_MERGE: len(matches),
            TIER_HUMAN_REVIEW: int(dvc.get(TIER_HUMAN_REVIEW, 0)) + n_review_confirmed,
            TIER_NO_MATCH: int(dvc.get(TIER_NO_MATCH, 0)),
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
        **(
            {
                "review_match_distribution": dict(
                    sorted(
                        review_matches["match_rule"].value_counts().items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                )
            }
            if review_matches is not None and not review_matches.empty
            else {}
        ),
    }


# ── PairClassifier adapter ──────────────────────────────────────────────────
_CLASSIFICATION_RESULTS_COLS = ("PATID_A", "PATID_B", "model_name", "score", "predicted_tier")


def _to_classification_results(
    frame: pd.DataFrame,
    *,
    model_name: str,
    tier: str | None = None,
    tier_col: str | None = None,
    score_col: str | None = None,
) -> pd.DataFrame:
    """Project a rules-stage frame to the shared 5-col `ClassificationResults`
    shape. Exactly one of `tier` (fixed for every row) or `tier_col` (a
    per-row tier column, e.g. `decided["decision"]`) is used."""
    n = len(frame)
    if n == 0:
        return pd.DataFrame(columns=list(_CLASSIFICATION_RESULTS_COLS))
    predicted_tier = frame[tier_col].to_numpy() if tier_col else tier
    score = frame[score_col].to_numpy() if score_col else np.nan
    return pd.DataFrame(
        {
            "PATID_A": frame["PATID_A"].to_numpy(),
            "PATID_B": frame["PATID_B"].to_numpy(),
            "model_name": model_name,
            "score": score,
            "predicted_tier": predicted_tier,
        }
    )


class DeterministicRulesClassifier:
    """Adapter exposing the deterministic-rules functional API through the
    shared `src.models.base.PairClassifier` interface.

    Delegates entirely to `apply_rules`/`classify_non_matches` — no new
    matching logic lives here. This exists for uniformity with the FS and ML
    matchers (e.g. any future cross-model tooling built against
    `PairClassifier`), not to replace `src/pipeline.py`'s Stage 3, which
    calls the underlying functions directly to produce the richer
    `Matches`/`NonMatches`/`Rejects` artifacts this adapter's minimal 5-col
    frame can't carry (cluster_id, rules_fired, is_suspicious, ...).
    """

    model_name = "deterministic_rules"

    def __init__(
        self,
        ssn_fanout_threshold: int = DEFAULT_SSN_FANOUT_THRESHOLD,
        min_contradictions: int = DEFAULT_REJECT_MIN_CONTRADICTIONS,
    ):
        self.ssn_fanout_threshold = ssn_fanout_threshold
        self.min_contradictions = min_contradictions

    def run(
        self, candidate_pairs: pd.DataFrame, df_clean: pd.DataFrame, **kwargs: object
    ) -> pd.DataFrame:
        """Classify every candidate pair; returns a frame satisfying
        `src.contracts.ClassificationResults`."""
        confirmed = apply_rules(candidate_pairs, df_clean, self.ssn_fanout_threshold)
        decided = classify_non_matches(
            candidate_pairs, confirmed, df_clean, self.min_contradictions
        )
        is_auto = confirmed["match_rule"].isin(AUTO_MERGE_RULES)
        auto = confirmed[is_auto]
        review_confirmed = confirmed[~is_auto]

        return pd.concat(
            [
                _to_classification_results(
                    auto, model_name=self.model_name,
                    tier=TIER_AUTO_MERGE, score_col="confidence",
                ),
                _to_classification_results(
                    review_confirmed, model_name=self.model_name,
                    tier=TIER_HUMAN_REVIEW, score_col="confidence",
                ),
                _to_classification_results(
                    decided, model_name=self.model_name, tier_col="decision",
                ),
            ],
            ignore_index=True,
        )


__all__ = [
    "DEFAULT_REJECT_MIN_CONTRADICTIONS",
    "RejectRule",
    "REJECT_RULES",
    "REJECT_RULE_NAMES",
    "classify_non_matches",
    "get_match_stats",
    "DeterministicRulesClassifier",
]
