"""Blocking-recall evaluation: deterministic rules as ground truth.

The deterministic rules are high precision, so any pair a rule confirms is
effectively a ground-truth positive. If a rule-confirmed pair is NOT among
production blocking's candidate pairs, blocking missed a true match — a recall
gap. This module quantifies that gap.

    1. Build a candidate set WIDER than production blocking, so it contains pairs
       the 8-block scheme would never emit:
         * `method="loose"`  — block the full dataset on single loose keys (exact
           DOB, Soundex(last name)) to surface pairs the 8-block scheme splits.
           Group-bounded, so it scales to the full dataset (the default workhorse).
         * `method="sample"` — take a random sample of S records and generate every
           within-sample pair (C(S,2)); an unbiased local ground truth. This is
           O(S^2) and fuzzy rule-matching runs per pair, so keep S small (a few
           hundred) — it is a cross-check, not a full-data scan.
    2. Run the deterministic rules over the wide set -> R (rule-confirmed pairs).
    3. Run production blocking over the same records -> B (candidate pairs).
    4. Blocking misses = R - B; recall = |R ∩ B| / |R|.

Reports estimated recall, which rule each missed pair fired (where blocking
under-covers), and which production blocks catch the hits.

CAVEAT: this measures recall only against matches the *deterministic rules* can
detect. True matches that neither blocking nor the rules catch are invisible to
it — it is a lower bound on the work the probabilistic stage must still do.

CLI:
    python -m src.evaluation.blocking_eval --run-id <run_id>                # loose
    python -m src.evaluation.blocking_eval --run-id <run_id> --method sample --n 400
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import jellyfish
import pandas as pd

from src.config.config import settings
from src.features.blocking import run_batch_blocking
from src.models.deterministic_rules import apply_rules

logger = logging.getLogger(__name__)

COL_PATID = "PATID"
COL_LAST_NM = "LastNM_clean"
COL_BIRTH_DT = "BirthDT_clean"


# ── Pair-set helpers (canonical (PATID_A, PATID_B) ordering, as blocking emits) ──
def _canonical(a: str, b: str) -> tuple[str, str]:
    """Canonical pair key: the two PATIDs in sorted order (matches blocking)."""
    return (a, b) if a <= b else (b, a)


def _pairset(pairs: pd.DataFrame) -> set[tuple[str, str]]:
    """Canonical (PATID_A, PATID_B) set from a candidate-pairs frame."""
    if pairs.empty:
        return set()
    return {_canonical(a, b) for a, b in zip(pairs["PATID_A"], pairs["PATID_B"])}


def _soundex(value: object) -> str | None:
    """Soundex of a name, or None when absent (mirrors blocking's loose key)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return jellyfish.soundex(text)
    except (ValueError, UnicodeDecodeError):
        return None


# ── Wide candidate generation ───────────────────────────────────────────────────
def _pairs_from_groups(
    df: pd.DataFrame, key: pd.Series, max_group: int
) -> set[tuple[str, str]]:
    """All within-group canonical pairs for a grouping key (governance-capped)."""
    out: set[tuple[str, str]] = set()
    work = pd.DataFrame({COL_PATID: df[COL_PATID].to_numpy(), "_key": key.to_numpy()})
    work = work.dropna(subset=["_key"])
    for _, grp in work.groupby("_key", sort=False):
        patids = sorted(grp[COL_PATID])
        if len(patids) < 2 or len(patids) > max_group:
            continue
        out.update(combinations(patids, 2))
    return out


def wide_candidate_pairs(
    cleaned: pd.DataFrame,
    method: str = "loose",
    sample_size: int = 500,
    seed: int = 7,
    max_group: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a candidate set wider than production blocking.

    Returns ``(wide_pairs, records)`` where ``records`` is the record subset the
    evaluation runs over (the sample for ``method="sample"``, the full frame for
    ``method="loose"``), so production blocking is compared over the same records.
    """
    max_group = max_group or settings.governance_threshold

    if method == "sample":
        n = min(sample_size, len(cleaned))
        records = cleaned.sample(n=n, random_state=seed).reset_index(drop=True)
        pairs = {
            _canonical(a, b)
            for a, b in combinations(sorted(records[COL_PATID]), 2)
        }
    elif method == "loose":
        records = cleaned
        dob_pairs = _pairs_from_groups(
            records, records[COL_BIRTH_DT].astype("string"), max_group
        )
        last_sx = records[COL_LAST_NM].map(_soundex)
        name_pairs = _pairs_from_groups(records, last_sx, max_group)
        pairs = dob_pairs | name_pairs
    else:
        raise ValueError(f"method must be 'sample' or 'loose', got {method!r}")

    wide = pd.DataFrame(sorted(pairs), columns=["PATID_A", "PATID_B"])
    logger.info(
        "Wide candidate set (%s): %d pairs over %d records.",
        method, len(wide), len(records),
    )
    return wide, records


# ── Report ──────────────────────────────────────────────────────────────────────
@dataclass
class BlockingEvalReport:
    """Blocking-recall evaluation result for one cleaned dataset."""

    headline: dict = field(default_factory=dict)
    recall: float | None = None
    missed_by_rule: dict = field(default_factory=dict)
    hits_by_block: dict = field(default_factory=dict)
    missed_examples: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_text(self) -> str:
        L = ["=" * 60, "  BLOCKING RECALL — RULES AS GROUND TRUTH", "=" * 60]
        for k, v in self.headline.items():
            L.append(f"  {k:<24}{v}")
        L += ["", f"  ESTIMATED BLOCKING RECALL: "
                  f"{'n/a' if self.recall is None else f'{self.recall:.1f}%'}"]
        if self.missed_by_rule:
            L += ["", "  MISSED CONFIRMED PAIRS BY RULE (blocking under-covers)"]
            L += [f"    {r:<20}{c}" for r, c in self.missed_by_rule.items()]
        if self.hits_by_block:
            L += ["", "  CAUGHT CONFIRMED PAIRS BY PRODUCTION BLOCK"]
            L += [f"    {b:<20}{c}" for b, c in self.hits_by_block.items()]
        if not self.missed_examples.empty:
            L += ["", "  SAMPLE OF MISSED PAIRS",
                  self.missed_examples.to_string(index=False)]
        L += ["", "  NOTE: recall is measured only against rule-detectable matches."]
        return "\n".join(L)


def evaluate_blocking(
    cleaned: pd.DataFrame,
    method: str = "loose",
    sample_size: int = 500,
    seed: int = 7,
    n_examples: int = 15,
) -> BlockingEvalReport:
    """Estimate production blocking's recall against rule-confirmed matches."""
    wide, records = wide_candidate_pairs(cleaned, method, sample_size, seed)

    confirmed = apply_rules(wide, records)  # R — ground-truth positives
    production = run_batch_blocking(records)  # B — production candidate pairs

    r_keys = _pairset(confirmed)
    b_keys = _pairset(production)
    caught = r_keys & b_keys
    missed = r_keys - b_keys

    recall = round(100 * len(caught) / len(r_keys), 1) if r_keys else None

    # Which rule fired on each pair, keyed canonically for set lookups.
    rule_by_pair: dict[tuple[str, str], str] = {}
    if not confirmed.empty:
        for a, b, rule in zip(
            confirmed["PATID_A"], confirmed["PATID_B"], confirmed["match_rule"]
        ):
            rule_by_pair[_canonical(a, b)] = rule

    missed_by_rule = (
        pd.Series([rule_by_pair[p] for p in missed])
        .value_counts().to_dict() if missed else {}
    )

    # Per-production-block attribution of the caught pairs.
    hits_by_block: dict[str, int] = {}
    if caught and "source_blocks" in production.columns:
        blocks_by_pair = {
            _canonical(a, b): sb
            for a, b, sb in zip(
                production["PATID_A"], production["PATID_B"],
                production["source_blocks"],
            )
        }
        for p in caught:
            for blk in str(blocks_by_pair.get(p, "")).split("|"):
                if blk:
                    hits_by_block[blk] = hits_by_block.get(blk, 0) + 1
        hits_by_block = dict(
            sorted(hits_by_block.items(), key=lambda kv: kv[1], reverse=True)
        )

    missed_examples = pd.DataFrame(
        [
            {"PATID_A": p[0], "PATID_B": p[1], "match_rule": rule_by_pair[p]}
            for p in sorted(missed)[:n_examples]
        ]
    )

    headline = {
        "method": method,
        "records_evaluated": len(records),
        "wide_pairs": len(wide),
        "production_pairs": len(production),
        "rule_confirmed (R)": len(r_keys),
        "caught (R ∩ B)": len(caught),
        "missed (R − B)": len(missed),
    }
    return BlockingEvalReport(
        headline=headline,
        recall=recall,
        missed_by_rule=missed_by_rule,
        hits_by_block=hits_by_block,
        missed_examples=missed_examples,
    )


def _load_cleaned(run_id: str, root: Path) -> pd.DataFrame:
    return pd.read_parquet(
        root / f"data/processed/MDM_Population_cleaned_{run_id}.parquet"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Blocking-recall evaluation (rules as ground truth)"
    )
    ap.add_argument("--run-id", required=True,
                    help="Run id of the cleaned-dataset artifact to evaluate")
    ap.add_argument("--method", choices=("loose", "sample"), default="loose",
                    help="Wide-candidate strategy (default: loose)")
    ap.add_argument("--n", type=int, default=500,
                    help="Sample size for method=sample (kept small: O(S^2))")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    cleaned = _load_cleaned(args.run_id, root)
    report = evaluate_blocking(
        cleaned, method=args.method, sample_size=args.n, seed=args.seed
    )
    print(report.to_text())


if __name__ == "__main__":
    main()
