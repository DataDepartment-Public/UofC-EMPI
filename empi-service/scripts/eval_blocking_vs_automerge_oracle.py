"""Blocking recall validated against an auto-merge "oracle" rather than labels.

Method: for each auto-merge-tier rule (SSN_DOB, NAME_DOB_EMAIL, NAME_DOB_PHONE),
build the *unblocked* set of pairs that share the rule's strong identifier
exactly (grouping the full cleaned population directly by SSN / email / phone,
with no governance cap and no dependency on the production blocking scheme),
then run the real deterministic rules over just those pairs to get the true
auto-merge-eligible set. This is a labeling-free ground truth specific to the
highest-confidence rule tier: it asks "of every pair the rules would auto-merge
if given the chance, how many did blocking actually surface as a candidate?"

Usage:
    python scripts/eval_blocking_vs_automerge_oracle.py \
        --cleaned data/processed/MDM_Population_cleaned_real_20260620.parquet \
        --candidates data/blocking/candidate_pairs_stacked_frozen.parquet
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import configure_logging
from src.models.deterministic_rules import apply_rules


def _canon(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pairset(df: pd.DataFrame) -> set[tuple[str, str]]:
    return {_canon(a, b) for a, b in zip(df["PATID_A"], df["PATID_B"])}


def _group_pairs(df: pd.DataFrame, id_col: str, key_col: str,
                  max_group: int = 2000) -> pd.DataFrame:
    """All within-group PATID pairs sharing an exact, non-null key value."""
    sub = df[[id_col, key_col]].dropna(subset=[key_col])
    sub = sub[sub[key_col] != ""]
    rows_a, rows_b = [], []
    for _, grp in sub.groupby(key_col):
        ids = grp[id_col].tolist()
        if len(ids) < 2 or len(ids) > max_group:
            continue  # skip pathological hubs (e.g. junk shared values)
        for a, b in itertools.combinations(sorted(ids), 2):
            rows_a.append(a)
            rows_b.append(b)
    return pd.DataFrame({"PATID_A": rows_a, "PATID_B": rows_b})


def _explode_phones(df: pd.DataFrame) -> pd.DataFrame:
    rec = df[["PATID", "Phones_set"]].copy()
    rec["phone"] = rec["Phones_set"].map(
        lambda v: list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else []
    )
    rec = rec.explode("phone").dropna(subset=["phone"])
    rec = rec[rec["phone"] != ""]
    return rec[["PATID", "phone"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path,
                    default=_ROOT / "data/processed/"
                    "MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--candidates", type=Path,
                    default=_ROOT / "data/blocking/candidate_pairs_stacked_frozen.parquet")
    args = ap.parse_args()
    configure_logging(level="WARNING")

    cleaned = pd.read_parquet(args.cleaned)
    valid = cleaned[cleaned["valid_record"]].copy() if "valid_record" in cleaned else cleaned
    candidates = pd.read_parquet(args.candidates)
    candidate_keys = _pairset(candidates)

    print(f"Population: {len(cleaned)} records ({len(valid)} valid)")
    print(f"Blocking candidate set: {len(candidates)} pairs "
          f"({args.candidates.name})\n")

    # SSN_DOB: no name check in this rule, so group by SSN alone is the whole story.
    ssn_pairs = _group_pairs(valid, "PATID", "SSN_clean")
    email_pairs = _group_pairs(valid, "PATID", "Email_clean")
    phone_recs = _explode_phones(valid)
    phone_pairs = _group_pairs(phone_recs, "PATID", "phone")

    oracle_by_rule = {}
    for name, raw_pairs in [("SSN_DOB", ssn_pairs),
                            ("NAME_DOB_EMAIL", email_pairs),
                            ("NAME_DOB_PHONE", phone_pairs)]:
        raw_pairs = raw_pairs.drop_duplicates(
            subset=None,
        )
        # canonicalize + dedupe
        keys = {_canon(a, b) for a, b in zip(raw_pairs.PATID_A, raw_pairs.PATID_B)}
        dedup = pd.DataFrame(list(keys), columns=["PATID_A", "PATID_B"])
        confirmed = apply_rules(dedup, cleaned)
        rule_pairs = confirmed[confirmed["match_rule"] == name]
        oracle_by_rule[name] = _pairset(rule_pairs)
        print(f"{name}: {len(raw_pairs)} raw same-key pairs -> "
              f"{len(rule_pairs)} true auto-merge-eligible pairs")

    all_oracle = set()
    for s in oracle_by_rule.values():
        all_oracle |= s

    print("\n" + "=" * 70)
    print("BLOCKING RECALL vs. AUTO-MERGE ORACLE")
    print("=" * 70)
    print(f"{'Rule':<20}{'Oracle pairs':<16}{'Found in blocking':<20}{'Recall':<10}")
    rows = []
    for name, keys in oracle_by_rule.items():
        found = len(keys & candidate_keys)
        recall = found / len(keys) if keys else None
        rows.append((name, len(keys), found, recall))
        r = f"{recall:.4%}" if recall is not None else "N/A"
        print(f"{name:<20}{len(keys):<16}{found:<20}{r:<10}")
    found_all = len(all_oracle & candidate_keys)
    recall_all = found_all / len(all_oracle) if all_oracle else None
    print(f"{'TOTAL (union)':<20}{len(all_oracle):<16}{found_all:<20}"
          f"{f'{recall_all:.4%}' if recall_all is not None else 'N/A':<10}")

    missing = all_oracle - candidate_keys
    if missing:
        print(f"\n{len(missing)} auto-merge-eligible pairs missing from blocking candidates:")
        for a, b in list(missing)[:10]:
            print(f"  {a}, {b}")


if __name__ == "__main__":
    main()
