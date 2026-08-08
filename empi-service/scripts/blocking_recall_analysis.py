"""Blocking-recall residual analysis + first-name-anchored recovery-block study.

ANALYSIS ONLY — this does NOT modify production blocking (`src/preprocessing/
blocking.py`). It quantifies the recall gap from two angles the rules-as-truth
RCA (`Blocking-Recall-RCA.md`) could not, and measures candidate B10 recovery
blocks head-to-head.

Part A — pattern of the misses
  * Synthetic: of the 16k planted duplicates, profile the ones blocking missed by
    `case_type` and by which fields are corrupted vs missing (independent of what
    the deterministic rules can confirm).
  * Real: re-derive the rules-as-truth residual (R − B over the valid population)
    and add a strong-identifier missingness breakdown.

Part B — recovery-block variants (first name + DOB, last-name-free)
  * B10a Soundex(first) + full DOB              (RCA recommendation; unconditional)
  * B10b FirstNM exact + full DOB               (unconditional)
  * B10c FirstNM exact + full DOB, SSN-missing gate   (the user's idea)
  * B10d Soundex(first) + full DOB, SSN-missing gate
  Each is scored on BOTH populations for matches-recovered vs candidate cost
  (gross + net-new after dedup against the existing 8-block output).

Usage:
    python scripts/blocking_recall_analysis.py [--skip-real]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import jellyfish
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import configure_logging, settings
from src.evaluation.blocking_eval import wide_candidate_pairs
from src.models.deterministic_rules import apply_rules
from src.preprocessing.blocking import run_batch_blocking
from scripts.eval_against_labels import _canon, _pairset, _synthetic_records

logger = logging.getLogger(__name__)

_CAP = settings.governance_threshold


# ── shared key helpers ──────────────────────────────────────────────────────────
def _dob_str(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def _soundex(v) -> str | None:
    if pd.isna(v) or not str(v).strip():
        return None
    try:
        return jellyfish.soundex(str(v))
    except (ValueError, UnicodeDecodeError):
        return None


def _first_dob_pairs(records: pd.DataFrame, first: str) -> set[tuple[str, str]]:
    """Canonical within-group pairs for a first-name + full-DOB block.

    `first` is "soundex" or "exact". Groups are governance-capped like production.
    """
    dob = _dob_str(records["BirthDT_clean"])
    if first == "soundex":
        fkey = records["FirstNM_clean"].map(_soundex)
    else:
        fkey = records["FirstNM_clean"].map(
            lambda x: str(x).upper() if pd.notna(x) and str(x).strip() else None
        )
    work = pd.DataFrame({
        "PATID": records["PATID"].to_numpy(),
        "k": (fkey.astype("string") + "|" + dob.astype("string")).to_numpy(),
    }).dropna(subset=["k"])
    out: set[tuple[str, str]] = set()
    for _, grp in work.groupby("k", sort=False):
        patids = sorted(grp["PATID"])
        if len(patids) < 2 or len(patids) > _CAP:
            continue
        out.update(combinations(patids, 2))
    return out


def _ssn_missing_map(records: pd.DataFrame) -> dict[str, bool]:
    """PATID -> True when SSN_clean is null (the 'missingness' gate field)."""
    return {
        p: pd.isna(s)
        for p, s in zip(records["PATID"], records["SSN_clean"])
    }


def _gate_ssn_missing(
    pairs: set[tuple[str, str]], ssn_missing: dict[str, bool]
) -> set[tuple[str, str]]:
    """Keep pairs where SSN is missing on at least one side."""
    return {
        (a, b) for a, b in pairs
        if ssn_missing.get(a, True) or ssn_missing.get(b, True)
    }


def _variants(records: pd.DataFrame) -> dict[str, set[tuple[str, str]]]:
    ssn_missing = _ssn_missing_map(records)
    a = _first_dob_pairs(records, "soundex")
    b = _first_dob_pairs(records, "exact")
    return {
        "B10a soundex(first)+DOB": a,
        "B10b first(exact)+DOB": b,
        "B10c first(exact)+DOB | SSN-missing": _gate_ssn_missing(b, ssn_missing),
        "B10d soundex(first)+DOB | SSN-missing": _gate_ssn_missing(a, ssn_missing),
    }


def _score_variants(
    records: pd.DataFrame,
    missed: set[tuple[str, str]],
    existing: set[tuple[str, str]],
) -> dict:
    out = {}
    for name, pairs in _variants(records).items():
        out[name] = {
            "recovered": len(pairs & missed),
            "of_missed": len(missed),
            "gross_pairs": len(pairs),
            "net_new_pairs": len(pairs - existing),
        }
    return out


# ── Part A field-profile helpers (synthetic) ─────────────────────────────────────
def _status(left, right) -> str:
    lp, rp = pd.notna(left), pd.notna(right)
    if not lp or not rp:
        return "missing"
    return "agree" if str(left) == str(right) else "differ"


def _phone_overlap(l, r) -> bool:
    sl = set(str(l).replace(",", " ").split()) if pd.notna(l) else set()
    sr = set(str(r).replace(",", " ").split()) if pd.notna(r) else set()
    return bool(sl & sr)


def _synthetic_miss_profile(syn: pd.DataFrame, missed: set[tuple[str, str]]) -> dict:
    pos = syn[syn["label"].astype(int) == 1].copy()
    pos["key"] = [_canon(a, b) for a, b in zip(pos.PATID_A, pos.PATID_B)]
    pos["missed"] = pos["key"].isin(missed)
    miss = pos[pos["missed"]]

    # by case_type: miss rate
    by_ct = []
    for ct, grp in pos.groupby("case_type"):
        m = int(grp["missed"].sum())
        if m:
            by_ct.append({"case_type": ct, "n": int(len(grp)), "missed": m,
                          "miss_pct": round(100 * m / len(grp), 1)})
    by_ct.sort(key=lambda d: d["missed"], reverse=True)

    # field-status tallies on the missed pairs
    def tally(colbase, cmp_phone=False):
        c = {}
        for _, r in miss.iterrows():
            if cmp_phone:
                st = "agree" if _phone_overlap(
                    r["Phones_set_l"], r["Phones_set_r"]) else "no-overlap"
            else:
                st = _status(r[f"{colbase}_l"], r[f"{colbase}_r"])
            c[st] = c.get(st, 0) + 1
        return c

    # does the missed pair have ANY agreeing strong identifier?
    def has_strong(r) -> bool:
        if _status(r["SSN_clean_l"], r["SSN_clean_r"]) == "agree":
            return True
        if _status(r["Email_clean_l"], r["Email_clean_r"]) == "agree":
            return True
        if _phone_overlap(r["Phones_set_l"], r["Phones_set_r"]):
            return True
        return False

    n_no_strong = int((~miss.apply(has_strong, axis=1)).sum()) if len(miss) else 0
    n_ssn_missing = int(sum(
        _status(r["SSN_clean_l"], r["SSN_clean_r"]) == "missing"
        for _, r in miss.iterrows()
    ))
    return {
        "n_missed": int(len(miss)),
        "by_case_type": by_ct,
        "last_name_status": tally("LastNM_clean"),
        "first_name_status": tally("FirstNM_clean"),
        "dob_status": tally("BirthDT_clean"),
        "phone_status": tally(None, cmp_phone=True),
        "no_agreeing_strong_id": n_no_strong,
        "ssn_missing_pairs": n_ssn_missing,
    }


# ── orchestration ────────────────────────────────────────────────────────────────
def run_synthetic(syn: pd.DataFrame) -> dict:
    records = _synthetic_records(syn)
    blocked = _pairset(run_batch_blocking(records))
    pos = syn[syn["label"].astype(int) == 1]
    pos_keys = _pairset(pos)
    missed = pos_keys - blocked
    return {
        "records": int(len(records)),
        "positive_pairs": len(pos_keys),
        "missed_positives": len(missed),
        "profile": _synthetic_miss_profile(syn, missed),
        "variants": _score_variants(records, missed, blocked),
    }


def run_real(cleaned: pd.DataFrame) -> dict:
    wide, records = wide_candidate_pairs(cleaned, method="loose")
    confirmed = apply_rules(wide, records)
    blocked = _pairset(run_batch_blocking(records))
    confirmed_keys = _pairset(confirmed)
    missed = confirmed_keys - blocked

    # strong-id missingness on the missed (rule-confirmed) pairs
    attrs = records.drop_duplicates("PATID").set_index("PATID")
    ssn = attrs["SSN_clean"]
    n_ssn_missing = sum(
        pd.isna(ssn.get(a)) or pd.isna(ssn.get(b)) for a, b in missed
    )
    return {
        "valid_records": int(len(records)),
        "rule_confirmed_R": len(confirmed_keys),
        "caught": len(confirmed_keys & blocked),
        "missed_R_minus_B": len(missed),
        "missed_with_ssn_missing_either_side": int(n_ssn_missing),
        "variants": _score_variants(records, missed, blocked),
    }


def _print(res: dict) -> None:
    def line(): print("-" * 70)

    print("=" * 70)
    print("  PART A/B — SYNTHETIC (planted duplicates, label-true)")
    print("=" * 70)
    s = res["synthetic"]
    print(f"  records={s['records']}  positives={s['positive_pairs']}  "
          f"missed={s['missed_positives']}")
    p = s["profile"]
    print(f"\n  Missed-positive field profile (n={p['n_missed']}):")
    print(f"    last name : {p['last_name_status']}")
    print(f"    first name: {p['first_name_status']}")
    print(f"    DOB       : {p['dob_status']}")
    print(f"    phone     : {p['phone_status']}")
    print(f"    pairs with NO agreeing strong id (SSN/email/phone): "
          f"{p['no_agreeing_strong_id']}/{p['n_missed']}")
    print("\n  Top missed case_types:")
    for d in p["by_case_type"][:12]:
        print(f"    {d['case_type']:<28} missed={d['missed']:<4} "
              f"of {d['n']:<4} ({d['miss_pct']}%)")
    print("\n  Recovery-block variants (synthetic):")
    line()
    print(f"  {'variant':<40}{'recov':>7}{'gross':>9}{'net-new':>9}")
    line()
    for name, d in s["variants"].items():
        print(f"  {name:<40}{d['recovered']:>4}/{d['of_missed']:<4}"
              f"{d['gross_pairs']:>9}{d['net_new_pairs']:>9}")

    if "real" in res:
        print("\n" + "=" * 70)
        print("  PART A/B — REAL DATA (rules-as-truth residual)")
        print("=" * 70)
        r = res["real"]
        print(f"  valid_records={r['valid_records']}  R={r['rule_confirmed_R']}  "
              f"caught={r['caught']}  missed={r['missed_R_minus_B']}")
        print("  SSN-related missingness metrics are redacted from console output.")
        print("\n  Recovery-block variants (real):")
        line()
        print(f"  {'variant':<40}{'recov':>7}{'gross':>9}{'net-new':>9}")
        line()
        for name, d in r["variants"].items():
            print(f"  {name:<40}{d['recovered']:>4}/{d['of_missed']:<4}"
                  f"{d['gross_pairs']:>9}{d['net_new_pairs']:>9}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cleaned", type=Path,
                    default=_ROOT / "data/processed/"
                    "MDM_Population_cleaned_real_20260620.parquet")
    ap.add_argument("--synthetic", type=Path,
                    default=_ROOT / "data/raw/synthetic_data.csv")
    ap.add_argument("--skip-real", action="store_true",
                    help="Skip the slow real-data rules-as-truth residual pass.")
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "data/runs/blocking_recall_analysis.json")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    syn = pd.read_csv(args.synthetic, dtype=str, keep_default_na=True, na_values=[""])
    syn["label"] = syn["label"].astype(int)
    res = {"synthetic": run_synthetic(syn)}

    if not args.skip_real:
        cleaned = pd.read_parquet(args.cleaned)
        res["real"] = run_real(cleaned)

    _print(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
