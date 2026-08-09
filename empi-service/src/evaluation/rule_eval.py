"""Automated evaluation & clerical adjudication of the deterministic rules.

This module turns the one-off analysis scripts into a reusable, importable
component. Given a confirmed-matches frame and the cleaned dataset it produces:

  * `agreement_profile`   — per-rule, how often the *other* identifying fields
                            also agree (the classic deterministic-rule audit).
  * `value_fanout`        — how many distinct patients share each SSN/email
                            value (placeholder / shared-account detection).
  * `cluster_profile`     — connected-component size distribution.
  * `adjudicate`          — AI-assisted clerical review of a sampled subset:
                            SAME / DIFFERENT / UNCERTAIN per pair from a
                            documented, holistic field-comparison rubric.
  * `precision_report`    — per-rule precision proxy = SAME / (SAME+DIFFERENT).
  * `evaluate`            — runs all of the above and returns one `RuleEvalReport`.

IMPORTANT — the adjudicator is a heuristic proxy, NOT ground truth. There are no
labels for this data. It encodes a reviewer's reasoning (name typos/nicknames,
DOB transposition, placeholder SSNs, shared-email families) to flag likely false
positives at scale; ambiguous pairs are returned as UNCERTAIN, not forced.

CLI:
    python -m src.evaluation.rule_eval --run-id eval_20260615 --n 300
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import jellyfish
import numpy as np
import pandas as pd

from src.config import configure_logging, settings

logger = logging.getLogger(__name__)

# Cleaned-frame column names this module reads.
COL_PATID = "PATID"
FIELDS = {
    "first": "FirstNM_clean",
    "last": "LastNM_clean",
    "dob": "BirthDT_clean",
    "ssn": "SSN_clean",
    "email": "Email_clean",
    "address": "AddressLine1_clean",
    "sex": "SexAtBirthDSC_clean",
}
COL_PHONES = "Phones_set"

# Adjudication thresholds (calibrated against a hand-labeled sample). Sourced
# from central config; override via EMPI_ADJ_* env vars.
_FN_STRONG = settings.adj_first_strong       # first-name sim treated as same name
_FN_OK = settings.adj_first_ok               # first-name sim ok with corroboration
_LN_OK = settings.adj_last_ok                # last-name sim ok (maiden-name)
_FN_DIFFERENT = settings.adj_first_different  # below this, distinct people


# ── value helpers ─────────────────────────────────────────────────────────────
def _s(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


def _parse_phones(v) -> set[str]:
    if isinstance(v, (set, frozenset, list, tuple, np.ndarray)):
        return {str(p).strip() for p in v if str(p).strip()}
    return set()


def _placeholder_ssn(s: str | None) -> bool:
    """Low-entropy SSN (<=2 distinct digits or one digit dominating)."""
    if not s:
        return False
    return len(set(s)) <= 2 or Counter(s).most_common(1)[0][1] >= 7


def _name_sim(a, b) -> float:
    a, b = _s(a), _s(b)
    if not a or not b:
        return float("nan")
    a, b = a.upper(), b.upper()
    if a == b:
        return 1.0
    if a in b or b in a:  # truncation / prefix
        return 0.92
    return jellyfish.jaro_winkler_similarity(a, b)


def _dob_match(a, b) -> bool:
    a, b = _s(a), _s(b)
    if not a or not b:
        return False
    da, db = a[:10], b[:10]
    if da == db:
        return True
    try:  # day/month transposition (data-entry error)
        ya, ma, dda = da.split("-")
        yb, mb, ddb = db.split("-")
        return ya == yb and {ma, dda} == {mb, ddb}
    except ValueError:
        return False


# ── data assembly ─────────────────────────────────────────────────────────────
def _materialize(matches: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FIELDS.values() if c in cleaned.columns]
    attrs = cleaned.drop_duplicates(COL_PATID).set_index(COL_PATID)[cols]
    m = matches.join(attrs.add_suffix("_A"), on="PATID_A").join(
        attrs.add_suffix("_B"), on="PATID_B"
    )
    if COL_PHONES in cleaned.columns:
        pm = {p: _parse_phones(v) for p, v in zip(cleaned[COL_PATID], cleaned[COL_PHONES])}
        m["_phone_overlap"] = [
            bool(pm.get(a, set()) & pm.get(b, set()))
            for a, b in zip(m["PATID_A"], m["PATID_B"])
        ]
    else:
        m["_phone_overlap"] = False
    return m


def _agree(m: pd.DataFrame, key: str) -> pd.Series:
    if key == "phone":
        return m["_phone_overlap"]
    col = FIELDS[key]
    a, b = f"{col}_A", f"{col}_B"
    if a not in m.columns:
        return pd.Series(False, index=m.index)
    return (m[a] == m[b]) & m[a].notna() & m[b].notna()


# ── public analyses ───────────────────────────────────────────────────────────
def agreement_profile(matches: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Per-rule agreement rate (%) for every identifying field."""
    m = _materialize(matches, cleaned)
    keys = ["first", "last", "dob", "ssn", "email", "phone", "address", "sex"]
    rows = []
    for rule, n in matches["match_rule"].value_counts().items():
        sub = m["match_rule"] == rule
        row = {"rule": rule, "n": int(n)}
        for k in keys:
            row[k] = round(100 * _agree(m, k)[sub].mean(), 1) if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def value_fanout(cleaned: pd.DataFrame, column: str, top: int = 10) -> dict:
    """Distinct-patient fan-out for a matching value (e.g. SSN_clean, Email_clean)."""
    if column not in cleaned.columns:
        return {}
    vc = cleaned[cleaned[column].notna()][column].value_counts()
    shared = vc[vc > 1]
    return {
        "distinct_shared": int(len(shared)),
        "max_fanout": int(vc.max()) if len(vc) else 0,
        "top": shared.head(top).to_dict(),
    }


def cluster_profile(matches: pd.DataFrame) -> dict:
    """Connected-component size distribution over confirmed matches."""
    if matches.empty or "cluster_id" not in matches.columns:
        return {}
    pid = pd.concat([
        matches[["PATID_A", "cluster_id"]].rename(columns={"PATID_A": "PATID"}),
        matches[["PATID_B", "cluster_id"]].rename(columns={"PATID_B": "PATID"}),
    ]).drop_duplicates()
    sizes = pid.groupby("cluster_id")["PATID"].nunique()
    return {
        "n_clusters": int(sizes.size),
        "size_2": int((sizes == 2).sum()),
        "size_3_5": int(sizes.between(3, 5).sum()),
        "size_6_20": int(sizes.between(6, 20).sum()),
        "size_gt_20": int((sizes > 20).sum()),
        "max_size": int(sizes.max()),
    }


# ── adjudication ──────────────────────────────────────────────────────────────
def _adjudicate_row(row: pd.Series) -> tuple[str, str]:
    """Holistic SAME / DIFFERENT / UNCERTAIN verdict for one pair."""
    fn = _name_sim(row.get("FirstNM_clean_A"), row.get("FirstNM_clean_B"))
    ln = _name_sim(row.get("LastNM_clean_A"), row.get("LastNM_clean_B"))
    fn = 0.0 if np.isnan(fn) else fn
    ln = 0.0 if np.isnan(ln) else ln
    dob = _dob_match(row.get("BirthDT_clean_A"), row.get("BirthDT_clean_B"))
    sa, sb = _s(row.get("SexAtBirthDSC_clean_A")), _s(row.get("SexAtBirthDSC_clean_B"))
    sex_conflict = bool(sa and sb and sa != sb)
    phones = bool(row.get("_phone_overlap"))
    aa, ab = _s(row.get("AddressLine1_clean_A")), _s(row.get("AddressLine1_clean_B"))
    addr = bool(aa and ab and aa == ab)
    na, nb = _s(row.get("SSN_clean_A")), _s(row.get("SSN_clean_B"))
    real_ssn = bool(na and nb and na == nb and not _placeholder_ssn(na))

    name_ok = fn >= _FN_OK and ln >= _LN_OK

    if name_ok and (dob or real_ssn or phones or addr):
        return "SAME", "name + corroborator"
    if fn >= _FN_STRONG and ln >= _FN_STRONG:
        return "SAME", "strong name match"
    if fn < _FN_DIFFERENT and (sex_conflict or not dob):
        return "DIFFERENT", "distinct first name + sex/DOB conflict"
    if _placeholder_ssn(na) and fn < _FN_OK:
        return "DIFFERENT", "placeholder SSN + name mismatch"
    if not (dob or phones or addr or real_ssn) and fn < _FN_OK:
        return "DIFFERENT", "no corroborator + name mismatch"
    return "UNCERTAIN", f"fn={fn:.2f} ln={ln:.2f} dob={dob}"


def adjudicate_pairs(pairs: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Assign a clerical verdict to an arbitrary `(PATID_A, PATID_B)` frame.

    Works on any pair set — confirmed matches OR rejected non-matches — so the
    same rubric measures both precision (on matches) and the false-negative rate
    (on pairs the rules declined).
    """
    if pairs.empty:
        return pairs.copy()
    m = _materialize(pairs, cleaned)
    verdicts = m.apply(_adjudicate_row, axis=1)
    m["verdict"] = verdicts.map(lambda t: t[0])
    m["verdict_reason"] = verdicts.map(lambda t: t[1])
    return m


def adjudicate(
    matches: pd.DataFrame,
    cleaned: pd.DataFrame,
    n: int = settings.adj_sample_size,
    seed: int = 7,
) -> pd.DataFrame:
    """Sample up to `n` matches per rule and assign a clerical verdict to each.

    Returns the sampled matches with `verdict` and `verdict_reason` columns plus
    the materialized A/B fields, so callers can export a review workbook.
    """
    out = []
    for rule in sorted(matches["match_rule"].unique()):
        sub = matches[matches["match_rule"] == rule]
        sub = sub.sample(n=min(n, len(sub)), random_state=seed)
        out.append(sub)
    if not out:
        return pd.DataFrame()
    return adjudicate_pairs(pd.concat(out), cleaned)


def precision_report(adjudicated: pd.DataFrame) -> pd.DataFrame:
    """Per-rule SAME/DIFFERENT/UNCERTAIN counts and precision proxy."""
    rows = []
    for rule in sorted(adjudicated["match_rule"].unique()):
        sub = adjudicated[adjudicated["match_rule"] == rule]["verdict"].value_counts()
        same = int(sub.get("SAME", 0))
        diff = int(sub.get("DIFFERENT", 0))
        unc = int(sub.get("UNCERTAIN", 0))
        decided = same + diff
        rows.append({
            "rule": rule, "n": same + diff + unc,
            "SAME": same, "DIFFERENT": diff, "UNCERTAIN": unc,
            "precision_pct": round(100 * same / decided, 1) if decided else None,
        })
    return pd.DataFrame(rows)


def precision_ci(
    adjudicated: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Per-rule precision proxy with a bootstrap 95% confidence interval.

    Resamples the decided (SAME/DIFFERENT) verdicts to quantify how much the
    point estimate could move given the sample size.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rule in sorted(adjudicated["match_rule"].unique()):
        v = adjudicated[adjudicated["match_rule"] == rule]["verdict"]
        decided = v[v.isin(["SAME", "DIFFERENT"])]
        n = len(decided)
        if n == 0:
            rows.append({"rule": rule, "decided": 0, "precision_pct": None,
                         "ci95_low": None, "ci95_high": None})
            continue
        same = (decided == "SAME").to_numpy()
        boot = [rng.choice(same, size=n, replace=True).mean() for _ in range(n_boot)]
        rows.append({
            "rule": rule, "decided": n,
            "precision_pct": round(100 * same.mean(), 1),
            "ci95_low": round(100 * float(np.percentile(boot, 2.5)), 1),
            "ci95_high": round(100 * float(np.percentile(boot, 97.5)), 1),
        })
    return pd.DataFrame(rows)


def confidence_calibration(
    adjudicated: pd.DataFrame, rule_confidence: dict[str, float]
) -> pd.DataFrame:
    """Compare each rule's assigned confidence to its adjudicated precision."""
    prec = precision_report(adjudicated).set_index("rule")
    rows = []
    for rule, conf in rule_confidence.items():
        if rule not in prec.index:
            continue
        p = prec.loc[rule, "precision_pct"]
        rows.append({
            "rule": rule,
            "assigned_confidence": conf,
            "adjudicated_precision_pct": p,
            "gap": round(conf * 100 - p, 1) if p is not None else None,
        })
    return pd.DataFrame(rows)


def rule_overlap(matches: pd.DataFrame) -> pd.DataFrame:
    """Co-occurrence counts of rules within `rules_fired`, plus sole-fire counts."""
    fired = matches["rules_fired"].str.split("|")
    rules = sorted({r for lst in fired for r in lst})
    idx = {r: i for i, r in enumerate(rules)}
    mat = np.zeros((len(rules), len(rules)), dtype=int)
    sole = Counter()
    for lst in fired:
        if len(lst) == 1:
            sole[lst[0]] += 1
        for a in lst:
            for b in lst:
                mat[idx[a], idx[b]] += 1
    df = pd.DataFrame(mat, index=rules, columns=rules)
    df["__sole__"] = [sole.get(r, 0) for r in rules]
    df["__sole_pct__"] = [
        round(100 * sole.get(r, 0) / df.loc[r, r], 1) if df.loc[r, r] else 0.0
        for r in rules
    ]
    return df


def suspicious_breakdown(matches: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Typology of suspicious matches: which field disagrees, and DOB severity."""
    if "is_suspicious" not in matches.columns or not matches["is_suspicious"].any():
        return {}
    susp = matches[matches["is_suspicious"]]
    m = _materialize(susp, cleaned)

    def _dis(key):
        c = FIELDS[key]
        a, b = m[f"{c}_A"], m[f"{c}_B"]
        return a.notna() & b.notna() & (a != b)

    last_d, dob_d, ssn_d = _dis("last"), _dis("dob"), _dis("ssn")

    # DOB-difference severity: transposition / single-day-or-month typo vs other.
    def _minor_dob(a, b):
        a, b = _s(a), _s(b)
        if not a or not b:
            return False
        da, db = a[:10], b[:10]
        if _dob_match(a, b):  # transposition counts as recoverable
            return True
        try:
            ya, ma, dda = da.split("-")
            yb, mb, ddb = db.split("-")
        except ValueError:
            return False
        if ya == yb and ma == mb and abs(int(dda) - int(ddb)) <= 1:
            return True
        if ya == yb and dda == ddb and abs(int(ma) - int(mb)) <= 1:
            return True
        return False

    dob_pairs = m[dob_d]
    minor = sum(
        _minor_dob(r[f"{FIELDS['dob']}_A"], r[f"{FIELDS['dob']}_B"])
        for _, r in dob_pairs.iterrows()
    )
    return {
        "total_suspicious": int(len(susp)),
        "last_name_only": int((last_d & ~dob_d & ~ssn_d).sum()),
        "dob_only": int((dob_d & ~last_d & ~ssn_d).sum()),
        "ssn_only": int((ssn_d & ~last_d & ~dob_d).sum()),
        "multi_field": int(((last_d.astype(int) + dob_d.astype(int)
                             + ssn_d.astype(int)) >= 2).sum()),
        "dob_diffs_total": int(dob_d.sum()),
        "dob_diffs_minor_typo": int(minor),
    }


def false_negative_estimate(
    non_matches: pd.DataFrame, cleaned: pd.DataFrame, n: int = 1000, seed: int = 7
) -> dict:
    """Adjudicate a sample of rejected pairs to estimate missed true matches.

    Pairs that blocking surfaced but no deterministic rule confirmed. A SAME
    verdict here is a deterministic-stage false negative — a pair the downstream
    probabilistic stage would need to recover. Reported as a rate, with a
    per-source-block breakdown of where the misses concentrate.
    """
    if non_matches.empty:
        return {}
    sample = non_matches.sample(n=min(n, len(non_matches)), random_state=seed)
    adj = adjudicate_pairs(sample, cleaned)
    vc = adj["verdict"].value_counts()
    same = int(vc.get("SAME", 0))
    diff = int(vc.get("DIFFERENT", 0))
    unc = int(vc.get("UNCERTAIN", 0))
    by_block = {}
    if "source_blocks" in adj.columns:
        for blk, grp in adj.groupby(adj["source_blocks"]):
            s = int((grp["verdict"] == "SAME").sum())
            by_block[blk] = {"n": int(len(grp)), "same": s}
    return {
        "sampled": int(len(sample)),
        "SAME": same, "DIFFERENT": diff, "UNCERTAIN": unc,
        "missed_match_rate_pct": round(100 * same / len(sample), 1),
        "top_blocks_by_same": dict(sorted(
            by_block.items(), key=lambda kv: kv[1]["same"], reverse=True)[:8]),
    }


# ── orchestration ─────────────────────────────────────────────────────────────
@dataclass
class RuleEvalReport:
    """Bundle of every evaluation artifact for one matches frame."""

    headline: dict = field(default_factory=dict)
    agreement: pd.DataFrame = field(default_factory=pd.DataFrame)
    precision: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    overlap: pd.DataFrame = field(default_factory=pd.DataFrame)
    suspicious: dict = field(default_factory=dict)
    false_negatives: dict = field(default_factory=dict)
    adjudicated: pd.DataFrame = field(default_factory=pd.DataFrame)
    ssn_fanout: dict = field(default_factory=dict)
    email_fanout: dict = field(default_factory=dict)
    clusters: dict = field(default_factory=dict)

    def to_text(self) -> str:
        L = ["=" * 60, "  DETERMINISTIC RULES — AUTOMATED EVALUATION", "=" * 60]
        for k, v in self.headline.items():
            L.append(f"  {k:<22}{v}")
        L += ["", "  PER-RULE AGREEMENT PROFILE (%)", self.agreement.to_string(index=False)]
        L += ["", "  PRECISION + 95% CI (adjudicator, NOT ground truth)",
              self.precision.to_string(index=False)]
        if not self.calibration.empty:
            L += ["", "  CONFIDENCE CALIBRATION", self.calibration.to_string(index=False)]
        if not self.overlap.empty:
            L += ["", "  RULE OVERLAP (diagonal = total fires; __sole__ = unique)",
                  self.overlap.to_string()]
        if self.suspicious:
            L += ["", "  SUSPICIOUS TYPOLOGY"]
            L += [f"    {k:<24}{v}" for k, v in self.suspicious.items()]
        if self.false_negatives:
            L += ["", "  FALSE-NEGATIVE ESTIMATE (rejected pairs adjudicated)"]
            L += [f"    {k:<24}{v}" for k, v in self.false_negatives.items()
                  if k != "top_blocks_by_same"]
            for b, d in self.false_negatives.get("top_blocks_by_same", {}).items():
                L += [f"      {b:<16} n={d['n']:<5} same={d['same']}"]
        L += ["", f"  SSN fan-out:   shared={self.ssn_fanout.get('distinct_shared')}, "
                  f"max={self.ssn_fanout.get('max_fanout')}"]
        L += [f"  Email fan-out: shared={self.email_fanout.get('distinct_shared')}, "
              f"max={self.email_fanout.get('max_fanout')}"]
        L += [f"  Clusters: {self.clusters}"]
        return "\n".join(L)


def evaluate(
    matches: pd.DataFrame,
    cleaned: pd.DataFrame,
    n_sample: int = settings.adj_sample_size,
    seed: int = 7,
    non_matches: pd.DataFrame | None = None,
    rule_confidence: dict[str, float] | None = None,
) -> RuleEvalReport:
    """Run the full evaluation suite and return a `RuleEvalReport`.

    Pass `non_matches` to include the recall / false-negative estimate, and
    `rule_confidence` (rule name -> confidence) to include calibration.
    """
    adjud = adjudicate(matches, cleaned, n=n_sample, seed=seed)
    matched = pd.concat([matches["PATID_A"], matches["PATID_B"]]).nunique() if len(matches) else 0
    headline = {
        "cleaned_records": len(cleaned),
        "confirmed_matches": len(matches),
        "patients_matched": matched,
        "suspicious_rate": (round(100 * matches["is_suspicious"].mean(), 1)
                            if "is_suspicious" in matches and len(matches) else None),
    }
    return RuleEvalReport(
        headline=headline,
        agreement=agreement_profile(matches, cleaned),
        precision=precision_ci(adjud) if len(adjud) else pd.DataFrame(),
        calibration=(confidence_calibration(adjud, rule_confidence)
                     if (len(adjud) and rule_confidence) else pd.DataFrame()),
        overlap=rule_overlap(matches) if len(matches) else pd.DataFrame(),
        suspicious=suspicious_breakdown(matches, cleaned),
        false_negatives=(false_negative_estimate(non_matches, cleaned, seed=seed)
                         if non_matches is not None else {}),
        adjudicated=adjud,
        ssn_fanout=value_fanout(cleaned, FIELDS["ssn"]),
        email_fanout=value_fanout(cleaned, FIELDS["email"]),
        clusters=cluster_profile(matches),
    )


def _load_run(run_id: str, root: Path):
    cleaned = pd.read_parquet(root / f"data/processed/MDM_Population_cleaned_{run_id}.parquet")
    matches = pd.read_parquet(root / f"data/auto_merge/matches_{run_id}.parquet")
    nm_path = root / f"data/non_matches/non_matches_{run_id}.parquet"
    non_matches = pd.read_parquet(nm_path) if nm_path.exists() else None
    return cleaned, matches, non_matches


def main() -> None:
    ap = argparse.ArgumentParser(description="Automated deterministic-rule evaluation")
    ap.add_argument("--run-id", required=True, help="Run id of pipeline artifacts to evaluate")
    ap.add_argument("--n", type=int, default=settings.adj_sample_size,
                    help="Adjudication sample size per rule")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--export", type=Path, default=None,
                    help="Optional .csv path to write the adjudicated sample")
    ap.add_argument("--log-level", type=str, default=None,
                    help="Override EMPI_LOG_LEVEL (DEBUG/INFO/WARNING/...).")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    root = Path(__file__).resolve().parents[2]
    cleaned, matches, non_matches = _load_run(args.run_id, root)
    # Rule confidences for the calibration table (kept in sync with the engine).
    try:
        from src.models.deterministic_rules import RULES
        rule_conf = {r.name: r.confidence for r in RULES}
    except ImportError:
        rule_conf = None
    report = evaluate(matches, cleaned, n_sample=args.n, seed=args.seed,
                      non_matches=non_matches, rule_confidence=rule_conf)
    print(report.to_text())
    if args.export:
        report.adjudicated.to_csv(args.export, index=False)
        print(f"\nAdjudicated sample written to {args.export}")


if __name__ == "__main__":
    main()
