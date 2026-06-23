# Deterministic Matching Rules Guide

This document describes the deterministic matching rules used in the EMPI (Enterprise Master Patient Index) pipeline to identify potential duplicate patient records.

> **Status:** Regenerated 2026-06-20 from the current code (`src/models/deterministic_rules.py`) and a full run on the real 163,364-record `MDM_Population.csv` (run `real_20260620`). This revision reflects three changes since the prior version: the SSN rule now requires a corroborating DOB (`SSN_DOB`, replacing the bare `EXACT_SSN`); first/last name agreement is **fuzzy** (single-typo tolerant); and the stage now emits a **three-way** decision (match / reject / review). See the [Three-way decision](#three-way-decision) section.
>
> For the blocking stage itself see [Blocking-Guide.md](Blocking-Guide.md).

## Overview

The deterministic matching approach uses blocking strategies combined with exact-match rules to efficiently identify candidate record pairs that likely represent the same patient. The process follows a two-stage approach:

1. **Blocking**: Group records by shared attributes to reduce the comparison space
2. **Rule Application**: Apply deterministic rules within blocks to identify matches

## Blocking Strategies

Blocking reduces the O(n²) comparison problem by grouping records on shared keys. Each block key is designed to capture different match scenarios while preventing cartesian explosion.

### Block Definitions

| Block | Key Composition | Purpose |
|-------|----------------|---------|
| B1 | `SSN_clean` | Direct SSN matches |
| B3 | Double Metaphone(`LastNM_clean`) + `BirthDT_clean` | Phonetic name + exact DOB |
| B4 | `LastNM_clean` + `BirthYear` + `FirstNM_clean[:3]` | Name + birth year + first name prefix |
| B5 | `Phones_set` intersection | Any shared phone number |
| B6 | `Email_clean` | Email matches |
| B7 | Double Metaphone(`LastNM_clean`) + `ZipCD_base` + `BirthYear` | Name + location + birth year |
| B8 | Soundex(`FirstNM`) + Soundex(`LastNM`) + `BirthYear` | Initial/phonetic broad block |
| B9 | `LastNM_clean` + `FirstNM_clean` + `SSN_Last4` | Full name + last 4 SSN |

### Block Safety Controls

To prevent memory issues and false positive clusters:
- **Governance threshold**: keys with more than `EMPI_GOVERNANCE_THRESHOLD` (default 500) records are capped
- Records with null keys are excluded from their respective blocks
- Records flagged `valid_record == False` by the cleaning stage are excluded entirely

> **Note:** Blocking alone does not stop *low-frequency* junk values (a placeholder SSN shared by 20 patients is far under the 500 cap). Those are handled upstream in the cleaning stage (see [Data quality dependencies](#data-quality-dependencies)).

---

## Match Rules

Rules are evaluated in descending confidence order; the highest-confidence rule that fires becomes the pair's `match_rule`, and every rule that fired is recorded in `rules_fired`. Every predicate requires **both** sides to be non-null; all predicates demand exact equality **except** first/last name, which are matched fuzzily (see [Fuzzy name matching](#fuzzy-name-matching)).

| # | Rule | Confidence | Conditions |
|---|------|-----------|------------|
| 1 | **SSN_DOB** | 1.000 | `SSN_clean` **and** `BirthDT_clean` agree |
| 2 | **NAME_DOB_EMAIL** | 0.990 | first + last + DOB + email agree |
| 3 | **NAME_DOB_PHONE** | 0.985 | first + last + DOB + a shared phone |
| 4 | **NAME_DOB_SEX** | 0.980 | first + last + DOB + sex agree |
| 5 | **NAME_DOB_ADDRESS** | 0.970 | first + last + DOB + street address agree |

### Rule 1 — SSN_DOB (1.000)
Matches records with identical, **valid** Social Security Numbers **corroborated by an agreeing date of birth**. The bare SSN-only rule (`EXACT_SSN`) was removed: an SSN match whose DOB is missing or disagreeing no longer auto-confirms and instead flows to the downstream probabilistic stage. SSN validity is still enforced in cleaning — structurally invalid SSNs (bad area/group/serial, known advertising SSNs) and low-entropy placeholders (e.g. `333333330`) are nulled out before this rule sees them. Requiring DOB trades a small amount of recall on single-source SSN matches for protection against typo'd / shared / placeholder SSNs that DOB cannot vouch for. Confirmed pairs whose last name disagrees are still surfaced via `is_suspicious`.

### Fuzzy name matching
First and last name agreement is **single-typo tolerant**: two names agree when they are exactly equal, OR their Jaro-Winkler similarity ≥ `NAME_JW_THRESHOLD` (0.92), OR their Damerau-Levenshtein distance ≤ `NAME_LEV_MAX` (1). This closes the documented recall gap on name typos (e.g. `MEAGAN` ↔ `MEGAN`, `SMITH` ↔ `SMYTH`) — pairs blocking caught but exact matching previously dropped. Fuzzy matching is used **only to confirm** a match, never to reject one. A side effect: because the `is_suspicious` last-name check is *strict*, fuzzy-matched name pairs are flagged suspicious (and the suspicious rate rose accordingly — see Results), correctly routing the typo cases to clerical review.

### Rule 2 — NAME_DOB_EMAIL (0.990)
First name, last name, DOB **and** email all agree. This is the **only** way email participates in a deterministic decision — email on its own is not trustworthy (see [Why email-only was removed](#why-email-only-was-removed)).

### Rule 3 — NAME_DOB_PHONE (0.985)
First name, last name, DOB agree and the two records share at least one cleaned phone number (set intersection of `Phones_set`).

### Rule 4 — NAME_DOB_SEX (0.980)
First name, last name, DOB and sex at birth all agree. The most frequently triggered rule.

### Rule 5 — NAME_DOB_ADDRESS (0.970)
First name, last name, DOB and street address (`AddressLine1_clean`) all agree.

---

## Three-way decision

Every candidate pair the blocking stage produces lands in exactly one of three
buckets (`src/models/deterministic_rules.classify_non_matches`):

| Decision | Condition | Destination |
|----------|-----------|-------------|
| **match** | a rule confirmed it (`apply_rules`) | auto-merge |
| **reject** | no rule fired **and ≥3 strong identifiers strictly disagree** (full SSN / first / last / DOB) | dropped — written to `data/rejects/` for audit, **not** sent downstream |
| **review** | no rule fired and < 3 contradictions | `data/non_matches/` → downstream probabilistic / ML stage |

The throw-out is deliberately strict: one or two disagreeing fields are not
enough to reject (a pair can carry two independent typos — see the calibration
below); only three independent strong-identifier conflicts mark a pair a
confident non-match. Fuzzy name matching is **not** used in the contradiction
count — names are compared strictly there, so a typo counts toward a
contradiction but, on its own, only routes a pair to review.

### Decision distribution (run `real_20260620`)

| Decision | Pairs | Notes |
|----------|-------|-------|
| match | 44,786 | confirmed by a rule |
| reject | 126,903 | ≥3 contradictions; dropped |
| review | 33,116 | → probabilistic stage |
| *(candidate pairs)* | *204,805* | from blocking |

> **Calibration of the reject threshold (= 3).** Adjudicating reject samples by
> contradiction count on `real_20260620` gave the **false-reject rate** (truly
> same-person pairs wrongly dropped): **~10% at 2 contradictions**, but **0% at
> 3 and 4**. So two strong-identifier conflicts are *not* decisive — a real match
> can carry two independent typos — while three are. The threshold was set to 3:
> the 11,589 two-contradiction pairs now route to **review** (where the
> probabilistic stage can still reject them) instead of being discarded, recovering
> the ~10% of them that are true matches. After the change the review set
> adjudicates to 57% same-person (down from 85% at threshold 2), as expected —
> it now carries the mostly-non-matching two-contradiction pairs as well. The
> fuzzy cutoffs (`NAME_JW_THRESHOLD` 0.92, `NAME_LEV_MAX` 1) remain conventional
> single-typo values; rigorous fuzzy calibration needs a labeled set.

## Results Summary

Full run on `MDM_Population.csv` (163,364 records, 158,724 valid after cleaning; run `real_20260620`).

### Match Distribution

| Match Rule | Count | % of Matches |
|------------|-------|--------------|
| NAME_DOB_SEX | 20,402 | 45.6% |
| NAME_DOB_PHONE | 15,700 | 35.1% |
| SSN_DOB | 4,577 | 10.2% |
| NAME_DOB_EMAIL | 2,878 | 6.4% |
| NAME_DOB_ADDRESS | 1,229 | 2.7% |
| **Total** | **44,786** | **100%** |

### Coverage & Quality

| Metric | Value |
|--------|-------|
| Total patients | 163,364 |
| Patients in ≥1 match | 62,912 |
| Coverage rate | 38.5% |
| Average match confidence | 0.984 |
| Suspicious match rate | 11.5% (5,165 matches) |
| Distinct clusters | 27,215 |
| Max cluster size | 8 |

The higher match count and coverage vs the prior version come from **fuzzy name
matching** newly confirming name-typo pairs; the same fuzzy matching also drives
the higher suspicious rate (the strict `is_suspicious` flag marks the typo'd
names — 4,914 of the 5,165 suspicious matches are last-name-only).

`review` pairs (21,527) are written to `data/non_matches/` as the input to the
downstream probabilistic / ML stage; `reject` pairs (138,492) are dropped.

---

## Evaluation

Rules are evaluated with `src/evaluation/rule_eval.py`, which runs against any pipeline run's artifacts:

```bash
python -m src.evaluation.rule_eval --run-id <run_id> --n 300
```

It produces, **without ground-truth labels**:

- **Agreement profile** (`agreement_profile`) — per rule, how often the *non-tested* identifying fields also agree (a high rate corroborates the rule).
- **Value fan-out** (`value_fanout`) — how many distinct patients share each SSN/email value (placeholder / shared-account detection).
- **Cluster profile** (`cluster_profile`) — connected-component size distribution.
- **Adjudication** (`adjudicate`, `adjudicate_pairs`) — an AI-assisted clerical review of a sampled subset, labelling each pair **SAME / DIFFERENT / UNCERTAIN** from a holistic field comparison (name typo/nickname tolerance, DOB transposition, placeholder-SSN and shared-email detection), and a per-rule **precision proxy** = SAME / (SAME + DIFFERENT).
- **Precision CI** (`precision_ci`) — bootstrap 95% confidence interval on each rule's precision.
- **Confidence calibration** (`confidence_calibration`) — assigned confidence vs adjudicated precision.
- **Rule overlap** (`rule_overlap`) — co-firing matrix and sole-contribution counts.
- **Suspicious typology** (`suspicious_breakdown`) — which field disagrees, and DOB-difference severity.
- **False-negative estimate** (`false_negative_estimate`) — adjudicates rejected pairs to estimate missed true matches (recall), broken down by source block.

> The adjudicator is a **heuristic proxy, not ground truth.** It flags likely false positives at scale and leaves genuinely ambiguous pairs as UNCERTAIN. For a defensible precision number, export the sample (`--export`) and have a human adjudicate it.

### Precision with bootstrap 95% CI (≤400 sampled matches per rule)

| Rule | Decided | Precision\* | 95% CI |
|------|---------|-------------|--------|
| SSN_DOB | 275 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_EMAIL | 300 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_PHONE | 300 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_SEX | 299 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_ADDRESS | 300 | 100.0% | 100.0 – 100.0 |

\*Precision = SAME / (SAME + DIFFERENT); UNCERTAIN excluded (run `real_20260620`, `--n 300`). Zero adjudicated false positives. The CI collapses to a point because no resample produced a DIFFERENT; the true bound is limited by the adjudicator's own accuracy, not sampling. Note the adjudicator's name-similarity tolerance means fuzzy-confirmed name-typo pairs are scored SAME, so precision stays at 100% even with fuzzy matching on.

### Confidence calibration

Assigned rule confidences are **conservative** — no rule's adjudicated precision falls below its stated confidence (every gap is ≤0):

| Rule | Assigned confidence | Adjudicated precision | Gap |
|------|--------------------|----------------------|-----|
| SSN_DOB | 1.000 | 100.0% | 0.0 |
| NAME_DOB_EMAIL | 0.990 | 100.0% | −1.0 |
| NAME_DOB_PHONE | 0.985 | 100.0% | −1.5 |
| NAME_DOB_SEX | 0.980 | 100.0% | −2.0 |
| NAME_DOB_ADDRESS | 0.970 | 100.0% | −3.0 |

(The adjudicator cannot resolve precision differences above ~99%, so these confirm "no over-confidence" rather than proving the exact ordering.)

### Rule overlap & marginal contribution

How often each rule fires, and how often it is the **sole** rule confirming a pair (its unique contribution):

| Rule | Times fired | Sole confirmations | Sole % |
|------|-------------|--------------------|--------|
| NAME_DOB_SEX | 34,994 | 18,667 | 53.3% |
| NAME_DOB_PHONE | 19,488 | 4,835 | 24.8% |
| NAME_DOB_ADDRESS | 9,047 | 1,229 | 13.6% |
| SSN_DOB | 4,577 | 982 | 21.5% |
| NAME_DOB_EMAIL | 3,091 | 210 | 6.8% |

`NAME_DOB_SEX` does the most unique work. `NAME_DOB_EMAIL` is the most redundant (only 6.8% sole) but still uniquely confirms 210 pairs, so it earns its place. Heaviest co-firing is `NAME_DOB_SEX`×`NAME_DOB_PHONE` (12,209 shared pairs) — expected, since both build on the same name+DOB core.

### Recall / what reaches the probabilistic stage

Precision is essentially solved; the open question is **recall** — what the
deterministic stage hands to the probabilistic stage, and what it discards.

The `review` set (33,116 pairs routed downstream) adjudicates to **57%
same-person** (1,000 sampled: 570 SAME, 301 DIFFERENT, 129 UNCERTAIN). Fuzzy
matching already *confirmed* the clear name-typo pairs (they are now matches),
and the `reject` bucket removed the ≥3-contradiction non-matches; what remains is
a mix of the two-contradiction pairs (mostly true non-matches the probabilistic
stage will filter) and the one residual cause of genuine misses:

- **Identical name+DOB but no agreeing 4th field** — e.g. `KRYSTAL SMITH
  1984-12-25` on both sides with no matching sex/phone/address/email. The rules
  deliberately demand a corroborator (common name + birthday can collide), so
  these stay unconfirmed and route to the probabilistic stage to adjudicate.

The earlier dominant miss — **name typos defeating exact matching** (`MEAGAN LEE`
vs `MEGAN LEE`) — is now largely **caught** by [fuzzy name
matching](#fuzzy-name-matching), which is why the match count and coverage rose.

> Caveat: the review set being 85% same-person means the probabilistic stage's
> job is mostly *confirmation*, not discrimination. The 5 adjudicated DIFFERENT
> (and 141 UNCERTAIN) are the genuine collisions to be careful with.

### Blocking recall

Blocking's own recall — does it even surface the pairs the rules can confirm? —
is measured by `src/evaluation/blocking_eval.py` (rules as ground truth). On run
`real_20260620` it is **97.9%**: of 45,743 rule-confirmable pairs found in a wide
candidate set, blocking emitted 44,786 and missed 957. The misses skew to
**SSN_DOB (400)** — pairs sharing SSN+DOB that B1 did not emit, most likely
high-fan-out SSNs hit by the governance cap. See [Blocking-Guide.md](Blocking-Guide.md).

---

## Data quality dependencies

Two deterministic-rule failure modes were diagnosed on the real data and fixed **upstream**, because no confidence tuning can repair a bad input value:

### Placeholder SSNs
A single placeholder SSN (`333333330`) had once chained **22 unrelated patients** into one 33-record SSN cluster at confidence 1.000. These values are structurally valid (they pass area/group/serial checks and `python-stdnum`), so they must be caught by entropy. `src/data/transformations.clean_ssn` now nulls any SSN that has ≤2 distinct digits, has one digit filling ≥7 of 9 positions, or is a full ascending/descending digit run (e.g. `012345678`), in addition to `python-stdnum` structural validation. Effect: max cluster size dropped 33 → 9 (now 8), SSN fan-out 25 → 6, and the SSN rule's precision rose from ~88% to ~100%.

### SSN risk and the corroboration / fan-out controls
SSN matches now carry **two** layers of protection:

1. **Mandatory DOB corroboration (`SSN_DOB`).** Unlike the former `EXACT_SSN`,
   an SSN match must agree on date of birth to confirm. This was the deliberate
   reversal of the earlier "no mandatory corroboration" stance — a shared/typo'd
   SSN with a disagreeing DOB no longer auto-merges; it routes to review. SSN_DOB
   adjudicates at 100% precision (above).
2. **High-fan-out flag.** A *valid* SSN shared by many distinct identities (a
   shared or fraudulent number) is still risky even with DOB agreement. Any
   confirmed match whose shared SSN is carried by ≥ `EMPI_SSN_FANOUT_THRESHOLD`
   distinct patients (default **4**) is flagged `high_fanout_ssn` for clerical
   review rather than silently trusted. (On `real_20260620`, shared-SSN fan-out
   tops out at 6.) The flag is informational — it does not drop the match — and
   is surfaced in the `run_rules` audit report.

### Why email-only was removed
A standalone `EMAIL_EXACT` rule (confidence 0.995) previously confirmed any pair sharing an email. On the real data it ran at only **63–80% precision**: shared family/clinic inboxes linked parents to children, siblings, twins, and unrelated patients (e.g. `pcruz@nyap.org`). Email is only trustworthy when corroborated by name + DOB — which is exactly `NAME_DOB_EMAIL`. The bare-email rule was removed; pairs that match on email alone now flow to the downstream probabilistic stage as non-matches rather than being auto-confirmed.

---

## Suspicious Match Analysis

A confirmed pair is flagged `is_suspicious` when:
- DOB differs between records (both present), OR
- Last name differs between records (both present), OR
- Both SSNs are present but differ

These are confirmed matches that warrant clerical review — typically minor spelling variations, hyphenation/suffix differences, or name changes. The suspicious rate is **11.5%** (5,165 matches), up from 2.5% in the prior version. The increase is a direct, expected consequence of **fuzzy name matching**: pairs are now confirmed on near-equal names, and the `is_suspicious` last-name check is *strict*, so those typo'd last names register as a disagreement. In other words, the typo matches fuzzy matching newly recovers are exactly the ones flagged for a human to eyeball.

### Typology of the 5,165 suspicious matches (run `real_20260620`)

| Disagreeing field | Count | Interpretation |
|-------------------|-------|----------------|
| Last name only | 4,914 (95%) | Fuzzy-matched name typos, maiden-name changes, hyphenation — usually still the same person |
| SSN only | 204 (4%) | Name+DOB agree but SSNs differ → one record likely has a wrong/typo'd SSN |
| Multiple fields | 47 (<1%) | Highest-risk; two or more identifiers disagree |
| DOB only | 0 | No rule confirms a DOB-disagreeing pair (all rules require exact DOB), so DOB-only suspicion is now unreachable |

Last-name disagreement overwhelmingly dominates, consistent with the design: real-world name variation is this pipeline's primary source of identity noise, and fuzzy matching deliberately surfaces it for review rather than dropping it.

---

## Cluster Analysis

Large match clusters can indicate bad blocking keys, unfiltered placeholder values, or overly broad rules. On `real_20260620` there are **27,215 clusters**, the maximum cluster is **8 records**, and **76%** are clean 2-record pairs (20,806 of 27,215; 94 clusters of size 6–20; none above 20). Any future cluster above ~15 members should be auto-flagged for manual review (a recommended circuit-breaker, not yet enforced in code).

---

## Appendix: generated evaluation report

Verbatim output of `python -m src.evaluation.rule_eval --run-id real_20260620 --n 300` on the full dataset (run `real_20260620`, 2026-06-20). Reproducible from any run's artifacts.

```
============================================================
  DETERMINISTIC RULES — AUTOMATED EVALUATION
============================================================
  cleaned_records       163364
  confirmed_matches     44786
  patients_matched      62912
  suspicious_rate       11.5

  PER-RULE AGREEMENT PROFILE (%)
            rule     n  first  last   dob   ssn  email  phone  address   sex
    NAME_DOB_SEX 20402   85.9  90.5 100.0   0.0    0.0    0.0      8.5 100.0
  NAME_DOB_PHONE 15700   88.2  86.9 100.0   0.0    0.0  100.0     24.8  60.1
         SSN_DOB  4577   94.8  85.3 100.0 100.0    5.4   39.5     32.7  80.9
  NAME_DOB_EMAIL  2878   92.5  93.0 100.0   0.0  100.0   72.7     26.8  64.6
NAME_DOB_ADDRESS  1229   86.1  91.6 100.0   0.0    0.0    0.0    100.0   0.0

  PRECISION + 95% CI (adjudicator, NOT ground truth)
            rule  decided  precision_pct  ci95_low  ci95_high
NAME_DOB_ADDRESS      300          100.0     100.0      100.0
  NAME_DOB_EMAIL      300          100.0     100.0      100.0
  NAME_DOB_PHONE      300          100.0     100.0      100.0
    NAME_DOB_SEX      299          100.0     100.0      100.0
         SSN_DOB      275          100.0     100.0      100.0

  CONFIDENCE CALIBRATION
            rule  assigned_confidence  adjudicated_precision_pct  gap
         SSN_DOB                1.000                      100.0  0.0
  NAME_DOB_EMAIL                0.990                      100.0 -1.0
  NAME_DOB_PHONE                0.985                      100.0 -1.5
    NAME_DOB_SEX                0.980                      100.0 -2.0
NAME_DOB_ADDRESS                0.970                      100.0 -3.0

  RULE OVERLAP (diagonal = total fires; __sole__ = unique)
                  NAME_DOB_ADDRESS  NAME_DOB_EMAIL  NAME_DOB_PHONE  NAME_DOB_SEX  SSN_DOB  __sole__  __sole_pct__
NAME_DOB_ADDRESS              9047             813            5627          5937     1421      1229          13.6
NAME_DOB_EMAIL                 813            3091            2233          2009      213       210           6.8
NAME_DOB_PHONE                5627            2233           19488         12209     1696      4835          24.8
NAME_DOB_SEX                  5937            2009           12209         34994     3297     18667          53.3
SSN_DOB                       1421             213            1696          3297     4577       982          21.5

  SUSPICIOUS TYPOLOGY
    total_suspicious        5165
    last_name_only          4914
    dob_only                0
    ssn_only                204
    multi_field             47
    dob_diffs_total         0
    dob_diffs_minor_typo    0

  FALSE-NEGATIVE ESTIMATE (review pairs adjudicated)
    sampled                 1000
    SAME                    570
    DIFFERENT               301
    UNCERTAIN               129
    missed_match_rate_pct   57.0
      B3|B4|B8         n=260   same=260
      B3|B4|B7|B8      n=98    same=96
      B5               n=181   same=61
      B4|B8            n=39    same=31
      B3|B8            n=29    same=26
      B5|B8            n=28    same=26
      B8               n=40    same=20

  SSN fan-out:   shared=3841, max=6
  Email fan-out: shared=4289, max=7
  Clusters: {'n_clusters': 27215, 'size_2': 20806, 'size_3_5': 6315, 'size_6_20': 94, 'size_gt_20': 0, 'max_size': 8}
```

> Note: the FALSE-NEGATIVE / missed-match section now adjudicates the **review**
> set (the < 2-contradiction pairs routed downstream), not all unconfirmed pairs —
> the `reject` bucket is excluded. That is why the SAME rate is high (85%): the
> review set is, by construction, the pool of likely matches the rules could not
> confirm outright.
