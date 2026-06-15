# Deterministic Matching Rules Guide

This document describes the deterministic matching rules used in the EMPI (Enterprise Master Patient Index) pipeline to identify potential duplicate patient records.

> **Status:** Regenerated 2026-06-15 from the current code (`src/models/deterministic_rules.py`) and a full run on the real 163,364-record `MDM_Population.csv`.

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

Rules are evaluated in descending confidence order; the highest-confidence rule that fires becomes the pair's `match_rule`, and every rule that fired is recorded in `rules_fired`. A predicate requires **both** sides to be non-null and equal.

| # | Rule | Confidence | Conditions |
|---|------|-----------|------------|
| 1 | **EXACT_SSN** | 1.000 | `SSN_clean` agrees |
| 2 | **NAME_DOB_EMAIL** | 0.990 | first + last + DOB + email agree |
| 3 | **NAME_DOB_PHONE** | 0.985 | first + last + DOB + a shared phone |
| 4 | **NAME_DOB_SEX** | 0.980 | first + last + DOB + sex agree |
| 5 | **NAME_DOB_ADDRESS** | 0.970 | first + last + DOB + street address agree |

### Rule 1 — EXACT_SSN (1.000)
Matches records with identical, **valid** Social Security Numbers. Validity is enforced in cleaning: structurally invalid SSNs (bad area/group/serial, known advertising SSNs) and low-entropy placeholders (e.g. `333333330`) are nulled out before this rule ever sees them, so they cannot fire. Remaining SSN matches with name/DOB discrepancies are still possible (name changes, data-entry typos) and are surfaced via `is_suspicious`.

### Rule 2 — NAME_DOB_EMAIL (0.990)
First name, last name, DOB **and** email all agree. This is the **only** way email participates in a deterministic decision — email on its own is not trustworthy (see [Why email-only was removed](#why-email-only-was-removed)).

### Rule 3 — NAME_DOB_PHONE (0.985)
First name, last name, DOB agree and the two records share at least one cleaned phone number (set intersection of `Phones_set`).

### Rule 4 — NAME_DOB_SEX (0.980)
First name, last name, DOB and sex at birth all agree. The most frequently triggered rule.

### Rule 5 — NAME_DOB_ADDRESS (0.970)
First name, last name, DOB and street address (`AddressLine1_clean`) all agree.

---

## Results Summary

Full run on `MDM_Population.csv` (163,364 records, 159,440 valid after cleaning).

### Match Distribution

| Match Rule | Count | % of Matches |
|------------|-------|--------------|
| NAME_DOB_SEX | 15,632 | 43.9% |
| NAME_DOB_PHONE | 11,921 | 33.5% |
| EXACT_SSN | 4,616 | 13.0% |
| NAME_DOB_EMAIL | 2,474 | 6.9% |
| NAME_DOB_ADDRESS | 963 | 2.7% |
| **Total** | **35,606** | **100%** |

### Coverage & Quality

| Metric | Value |
|--------|-------|
| Total patients | 163,364 |
| Patients in ≥1 match | 51,014 |
| Coverage rate | 31.2% |
| Average match confidence | 0.984 |
| Suspicious match rate | 2.5% (895 matches) |
| Distinct clusters | 22,246 |
| Max cluster size | 9 |

Pairs that no rule confirms (≈171k) are written to `data/non_matches/` as the input to the downstream probabilistic / ML matching stage.

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
| EXACT_SSN | 370 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_EMAIL | 400 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_PHONE | 400 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_SEX | 400 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_ADDRESS | 400 | 100.0% | 100.0 – 100.0 |

\*Precision = SAME / (SAME + DIFFERENT); UNCERTAIN excluded. Zero adjudicated false positives across ~1,970 decided pairs. The CI collapses to a point because no resample produced a DIFFERENT; the true bound is limited by the adjudicator's own accuracy, not sampling.

### Confidence calibration

Assigned rule confidences are **conservative** — no rule's adjudicated precision falls below its stated confidence (every gap is ≤0):

| Rule | Assigned confidence | Adjudicated precision | Gap |
|------|--------------------|----------------------|-----|
| EXACT_SSN | 1.000 | 100.0% | 0.0 |
| NAME_DOB_EMAIL | 0.990 | 100.0% | −1.0 |
| NAME_DOB_PHONE | 0.985 | 100.0% | −1.5 |
| NAME_DOB_SEX | 0.980 | 100.0% | −2.0 |
| NAME_DOB_ADDRESS | 0.970 | 100.0% | −3.0 |

(The adjudicator cannot resolve precision differences above ~99%, so these confirm "no over-confidence" rather than proving the exact ordering.)

### Rule overlap & marginal contribution

How often each rule fires, and how often it is the **sole** rule confirming a pair (its unique contribution):

| Rule | Times fired | Sole confirmations | Sole % |
|------|-------------|--------------------|--------|
| NAME_DOB_SEX | 27,500 | 14,292 | 52.0% |
| NAME_DOB_PHONE | 15,331 | 3,542 | 23.1% |
| EXACT_SSN | 4,615 | 1,299 | 28.1% |
| NAME_DOB_ADDRESS | 7,541 | 963 | 12.8% |
| NAME_DOB_EMAIL | 2,676 | 187 | 7.0% |

`NAME_DOB_SEX` does the most unique work. `NAME_DOB_EMAIL` is the most redundant (only 7% sole) but still uniquely confirms 187 pairs, so it earns its place. Heaviest co-firing is `NAME_DOB_SEX`×`NAME_DOB_PHONE` (9,761 shared pairs) — expected, since both build on the same name+DOB core.

### Recall / false-negative estimate (the limiting factor)

Precision is essentially solved; **recall is where the deterministic stage is intentionally incomplete.** Adjudicating 1,500 *rejected* pairs (blocking surfaced them, no rule confirmed) estimates how many true matches the rules miss:

- **18.9%** of rejected pairs look like the same person overall, but the rate is wildly uneven by block:
  - **Phone-only (B5) rejections: 2.2%** same-person — correctly rejected; a shared phone alone is mostly *different* people (households, clinics). This validates having no phone-only rule.
  - **Non-phone (name/DOB-block) rejections: 67%** same-person — these are genuine deterministic-stage misses.

Two causes, both by design routed to the downstream probabilistic stage:
1. **Name typos/variants defeat exact matching** — e.g. `MEAGAN LEE` vs `MEGAN LEE`, `POE HTOO` vs `POE TOO` (same DOB). The rules require *exact* first+last equality, so phonetic near-matches blocking caught are not confirmed.
2. **Identical name+DOB but no agreeing 4th field** — e.g. `KRYSTAL SMITH 1984-12-25` on both sides with no matching sex/phone/address/email. The rules deliberately demand a corroborator (because common name + birthday can collide), so these stay unconfirmed.

> This 67% is an **upper bound** on missed matches: for common surnames some name+DOB pairs are genuine collisions the rules *correctly* declined. The unambiguous misses are the name-typo cases — the strongest argument for the planned fuzzy/probabilistic stage (Stage 4).

---

## Data quality dependencies

Two deterministic-rule failure modes were diagnosed on the real data and fixed **upstream**, because no confidence tuning can repair a bad input value:

### Placeholder SSNs
A single placeholder SSN (`333333330`) had chained **22 unrelated patients** into one 33-record `EXACT_SSN` cluster at confidence 1.000. These values are structurally valid (they pass area/group/serial checks and `python-stdnum`), so they must be caught by entropy. `src/data/transformations.clean_ssn` now nulls any SSN that has ≤2 distinct digits, has one digit filling ≥7 of 9 positions, or is a full ascending/descending digit run (e.g. `012345678`), in addition to `python-stdnum` structural validation. Effect: max cluster size dropped 33 → 9, SSN fan-out 25 → 6, and EXACT_SSN precision rose from ~88% to ~100%.

### Residual EXACT_SSN risk and the fan-out control
After the placeholder fixes, EXACT_SSN runs at ~99.95% on the proxy: of ~4,616 matches only **8** have last-name *and* DOB disagreeing and just **2** disagree on all of first/last/DOB — and the discordant cases are already flagged `is_suspicious`, so they route to review. Mandatory field corroboration was deliberately **not** added: it would drop legitimate single-source SSN matches to fix ~2 errors.

The remaining frequency risk is a *valid* SSN shared by many distinct identities (a shared/fraudulent number). This is now handled by the **`high_fanout_ssn`** flag: any confirmed match whose shared SSN is carried by at least `EMPI_SSN_FANOUT_THRESHOLD` distinct patients (default **4**) is flagged for clerical review rather than silently trusted at confidence 1.000. On the full dataset this flags **457** matches, **372** of which `is_suspicious` does *not* catch (the SSN is shared but the other fields agree or are null, so disagreement cannot be proven). The flag is informational — it does not drop the match — and is surfaced in the `run_rules` audit report and the evaluation workbook.

### Why email-only was removed
A standalone `EMAIL_EXACT` rule (confidence 0.995) previously confirmed any pair sharing an email. On the real data it ran at only **63–80% precision**: shared family/clinic inboxes linked parents to children, siblings, twins, and unrelated patients (e.g. `pcruz@nyap.org`). Email is only trustworthy when corroborated by name + DOB — which is exactly `NAME_DOB_EMAIL`. The bare-email rule was removed; pairs that match on email alone now flow to the downstream probabilistic stage as non-matches rather than being auto-confirmed.

---

## Suspicious Match Analysis

A confirmed pair is flagged `is_suspicious` when:
- DOB differs between records (both present), OR
- Last name differs between records (both present), OR
- Both SSNs are present but differ

These are confirmed matches that warrant clerical review — typically minor spelling variations, single-digit DOB typos, hyphenation/suffix differences, or name changes. After the upstream fixes the suspicious rate is **2.5%** (down from 23.7% when placeholder-SSN and email-only false positives inflated it).

### Typology of the 894 suspicious matches

| Disagreeing field | Count | Interpretation |
|-------------------|-------|----------------|
| Last name only | 674 (75%) | Maiden-name changes, hyphenation, spelling — usually still the same person |
| SSN only | 183 (20%) | Name+DOB agree but SSNs differ → one record likely has a wrong/typo'd SSN |
| DOB only | 30 (3%) | Name agrees, DOB differs |
| Multiple fields | 7 (<1%) | Highest-risk; two or more identifiers disagree |

Of the **37** matches with a DOB disagreement, only **7** are recoverable minor typos (day/month transposition or ±1 day/month); the other 30 are larger differences where the rest of the identity still matched. Last-name disagreement dominates suspicious matches, consistent with the recall analysis: real-world name variation is this pipeline's primary source of identity noise.

---

## Cluster Analysis

Large match clusters can indicate bad blocking keys, unfiltered placeholder values, or overly broad rules. The current maximum cluster is **9 records**, with 77% of clusters being clean 2-record pairs. Any future cluster above ~15 members should be auto-flagged for manual review (a recommended circuit-breaker, not yet enforced in code).

---

## Appendix: generated evaluation report

Verbatim output of `python -m src.evaluation.rule_eval --run-id eval_fanout --n 400` on the full dataset (run `eval_fanout`, 2026-06-15). Reproducible from any run's artifacts.

```
============================================================
  DETERMINISTIC RULES — AUTOMATED EVALUATION
============================================================
  cleaned_records       163364
  confirmed_matches     35605
  patients_matched      51012
  suspicious_rate       2.5

  PER-RULE AGREEMENT PROFILE (%)
            rule     n  first  last   dob   ssn  email  phone  address   sex
    NAME_DOB_SEX 15632  100.0 100.0 100.0   0.0    0.0    0.0      8.6 100.0
  NAME_DOB_PHONE 11921  100.0 100.0 100.0   0.0    0.0  100.0     26.9  60.7
       EXACT_SSN  4615   94.7  85.2  99.2 100.0    5.4   39.5     32.9  80.8
  NAME_DOB_EMAIL  2474  100.0 100.0 100.0   0.0  100.0   72.3     26.6  64.3
NAME_DOB_ADDRESS   963  100.0 100.0 100.0   0.0    0.0    0.0    100.0   0.0

  PRECISION + 95% CI (adjudicator, NOT ground truth)
            rule  decided  precision_pct  ci95_low  ci95_high
       EXACT_SSN      370          100.0     100.0      100.0
NAME_DOB_ADDRESS      400          100.0     100.0      100.0
  NAME_DOB_EMAIL      400          100.0     100.0      100.0
  NAME_DOB_PHONE      400          100.0     100.0      100.0
    NAME_DOB_SEX      400          100.0     100.0      100.0

  CONFIDENCE CALIBRATION
            rule  assigned_confidence  adjudicated_precision_pct  gap
       EXACT_SSN                1.000                      100.0  0.0
  NAME_DOB_EMAIL                0.990                      100.0 -1.0
  NAME_DOB_PHONE                0.985                      100.0 -1.5
    NAME_DOB_SEX                0.980                      100.0 -2.0
NAME_DOB_ADDRESS                0.970                      100.0 -3.0

  RULE OVERLAP (diagonal = total fires; __sole__ = unique)
                  EXACT_SSN  NAME_DOB_ADDRESS  NAME_DOB_EMAIL  NAME_DOB_PHONE  NAME_DOB_SEX  __sole__  __sole_pct__
EXACT_SSN              4615              1374             202            1621          3046      1299          28.1
NAME_DOB_ADDRESS       1374              7541             700            4814          5029       963          12.8
NAME_DOB_EMAIL          202               700            2676            1922          1731       187           7.0
NAME_DOB_PHONE         1621              4814            1922           15331          9761      3542          23.1
NAME_DOB_SEX           3046              5029            1731            9761         27500     14292          52.0

  SUSPICIOUS TYPOLOGY
    total_suspicious        894
    last_name_only          674
    dob_only                30
    ssn_only                183
    multi_field             7
    dob_diffs_total         37
    dob_diffs_minor_typo    7

  FALSE-NEGATIVE ESTIMATE (rejected pairs adjudicated)
    sampled                 1000
    SAME                    192
    DIFFERENT               749
    UNCERTAIN               59
    missed_match_rate_pct   19.2
      B3|B4|B8         n=79    same=79
      B3|B4|B7|B8      n=17    same=16
      B5               n=739   same=16
      B3|B8            n=19    same=15
      B5|B8            n=12    same=12
      B3|B4|B5|B7|B8   n=6     same=6
      B3|B7|B8         n=6     same=6
      B4               n=12    same=6

  SSN fan-out:   shared=3841, max=6
  Email fan-out: shared=4289, max=7
  Clusters: {'n_clusters': 22245, 'size_2': 17250, 'size_3_5': 4933, 'size_6_20': 62, 'size_gt_20': 0, 'max_size': 9}
```

> Note: the CLI's false-negative sample defaults to n=1000 (shown here); the 1,500-sample run cited in the [Recall](#recall--false-negative-estimate-the-limiting-factor) section gives the same ~19% headline and the same block ranking.
