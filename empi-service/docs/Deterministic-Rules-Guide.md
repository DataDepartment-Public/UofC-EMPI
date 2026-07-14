# Deterministic Matching Rules Guide

This document describes the deterministic matching rules used in the EMPI (Enterprise Master Patient Index) pipeline to identify potential duplicate patient records.

> **Status:** Updated 2026-06-29. Originally regenerated 2026-06-20 from the current code (`src/models/deterministic_rules.py`) and a full run on the real 163,364-record `MDM_Population.csv` (run `real_20260620`). Changes since the prior version: the SSN rule now requires a corroborating DOB (`SSN_DOB`, replacing the bare `EXACT_SSN`); first/last name agreement is **fuzzy** (single-typo tolerant); and the stage emits a **three-way** decision (match / reject / review).
>
> **2026-06-29 — rule demotion.** `NAME_DOB_SEX` and `NAME_DOB_ADDRESS` are now **review-tier**: they still fire and record full provenance, but a pair confirmed *only* by one of them is **routed to review, not auto-merged**. The silver evaluation ([Method 2](#method-2--silver-labels)) showed both adjudicate at only ~65% / ~67% precision and carry essentially all of the false merges; demoting them lifts **auto-merge precision from 83.1% to 99.8%** on the silver set (false merges 7,578 → 39) while preserving the ~14k true matches they confirm (those flow to the review band for the downstream probabilistic / FS stage). See [Match Rules](#match-rules), [Three-way decision](#three-way-decision), and the rule-tier note in [Method 2](#method-2--silver-labels).
>
> For the blocking stage itself see [Blocking-Guide.md](Blocking-Guide.md).

## Overview

The deterministic matching approach uses blocking strategies combined with exact-match rules to efficiently identify candidate record pairs that likely represent the same patient. The process follows a two-stage approach:

1. **Blocking**: Group records by shared attributes to reduce the comparison space
2. **Rule Application**: Apply deterministic rules within blocks to identify matches

## Blocking Strategies

Blocking reduces the O(n²) comparison problem by grouping records on shared keys. Each block key is designed to capture different match scenarios while preventing cartesian explosion.

> **Note (2026-06-29):** production blocking is now the **stacked blocker** —
> 8-block ∪ q-gram → CNP/ARCS prune (`src/preprocessing/stacked_blocking.py`),
> −46.7% candidates vs the 8-block scheme at 99.96% silver recall. The 8 blocks
> below are the base leg. See [Blocking-Guide.md](Blocking-Guide.md#the-stacked-blocker-production-path).

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

Each rule carries a **tier** that decides what a confirmed pair *does* (not whether it fires):

- **auto-merge** — the pair is auto-merged (the `match` decision).
- **review** — the pair is confirmed but routed to the downstream review / probabilistic stage instead of being auto-merged.

The tiers are derived from the code (`AUTO_MERGE_RULES` / `REVIEW_RULES`). A pair confirmed only by a review-tier rule routes to **review**; a pair that also fires an auto-merge rule auto-merges (the higher-confidence auto-merge rule wins).

| # | Rule | Confidence | Tier | Conditions |
|---|------|-----------|------|------------|
| 1 | **SSN_DOB** | 1.000 | auto-merge | `SSN_clean` **and** `BirthDT_clean` agree |
| 2 | **NAME_DOB_EMAIL** | 0.990 | auto-merge | first + last + DOB + email agree |
| 3 | **NAME_DOB_PHONE** | 0.985 | auto-merge | first + last + DOB + a shared phone |
| 4 | **NAME_DOB_SEX** | 0.980 | **review** | first + last + DOB + sex agree |
| 5 | **NAME_DOB_ADDRESS** | 0.970 | **review** | first + last + DOB + street address agree |

### Rule 1 — SSN_DOB (1.000)
Matches records with identical, **valid** Social Security Numbers **corroborated by an agreeing date of birth**. The bare SSN-only rule (`EXACT_SSN`) was removed: an SSN match whose DOB is missing or disagreeing no longer auto-confirms and instead flows to the downstream probabilistic stage. SSN validity is still enforced in cleaning — structurally invalid SSNs (bad area/group/serial, known advertising SSNs) and low-entropy placeholders (e.g. `333333330`) are nulled out before this rule sees them. Requiring DOB trades a small amount of recall on single-source SSN matches for protection against typo'd / shared / placeholder SSNs that DOB cannot vouch for. Confirmed pairs whose last name disagrees are still surfaced via `is_suspicious`.

### Fuzzy name matching
First and last name agreement is **single-typo tolerant**: two names agree when they are exactly equal, OR their Jaro-Winkler similarity ≥ `NAME_JW_THRESHOLD` (0.92), OR their Damerau-Levenshtein distance ≤ `NAME_LEV_MAX` (1). This closes the documented recall gap on name typos (e.g. `MEAGAN` ↔ `MEGAN`, `SMITH` ↔ `SMYTH`) — pairs blocking caught but exact matching previously dropped. Fuzzy matching is used **only to confirm** a match, never to reject one. A side effect: because the `is_suspicious` last-name check is *strict*, fuzzy-matched name pairs are flagged suspicious (and the suspicious rate rose accordingly — see Results), correctly routing the typo cases to clerical review.

### Rule 2 — NAME_DOB_EMAIL (0.990)
First name, last name, DOB **and** email all agree. This is the **only** way email participates in a deterministic decision — email on its own is not trustworthy (see [Why email-only was removed](#why-email-only-was-removed)).

### Rule 3 — NAME_DOB_PHONE (0.985)
First name, last name, DOB agree and the two records share at least one cleaned phone number (set intersection of `Phones_set`).

### Rule 4 — NAME_DOB_SEX (0.980) — **review-tier**
First name, last name, DOB and sex at birth all agree. The most frequently triggered rule — but **demoted to the review tier** (2026-06-29). Against silver labels it adjudicates at only ~65% precision: name + DOB + sex is not a unique identity, because a common name + shared birthday + same sex collides for genuinely different people. A pair confirmed only by this rule now routes to **review** (the downstream probabilistic / FS stage) rather than auto-merging. See [Method 2](#method-2--silver-labels).

### Rule 5 — NAME_DOB_ADDRESS (0.970) — **review-tier**
First name, last name, DOB and street address (`AddressLine1_clean`) all agree. Also **demoted to the review tier** (2026-06-29): a shared street address is a *household* identifier, so co-resident relatives who share a birthday collide (~67% silver precision). A pair confirmed only by this rule routes to **review**, not auto-merge.

---

## Three-way decision

Every candidate pair the blocking stage produces lands in exactly one of three
buckets. The split is by **rule tier** (`apply_rules` + the `AUTO_MERGE_RULES` /
`REVIEW_RULES` sets) for confirmed pairs, and by contradiction count
(`classify_non_matches`) for the rest:

| Decision | Condition | Destination |
|----------|-----------|-------------|
| **match** | confirmed by an **auto-merge-tier** rule (`SSN_DOB` / `NAME_DOB_EMAIL` / `NAME_DOB_PHONE`) | auto-merge |
| **review** | confirmed **only** by a review-tier rule (`NAME_DOB_SEX` / `NAME_DOB_ADDRESS`), **OR** no rule fired and < 3 contradictions | `data/non_matches/` → downstream probabilistic / ML stage |
| **reject** | no rule fired **and ≥3 strong identifiers strictly disagree** (full SSN / first / last / DOB) | dropped — written to `data/rejects/` for audit, **not** sent downstream |

A review-tier rule confirmation is **never** reject-scored: `apply_rules` returns it
as a confirmed pair (so `classify_non_matches` excludes it from the contradiction
split), and the pipeline routes it straight to review. This is why demoting
`NAME_DOB_SEX` / `NAME_DOB_ADDRESS` grows the review band rather than the reject
pile — the ~14k true matches they confirm stay recoverable downstream.

The reject decision is itself a deterministic rule — the **`STRONG_ID_CONFLICT`
reject rule** (`REJECT_RULES[0]`), the third tier symmetric with the auto-merge and
review match rules. The throw-out is deliberately strict: one or two disagreeing
fields are not enough to reject (a pair can carry two independent typos — see the
calibration below); only three independent strong-identifier conflicts mark a pair
a confident non-match. (A *single* strong-ID conflict is deliberately not a reject
rule — the FS conflict-veto analysis showed true duplicates conflict on SSN/email/
phone nearly as often as false merges, so a one-conflict veto destroys more true
matches than it saves.) Fuzzy name matching is **not** used in the contradiction
count — names are compared strictly there, so a typo counts toward a contradiction
but, on its own, only routes a pair to review.

### Decision distribution (run `real_20260620`, stacked blocker + tiered rules)

Reflecting **both** 2026-06-29 changes — the [stacked blocker](Blocking-Guide.md#the-stacked-blocker-production-path)
(8-block ∪ q-gram → CNP prune, 109,061 candidate pairs) and the rule demotion:

| Decision | Pairs | Notes |
|----------|-------|-------|
| match (auto-merge) | 23,155 | confirmed by an auto-merge-tier rule |
| review | 57,666 | 21,842 review-tier rule confirmations + 35,824 unconfirmed (<3 contradictions) → probabilistic stage |
| reject | 28,240 | `STRONG_ID_CONFLICT` (≥3 contradictions); dropped |
| *(candidate pairs)* | *109,061* | from the stacked blocker |

> **How the two changes compose.** On the raw 8-block candidate set (204,805) the
> tiered rules split match 23,155 / review 54,747 / reject 126,903. The stacked
> blocker then prunes ~96k pairs — almost entirely the **weak edges that would have
> been rejected anyway** (reject 126,903 → 28,240), leaving **auto-merge unchanged**
> (the auto-merge rules' pairs are high-ARCS-weight and never pruned) and review
> roughly flat (q-gram adds a few typo pairs to the review-tier rules: 21,631 →
> 21,842). Net: the prune removes downstream comparison load without touching the
> confirmed matches.

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

### Match Distribution (auto-merge, post-demotion)

The auto-merge set is now the three auto-merge-tier rules only:

| Match Rule | Count | % of Auto-merges |
|------------|-------|------------------|
| NAME_DOB_PHONE | 15,700 | 67.8% |
| SSN_DOB | 4,577 | 19.8% |
| NAME_DOB_EMAIL | 2,878 | 12.4% |
| **Total auto-merge** | **23,155** | **100%** |

The two review-tier rules still fire but route to review, not auto-merge (counts
on the stacked-blocker candidate set):

| Review-tier Rule | Count |
|------------------|-------|
| NAME_DOB_SEX | 20,578 |
| NAME_DOB_ADDRESS | 1,264 |
| **Total review-tier** | **21,842** |

### Coverage & Quality (auto-merge)

| Metric | Value |
|--------|-------|
| Total patients | 163,364 |
| Patients in ≥1 auto-merge | 37,537 |
| Coverage rate (auto-merge) | 22.98% |
| Average match confidence | 0.989 |
| Suspicious match rate | 12.9% (2,994 of 23,155 auto-merges) |
| High-fanout SSN matches | 445 |
| Distinct clusters | 17,241 |
| Max cluster size | 7 |

Coverage and cluster counts dropped from the pre-demotion figures (38.5% / 62,912
patients / 27,215 clusters / max 8) because `NAME_DOB_SEX` and `NAME_DOB_ADDRESS`
no longer auto-merge — those pairs are now in the review band. This is the intended
trade: auto-merge precision rose to **99.8%** on the silver set (false merges
7,578 → 39) at the cost of auto-merge *coverage*, with the demoted true matches
recoverable downstream. Auto-merges still carry **fuzzy name matching**, which is
why the suspicious rate stays elevated (the strict `is_suspicious` flag marks the
typo'd names it confirms).

The 57,666 `review` pairs (21,842 review-tier rule confirmations + 35,824
unconfirmed) are written to `data/non_matches/` as the input to the downstream
probabilistic / ML stage; the 28,240 `reject` pairs are dropped. (The reject pile
is far smaller than the 126,903 on the raw 8-block set because the stacked
blocker's CNP prune already removed most of the weak edges that would have been
rejected — see [Decision distribution](#decision-distribution-run-real_20260620-stacked-blocker--tiered-rules).)

---

## Evaluation

The rules are evaluated **four ways**, in increasing order of label
independence. Methods 1–3 exist today; Method 4 is the open gap.

| # | Method | Labels | Precision | Recall | Notes |
|---|--------|--------|:---------:|:------:|-------|
| 1 | [Rules as their own silver standard](#method-1--rules-as-their-own-silver-standard-current-approach) (current) | none (AI adjudicator) | 100%* | 57% review-SAME proxy | circular; precision-only |
| 2 | [Silver labels](#method-2--silver-labels) | adjudicated blocking output | **83.1%** | **72.9%** | real records; **NAME_DOB_SEX → 65%** |
| 3 | [Synthetic data](#method-3--synthetic-data) | generator-true | **99.8%** | **77.1%** | adversarial negatives; recall floor |
| 4 | [Gold labels](#method-4--gold-labels-tbd) | hand-adjudicated | *TBD* | *TBD* | the open gap |

\*The Method-1 adjudicator scores **100%** for every rule — which Methods 2–3
show is **over-stated** for `NAME_DOB_SEX` and `NAME_DOB_ADDRESS` (the
adjudicator shares the rules' own name/DOB blind spot). Methods 2–3 are
reproducible with `python scripts/eval_against_labels.py` (writes
`data/runs/eval_against_labels.json`).

### Method 1 — Rules as their own silver standard (current approach)

> **Note (2026-06-29):** the counts in this section and the [Appendix](#appendix-generated-evaluation-report) are the verbatim `rule_eval.py` output from run `real_20260620`, **before** the `NAME_DOB_SEX` / `NAME_DOB_ADDRESS` demotion — so "matches" here means all five rules firing (44,786), not the current 23,155 auto-merges. Method 1 is circular by construction (it scores the rules with the rules' own tolerance) and is exactly what [Method 2](#method-2--silver-labels) overturns; it is kept for provenance. Re-run `rule_eval.py` against a post-demotion run to refresh these numbers.

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

#### Precision with bootstrap 95% CI (≤400 sampled matches per rule)

| Rule | Decided | Precision\* | 95% CI |
|------|---------|-------------|--------|
| SSN_DOB | 275 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_EMAIL | 300 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_PHONE | 300 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_SEX | 299 | 100.0% | 100.0 – 100.0 |
| NAME_DOB_ADDRESS | 300 | 100.0% | 100.0 – 100.0 |

\*Precision = SAME / (SAME + DIFFERENT); UNCERTAIN excluded (run `real_20260620`, `--n 300`). Zero adjudicated false positives. The CI collapses to a point because no resample produced a DIFFERENT; the true bound is limited by the adjudicator's own accuracy, not sampling. Note the adjudicator's name-similarity tolerance means fuzzy-confirmed name-typo pairs are scored SAME, so precision stays at 100% even with fuzzy matching on.

#### Confidence calibration

Assigned rule confidences are **conservative** — no rule's adjudicated precision falls below its stated confidence (every gap is ≤0):

| Rule | Assigned confidence | Adjudicated precision | Gap |
|------|--------------------|----------------------|-----|
| SSN_DOB | 1.000 | 100.0% | 0.0 |
| NAME_DOB_EMAIL | 0.990 | 100.0% | −1.0 |
| NAME_DOB_PHONE | 0.985 | 100.0% | −1.5 |
| NAME_DOB_SEX | 0.980 | 100.0% | −2.0 |
| NAME_DOB_ADDRESS | 0.970 | 100.0% | −3.0 |

(The adjudicator cannot resolve precision differences above ~99%, so these confirm "no over-confidence" rather than proving the exact ordering.)

#### Rule overlap & marginal contribution

How often each rule fires, and how often it is the **sole** rule confirming a pair (its unique contribution):

| Rule | Times fired | Sole confirmations | Sole % |
|------|-------------|--------------------|--------|
| NAME_DOB_SEX | 34,994 | 18,667 | 53.3% |
| NAME_DOB_PHONE | 19,488 | 4,835 | 24.8% |
| NAME_DOB_ADDRESS | 9,047 | 1,229 | 13.6% |
| SSN_DOB | 4,577 | 982 | 21.5% |
| NAME_DOB_EMAIL | 3,091 | 210 | 6.8% |

`NAME_DOB_SEX` does the most unique work. `NAME_DOB_EMAIL` is the most redundant (only 6.8% sole) but still uniquely confirms 210 pairs, so it earns its place. Heaviest co-firing is `NAME_DOB_SEX`×`NAME_DOB_PHONE` (12,209 shared pairs) — expected, since both build on the same name+DOB core.

#### Recall / what reaches the probabilistic stage

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

#### Blocking recall

Blocking's own recall — does it even surface the pairs the rules can confirm? —
is measured by `src/evaluation/blocking_eval.py` (rules as ground truth). On run
`real_20260620` it is **97.9%**: of 45,743 rule-confirmable pairs found in a wide
candidate set, blocking emitted 44,786 and missed 957. The misses skew to
**SSN_DOB (400)** — pairs sharing SSN+DOB that B1 did not emit, most likely
high-fan-out SSNs hit by the governance cap. See [Blocking-Guide.md](Blocking-Guide.md).

### Method 2 — Silver labels

`data/raw/silver_labels.csv` is the production blocking candidate set
(**204,805** pairs over the real `MDM_Population`, run `real_20260620`) with a
True/False `silver_label` adjudication per pair — **51,067 True / 153,738
False**. This is the first measurement of rule precision **and** recall against
labels produced *independently of the rules* (the silver adjudicator credits
51,067 pairs as matches vs the 44,786 the rules confirm — i.e. fire a rule, of
which 23,155 now auto-merge and 21,631 route to review — so it is more permissive
than the rules; rule recall < 100% against it is meaningful, not circular). This
analysis scores every rule, including the demoted review-tier ones, on the labeled
8-block candidate set; it is independent of the production stacked blocker.

Apply the rules to all 204,805 labeled pairs and score against `silver_label`:

> **Precision 83.1% · Recall 72.9% · F1 77.6%**
> (TP 37,208 · FP 7,578 · FN 13,859 · TN 146,160)

Per-rule precision (by winning `match_rule`):

| Rule | Fired | True | Precision |
|------|------:|-----:|:---------:|
| SSN_DOB | 4,577 | 4,577 | **100.0%** |
| NAME_DOB_EMAIL | 2,878 | 2,877 | 99.97% |
| NAME_DOB_PHONE | 15,700 | 15,662 | 99.76% |
| **NAME_DOB_SEX** | 20,402 | 13,272 | **65.05%** |
| **NAME_DOB_ADDRESS** | 1,229 | 820 | **66.72%** |

**Headline finding.** The corroborated rules (SSN, email, phone) hold at
≥99.8%, but `NAME_DOB_SEX` — the **highest-volume rule** (45.6% of all matches) —
adjudicates at only **65%**, and `NAME_DOB_ADDRESS` at **67%**. Name + DOB + sex
collides on real data (a common name + shared birthday + same sex is not a
unique identity), and a shared street address links co-resident family members
who share a birthday far more often than the synthetic negatives suggest. These
two rules carry essentially all of the 7,578 false positives.

**What Method 1 over-stated.** The rules-as-ground-truth adjudicator scored every
rule at **100%** precision — because it encodes the *same* name/DOB tolerance the
rules use, so it cannot see a name+DOB+sex collision as a false positive. The
silver labels expose it: a **−35 pp** precision over-statement on `NAME_DOB_SEX`
and **−33 pp** on `NAME_DOB_ADDRESS`. This also overturns the [confidence
calibration](#confidence-calibration) conclusion: against silver, `NAME_DOB_SEX`
(assigned 0.980) and `NAME_DOB_ADDRESS` (0.970) are **over-confident by ~33 pp**,
not conservative.

> **Action taken (2026-06-29): demoted, not recalibrated.** Rather than re-tune
> confidences (which still auto-merges the false positives) or demand a second
> corroborator (which would also drop the ~14k true matches these rules confirm),
> both rules were moved to the **review tier** — they still fire and keep
> provenance, but a pair confirmed only by one of them routes to review instead of
> auto-merging. Restricting auto-merge to the three corroborated rules lifts
> **auto-merge precision from 83.1% to 99.8%** on this silver set (false merges
> **7,578 → 39**; the 39 residual are 38 `NAME_DOB_PHONE` + 1 `NAME_DOB_EMAIL`),
> while the 14,092 silver-True pairs they confirmed flow to the review band for the
> downstream probabilistic / FS stage. The precision/recall table above is the
> per-rule analysis that *motivated* the demotion; it scores each rule as if it
> auto-merged. **The pre-demotion 83.1% / 72.9% is the old auto-merge operating
> point; the post-demotion auto-merge operating point is 99.8% precision** (recall
> shifts from auto-merge into the review band, not lost). Reproduce with
> `scripts/eval_against_labels.py` or by splitting `apply_rules` output on
> `AUTO_MERGE_RULES`.

**Worked examples — the collisions Method 1 can't see.** Ten pairs (each) where the
rule fires but silver labels it a non-match. All agree on the rule's fields; the
"what separates them" column is the conflicting identifier the rule ignores and
Method 1's adjudicator never inspects. (Reproduce via `apply_rules` on
`silver_labels.csv`, filtered to `match_rule == <rule>` and `silver_label == False`.)

`NAME_DOB_SEX` — all agree on name + DOB + sex:

| Name | DOB | Sex | What separates them |
|------|-----|:---:|---------------------|
| ISABELLA HIDALGO | 2010-09-21 | F=F | different address; contact info one side only |
| AMAIRANI ~ AMAYRANI SANCHEZ | 1995-08-08 | F=F | phone differs, address differs |
| TIMEIKA JOHNSON | 1979-06-19 | F=F | phone differs, address differs |
| CRISTOBAL SOLIS LOPEZ | 1990-08-31 | M=M | phone differs, address differs (one email is a different person's) |
| LOUISE BRAXTON | 1941-10-18 | F=F | phone differs, address differs |
| REMEKA KIMBLE | 1983-02-02 | F=F | phone differs, address differs |
| JOSE VENCES | 1998-01-26 | M=M | phone differs, address differs |
| DEMETRIUS GOSHA | 1991-09-18 | M=M | phone differs |
| GLORIA HOWARD | 1998-05-22 | F=F | phone differs |
| YASMIN YAWAR | 1960-12-05 | F=F | different address; contact info one side only |

…and **7,130 such pairs in total**.

`NAME_DOB_ADDRESS` — all agree on name + DOB + address (a *household* identifier, so
co-resident relatives sharing a birthday collide) — **409 such pairs in total**:

| Name | DOB | What separates them |
|------|-----|---------------------|
| MAATI YOUNG | 1994-10-22 | contact info one side only |
| LUCINA NUNEZ | 1959-06-30 | phone differs |
| JESSICA VALENCIA | 1999-05-29 | SSN / email one side only |
| KADREE THORNE | 1994-09-29 | phone differs |
| IGNACIO SILVA | 2007-03-31 | contact info one side only |
| MARILYN ZUNIGA | 2004-02-13 | phone differs |
| RASHID MOTIWALA | 1944-09-29 | phone differs |
| SHANA FOUNTAIN | 1984-08-26 | phone differs |
| LENELVER ~ LANELVER COLEMAN | 1971-06-07 | phone differs |
| SHEILA ~ SHELIA SCOTT | 1979-08-14 | SSN / phone one side only |

The dominant signature is **same name + same DOB, but a genuinely different phone
and/or address** — two distinct people whom silver separates on the conflicting
identifier. Because Method 1 re-applies the rule's own name/DOB tolerance, it rubber-
stamps every one of these as a true match. A couple (AMAIRANI/AMAYRANI SANCHEZ,
SHEILA/SHELIA SCOTT) are genuine name-variant ambiguities silver resolved against the
merge — themselves an argument for routing such pairs to a review/probabilistic stage
rather than auto-merging on name + DOB + sex alone.

**Caveats.** Silver labels are a heuristic/model adjudication of the blocking
output, not gold — the absolute 65% should be confirmed against Method 4 before
re-tuning confidences, though the *ordering* (SEX/ADDRESS weakest) is robust.
Silver cannot score blocking recall (it only labels pairs blocking emitted) — see
[Blocking-Guide.md](Blocking-Guide.md) Method 2.

### Method 3 — Synthetic data

`data/raw/synthetic_data.csv` is **40,000** pre-constructed pairs — **16,000
planted duplicates** (one entity corrupted into two records across 110
`case_type`s) and **24,000 hard negatives** — with generator-true labels and the
cleaned `*_l` / `*_r` fields already attached. Applying the rules pairwise:

> **Precision 99.8% · Recall 77.1% · F1 87.0%**
> (TP 12,335 · FP 28 · FN 3,665 · TN 23,972)

Per-rule precision is ≥98.6% across the board, and only **28 of 24,000** hard
negatives wrongly fire any rule (**0.12%** false-positive rate even on
adversarial negatives). The **recall of 77.1%** is the informative number: the
deterministic rules confirm ~77% of true duplicates, and the missed 23% carry
corruptions that break every rule's required field-agreement at once (e.g. a name
typo *and* an edited DOB) — exactly the cases the downstream probabilistic stage
must recover. By family: **77.1%** of duplicates fire a rule vs **0.12%** of
non-matches.

**Synthetic vs silver — why precision differs.** Synthetic puts `NAME_DOB_SEX` at
98.6% but silver at 65%. The synthetic hard negatives are built by *corrupting
fields* (typos, transpositions), which rarely leave name + DOB + sex all
agreeing; the real-world failure mode — two *different* people who genuinely share
a common name, birthday, and sex — is under-represented by the generator. So
**synthetic validates the rules against typo/field-corruption negatives (precision
floor), while silver exposes the identity-collision risk (precision ceiling).**
Use them together: synthetic for the recall floor, silver for the real-data
precision weak points.

### Method 4 — Gold labels (TBD)

*Placeholder.* A hand-adjudicated gold-standard sample — pairs stratified across
rules, blocks, and the silver True/False boundary, each reviewed by a human — is
the only way to settle whether `NAME_DOB_SEX`'s real precision is the silver-
estimated ~65% (and to measure true recall without the generator-shaped bias of
Method 3). This is the #1 ground-truth gap tracked in `to-do.md`; the silver
disagreements (the 7,578 rule-confirmed / silver-False pairs) are the natural
stratum to adjudicate first. Numbers to be filled in once the labeling exists.

---

## Data quality dependencies

Two deterministic-rule failure modes were diagnosed on the real data and fixed **upstream**, because no confidence tuning can repair a bad input value:

### Placeholder SSNs
A single placeholder SSN (`333333330`) had once chained **22 unrelated patients** into one 33-record SSN cluster at confidence 1.000. These values are structurally valid (they pass area/group/serial checks and `python-stdnum`), so they must be caught by entropy. `src/preprocessing/transformations.clean_ssn` now nulls any SSN that has ≤2 distinct digits, has one digit filling ≥7 of 9 positions, or is a full ascending/descending digit run (e.g. `012345678`), in addition to `python-stdnum` structural validation. Effect: max cluster size dropped 33 → 9 (now 7 after the rule demotion), SSN fan-out 25 → 6, and the SSN rule's precision rose from ~88% to ~100%.

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

These are confirmed matches that warrant clerical review — typically minor spelling variations, hyphenation/suffix differences, or name changes. Over the post-demotion **auto-merge** set (23,155 matches; `real_20260620`, stacked blocker) the suspicious rate is **12.9%** (2,994 matches). The flag is driven by **fuzzy name matching**: pairs are confirmed on near-equal names, and the `is_suspicious` last-name check is *strict*, so those typo'd last names register as a disagreement — exactly the cases a human should eyeball. (The rate is essentially unchanged by the demotion: the demoted `NAME_DOB_SEX` / `NAME_DOB_ADDRESS` pairs left the auto-merge set, so this now reports only the auto-merged matches.)

### Typology of the 2,994 suspicious matches (run `real_20260620`, auto-merge set)

| Disagreeing field | Count | Interpretation |
|-------------------|-------|----------------|
| Last name only | 2,915 (97%) | Fuzzy-matched name typos, maiden-name changes, hyphenation — usually still the same person |
| SSN only | 66 (2%) | Name+DOB agree but SSNs differ → one record likely has a wrong/typo'd SSN |
| Multiple fields | 13 (<1%) | Highest-risk; two or more identifiers disagree |
| DOB only | 0 | No rule confirms a DOB-disagreeing pair (all rules require exact DOB), so DOB-only suspicion is unreachable |

Last-name disagreement overwhelmingly dominates, consistent with the design: real-world name variation is this pipeline's primary source of identity noise, and fuzzy matching deliberately surfaces it for review rather than dropping it.

---

## Cluster Analysis

Large match clusters can indicate bad blocking keys, unfiltered placeholder values, or overly broad rules. Over the post-demotion **auto-merge** set (`real_20260620`, stacked blocker) there are **17,241 clusters**, the maximum cluster is **7 records**, and **85%** are clean 2-record pairs (14,693 of 17,241; 2,531 of size 3–5; 17 of size 6–20; none above 20). The cluster count is lower than the pre-demotion 27,215 (max 8) because `NAME_DOB_SEX` / `NAME_DOB_ADDRESS` no longer auto-merge. Any future cluster above ~15 members should be auto-flagged for manual review (a recommended circuit-breaker, not yet enforced in code).

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
