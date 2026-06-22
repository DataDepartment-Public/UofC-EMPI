# Blocking Guide

How the eMPI pipeline reduces the O(n²) record-pair comparison problem to a
tractable candidate set, and how that candidate set's **recall** is evaluated.

> **Scope:** the blocking stage (`src/features/blocking.py`) and its evaluation
> (`src/evaluation/blocking_eval.py`). For what happens to the candidate pairs
> afterwards, see [Deterministic-Rules-Guide.md](Deterministic-Rules-Guide.md).

## Pipeline position

```
raw → src/data/clean.py → cleaned dataset (PATID + *_clean fields)
    → src/features/blocking.py → candidate pairs (PATID_A, PATID_B, source_blocks, n_blocks)
    → src/models/deterministic_rules.py → match / reject / review
```

Blocking decides **which pairs are even compared**. A pair the rules would
confirm is invisible to the whole pipeline if blocking never emits it — so
blocking's *recall* (does it surface the true matches?) is the property that
caps everything downstream.

## Why blocking

Comparing all pairs of N records is N·(N−1)/2 comparisons — ~13 billion for the
163k production dataset. Blocking groups records on shared **keys** and only
emits pairs that co-occur in at least one key's group, cutting the comparison
space by orders of magnitude while aiming to keep every true match in at least
one block.

The output is one row per candidate pair with canonical ordering
(`PATID_A < PATID_B`), the `source_blocks` that produced it (pipe-delimited), and
`n_blocks` (how many blocks independently surfaced it).

## The 8-block scheme

Each block targets a different match scenario; together they trade recall for a
manageable candidate count. (There is no B2 — the numbering is historical.)

| Block | Key composition | Targets |
|-------|----------------|---------|
| **B1** | `SSN_clean` | Exact SSN matches |
| **B3** | DoubleMetaphone(`LastNM_clean`) + `BirthDT_clean` | Phonetic last name + exact DOB |
| **B4** | `LastNM_clean` + `BirthYear` + `FirstNM_clean[:3]` | Name + birth year + first-name prefix (DOB-typo tolerant) |
| **B5** | `Phones_set` intersection | Any shared phone number |
| **B6** | `Email_clean` | Exact email |
| **B7** | DoubleMetaphone(`LastNM_clean`) + `ZipCD_base` + `BirthYear` | Phonetic name + location + birth year |
| **B8** | Soundex(`FirstNM_clean`) + Soundex(`LastNM_clean`) + `BirthYear` | Coarse phonetic catch-all |
| **B9** | `LastNM_clean` + `FirstNM_clean` + `SSN_Last4` | Full name + last-4 SSN (front-digit SSN typos) |

A record joins a block only when **every** component of that block's key is
non-null; records missing a component are simply absent from that block.

### Derived fields used by the keys

Computed once in `_compute_derived_columns`:

- **Double Metaphone** of the last name — primary code via `phonetics.dmetaphone`
  (B3, B7). Tolerates spelling variation (SMITH ≈ SMYTH).
- **Soundex** of first and last name — via `jellyfish.soundex` (B8). Coarser than
  Metaphone, so it catches variants B3 splits apart.
- **BirthYear** — year component of `BirthDT_clean` (B4, B7, B8). Survives
  month/day transposition typos that break an exact-DOB block.
- **First-name prefix** — first three characters of `FirstNM_clean` (B4).
- **`ZipCD_base`** — the normalized 5-digit ZIP (B7).
- **`SSN_Last4`** — last four SSN digits (B9), catching typos in the first five.

## Safety controls

To prevent runaway blocks and false-positive clusters:

- **Governance cap.** A key value shared by more than `EMPI_GOVERNANCE_THRESHOLD`
  records (default **500**) is capped — only the first 500 records in the group
  produce pairs. This stops a junk key (a default clinic phone, a placeholder
  value) from generating a quadratic blow-up. Capped blocks are logged.
- **Null keys excluded.** Records with any null key component are left out of that
  block (above).
- **Invalid records excluded entirely.** Records the cleaning stage flagged
  `valid_record == False` (test/junk/invalid-marker rows) are dropped before
  blocking via `_filter_valid_records`.

> Blocking does **not** stop *low-frequency* junk (a placeholder SSN shared by 20
> records is far under the 500 cap). Those are handled upstream in cleaning — see
> [Data-Cleaning-Guide.md](Data-Cleaning-Guide.md).

## Public API

- `run_batch_blocking(df_clean)` → candidate pairs for the whole cleaned dataset
  (batch pipeline).
- `build_blocking_index(df_clean)` + `run_inference_blocking(record, index)` →
  block a single incoming record against a pre-built index (online / inference).
- `get_blocking_stats(candidate_pairs)` → per-block counts and pair distribution.

## Evaluating blocking recall

Blocking precision doesn't matter much (the rules filter false candidates
downstream); **recall is the metric that matters** — does blocking surface the
true matches at all? `src/evaluation/blocking_eval.py` estimates it using the
**deterministic rules as ground truth**: the rules are high-precision, so any
pair a rule confirms is effectively a true positive. If a rule-confirmed pair is
*not* in blocking's output, blocking missed a true match.

```
R = rules confirmed over a WIDER-than-production candidate set
B = production blocking output (run_batch_blocking)
recall = |R ∩ B| / |R|        misses = R − B
```

### Methods for the wider candidate set

- **`loose`** (default) — block the full dataset on single loose keys (exact DOB,
  Soundex(last name)) to surface pairs the 8-block scheme splits across blocks.
  Group-bounded, so it scales to the full dataset.
- **`sample`** — take a random sample of S records and generate every within-sample
  pair (C(S,2)). An unbiased local cross-check; O(S²) with per-pair fuzzy rule
  matching, so keep S small (a few hundred).

### Running it

```bash
python -m src.evaluation.blocking_eval --run-id <run_id>                 # loose
python -m src.evaluation.blocking_eval --run-id <run_id> --method sample --n 400
```

### Reading the report

- **ESTIMATED BLOCKING RECALL** — `|R ∩ B| / |R|`.
- **MISSED CONFIRMED PAIRS BY RULE** — for the misses (`R − B`), which rule fired,
  showing *which* signals blocking under-covers (e.g. many missed `NAME_DOB_*`
  pairs point at phonetic-key gaps).
- **CAUGHT CONFIRMED PAIRS BY PRODUCTION BLOCK** — which blocks actually carry the
  recall, so low-value blocks can be questioned.
- **SAMPLE OF MISSED PAIRS** — concrete `(PATID_A, PATID_B, match_rule)` to inspect.

### Measured recall (run `real_20260620`)

Loose method on the full 163,364-record dataset:

| Metric | Value |
|--------|-------|
| Wide candidate pairs (DOB + Soundex-last) | 9,631,646 |
| Production candidate pairs | 204,805 |
| Rule-confirmed in wide set (R) | 45,743 |
| Caught by blocking (R ∩ B) | 44,786 |
| Missed (R − B) | 957 |
| **Estimated blocking recall** | **97.9%** |

Missed pairs by rule: **SSN_DOB 400**, NAME_DOB_SEX 322, NAME_DOB_PHONE 160,
NAME_DOB_ADDRESS 40, NAME_DOB_EMAIL 35. The SSN_DOB misses are the notable
finding — pairs that share SSN *and* DOB that B1 (exact SSN) did not emit. The
most likely cause is the **governance cap**: a high-fan-out SSN whose group
exceeds 500 records is truncated, so not all of its pairs are generated. Worth a
look if SSN recall matters.

Caught pairs by block (where recall actually comes from): B8 42,993 · B3 42,803 ·
B4 38,323 · B7 20,268 · B5 19,598 · B1 4,577 · B9 3,786 · B6 3,127. The coarse
phonetic blocks (B8, B3, B4) carry the bulk of recall.

> The `sample` method is low-power on a dataset this size: a 600-record sample
> contains essentially no within-sample duplicate pairs (R ≈ 0), so use `loose`
> for real recall and `sample` only as a quick sanity check on small data.

### Caveats

- Recall is measured **only against rule-detectable matches**. True matches that
  neither blocking nor the deterministic rules catch are invisible to this tool —
  it is a *lower bound* on the work the downstream probabilistic stage must do.
- The `loose` method runs the (fuzzy) rules over a large candidate set (here 9.6M
  pairs); on the full real dataset this is the slow step (~minutes). Use `sample`
  only for quick checks on small data.
