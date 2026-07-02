# Blocking Guide

How the eMPI pipeline reduces the O(n²) record-pair comparison problem to a
tractable candidate set, and how that candidate set's **recall** is evaluated.

> **Scope:** the blocking stage and its evaluation
> (`src/evaluation/blocking_eval.py`). Since 2026-06-29 the production blocker is a
> **stacked** pipeline (`src/preprocessing/stacked_blocking.py`): the 8-block scheme
> (`src/preprocessing/blocking.py`) unioned with a typo-tolerant q-gram pass
> (`src/preprocessing/qgram_blocking.py`), then pruned by graph meta-blocking
> (`src/preprocessing/meta_blocking.py`). For what happens to the candidate pairs
> afterwards, see [Deterministic-Rules-Guide.md](Deterministic-Rules-Guide.md).

## Pipeline position

```
raw → src/preprocessing/clean.py → cleaned dataset (PATID + *_clean fields)
    → src/preprocessing/stacked_blocking.py
          8-block (blocking.py)  ∪  q-gram (qgram_blocking.py)  →  CNP prune (meta_blocking.py)
        → candidate pairs (PATID_A, PATID_B, source_blocks, n_blocks)
    → src/models/deterministic_rules.py → match / review / reject
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

## The stacked blocker (production path)

Every name+DOB block (B3/B4/B7/B8/B9) anchors on the **last name**, so a last-name
typo that breaks *both* Soundex and Double Metaphone escapes all of them at once —
the documented 212-pair residual ([measured recall](#measured-recall-run-real_20260620)
below). The production blocker (`src/preprocessing/stacked_blocking.run_stacked_blocking`)
closes that gap by stacking three stages:

```
8-block (run_batch_blocking)  ∪  q-gram (run_qgram_blocking)  →  CNP/ARCS prune (prune_candidate_pairs)
```

### 1. q-gram pass — `src/preprocessing/qgram_blocking.py`

A **character-n-gram TF-IDF cosine** on the full name, restricted to within a cheap
coarse block (exact birth date by default). Restricting the O(g²) cosine to small
within-DOB groups keeps it sub-quadratic. Any pair whose name cosine ≥ the
threshold is emitted with `source_blocks = "QGRAM"`. This recovers exactly the
last-name-typo / phonetic-miss pairs the 8-block scheme splits apart. Configurable
via `settings.qgram_*`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `qgram_block_kind` | `fulldob` | coarse block (`fulldob` or `birthyear`) |
| `qgram_threshold` | `0.30` | min cosine — **pending gold-label confirmation** |
| `qgram_ngram_min/max` | `2` / `4` | char n-gram sizes (`char_wb`) |
| `qgram_min_df` | `2` | drop n-grams rarer than this |

### 2. Graph meta-blocking — `src/preprocessing/meta_blocking.py`

The q-gram pass and the union both over-generate, so the combined set is pruned by
**Cardinality Node Pruning** over **ARCS**-weighted edges:

- **ARCS weight** — an edge's weight is the sum over the blocks that produced it of
  `1 / (pairs that block emitted)`. Specific blocks (a shared SSN) score high;
  coarse blocks (common Soundex + birth year) score low. q-gram-only edges carry no
  block-cardinality signal, so they get the flat `settings.cnp_qgram_only_weight`
  (default 0.5).
- **CNP** — keep an edge if it ranks in the top-`settings.cnp_top_k` (default **10**)
  ARCS-weighted edges of **either** endpoint. This bounds each record to ~k of its
  strongest links.

### Stacked-blocker result (run `real_20260620`)

| Stage | Candidate pairs | vs 8-block |
|-------|----------------:|-----------:|
| 8-block alone | 204,805 | — |
| 8-block ∪ q-gram (union) | ~290k | over-generates |
| **stacked (after CNP prune)** | **109,061** | **−46.7%** |

The prune cuts the candidate set nearly in half while retaining **99.96%** of
silver-True pairs (51,048 / 51,067 — 19 dropped) and 99.5% of the 8-block recall
residual. Net: fewer comparisons for the downstream rules **and** the typo residual
recovered. The q-gram threshold and CNP top-k are still to be locked against
[gold labels](#method-4--gold-labels-tbd).

> **Online inference is still 8-block only.** The q-gram pass fits its TF-IDF
> vocabulary on the whole corpus, so it is batch-only; `run_inference_blocking`
> (single incoming record vs a pre-built index) continues to use the 8-block index.

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

- `run_stacked_blocking(df_clean)` (`stacked_blocking.py`) → the **production**
  candidate pairs: 8-block ∪ q-gram → CNP prune. Use this in the batch pipeline.
- `run_batch_blocking(df_clean)` (`blocking.py`) → the 8-block scheme alone (the
  base leg; still used directly by the blocking-recall evaluation).
- `run_qgram_blocking(df_clean)` (`qgram_blocking.py`) → the q-gram leg alone.
- `prune_candidate_pairs(pairs)` / `arcs_weights` / `cnp_prune`
  (`meta_blocking.py`) → the meta-blocking prune.
- `union_candidate_pairs(*frames)` (`stacked_blocking.py`) → merge candidate-pair
  frames on `(PATID_A, PATID_B)`, unioning `source_blocks` and counting `n_blocks`.
- `build_blocking_index(df_clean)` + `run_inference_blocking(record, index)` →
  block a single incoming record against a pre-built 8-block index (online /
  inference; the q-gram leg is batch-only).
- `get_blocking_stats(candidate_pairs)` → per-block counts and pair distribution.

## Evaluation

Blocking is evaluated **four ways**; each answers a different question with a
different trust level. **Recall** — does blocking surface the true matches at
all? — is the property that caps everything downstream, so it is the focus.

| # | Method | What it scores | Recall measurable? | Headline | Trust |
|---|--------|----------------|:------------------:|----------|-------|
| 1 | Rules as ground truth (current) | recall vs rule-confirmable pairs | yes (lower bound) | **99.5%** | silver, circular |
| 2 | [Silver labels](#method-2--silver-labels) | pair-quality of the emitted set | no (silver ⊆ blocking output) | PQ **24.9%** | silver |
| 3 | [Synthetic data](#method-3--synthetic-data) | recall vs independently-planted duplicates | yes (label-true) | PC **96.9%** | synthetic |
| 4 | [Gold labels](#method-4--gold-labels-tbd) | recall + PQ vs hand-adjudicated truth | yes | *TBD* | gold |

Methods 2–3 are reproducible with `python scripts/eval_against_labels.py`
(writes `data/runs/eval_against_labels.json`).

### Method 1 — Rules as ground truth (current approach)

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

#### Methods for the wider candidate set

- **`loose`** (default) — block the full dataset on single loose keys (exact DOB,
  Soundex(last name)) to surface pairs the 8-block scheme splits across blocks.
  Group-bounded, so it scales to the full dataset.
- **`sample`** — take a random sample of S records and generate every within-sample
  pair (C(S,2)). An unbiased local cross-check; O(S²) with per-pair fuzzy rule
  matching, so keep S small (a few hundred).

By default the evaluation builds **R** over the *valid-record* population only
(`valid_record != False`) — the same records production blocking sees — so the
recall denominator never counts pairs blocking was filtered from emitting. Pass
`--include-invalid` for the diagnostic all-records view. Measuring over all records
re-introduces the validity-filter artifact diagnosed in
[Blocking-Recall-RCA.md](../Blocking-Recall-RCA.md): ~745 of the apparent "misses"
are pairs where one record was dropped as invalid before blocking, which deflates
the headline number.

#### Running it

```bash
python -m src.evaluation.blocking_eval --run-id <run_id>                 # loose
python -m src.evaluation.blocking_eval --run-id <run_id> --method sample --n 400
```

#### Reading the report

- **ESTIMATED BLOCKING RECALL** — `|R ∩ B| / |R|`.
- **MISSED CONFIRMED PAIRS BY RULE** — for the misses (`R − B`), which rule fired,
  showing *which* signals blocking under-covers (e.g. many missed `NAME_DOB_*`
  pairs point at phonetic-key gaps).
- **CAUGHT CONFIRMED PAIRS BY PRODUCTION BLOCK** — which blocks actually carry the
  recall, so low-value blocks can be questioned.
- **SAMPLE OF MISSED PAIRS** — concrete `(PATID_A, PATID_B, match_rule)` to inspect.

#### Measured recall (run `real_20260620`)

Loose method on the full 163,364-record dataset (158,724 valid), 8-block scheme:

| Metric | All records | Valid records (default) |
|--------|------------:|------------------------:|
| Production candidate pairs | 204,805 | 204,805 |
| Rule-confirmed in wide set (R) | 45,743 | 44,998 |
| Caught by blocking (R ∩ B) | 44,786 | 44,786 |
| Missed (R − B) | 957 | 212 |
| **Estimated blocking recall** | 97.9% | **99.5%** |

Measured apples-to-apples over the valid population (the default), recall is
**99.5%**. The 745-pair difference is the validity-filter artifact (one record
dropped as invalid before blocking), not a blocking defect — see
[Blocking-Recall-RCA.md](../Blocking-Recall-RCA.md).

The remaining **212** valid-population misses are the one genuine blocking-scheme
gap: pairs that share a first name + exact DOB but whose **last name differs by a
typo that breaks both Soundex and Double Metaphone**. Every name+DOB block
(B3/B4/B7/B8/B9) anchors on the last name, so such a typo escapes all of them at
once. The robust fix is **q-gram / n-gram (sub-quadratic, typo-tolerant)
blocking**, tracked under the embedding/graph blocking research; an in-block fuzzy
patch on B8 was evaluated and rejected as not worth the quadratic cost for the
residual (see that research item). The earlier hypothesis that the SSN_DOB misses
were a governance-cap artifact was **disproven** by the RCA — the largest SSN
group on this data is 6, far under the 500 cap; those misses are all
invalid-record pairs.

> The `sample` method is low-power on a dataset this size: a 600-record sample
> contains essentially no within-sample duplicate pairs (R ≈ 0), so use `loose`
> for real recall and `sample` only as a quick sanity check on small data.

#### Caveats

- Recall is measured **only against rule-detectable matches**. True matches that
  neither blocking nor the deterministic rules catch are invisible to this tool —
  it is a *lower bound* on the work the downstream probabilistic stage must do.
- The `loose` method runs the (fuzzy) rules over a large candidate set (here 9.6M
  pairs); on the full real dataset this is the slow step (~minutes). Use `sample`
  only for quick checks on small data.

### Method 2 — Silver labels

`data/raw/silver_labels.csv` is the production blocking candidate set (the same
**204,805** pairs from run `real_20260620`) with a True/False `silver_label`
adjudication per pair — **51,067 True / 153,738 False**. Because every silver
pair is one blocking *already emitted*, this set **cannot measure blocking
recall**: there are no true pairs living outside the blocking output for it to
catch blocking missing, so recall against it is trivially 100% by construction.

What it gives instead is blocking **pair-quality (PQ)** — the share of emitted
candidates that are real matches:

> **PQ = 51,067 / 204,805 = 24.9%**

A low PQ is **expected and not a defect**: blocking deliberately over-generates
and lets the deterministic + probabilistic stages filter. PQ is the *quality*
side of the recall/quality trade-off — useful for sizing the downstream
comparison load, not for judging recall. To score blocking recall against labels
you need true pairs that live *outside* the candidate set; that is what the
synthetic set (Method 3) supplies.

### Method 3 — Synthetic data

`data/raw/synthetic_data.csv` is **40,000** pre-constructed pairs — **16,000
planted duplicates** (one entity corrupted into two records: name typos, DOB
edits, address moves, SSN/phone changes, across 110 `case_type`s) and **24,000
hard negatives** — with known labels. Because the duplicates are generated
**independently of blocking**, they are true pairs blocking can genuinely miss,
so this set measures real **pair-completeness (PC = recall)** — and, unlike
Method 1, it is *not* capped by what the deterministic rules can confirm, so it
credits typo-tolerant recall the rules cannot.

`scripts/eval_against_labels.py` rebuilds an 80,000-record table (both sides of
every pair; each record appears in exactly one pair) and runs the 8-block scheme
over it:

| Metric | Value |
|--------|------:|
| Records blocked | 80,000 |
| Candidate pairs emitted | 89,870 |
| True duplicate pairs | 16,000 |
| **Duplicates surfaced (PC / recall)** | 15,509 — **96.9%** |
| Hard negatives also surfaced | 38% (rules reject them downstream) |

PC by `case_type` is 96–100% for almost every scenario; the only weak spots are
**heavy multi-field corruption** — **M-NOSSN-04 (51%)** and **M-MIX-02 (68%)** —
where enough anchoring fields are corrupted at once that no block's key survives.
This independently corroborates the 99.5% rules-as-ground-truth figure on a
*harder, label-true* population, and the residual misses line up with the known
last-name / multi-field-typo gap that q-gram blocking targets (see [the
embedding/graph research](../Blocking-Research-Embedding-Graph.md)).

> **What Method 1 over/under-stated.** Method 1's 99.5% is a recall *lower bound
> against only rule-confirmable matches*; Method 3's 96.9% is true-label recall
> over a deliberately harder, corrupted population. The two are not in conflict —
> they measure different denominators — but Method 3 is the first recall number
> that is not capped by the rules.

### Method 4 — Gold labels (TBD)

*Placeholder.* A hand-adjudicated gold-standard sample — stratified across blocks
and the q-gram thresholds, every pair reviewed by a human — is the only way to
measure blocking recall **and** pair-quality without the silver/synthetic
caveats (silver only labels what blocking already emitted; synthetic is
generator-shaped, not real records). This is the #1 ground-truth gap tracked in
`to-do.md`; numbers to be filled in once the labeling exists.
