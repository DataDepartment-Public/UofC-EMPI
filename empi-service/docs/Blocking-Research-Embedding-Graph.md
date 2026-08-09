# Blocking Research — q-gram & Graph Techniques

**Status:** exploratory research, **analysis only — not integrated into the
pipeline.** Every experiment is read-only on `data/processed/` and `data/raw/`; no
production code was changed. Three committed harnesses reproduce every number
(`scripts/research/research_blocking_labels.py`, `scripts/research/research_blocking_scalable.py`,
`scripts/research/research_blocking_round4.py`, plus `scripts/eval_against_labels.py`); see
[§9](#9-reproducibility).

This document is the **final research synthesis** before an implementation decision.
It evaluates alternatives to the current hand-built 8-block scheme —
**q-gram (character-similarity) blocking**, **graph techniques (meta-blocking,
connectivity, community detection, hub analysis)**, and two **alternative blockers
that were tested and ruled out (Sorted Neighbourhood, super-string LSH)** — measured
against three yardsticks of ground truth, analyses how each would **scale**, and ends
with the **stacked pipeline** (q-gram → meta-blocking) that combines the two winners.
It folds together four rounds of work: the original exact-method benchmark, a
re-scoring against real labels, a scalable production-viable implementation, and this
final round of graph-internal analysis, dominated-alternative elimination, and
stacking.

---

## TL;DR

- **The recommended design is a stack: 8-block ∪ block-then-q-gram, then CNP
  meta-blocking.** Measured end-to-end (§6) it lands at **109,098 candidates
  (−46.7% vs the 8-block baseline)** while **retaining 99.96% of real (silver-True)
  matches** *and* **recovering 99.5% of the structural residual** — the recall win
  and the efficiency win at the same time, neither cannibalising the other. This is
  the headline of the final round: the two techniques were previously measured
  separately; stacked, the q-gram residual pairs survive the prune.
- **q-gram blocking is the recall win, in a shippable form.** Instead of eight
  hand-tuned keys, run one rule: *compare two records' names character-chunk by
  character-chunk and pair them if similar enough.* Validated against **real labels**
  it **recovers ~100% of the duplicates the 8-block scheme structurally misses** and
  catches **99%+** of real matches. Run inside the existing date-of-birth key
  (**block-then-q-gram**) it reproduces those wins at **fewer candidates and ~10 s of
  near-linear compute** — not the O(N²) exact pass. Its single hardest miss (§3.5) is
  a record pair with a typo in *both* the first and last name at once.
- **…but it is not a drop-in replacement.** q-gram only "sees" name spelling, so it
  misses the ~0.8% of matches that link on a **shared phone/email/SSN with a
  different name**. Adopt it as an **add-on** that keeps the strong-identifier blocks.
- **Graph meta-blocking is the efficiency win, and we now understand *why*.** The
  candidate graph is extremely heavy-tailed — median degree **1**, but a tail of
  **~2,000 "hub" records with degree > 20** (placeholder-SSN chains, common name+DOB
  clusters) whose edges are **>98% false** (§4.5). Node-centric pruning (CNP) is
  exactly the right scalpel: it slashes a degree-340 junk hub to its top 10 edges
  while **keeping 100% of that hub's real matches**, cutting the candidate set
  **−48%** at **99.96% real-match retention** and **2× precision**.
- **Graph connectivity is a weak recall tool — confirmed three ways.** 2-hop
  expansion, connected-components, and Louvain community detection all plateau at
  **~43% of the residual** and **0% on isolated pairs**, because connectivity can
  only reach a miss that already sits in a connected cluster. Community detection
  costs **9.6 M candidates** (connected-components) for that 43% — far worse than
  q-gram. Keep connectivity, at most, as a supplementary signal.
- **Two alternative blockers were tested and ruled out (§5).** **Sorted
  Neighbourhood (SNM)** can recover the residual (via a DOB sort key) but only at
  **40–130× the candidates** and **10–40× worse precision** than block-then-q-gram —
  strictly dominated. **Super-string LSH** (one concatenated PII field, MinHash-LSH)
  is the only *scalable* super-string variant, but low-entropy concatenations
  (lastname|year|sex) collapse into pathologically large LSH buckets; the all-pairs
  cosine form is the same O(N²) class as the retired exact pass. Dropped.
- **Scalability:** the 8-block baseline and meta-blocking are linear and safe at any
  size; **block-then-q-gram is the linear, measured form of q-gram**; the exact
  all-pairs cosine and super-string cosine are retired to correctness references,
  with ANN/LSH as the large-scale fallback.

---

## 1. The problem, in plain terms

Comparing every patient record to every other is **N·(N−1)/2** comparisons — about
**13 billion** for our 163k records. **Blocking** avoids that: it groups records
that share something (a phonetic last name, a birth date, a phone number) and only
compares records in the same group. Blocking's one job is **recall** — make sure
the two records of a true duplicate land in *at least one* group together. Whatever
blocking fails to pair is invisible to every later stage, so **blocking's recall
caps the entire pipeline**.

The production system uses **8 hand-built blocks** (SSN, phonetic last-name + birth
date, phone, email, …). It works well (99.5% recall) but has a known blind spot: a
residual of **212 pairs** where a last-name typo breaks *every* name-based key at
once (a `SMITH`→`SMYTH` edit changes the exact spelling, the Soundex, *and* the
Double-Metaphone code simultaneously; see `Blocking-Recall-RCA.md`). This research
asks whether character-similarity and graph techniques can close that gap and/or
cut cost — and whether any of it is fast enough to actually use.

---

## 2. How we measure

A blocking method produces a **candidate set C** — the pairs it thinks are worth
comparing. We score it with the standard blocking metrics:

| Metric | Plain meaning | Formula |
|---|---|---|
| **PC** (Pairs Completeness) | **recall** — of the true matches, what fraction did we pair? | `\|C ∩ truth\| / \|truth\|` |
| **PQ** (Pairs Quality) | **precision** — of the pairs we emitted, what fraction are real? | `\|C ∩ truth\| / \|C\|` |
| **RR** (Reduction Ratio) | how much of the 13-billion-pair space we pruned | `1 − \|C\| / \|all pairs\|` |

The hard part is **"truth."** We use three sources of increasing trust (the same
four-method framework written up in the guides; the fourth — gold labels — is still
to come):

1. **Rules-as-ground-truth (current method).** The deterministic rules are high
   precision, so any pair a rule confirms is treated as a true match (**R**).
   *Limitation:* the rules only confirm exact-DOB + near-exact-name pairs, so a
   *more* tolerant blocker (q-gram) gets no credit for the very matches it's built
   to catch — its recall here is a **floor, not the real number**.
2. **Silver labels** (`data/raw/silver_labels.csv`). The 204,805 production
   candidate pairs, each adjudicated True/False (**51,067 True**). Real records;
   measures recall of *real* matches and precision — but only on pairs the 8-block
   scheme already emitted.
3. **Synthetic data** (`data/raw/synthetic_data.csv`). 16,000 planted duplicates +
   24,000 hard negatives, generated **independently of blocking** — the one source
   that can credit q-gram for matches the 8-block scheme *and* the rules miss.

**Baseline — the current 8-block scheme (real data, N = 158,724 valid records):**

| | candidates | PC (recall) | PQ (precision) | RR |
|---|---:|---:|---:|---:|
| **8-block scheme** | 204,805 | **99.53%** | 21.9% | 99.9984% |

`\|R\| = 44,998` rule-confirmed pairs; the scheme misses **212** of them (the
last-name-typo residual). `silver-True = 51,067`. Naïve all-pairs ≈ 1.26 × 10¹⁰.

---

## 3. Method A — q-gram blocking (the recall play)

### 3.1 The idea, in plain terms

Write each record as one string — e.g. `john smith` (name) or `john smith 19840312`
(name + birth date) — and chop it into overlapping **character chunks** of 2–4
letters ("q-grams": `joh`, `ohn`, `hn `, …). Two records are paired if their
chunk-sets overlap enough (cosine similarity ≥ a threshold). This is exactly how a
fuzzy search box works.

**Why it fits the residual.** A one-letter typo (`SMITH` → `SMYTH`) changes only a
couple of chunks, so the two strings still look ~95% the same — whereas it breaks
exact match, Soundex, *and* Double-Metaphone all at once. q-gram **degrades
gracefully** under precisely the typos that defeat the 8-block keys, and replaces
eight hand-built keys with one threshold dial. It uses a plain character-n-gram
TF-IDF vector — no embedding model, no neural network.

### 3.2 The exact pass — a clean precision/recall dial

Sweeping the similarity threshold on the full identity string `name + birthdate`
(rules-as-ground-truth; baseline = 204,805 cands, PC 99.53%, residual recovered 0%):

| cosine ≥ | candidates | PC (recall) | PQ | residual recovered |
|---:|---:|---:|---:|---:|
| 0.50 | 462,180 | **99.72%** | 9.7% | **99.5%** (211/212) |
| **0.55** | **187,984** | **99.46%** | 23.8% | 97.2% (206/212) |
| 0.60 | 106,369 | 98.90% | 41.8% | 93.9% (199/212) |
| 0.65 | 80,454 | 97.79% | 54.7% | 86.3% |
| 0.70 | 67,959 | 95.68% | 63.4% | 77.4% |
| 0.80 | 54,883 | 88.03% | 72.2% | 39.6% |

The threshold is a smooth dial: at ≥ 0.55 one q-gram blocker matches the 8-block
recall (99.46% vs 99.53%) at *fewer* candidates and slightly higher precision; at
≥ 0.50 it *beats* the baseline and recovers the residual; tighten it for ~2×
precision at a small recall cost. But these PC numbers are a **floor** — the rules
can't credit q-gram for matches they themselves can't confirm (§2.1). The real test
is the next section.

### 3.3 Validated against real labels (silver + synthetic)

Re-scoring the q-gram set against silver and synthetic removes the rules cap.

**Real records, vs silver:**

| method | candidates | rules PC | rules PQ | residual rec. | **silver PC** | **silver PQ** | q-gram-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8-block | 204,805 | 99.53% | 21.9% | 0% | 100%\* | 24.9% | 0 |
| **q-gram ≥ 0.55** | 151,672 | 99.75% | 29.6% | 99.5% | **99.2%** | **75.1%** | 84,207 |
| q-gram ≥ 0.60 | 100,869 | 99.55% | 44.4% | 98.6% | 98.3% | 76.0% | 34,802 |

\*8-block silver PC is trivially 100% — silver *is* the adjudicated 8-block output.
silver PC/PQ for q-gram are over the labelled subset (pairs also in the 8-block
output); q-gram-only pairs are silver-unlabelled (last column). Candidate counts
differ slightly from §3.2 because this run used a top-100-neighbour float-32 kernel
for tractability; conclusions are identical.

**Synthetic duplicates (independent of blocking — the uncapped recall test):**

| method | candidates | PC (recall) | residual recovered (491) | hard-neg surfaced |
|---|---:|---:|---:|---:|
| 8-block | 89,870 | 96.93% | 0% | 38.0% |
| **q-gram ≥ 0.55** | 511,546 | **99.44%** | **100%** (491/491) | 21.8% |
| q-gram ≥ 0.60 | 222,869 | 98.68% | 99.6% | 19.0% |

**What this establishes:**

1. **q-gram recovers the structural residual completely — better than the
   deterministic fix.** On the synthetic duplicates it recovers **all 491** pairs
   the 8-block scheme misses, lifting recall 96.9% → 99.4%, while surfacing *fewer*
   hard negatives. (The best *deterministic* recovery block, B10 in
   `Blocking-Recall-RCA.md` §8, reached only 473/491 — q-gram reaches all, because
   character chunks survive even when *both* names are edited.)
2. **It catches real matches at much higher precision.** Against silver it catches
   **99.2%** of the 51,067 real matches, and **75%** of its labelled candidates are
   true — vs **25%** for the 8-block scheme.
3. **But q-gram is NOT a superset of the 8-block scheme.** silver PC is 99.2%, not
   100%: it misses ~0.8% (~408) of the matches the 8-block scheme catches. These are
   **strong-identifier-only** matches — two records linked by a shared
   phone/email/SSN whose *names differ* — which a name-similarity blocker cannot
   see. **So q-gram complements the strong-ID blocks (SSN/phone/email); it does not
   replace them.**
4. **Its upside beyond the 8-block scheme is real but only partly measured.** The
   **84,207** "q-gram-only" pairs (emitted by q-gram, never by the 8-block scheme)
   are silver-unlabelled — silver only covers what the 8-block scheme emitted. The
   synthetic result is the positive signal this region holds real matches; valuing
   it directly needs a **gold** adjudication (the open #1 gap).

### 3.4 The scalable, production-viable form: block-then-q-gram

§3.2–3.3 use an **exact** all-pairs cosine — a research yardstick, not shippable (it
trends quadratic; §7). The fix is to restrict the cosine to within a **cheap coarse
block**: group records by exact date of birth, then run the name-only character
cosine *inside each DOB group*. The residual pairs all share an exact DOB (they're
last-name typos), so DOB-blocking keeps them together while the comparison count
collapses from ~N² to **1.28 M** (≈20,000× fewer) — essentially linear.

| method | candidates | rules PC | silver PC | residual rec. | compute |
|---|---:|---:|---:|---:|---:|
| exact global q-gram ≥ 0.55 | 151,672 | 99.75% | 99.2% | 99.5% | ~minutes |
| **block-then-q-gram (DOB) ≥ 0.30** | **68,849** | **99.87%** | **99.4%** | **99.5%** (211/212) | **~10 s** |
| block-then-q-gram (birth-year) ≥ 0.30 | 680,716 | 99.87% | 99.6% | 99.5% | ~10 s |

**The DOB-blocked version reproduces the exact pass's wins — same recall, same
residual recovery — at *fewer* candidates and ~10 s of near-linear compute**, and
it slots onto the DOB key the pipeline already computes.

- *On real data it loses nothing:* the rules require an exact DOB anyway, so
  DOB-blocking can't drop a rule-confirmable match (rules PC even ticks up, 99.75 →
  99.87, because the tighter blocking trims noise).
- *On synthetic data* it recovers 99.8% of the residual but only ~93% of *all*
  positives — the gap is the planted **DOB-typo** duplicates, which fall in
  different DOB groups. Those are already caught by the existing birth-year blocks
  (B4/B8), so as an **add-on** the DOB-blocked backend loses nothing the 8-block
  scheme didn't already have.
- The **birth-year** variant tolerates month/day DOB typos standalone, but at ~10×
  the candidates — so DOB-blocking is the efficient default; birth-year (or an
  ANN/LSH backend, §7) is the fallback if DOB-typo recall must stand alone.

> **Name-only threshold.** Block-then-q-gram scores names alone (DOB is the block,
> not part of the string), so the right cosine cutoff is lower (~0.30) than the
> ~0.55 used on the `name+DOB` string — the DOB digits no longer inflate the score.

### 3.5 Residual deep-dive — what's the single hardest case?

At its operating point (DOB ≥ 0.30) block-then-q-gram recovers **211 of the 212**
residual pairs. Characterising the **one** it still misses tells us where the floor
of name-similarity blocking lies:

| threshold | residual | recovered | still missed | trait of the misses |
|---|---:|---:|---:|---|
| block-then-q-gram ≥ 0.30 | 212 | **211** | **1** | same DOB, *both* name fields corrupted |
| block-then-q-gram ≥ 0.35 | 212 | 211 | 1 | same DOB, *both* name fields corrupted |

The lone hold-out is a **double typo**:

> `LUBSY` ↔ `LUSBY` (last name — an `SB`↔`BS` transposition) **and**
> `TITIANA` ↔ `TATIANA` (first name — an `I`↔`A` edit), on an identical DOB.

Both name fields are simultaneously corrupted, so the shared character-chunk mass
drops below the cosine threshold even at 0.30. Every still-missed residual pair has
the same signature — **same DOB, *different* last name** (the transposition makes the
last names not-equal) — confirming the hardest residual is *not* a strong-ID case but
a **compound name corruption**. This is the natural boundary where the eventual
probabilistic stage (which can weigh a shared DOB + partial name overlap jointly)
takes over from blocking. One pair in 212 is a ~0.5% name-blocking floor — negligible
for an add-on, but the precise reason a single name-similarity dial can never reach
exactly 100%.

---

## 4. Method B — graph techniques (efficiency, and a partial recall tool)

### 4.1 Meta-blocking — the idea, in plain terms

The 8 blocks emit a **redundant** candidate set: many pairs are flagged by several
blocks, many by only one. Picture the candidates as a **network** — patients are
dots, each candidate pair is a line. Strong matches sit on lines flagged by many
blocks; the weak long tail sits on single-block lines. Score each line by how much
evidence backs it, then drop the weak ones.

- **How much evidence? (two scores).** *Count* (CBS) = how many blocks flagged the
  pair. *Quality-weighted* (ARCS) = give more credit to agreement on **rare,
  distinctive** details (a full SSN) than on common ones (a popular phonetic
  surname).
- **Which to drop? (two strategies).** *One global cutoff* (WEP): set a single
  passing score and drop everything below it — simple but blunt, it throws away real
  matches flagged by only one (still-valid) block. *Keep each record's best leads*
  (CNP, recommended): for **every** patient keep its few strongest lines, so no true
  match is ever left without a surviving candidate — this protects recall while
  cutting the weak tail.

### 4.2 Meta-blocking results

(N = 158,724; baseline B = 204,805 cands, PC 99.53%, PQ 21.87%)

| Strategy | candidates | Δ vs B | PC (recall) | ΔPC | PQ |
|---|---:|---:|---:|---:|---:|
| **CNP, ARCS, top-10/node** | **105,862** | **−48.3%** | **99.53%** | **0.00** | **42.3%** |
| CNP, ARCS, top-5/node | 95,043 | −53.6% | 99.45% | −0.08 | 47.1% |
| WEP, CBS ≥ mean (n_blocks ≥ 2) | 61,636 | −69.9% | 96.41% | −3.12 | 70.4% |
| WEP, ARCS ≥ mean | 56,499 | −72.4% | 92.36% | −7.16 | 73.6% |
| WEP, CBS, n_blocks ≥ 3 | 54,111 | −73.6% | 90.78% | −8.75 | 75.5% |

**Keep-each-record's-best-leads (CNP, top-10) is the right tool:** it removes
**half** the candidates with **no measurable recall loss** and nearly **doubles
precision** (21.9% → 42.3%). Global cutoffs cut more but cost 3–9 points of recall —
the wrong trade for a recall-critical stage. Meta-blocking only prunes existing
pairs, so it **cannot add recall** (it recovers 0 of the 212 residual) — it is an
**efficiency** technique.

### 4.3 Validating the prune against real labels (silver)

§4.2's "0 recall loss" is measured against the *rules*. Silver labels every 8-block
pair True/False, and meta-blocking only prunes 8-block pairs, so silver gives the
**real** recall of the cut — does pruning drop any *adjudicated* match?

| prune | candidates | Δ vs B | rules PC | **silver-True retained** | silver PQ |
|---|---:|---:|---:|---:|---:|
| CNP, ARCS, top-10/node | 105,890 | −48.3% | 99.53% | **99.96%** | 48.2% |
| CNP, ARCS, top-5/node | 95,058 | −53.6% | 99.45% | 99.87% | 53.7% |

The top-10 prune **retains 99.96% of the 51,067 silver-True real matches** — it
drops ~20 of them while removing ~99,000 candidates — and **doubles** precision
(silver PQ 24.9% → 48.2%). So the −48% cut is genuinely (not just vs-rules)
recall-free.

### 4.4 Graph for *recall*: 2-hop expansion

Meta-blocking only prunes; can graph **connectivity** instead *recover* misses? Test
**2-hop transitive expansion**: if A–B and B–C are both candidate pairs, add A–C
(a match no single block emitted directly), expanding through low-degree "bridge"
records.

| population | new candidates | residual recovered |
|---|---:|---:|
| real | 44,702 | **41.5%** (88 / 212) |
| synthetic | 139,230 | **0%** (0 / 491) |

2-hop is a **real but partial** recall tool: on real data it reaches 41.5% of the
residual through shared bridge records, at ~45k new candidates. But it recovers
**nothing** on the synthetic set, where the planted duplicates are isolated pairs
with no bridge — and it is far behind q-gram's 99.5% residual recovery.

### 4.5 Hub / degree-distribution analysis — *why* node-centric pruning is the right tool

The whole case for CNP (keep each record's best leads) rests on the candidate graph
being **heavy-tailed** — a few junk records connected to everything, most records
connected to almost nothing. We measured the degree distribution of the 204,805-edge
8-block graph (95,915 nodes) directly:

| statistic | value | reading |
|---|---:|---|
| median degree | **1** | the typical record has a *single* candidate |
| mean degree | 4.27 | dragged up by the tail |
| 90th / 95th percentile | 4 / 5 | 90% of records have ≤ 4 candidates |
| 99th percentile | **81** | the top 1% explode |
| max degree | **340** | one record paired with 340 others |

| degree bucket | nodes | |
|---|---:|---|
| 1 | 51,204 | the bulk — singletons |
| 2–5 | 40,130 | normal |
| 6–10 | 2,121 | |
| 11–20 | 436 | |
| 21–50 | 742 | tail begins |
| 51–100 | 449 | junk hubs |
| **> 100** | **833** | **extreme hubs** |

**The tail is junk.** The ~2,000 records with degree > 20 are the placeholder-SSN
chains and common-name+DOB clusters the rules-eval flagged. Profiling the top-50
hubs: a degree-**340** node sits on **340** candidate edges of which only **6** are
silver-True — a **~1.8% real-match rate**. The whole top tail looks like this
(1.2–1.8% true): these records pull in hundreds of spurious comparisons each.

**CNP is a precise scalpel — it tames the junk without cutting muscle.** Tracking
the same hubs through the CNP top-10 prune:

| hub original degree | edges kept | % kept | silver-True before → after |
|---:|---:|---:|---|
| 340 | 10 | 3% | 6 → **6** (100% retained) |
| 339 | 10 | 3% | 5 → **5** (100% retained) |
| 339 | **325** | 96% | 4 → **4** (100% retained) |
| 338 | 10 | 3% | 6 → **6** (100% retained) |

The degree-340 junk hub is cut to **3%** of its edges — and **keeps every one of its
real matches**. Meanwhile the one node that *kept* 96% of its edges (339 → 325) is a
**legitimately** high-degree record: it survives because its neighbours each rank it
in *their* top-10 (CNP is reciprocal-OR), which is exactly the behaviour you want —
real clusters are preserved, spurious fans are clipped. Globally this is the
**99.96% silver-True retention at −48% candidates** of §4.3, now explained at the
node level: pruning removes ~99k overwhelmingly-false hub edges and almost no real
ones. **This is the mechanistic confirmation that node-centric pruning is safe** —
it isn't "top-10 happened to work," it's that the cut falls entirely on a measurable
junk tail.

### 4.6 Community detection for recall — connected components & Louvain

2-hop (§4.4) is one connectivity signal; **community detection** is the other natural
one. Partition the candidate graph into clusters, then add every within-cluster pair
as a candidate. We tried both the loosest (connected components) and a modularity
clustering (Louvain):

| method | total candidates | new vs 8-block | residual recovered | silver PC | silver PQ |
|---|---:|---:|---:|---:|---:|
| 2-hop (§4.4, for reference) | ~250k | 44,702 | 41.5% | — | — |
| **connected components** | **9,834,731** | 9,629,926 | **43.4%** | 100% | **0.5%** |
| **Louvain** | 758,514 | 554,215 | **43.4%** | 99.4% | 6.7% |

The graph breaks into **33,473 connected components** (median size 2), but with one
**giant component of 4,345 nodes** — the placeholder-SSN/common-name hubs fuse a huge
blob together. Expanding all within-component pairs therefore detonates to **9.6 M
candidates** (the giant component alone is ~9.4 M pairs) for a silver precision of
**0.5%**. Louvain breaks the blob into tighter communities (758k candidates, 6.7%
precision) but still only reaches the same **43.4%** of the residual.

**Verdict: community detection confirms — and does not beat — the 2-hop finding.**
All three connectivity methods plateau at **~43% residual recovery** because they
share the same ceiling: connectivity can only reach a miss whose two records already
sit in a connected cluster. The ~57% of residual pairs that are **isolated** (the
typo broke every key, so neither record has *any* other candidate to bridge through)
are unreachable by *any* connectivity method — exactly the 0% on the isolated
synthetic pairs. Community detection pays 2–40× more candidates than 2-hop for the
same recall and far worse precision. **Graph connectivity is not the recall
mechanism; q-gram is.** Keep connectivity only as a supplementary signal.

---

## 5. Alternative blockers tested and ruled out

Two further blocking families were on the table. Both were implemented and measured;
both are **dominated** by block-then-q-gram and are documented here so the decision is
on the record, not assumed.

### 5.1 Sorted Neighbourhood Method (SNM)

SNM sorts all records on a key, slides a fixed window down the sorted list, and
compares only records inside the window. It is cheap (one sort + a window pass) and
classically strong on phonetic name variants. We tested four sort keys at windows
w = 10/20/50, plus multi-pass (union of keys) and an adaptive (density-varying)
window. *(SNM's candidate count depends only on N and window size, not the key:
w = 10/20/50 → 1.43 M / 3.0 M / 7.8 M candidates.)*

| sort key (real, w=20) | candidates | rules PC | silver PC | silver PQ | residual rec. |
|---|---:|---:|---:|---:|---:|
| Soundex(Last)+DOB | 3,015,566 | 96.4% | 91.0% | 1.5% | **3.3%** |
| Soundex(First)+ZIP | 3,015,566 | 58.2% | 58.1% | 1.0% | 67.9% |
| **DOB+Sex** | 3,015,566 | **99.9%** | **99.8%** | 1.7% | **100%** |
| Last3+DOB | 3,015,566 | 95.8% | 92.4% | 1.6% | 51.9% |
| multi-pass (3 keys, w=20) | 8,940,390 | 100% | 99.97% | 0.6% | 100% |

**SNM *can* recover the residual — but only by becoming a worse block-then-q-gram.**
The story is clean:

- **A name-based sort key fails on exactly the residual.** Soundex(Last)+DOB reaches
  only **3.3%** of the residual (and 0% on the synthetic residual) — because the
  residual *is* last-name typos, which change the Soundex code and so scatter the two
  records to distant positions in the sorted list. SNM-on-name has the same blind
  spot as the 8-block keys it would replace.
- **The only key that recovers the residual is DOB+Sex** — which reaches 100%, but
  *because sorting on DOB places same-DOB records adjacent, it is just a coarse,
  windowed re-derivation of the DOB block.* It gets the recall at **3.0 M candidates
  and 1.7% precision** — versus block-then-q-gram's **68,849 candidates at the same
  recall**. SNM-on-DOB is block-then-q-gram **without the name-similarity filter**:
  same coarse key, but it compares *every* adjacent record instead of only the
  name-similar ones, so it pays **~44× the candidates** for an identical residual
  recovery.
- **Adaptive windows didn't rescue it** (6.6% residual on the Soundex key — the
  adaptivity can't fix a key that sorts the typos apart in the first place), and
  **multi-pass** only reaches 100% because its DOB pass does all the work, at **8.9 M
  candidates**.

**Verdict: dominated.** Every SNM configuration that recovers the residual costs
40–130× the candidates and 10–40× the false pairs of block-then-q-gram, for the same
recall. SNM's one structural advantage — robustness to a field being mis-keyed — is
not worth that blow-up here, and is better served by the strong-ID blocks we are
keeping anyway.

### 5.2 Super-string LSH

A "super-string" concatenates several PII fields into one string
(`lastname|dob|sex`) and runs a single similarity index over it, hoping for
field-swap robustness and "one index to rule them all." Two ways to index it:

1. **All-pairs cosine on the super-string** — this is the *same O(N²) cost class as
   the exact q-gram pass we already retired* (§7); running it at scale would only
   re-prove a scalability verdict we have. Not run.
2. **MinHash-LSH on the super-string** — the only *scalable* super-string form
   (LSH is ~linear). We implemented it (datasketch, char-3-gram MinHash, 64 perms)
   across three weighting schemes — `equal` (lastname|dob|sex), `weighted` (last and
   DOB duplicated to fight dilution), and `minimal` (lastname|birthyear|sex).

**Result: dropped — it does not scale on this data.** Identity super-strings are
**low-entropy**: huge numbers of records share the same `lastname|birthyear|sex`
prefix, so at any usable LSH threshold they collapse into **pathologically large LSH
buckets**, and the query phase degenerates toward all-pairs *inside* those buckets
(candidate generation ran into the millions and memory climbed without converging).
The classic super-string failure modes both bite here: **dilution** (a long address
field would drown a name typo — so we excluded address, which leaves the string too
short and generic) and the **apartment-complex false positive** (everyone in a ZIP +
birth-year bucket collides). The weighting/prefix tricks that are supposed to fix
dilution only sharpen the bucket-collision problem.

**Verdict: the scalable super-string variant is not viable on identity data, and the
quality variant is non-shippable by construction.** Block-then-q-gram already gives
the field-robustness benefit that matters (it uses an exact, cheap DOB key plus a
*focused* name comparison) without the bucket blow-up. Code retained behind
`--with-superstring` for the record.

---

## 6. The stacked pipeline — q-gram → meta-blocking (the recommended design)

q-gram (recall) and meta-blocking (efficiency) were measured **separately** in every
prior round. The obvious question for deployment is whether they **compose**: run
block-then-q-gram for robust recall, union it with the 8-block output, then apply CNP
to prune the combined volume — *does the prune throw away the very residual pairs
q-gram just recovered?* This round measured it end-to-end for the first time.

**Pipeline:** `8-block ∪ block-then-q-gram(DOB ≥ 0.30)` → ARCS edge-weighting →
`CNP top-k`. (q-gram-only edges, which appear in no 8-block, are given a single-signal
default weight so CNP can rank them.)

| stage | candidates | silver PC | silver PQ | residual rec. | Δ vs 8-block |
|---|---:|---:|---:|---:|---:|
| 8-block baseline | 204,805 | 100%\* | 24.9% | 0% | — |
| **∪ block-then-q-gram ≥ 0.30** | 208,094 | **100%** | — | **99.5%** | +1.6% |
| → CNP top-5 | 98,197 | 99.82% | 52% | **99.5%** | **−52.1%** |
| → **CNP top-10** | **109,098** | **99.96%** | 47% | **99.5%** | **−46.7%** |
| → CNP top-15 | 117,481 | 99.96% | 44% | 99.5% | −42.6% |

\*union silver PC is 100% because the union is a *superset* of the 8-block output;
the meaningful recall number is what survives the prune.

**The stack works — the two wins compose without trading off:**

1. **The union adds q-gram's recall for almost nothing.** block-then-q-gram ≥ 0.30
   contributes only **3,289 new pairs** on top of the 8-block set (most of its 68,849
   candidates already overlap the DOB+name blocks), lifting the union to 208,094 and
   carrying the **99.5% residual recovery** with it.
2. **CNP then prunes the *combined* set −46.7%** (to 109,098) — essentially the same
   −48% cut it achieved on the 8-block set alone (§4.3) — while **retaining 99.96% of
   real silver matches.**
3. **Crucially, the residual recovery *survives the prune*: 99.5% before and after
   CNP at every k.** The q-gram residual pairs are last-name-typo pairs between two
   *low-degree* records, so each is comfortably in its endpoints' top-k and CNP never
   drops it. The fear — that pruning would undo q-gram's recall gain — does not
   materialise.

**Net:** the stacked design simultaneously **(a) recovers 99.5% of the structural
residual** the production scheme misses, **(b) cuts the candidate volume −46.7%**, and
**(c) keeps 99.96% of adjudicated real matches**, at ~10 s of added q-gram compute.
top-10 is the recommended operating point (the −46.7% / 99.96% / 99.5% row); top-5
trades half a point of silver recall for a further ~5% candidate cut.

---

## 7. Scalability

The cost that matters is **candidate generation** (finding the pairs); everything
downstream scales with the candidate count.

| Method | Build cost | Pair-generation cost | Grows like | Safe to… |
|---|---|---|---|---|
| **8-block (baseline)** | hash keys, O(N) | within-bucket pairs, ~O(N) with caps | **linear** | very large |
| **Graph meta-blocking (CNP)** | — (post-pass) | one pass over edges + top-k/node, O(E) | **linear** (E ≈ c·N) | very large |
| **q-gram — block-then-q-gram** ⭐ | TF-IDF, O(N) | cosine within DOB groups, Σ\|group\|² ≈ **1.28 M** | **~linear** | very large |
| **Stacked (q-gram → CNP)** ⭐ | TF-IDF + edge weights, O(N+E) | union + top-k/node | **~linear** | very large |
| **q-gram — exact all-pairs** | TF-IDF, O(N) | sparse cosine product, ~O(N·neighbours) | **super-linear → O(N²)** | ~hundreds of k |
| **Super-string LSH** | MinHash sigs, O(N) | LSH bucket query — **degenerate on low-entropy keys** | **bucket-bound, blows up** | not viable here |
| **q-gram — ANN/LSH (name)** | build index, ~O(N log N) | approx. near-neighbour query, ~O(N) | **~linear** | millions |

**8-block, meta-blocking, and the stacked pipeline are linear and safe at any
realistic size.** Blocking hashes records into buckets (governance cap prevents
blow-up); meta-blocking is one pass over the candidate **edges** (E ≈ 205k today,
growing ~linearly with N) and runs sub-second; the stack adds only the linear
block-then-q-gram pass — the lowest-risk path to adopt.

**q-gram needs care, but the scalable form is built and measured.** Building the
TF-IDF matrix is linear and cheap. The danger is the **exact** all-pairs pass: each
record is compared against everything sharing a character chunk, and that fan-out
grows toward O(N²) on dense data (~minutes at 158k; **1 M+ records would be
impractical**). The fix is to never run the exact pass in production:

1. **Block-then-q-gram — the recommended default (§3.4).** Cosine within an exact-DOB
   block: Σ|group|² ≈ 1.28 M (≈linear), **~10 s**, reusing the DOB key the pipeline
   already has.
2. **Approximate nearest neighbours (LSH / ANN) *on the name vectors*** — MinHash-LSH
   or HNSW/IVF — retrieve near-neighbours in ~O(N) with a small recall loss. The
   standard answer when even DOB-blocking groups get large, or when DOB-typo tolerance
   must be global. *(Note: this is ANN on the focused **name** vector, not on the
   low-entropy super-string — §5.2 is why the distinction matters.)*
3. **Tighten threshold / prune common n-grams** to shrink each record's neighbourhood
   at some recall cost.

**Online / single-record inference.** A new record's name-q-gram retrieval is one
vector against the index — O(N) exact, ~O(log N) with an ANN index — and the index
updates incrementally, so q-gram is feasible for the `run_inference_blocking` path.
Meta-blocking on a new record only touches its own few edges (O(degree)) — trivial.

> **Remaining empirical item (de-scoped for now).** The qualitative analysis above is
> backed by the measured block-then-q-gram (~10 s, ~linear). A full scaling *curve*
> (timing at 40k/80k/160k subsamples to fit the growth and project the ANN crossover)
> was judged unnecessary for the deployment decision — the backend is already
> sub-quadratic by construction and measured at production size.

---

## 8. Synthesis & recommendations

| Approach | Role | Headline | Recommendation |
|---|---|---|---|
| **Stacked: 8-block ∪ block-then-q-gram → CNP** | **the design** | **−46.7% candidates, 99.96% silver retained, 99.5% residual recovered — all at once** | **Adopt** — composes the two wins with no trade-off |
| **Block-then-q-gram** (scalable q-gram) | recall / robustness | reproduces the exact pass at 69k candidates, ~10 s, ~linear; misses only a double-typo pair | Adopt as the recall add-on, on the existing DOB key; keep strong-ID blocks |
| **Graph meta-blocking (CNP)** | efficiency | −48% candidates, 99.96% silver-True retained, 2× precision — and the junk-hub tail explains *why* | Adopt — free, low-risk, mechanistically understood |
| **Graph connectivity (2-hop / community)** | partial recall | all three methods plateau at ~43% residual; 0% on isolated pairs | Supplementary only — q-gram dominates |
| **Sorted Neighbourhood (SNM)** | alternative blocker | recovers residual only via a DOB sort, at 40–130× candidates | **Rule out** — dominated by block-then-q-gram |
| **Super-string LSH** | alternative blocker | low-entropy keys → degenerate LSH buckets; cosine form is O(N²) | **Rule out** — not viable on identity data |

1. **Adopt the stacked pipeline (§6).** `8-block ∪ block-then-q-gram(DOB ≥ 0.30)` →
   `CNP/ARCS top-10` is the recommended design: it recovers 99.5% of the structural
   residual, cuts candidates −46.7%, and retains 99.96% of adjudicated real matches —
   the recall win and the efficiency win, measured together and shown not to trade
   off. **Keep the strong-ID blocks** (SSN/phone/email) in the union — q-gram can't
   see shared-identifier matches with divergent names (§3.3).
2. **The mechanism is understood, not just observed.** CNP is safe because the
   candidate graph is heavy-tailed and the tail is junk (§4.5): pruning falls almost
   entirely on ~2,000 hub records whose edges are >98% false, while real clusters
   (reciprocally high-degree) are preserved. This is the evidence to set the per-node
   `k` on, not a lucky default.
3. **Connectivity is not the recall mechanism.** 2-hop, connected-components, and
   Louvain all cap at ~43% of the residual and 0% on isolated pairs (§4.4, §4.6) —
   keep them, at most, as supplementary signals behind q-gram.
4. **The alternatives are ruled out on evidence (§5).** SNM is dominated (40–130×
   candidates for the same recall); super-string LSH is non-viable on low-entropy
   identity data. Neither needs further investigation.
5. **Close the last measurement gap with gold labels.** Silver + synthetic moved
   q-gram's recall from a rules-capped *floor* to a real, label-true number; the
   remaining unknown is the value of the 84k q-gram-only pairs (§3.3-4), which needs a
   hand-adjudicated **gold** sample. This is the one open item before — and the
   threshold-locking input for — implementation.

---

## 9. Reproducibility

All harnesses are committed, read-only on the data, and write a JSON artifact:

| Script | Produces | Covers |
|---|---|---|
| `scripts/research/research_blocking_labels.py` | `data/runs/research_blocking_labels.json` | q-gram (exact) vs rules + silver + synthetic (§3.2–3.3) |
| `scripts/research/research_blocking_scalable.py` | `data/runs/research_blocking_scalable.json` | block-then-q-gram (§3.4), meta-blocking-vs-silver (§4.3), 2-hop (§4.4) |
| `scripts/research/research_blocking_round4.py` | `data/runs/research_blocking_round4.json` | hub/degree (§4.5), community detection (§4.6), SNM (§5.1), super-string LSH (§5.2, behind `--with-superstring`), stacking (§6), residual deep-dive (§3.5) |
| `scripts/eval_against_labels.py` | (reused) | synthetic record reconstruction + label scoring |

- **Caches:** rules-as-truth **R** at `data/runs/_cache_rules_R_real.parquet`; the
  Round-1/2 8-block baseline (with `source_blocks` for ARCS) was cached at
  `/tmp/empi_research/B_8block.parquet`. **Note:** `research_blocking_round4.py`
  recomputes the 8-block set live via `run_batch_blocking` (it does not depend on the
  `/tmp` cache), so it is self-contained.
- **q-gram implementations:** char-n-gram(2–4) TF-IDF; the exact pass extracts pairs
  with a `sparse_dot_topn` blocked top-n kernel (the dense product OOMs); the scalable
  pass does dense cosine within DOB/birth-year groups. ARCS = Σ 1/(block comparisons);
  CNP = keep each node's top-k edges (reciprocal-OR).
- **Round-4 specifics:** degree/hub stats and community detection use `networkx`
  (connected components + `louvain_communities`, seed 42); SNM uses `jellyfish`
  Soundex on the sort key; super-string LSH uses `datasketch` (MinHash, 64 perms,
  char-3-gram). Thresholds for the stacking/residual sections are the **name-only**
  block-then-q-gram cutoffs (0.30/0.35), *not* the global name+DOB cutoffs (§3.4 note).
- **Research dependencies** — `scikit-learn`, `scipy`, `sparse_dot_topn`, `networkx`,
  `jellyfish`, `datasketch` — are installed in the venv (`uv pip install --python
  .venv …`) but **intentionally not in the project manifest** (analysis-only).
  Reinstall if the venv is rebuilt.
- **Compute (single CPU):** meta-blocking + 2-hop sub-second to seconds;
  **block-then-q-gram ~9 s**; hub analysis + community detection a few minutes (the
  giant-component pair expansion dominates); the exact q-gram pass ~minutes per
  threshold; the full Round-4 run (hub + community + SNM + stacking + deep-dive, real +
  synthetic) ~6–8 min. Super-string LSH does **not** converge at usable thresholds on
  this data (§5.2) and is off by default.
- **Caveats:** rules-as-truth recall is a lower bound (§2.1); silver only labels the
  8-block output; synthetic is generator-shaped. A gold sample is the remaining
  yardstick.
