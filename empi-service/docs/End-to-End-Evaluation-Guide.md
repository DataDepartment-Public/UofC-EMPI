# End-to-End Evaluation Guide

How to measure the quality of what the pipeline actually ships — the Stage-5
cluster assignments — using the project's pairwise label sets.

Everything else in `src/evaluation/` scores **one stage**: `blocking_eval.py`
measures blocking against the deterministic rules, `rule_eval.py` measures the
rules against heuristic adjudication. Neither reads `data/clusters/`. This guide
covers the modules that close that gap.

| Piece | What it is |
|---|---|
| `src/evaluation/cluster_eval.py` | Metric core — pair labels ↔ clusters. No I/O. |
| `src/evaluation/pipeline_eval.py` | Scores one run's `RunManifest` against a label set. |
| `src/evaluation/holdout.py` | Reproduces the training notebooks' test folds (leakage control). |
| `src/evaluation/synthetic.py` | Rebuilds a record population from the synthetic *pair* file. |
| `src/evaluation/report_io.py` | Reads stored reports back into tidy frames. |
| **`scripts/evaluate_all.py`** | **The entry point — runs everything and scores it.** |
| `scripts/eval_end_to_end.py` | Granular: one existing run, one label file, one holdout. |
| `notebooks/evaluation/end_to_end_eval.ipynb` | Tables + trend charts over stored reports. |

---

## 1. The shape problem

The labels are **pairs**; the output is **clusters**. There are exactly two
defensible ways to reconcile them, and the reports carry both because they have
different blind spots.

### Restricted pairwise — the headline

For every labeled pair, ask whether the run put the two records in the same
cluster, and score that against the label. No assumption beyond the labels
themselves, and it automatically credits or blames **transitive** merges, which
is the whole reason clustering differs from the pair classifiers upstream.

*Blind spot:* it only sees pairs someone labeled. A false merge between two
records that were never a candidate pair is invisible.

### Cluster-level — B-cubed + exact recovery

Lift the pair labels to truth *clusters* (connected components of the positive
pairs) and compare partition to partition. This **does** see the over-merging
that restricted pairwise misses.

*Blind spot — read this before quoting the numbers:* the closure asserts pairs
nobody labeled. If the labels say A~B and B~C, the closure asserts A~C **even
when the labeler explicitly marked A~C a non-match**. `ClosureDiagnostics`
counts exactly those (`n_contradicted` / `contradiction_rate`) and the report
prints them. When that count is not near zero, the cluster-level numbers are
directional — trust the headline.

The synthetic set is the exception: it carries real `entity_id_*` ground truth,
so `evaluate_run(..., truth_partition=...)` skips the closure entirely and its
cluster-level metrics are **exact**.

Both restrict to a universe of PATIDs (the records appearing in the labels) and
*induce* the predicted partition onto it — truth is undefined for records nobody
labeled.

### Three-class triage — the fair recall view

Both views above are **binary**: merged or not. That scores an *ambiguous* pair
routed to a reviewer as a miss, when routing it there is precisely the correct
behavior — the labeler themselves could not resolve it from the record. Binary
recall therefore understates the system by the whole ambiguous population.

The `triage` block scores the decision the pipeline actually makes. Both sides
speak `CLASSIFICATION_TIERS`:

| gold class | expected route | the pipeline's route is derived from |
|---|---|---|
| non-match (neither flag) | `no_match` | rules reject, gate drop, ML `no_match`, or never blocked |
| ambiguous (`ambiguous_pair`) | `human_review` | survived every stage, not merged |
| match (`final_gold_label`) | `auto_merge` | in the same cluster |

The result is a square confusion matrix whose diagonal is correct routing, plus
a standard per-class precision/recall/F1 report.

Three things to know before quoting it:

- **`ambiguous` outranks `match`** when a pair carries both flags. The class is
  a claim about the *evidence*, and evidence too weak for a confident call
  belongs with a human whatever the eventual adjudication was; scoring such a
  pair as a required auto-merge penalizes correct caution. The report records
  `ambiguous_precedence` so the number is never read without it. Pass
  `ambiguous_precedence=False` to `triage_evaluation` for the other convention.
- **It measures the *shipped* decision.** While `ml_feeds_clustering` is off the
  ML matcher's `auto_merge` tier changes no output, so those pairs are scored as
  `human_review` — which is where they genuinely end up. `stage_pairwise` is
  where a stage's own decision is scored in isolation.
- **Only gold defines the ambiguous class.** Synthetic and silver reports
  degrade to two classes and leave `human_review` unpopulated — honest rather
  than degenerate, since those sources make no undecidability claim.

Read `auto_merge` recall as "share of confident matches resolved without a
human" and `human_review` recall as "are undecidable pairs actually reaching a
reviewer". A low value on the latter is the more serious failure: a pair
misrouted to `no_match` is unrecoverable, while one left in review is merely
slow.

---

## 2. Leakage — mandatory reading for gold

**The Stage-4.25 gate and the served Stage-4.5 matcher were both trained on
`data/gold_labels/final_gold_labels_v1_2026_07_05.csv`.** Both notebooks take a
plain random stratified 60/20/20 split, so ~80% of the gold pairs are training
data. Scoring the pipeline on all 204,805 gold pairs reports a number that is
substantially memorized.

Neither notebook persists its split, but the split depends on nothing except the
gold label columns and the CSV's row order (`np.arange(n)`, `RANDOM_STATE=42`,
stratified on the target), so `holdout.py` reproduces it exactly — no features,
no model loading.

The two folds are **not the same pairs**:

| Fold | Population | Stratified on | Size |
|---|---|---|---|
| `gate` | all 204,805 | `final_gold_label \| ambiguous_pair` | 40,961 |
| `matcher` | the 62,610 plausible pairs | `ambiguous_pair` | 12,522 |
| **`strict`** | gate-held-out ∧ (matcher-held-out ∨ never in the matcher's population) | — | the safe set |

`--holdout strict` is the default and the only leakage-safe choice. `--holdout
none` runs, but the report's `leakage` block will say so in writing, and its
ML-stage numbers are memorization.

Stages **not** fit on gold — blocking and the deterministic rules — are clean at
any holdout setting.

**Which means the HEADLINE is leakage-free even at `--holdout none`, as long as
`fs_feeds_clustering` and `ml_feeds_clustering` are both off.** With both off,
clustering unions only the deterministic auto-merge edges, so nothing that
touched gold during training influences the final clusters. Only the `gate_pass`
and `ml_auto_merge` rows need the holdout.

That makes the recommended reading:

| Question | Run |
|---|---|
| How good is the clustering? | `--holdout none` — 8× more labeled positives, so a tighter estimate |
| How good are the gate / ML matcher? | `--holdout strict` |

`strict` is still the safe default because it is correct under *every*
configuration, including someone turning a feed toggle on. But `strict` keeps
only the ~20% of plausible pairs the matcher held out, so it is heavily depleted
of positives (~2.5k of ~62.5k) and enriched in easy confident non-matches — fine
for recall, mildly optimistic for precision. Cross-check against `--holdout
none`.

`tests/unit/evaluation/test_holdout.py` pins the fold sizes and class balances
the notebooks printed. If a notebook's seed, ratios, stratification target, or
row ordering changes, those tests fail rather than the folds silently drifting.

---

## 3. Running it

### The usual case — one command

```bash
python scripts/evaluate_all.py
```

One invocation is one **evaluation session**, and it does the whole thing:

1. run the pipeline over the real input → score vs gold at `none` **and** `strict`
2. run the pipeline over the synthetic set → score vs entity ground truth

Every report lands in `data/evaluations/` under one `session_id`. That session —
not a `run_id` — is the unit of comparison over time: the real-data run and the
synthetic run are necessarily different pipeline runs, but they are one
measurement of one state of the system, so they share a timeline point.

```bash
# name it when it is a baseline worth remembering
python scripts/evaluate_all.py --session-id gate_v2_baseline

# reuse an existing real-data run (the pipeline run is the slow part)
python scripts/evaluate_all.py --reuse-real-run 20260728T191705Z

# one half only
python scripts/evaluate_all.py --skip-synthetic
python scripts/evaluate_all.py --skip-real
```

The two halves run independently: a missing gold file or a failed real run does
not cost you the synthetic result. Failures are reported at the end and set a
non-zero exit code.

### The granular tool

`scripts/eval_end_to_end.py` scores **one existing run** against **one label
file** at **one holdout** — for silver, a non-standard label path, or a single
`--holdout gate` comparison:

```bash
python scripts/eval_end_to_end.py --run-id <run_id> \
    --labels data/silver_labels/silver_labels_v1_2026_06_21.csv \
    --label-col silver_label --source silver --holdout none
```

Silver has no `ambiguous_pair`, so every stage is scored against `silver_label`,
and `--holdout` is ignored with a warning (the folds are not defined over it).
Silver is also the FS matcher's training data — same leakage caveat, without a
reconstructible fold.

### What gold's extra column buys

`ambiguous_pair` lets each stage be scored against **its own** target rather
than the match label — keeping an ambiguous non-match is correct behavior for
the gate and would otherwise count as a false positive:

- gate → `plausible` = `final_gold_label | ambiguous_pair`
- ML matcher → `confident_match` = `final_gold_label & ~ambiguous_pair`
- rules, blocking, clustering → `final_gold_label`

It is also what makes the three-class **triage** block possible: without it the
report cannot tell "we should have merged this" from "we should have sent this
to a human", and recall is measured against a target the pipeline was never
asked to hit. Only gold has the column today.

### What the synthetic half measures

`data/synthetic_data/synthetic_test_v3.csv` is a *pair* file carrying
already-cleaned `*_l`/`*_r` attributes, so it cannot be fed to `--input`.
`src/evaluation/synthetic.py` reconstructs a record frame from both sides and
runs the **real** pipeline over it through `run_pipeline(cleaned_input=...)` —
Stages 2–5 exactly as production runs them, not a reimplementation.

- **Stage 1 (cleaning) is skipped by construction** — the inputs are already
  cleaned, and the planted corruptions live at the cleaned level.
- **Blocking recall here is real**, unlike gold/silver: the positives were built
  independently of blocking, whereas the gold/silver label universe *is* a
  blocking output.
- 10,000 pairs over 20,000 records / 18,000 entities, max true cluster size 2 —
  so any predicted cluster larger than 2 is a visible false merge.
- No ambiguous class, so the gate and matcher collapse to match / not-match.

### `--cleaned` on the pipeline

`run_pipeline(cleaned_input=...)` / `python -m src.pipeline --cleaned <parquet>`
skips Stage 1 and starts from an already-cleaned frame. It exists for evaluation
harnesses feeding pre-cleaned records; it is not a shortcut for production runs.
The frame must satisfy `CleanedRecords`, and the manifest records it as the run's
input so lineage stays intact.

---

## 4. Reading the report

Every report is printed and also written to
`data/evaluations/eval_<session_id>__<source>__<holdout>.{json,txt}`.

**HEADLINE** — labeled pairs vs. final clustering. `coverage` separates *pairs
the run never clustered* (a record that failed the Stage-1 validity filter has
no prediction at all) from *scored non-merges*; conflating them understates
recall. `uncovered_as_negative` is the reviewer's-eye view, where an unclustered
record is simply never surfaced.

**TRIAGE** — the 3-class routing matrix and its classification report (§1). For
gold this is the recall number to quote; the binary headline above it is the
precision number to quote. They answer different questions and neither replaces
the other.

**PER-STAGE** — each stage's own decision, so a bad headline can be attributed
instead of guessed at. Each stage is scored **only on the pairs it actually
saw**: the gate sees just the rules' `non_matches` pool, and the ML matcher just
the gate's survivors, so a true pair the rules already auto-merged never reaches
either. Scoring those as stage misses would understate recall by exactly the
number of pairs the previous stage resolved. When a stage's population is a
subset, the report prints `scored on N/M labeled pairs`.

For gold, the ML matcher gets two rows. `ml_auto_merge` scores it against
`confident_match`, its own training target. `ml_auto_merge (vs match label)`
scores it against `final_gold_label` — that is the row that answers "should
`ml_feeds_clustering` be turned on?", because merging a pair that is a true
match but was labeled *ambiguous* is a correct merge, and the first row counts
it as a false positive.

**FUNNEL** — labeled pairs surviving each boundary. Note the last row,
`clustered`, is **not** a subset of the row above it: transitivity can merge a
pair that never even blocked.

**LOSS ATTRIBUTION** — the actionable output. For every true pair the run failed
to cluster, the *first* stage that dropped it (a pair the rules rejected is never
also blamed on the gate). This is what turns "recall is 0.83" into "6,100 of the
misses were gate drops". `left in review (no auto_merge edge)` means every stage
kept the pair but none emitted a merge edge — the expected bucket while
`ml_feeds_clustering` is off, and the number to watch when deciding whether to
turn it on.

**TRANSITIVITY** — merges that exist only because clustering takes connected
components. No pair classifier ever scored them, so their false positives are
invisible to every upstream metric. If `transitive_only.precision` is poor, the
edge set is chaining unrelated records together.

**CLUSTER-LEVEL** — B-cubed and exact-cluster recovery, with the closure
diagnostics from §1. B-cubed is used rather than pairwise counting because it is
not dominated by singletons and degrades gracefully: one over-merge costs
precision proportional to the blob it created. `non_singleton` recovery is
reported separately because in a mostly-singleton population the overall rate is
~1.0 and tells you nothing.

---

## 5. Comparing results over time — the notebook

`notebooks/evaluation/end_to_end_eval.ipynb` renders the stored reports as
tables and trend charts. **The JSON files in `data/evaluations/` are the
history** — the notebook only reads them (`src/evaluation/report_io.py`), so
comparing past sessions needs no re-running, which matters because a past run's
artifacts may no longer exist.

Sections: what results exist → one report in detail (**triage**, per-stage,
funnel, loss attribution, transitivity, cluster-level) → trend across sessions →
gold vs. synthetic side by side. §1 optionally shells out to `evaluate_all.py`
to add a new session; everything else works offline.

The triage section (§3.1) is the one to open first on a gold report: it renders
`triage_report()` (the classification report), `triage_matrix()` (raw counts, or
`normalize=True` for row shares), an annotated heatmap of the matrix, and
`triage_history()` across sessions. The heatmap shades by **row share** rather
than raw count — non-matches outnumber matches several to one, so a
count-shaded grid is one dark corner and two invisible rows — while printing
each cell's absolute count, because a rate without its denominator is not
actionable.

Re-evaluating the same (session, source, holdout) triple **overwrites** rather
than accumulating — that is the same measurement, and keeping both would put a
duplicate point on every trend. So pass `--session-id` something memorable when
a session is a comparison point you want to keep.

Each `report_io` frame is display-oriented, not a contract: a report from an
older schema yields `NaN` for missing fields rather than raising, an unreadable
file is skipped with a warning instead of sinking the comparison, and a report
predating sessions falls back to its `run_id` as the timeline key.

---

## 6. HIPAA

The reports contain aggregate counts and metrics only — no PATIDs, no field
values, no example rows — so a JSON report is safe to copy off the VM. The label
files themselves are PHI and are gitignored.
