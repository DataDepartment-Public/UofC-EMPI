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
| `scripts/eval_end_to_end.py` | CLI for gold / silver against a completed run. |
| `scripts/eval_synthetic_pipeline.py` | Runs the pipeline over the synthetic set and scores it. |

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

`tests/unit/evaluation/test_holdout.py` pins the fold sizes and class balances
the notebooks printed. If a notebook's seed, ratios, stratification target, or
row ordering changes, those tests fail rather than the folds silently drifting.

---

## 3. Running it

### Gold (real records, VM only)

```bash
# leakage-safe headline
python scripts/eval_end_to_end.py --run-id <run_id>

# same run over every gold pair, for comparison only
python scripts/eval_end_to_end.py --run-id <run_id> --holdout none

# just the gate's fold, or just the matcher's
python scripts/eval_end_to_end.py --run-id <run_id> --holdout gate
```

Gold's extra `ambiguous_pair` column lets each stage be scored against **its
own** target rather than the match label — keeping an ambiguous non-match is
correct behavior for the gate and would otherwise count as a false positive:

- gate → `plausible` = `final_gold_label | ambiguous_pair`
- ML matcher → `confident_match` = `final_gold_label & ~ambiguous_pair`
- rules, blocking, clustering → `final_gold_label`

### Silver

```bash
python scripts/eval_end_to_end.py --run-id <run_id> \
    --labels data/silver_labels/silver_labels_v1_2026_06_21.csv \
    --label-col silver_label --source silver --holdout none
```

Silver has no `ambiguous_pair`, so every stage is scored against
`silver_label`, and `--holdout` is ignored with a warning (the folds are not
defined over it). Silver is also the FS matcher's training data — same leakage
caveat, without a reconstructible fold.

### Synthetic (leakage-free, exact cluster truth)

```bash
python scripts/eval_synthetic_pipeline.py
python scripts/eval_synthetic_pipeline.py --reuse-run <run_id>   # score only
```

`data/synthetic labels/synthetic_test_v3.csv` is a *pair* file carrying
already-cleaned `*_l`/`*_r` attributes, so it cannot be fed to `--input`. The
script reconstructs a record frame from both sides, writes it as a cleaned
Parquet, and runs the **real** pipeline over it through
`run_pipeline(cleaned_input=...)` — Stages 2–5 exactly as production runs them,
not a reimplementation.

What that measures and what it doesn't:

- **Stage 1 (cleaning) is skipped by construction** — the inputs are already
  cleaned, and the planted corruptions live at the cleaned level. Re-running the
  cleaning rules over them would be lossy.
- **Blocking recall here is real**, unlike gold/silver: the positives were built
  independently of blocking, whereas the gold/silver label universe *is* a
  blocking output (which is why their blocking recall is `N/A`).
- The set has 10,000 pairs over 20,000 records / 18,000 entities, max true
  cluster size 2 — so any predicted cluster larger than 2 is a visible false
  merge.
- It has no ambiguous class, so the gate and matcher collapse to match /
  not-match.

### `--cleaned` on the pipeline

`run_pipeline(cleaned_input=...)` / `python -m src.pipeline --cleaned <parquet>`
skips Stage 1 and starts from an already-cleaned frame. It exists for evaluation
harnesses feeding pre-cleaned records; it is not a shortcut for production runs.
The frame must satisfy `CleanedRecords`, and the manifest records it as the run's
input so lineage stays intact.

---

## 4. Reading the report

Both scripts print a text report and write it plus a JSON sibling to
`data/runs/eval_end_to_end_<run_id>_<source>_<holdout>.{json,txt}`.

**HEADLINE** — labeled pairs vs. final clustering. `coverage` separates *pairs
the run never clustered* (a record that failed the Stage-1 validity filter has
no prediction at all) from *scored non-merges*; conflating them understates
recall. `uncovered_as_negative` is the reviewer's-eye view, where an unclustered
record is simply never surfaced.

**PER-STAGE** — the same labeled pairs, each stage's own decision, so a bad
headline can be attributed instead of guessed at.

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

## 5. HIPAA

The reports contain aggregate counts and metrics only — no PATIDs, no field
values, no example rows — so a JSON report is safe to copy off the VM. The label
files themselves are PHI and are gitignored.
