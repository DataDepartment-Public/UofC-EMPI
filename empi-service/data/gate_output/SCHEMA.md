# data/gate_output/ — Stage 4.25 output: non-match gate audit frame

Full contract: `docs/Data-Contract.md` → "Stage 4.25 — non-match gate".
Model + operating point: `docs/Nonmatch-Gate-Guide.md`.

- **Producer:** `src/models/nonmatch_gate/` (`NonMatchGate.apply`), invoked from
  `src/pipeline.py` when an active gate model is resolvable and the
  `non_matches` pool is non-empty. Skipped otherwise — the pipeline then falls
  back to the legacy FS gate, or runs ungated with a warning.
- **Role:** the pipeline's **confident-non-match filter** — the pairs it tiers
  `no_match` are discarded and never reach the Stage-4.5 ML matcher. It makes
  no merge decision and never feeds clustering.
- **Trained model artifacts** live separately in `models/nonmatch_gate/` — not
  under `data/`.

## gate_results (full audit frame, ClassificationResults-shaped)

- **Contract:** `contracts.ClassificationResults` (the shared 5-column shape
  every classifier stage emits: `PATID_A`, `PATID_B`, `model_name`, `score`,
  `predicted_tier`)
- **File:** `gate_results_<run_id>.parquet`
- **Grain:** every scored `non_matches` pair — **including the dropped ones**,
  which exist nowhere else (`data/no_match/` holds the deterministic rules'
  rejects, not the gate's)
- **`score`:** `P(plausible)` = `P(match ∪ ambiguous)`; high = keep
- **Tiers:** only `human_review` (passed the gate) and `no_match` (dropped).
  Never `auto_merge`.
- **Status:** audit frame; the survivors are passed in-memory to Stage 4.5
