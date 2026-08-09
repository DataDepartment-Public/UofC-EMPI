# data/no_match/ — Stage 3c output: terminal audit-only rejects `[IMPLEMENTED]`

Full contract: `docs/Data-Contract.md` → "Stage 3 — ..." (§3c).

- **Producer:** `src/models/deterministic_rules`, invoked from `src/pipeline.py`
- **Contract:** `contracts.Rejects` (subclasses `CandidatePairs`; `strict=True`)
- **Consumers:** **none** — dropped from the pipeline; this artifact exists
  purely for audit/compliance. Nothing downstream reads it back.
- **File:** `rejects_<run_id>.parquet` — the file stem stays `rejects_`
  (matches `contracts.Rejects`, the pandera schema class); only the
  containing directory was renamed, to `no_match` per `contracts.TIER_NO_MATCH`
  — the `decision` column below already carries that exact value.
- **Naming note:** `data/auto_merge/` is named the same way, after
  `contracts.TIER_AUTO_MERGE`. `data/non_matches/` is deliberately **not**
  tier-named — see its own SCHEMA.md.
- **Grain:** unconfirmed pairs with **≥ 3** of {full SSN, first, last, DOB}
  strictly disagreeing (calibrated on a real run: 2 conflicts still carry
  ~10% true matches, 3 carry ~0%).

### Output schema

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` / `PATID_B` / `source_blocks` / `n_blocks` | (as `CandidatePairs`) | no | Passthrough from blocking. |
| `n_contradictions` | int64 | no | Count of strong-identifier disagreements. |
| `decision` | string | no | Always `"no_match"` in this artifact. |
| `reject_rule` | string | no | The reject rule that fired (`"STRONG_ID_CONFLICT"`) — always populated here. |

**Invariant:** `matches ⊎ non_matches ⊎ rejects == candidate_pairs`
(disjoint union), keyed on `(PATID_A, PATID_B)`.
