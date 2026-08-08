# data/non_matches/ — Stage 3b output: the FS matcher's input pool `[IMPLEMENTED]`

Full contract: `docs/Data-Contract.md` → "Stage 3 — ..." (§3b).

- **Producer:** `src/models/deterministic_rules::classify_non_matches`,
  invoked from `src/pipeline.py`
- **Contract:** `contracts.NonMatches` (identical schema to `CandidatePairs`)
- **Consumers:** Stage 4 (FS matcher scores this pool), `src/api/ingest/publish.py`
- **File:** `non_matches_<run_id>.parquet`
- **Naming note:** unlike `data/auto_merge/` and `data/no_match/` (named
  after `contracts.TIER_AUTO_MERGE`/`TIER_NO_MATCH`), this folder is
  deliberately **not** renamed to tier vocabulary — it isn't a decided tier
  at all, it's the pre-scoring pool below. Stage 4/4.5 sort it into all
  three tiers (`auto_merge`/`human_review`/`no_match`); the tier only exists
  once a classifier assigns it.

### Two provenances, scored indistinguishably by Stage 4

This is easy to miss if you only look at the `review_evidence` companion file
below — the FS matcher scores **both** of these the same way:

1. **Review-tier rule-confirmed pairs** — none possible today (no
   review-tier rule is defined; `NAME_DOB_SEX` at ~65% adjudicated precision
   and `NAME_DOB_ADDRESS` at ~67% held this tier briefly before removal).
2. **Genuine rule-undecided pairs** — no rule fired at all, and
   `classify_non_matches` found fewer than 3 strong-identifier
   contradictions (not confident enough to reject either).

### `review_evidence_<run_id>.parquet` (companion, same directory)

The full column set for provenance (1) above — `match_rule`, `confidence`,
`rules_fired`, etc. — that the closed `NonMatches`/`CandidatePairs` schema
trims. Not part of a strict pandera contract (nothing in the pipeline reads
it back), but it has exactly one consumer: `src/api/ingest/publish.py`, which
surfaces *why* a review-tier pair was flagged rather than just that it was.

### Output schema (`non_matches_*.parquet`)

Same as `data/blocking/`'s `CandidatePairs` — see that folder's `SCHEMA.md`:
`PATID_A`, `PATID_B`, `source_blocks`, `n_blocks`.

**Invariant:** `matches ⊎ non_matches ⊎ rejects == candidate_pairs`
(disjoint union), keyed on `(PATID_A, PATID_B)`.
