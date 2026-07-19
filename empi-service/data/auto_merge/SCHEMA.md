# data/auto_merge/ — Stage 3a output: auto-merge-tier matches `[IMPLEMENTED]`

Full contract: `docs/Data-Contract.md` → "Stage 3 — Deterministic matches,
non-matches & rejects" (§3a) and `docs/Deterministic-Rules-Guide.md`.

- **Producer:** `src/models/deterministic_rules::apply_rules`, invoked from
  `src/pipeline.py` (which owns the auto-merge/review tier split)
- **Contract:** `contracts.Matches` (`strict=True`; empty frames skip validation)
- **Consumers:** clustering (Stage 5), `src/api/ingest/publish.py`,
  `src/evaluation/rule_eval.py`
- **File:** `matches_<run_id>.parquet`
- **Grain:** one row per pair confirmed by an **auto-merge-tier** rule only.
  `NAME_DOB_SEX` and `NAME_DOB_ADDRESS` never appear here — they're
  review-tier; a pair they confirm routes to `data/non_matches/` instead.
- **Naming note:** the folder name matches `contracts.TIER_AUTO_MERGE`
  (`"auto_merge"`) — the same tier vocabulary used by the FS/ML classifier
  stages' `predicted_tier` column. This folder *is* that tier's output.
  `data/no_match/` mirrors `TIER_NO_MATCH` the same way; `data/non_matches/`
  is deliberately **not** renamed to tier vocabulary — it isn't a decided
  tier, it's the pre-scoring pool Stage 4/4.5 consume (see its own SCHEMA.md).

### The five rules (descending confidence; first to fire wins `match_rule`)

| Rule | Confidence | Tier | Agreement predicate |
|---|---|---|---|
| `SSN_DOB` | 1.000 | auto-merge | `SSN_clean` + `BirthDT_clean` |
| `NAME_DOB_EMAIL` | 0.990 | auto-merge | First + Last + DOB + Email |
| `NAME_DOB_PHONE` | 0.985 | auto-merge | First + Last + DOB + phone-set intersection |
| `NAME_DOB_SEX` | 0.980 | review | First + Last + DOB + Sex |
| `NAME_DOB_ADDRESS` | 0.970 | review | First + Last + DOB + `AddressLine1_clean` |

First/last name predicates are fuzzy (Jaro-Winkler ≥ 0.92 or
Damerau-Levenshtein ≤ 1); all other predicates are exact.

### Output schema

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` / `PATID_B` | string | no | Carried unchanged from blocking. |
| `match_rule` | string | no | Highest-confidence **auto-merge-tier** rule that fired. |
| `confidence` | float64 | no | `[0.985, 1.000]` — the auto-merge floor, not the full 5-rule range. |
| `rules_fired` | string | no | Pipe-delimited list of every rule that fired (any tier). |
| `is_suspicious` | bool | no | `True` if DOB, last name, or (both-present) SSN disagree. |
| `high_fanout_ssn` | bool | no | `True` if the pair's shared SSN is carried by ≥ `EMPI_SSN_FANOUT_THRESHOLD` patients. |
| `cluster_id` | int64 | no | Connected-component id, stamped by the writer. |
| `source_blocks` / `n_blocks` | string / int64 | no | Passthrough from blocking. |

**Invariant:** `matches ⊎ non_matches ⊎ rejects == candidate_pairs`
(disjoint union), keyed on `(PATID_A, PATID_B)`.
