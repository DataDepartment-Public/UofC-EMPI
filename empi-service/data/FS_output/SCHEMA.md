# data/fs_output/ — Stage 4 output: FS matcher audit frame + candidates `[IMPLEMENTED]`

Full contract: `docs/Data-Contract.md` → "Stage 4 — FS matcher" (§4a, §4b).

Merged folder — holds both artifacts the FS matcher produces per run.
Previously split across two directories (`data/matches_model/` for the audit
frame, `data/FS_output/` for the candidate deliverable); consolidated since
both are the FS matcher's output as a whole and always written together.

- **Producer:** `src/models/fs_matcher/` (`FSMatcher.score`, invoked from
  `src/pipeline.py`), and `python -m src.models.fs_matcher.train` for the
  labeled training set
- **Trained model artifacts** (Splink JSON + `.meta.json` + `active.json`)
  live separately in `models/fs/` — not under `data/`. See
  `docs/FS-Matcher-Production-Guide.md`.
- **Training input resolution:** `train.py` resolves `cleaned` +
  `candidate_pairs` from the **latest `RunManifest`**
  (`data/runs/run_<run_id>.json`), not directory-globbing — guarantees
  same-run lineage and the stacked blocker's output (never
  `run_blocking.py`'s narrower pool).
- **Naming note:** the folder is not tier-named (unlike `data/auto_merge/`/
  `data/no_match/`) because it contains **all three tiers mixed** — the tier
  lives in the `classification_tier` column of each row, not the folder.

## 4a — ProbabilisticMatches (full audit frame)

- **Contract:** `contracts.ProbabilisticMatches` (`strict=True`)
- **File:** `matches_model_<run_id>.parquet`
- **Grain:** every scored `non_matches` pair, all tiers including `no_match`
- **Status:** audit frame; no downstream reader other than Stage 5's optional
  edge union when `settings.fs_feeds_clustering` is on

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` / `PATID_B` | string | no | Canonical pair. |
| `match_source` | string | no | Always `"model"`. |
| `score` | float64 | no | Match probability, `[0.0, 1.0]`. |
| `match_weight` | float64 | no | Log-Bayes-factor match weight. |
| `classification_tier` | string | no | ∈ `{auto_merge, human_review, no_match}` (`contracts.CLASSIFICATION_TIERS`) — routes into clustering only when `settings.fs_feeds_clustering` is on; otherwise informational/audit only. |
| `veto_reason` | string | yes (optional) | Present only if the producer applies a veto layer; the production matcher omits the column entirely. |
| `source_blocks` / `n_blocks` | string / int64 | yes | Passthrough from blocking, where available. |

## 4b — FSFeatures (the candidate deliverable)

- **Contract:** `contracts.FSFeatures` (`strict=False` — `gamma_<field>` /
  `bf_<field>` feature columns are dynamic extras, checked for presence by
  `validate_fs_features` rather than pandera's closed-schema mode)
- **Files:** `fs_features_<run_id>.parquet` (pipeline, candidate-filtered)
  and `fs_features_train_<version>.parquet` (train CLI, labeled)
- **Grain:** candidates only — filtered to `match_probability >=
  settings.fs_review_floor` (0.40 default; doubles as the tier boundary *and*
  the candidate cutoff)
- **Status:** its intended consumer is a downstream trained model — either
  the ML matcher (Stage 4.5, which can join on this frame via its optional
  `fs_features` input) or a bespoke offline GBT training run

| Column | Dtype | Nullable | Notes |
|---|---|---|---|
| `PATID_A` / `PATID_B` | string | no | Canonical pair. |
| `match_probability` | float64 | no | Same as `ProbabilisticMatches.score`. |
| `match_weight` | float64 | no | Same as `ProbabilisticMatches.match_weight`. |
| `classification_tier` | string | no | Same as `ProbabilisticMatches.classification_tier`, above. |
| `label` | float64 (0.0/1.0) | yes | Present (non-null) only on the training feature set; absent/null when scoring. |
| `gamma_<field>` (×7) | int | no | Per-comparison level index (one per Splink comparison: FirstNM, LastNM, BirthDT, SSN, Email, Phones, Address). |
| `bf_<field>` (×N) | float64 | no | Per-comparison Bayes-factor bits, incl. `bf_tf_adj_*` for term-frequency-adjusted fields. |

Full detail (including how these fields feed the GBT and Stage 4.5):
`docs/Data-Contract.md`.
