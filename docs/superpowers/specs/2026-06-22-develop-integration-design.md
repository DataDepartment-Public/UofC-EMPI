# Phase E2-5b — Develop Integration Design

**Date:** 2026-06-22
**Branch:** `feature/fs-baseline-splink`
**Supersedes:** Phase E2-5-fix5 (corroboration gate) — dropped in favor of develop's `classify_non_matches` upstream pattern.

## Context

While `feature/fs-baseline-splink` was completing E2-0..E2-5 of the FS Enhanced_2 plan, `origin/develop` advanced by two commits:

```
14a8800 Merge pull request #11 from DataDepartment-Public/jason-rules
a892bc4 added logging and eval framework, cleaned up the repo
04d5414 Deterministic Rule Update & Blocking Evaluation
```

The diff is +2096/-335 lines across 39 files. The substantive changes:

1. **Layout rename:** `src/data/` + `src/features/` → `src/preprocessing/` (one combined package). `src/config/config.py` flattened to `src/config.py`.
2. **`classify_non_matches` added to `src/models/deterministic_rules.py`** — a three-way split (confirmed / reject / review) that drops candidate pairs with ≥`DEFAULT_REJECT_MIN_CONTRADICTIONS` strong-identifier disagreements *before* Stage 4 sees them. This is the team's architectural answer to the upstream-veto debate: contradiction-counting, not corroboration.
3. **New evaluation framework:** `src/evaluation/blocking_eval.py`, `src/evaluation/run_eval.py`, plus matching tests.
4. **Logging convergence:** `configure_logging()` + new log_* Settings fields replace per-module `logging.basicConfig`.
5. **Settings additions:** `rejects_dir`, `log_dir`, `log_*`, `deploy_*` (eval-report tunables).
6. **Settings deletion:** `matches_model_dir` is gone (develop hasn't touched FS Stage 4; our enhanced_2 work depends on it).

A `git merge develop` attempt surfaced three conflicts (`docs/Data-Cleaning-Guide.md` content, `docs/Data-Contract.md` auto-merged, `src/config/config.py` delete-vs-modify), aborted cleanly.

## Decision: drop E2-5-fix5

Develop's `classify_non_matches` introduces contradiction-filtering at Stage 3, which materially reduces the population that flows into Stage 4. The 105k auto_merges observed in the pre-merge real-data run are heavily populated by household-FP pairs (Address+Phones agree, name+DOB disagree) — exactly the pairs `classify_non_matches` is built to send to `reject` rather than `review`. The corroboration gate planned as E2-5-fix5 becomes redundant; we adopt the team's chosen pattern instead.

This decision is reversible: if post-merge measurement shows Stage 4 still over-scoring on the survivors, corroboration can be revisited as a future phase.

## Scope (single commit, on `feature/fs-baseline-splink`)

### 1. Merge develop

```bash
git merge develop
```

Merge commit, not rebase. Preserves the E2-0..E2-5 commit boundaries intact.

### 2. Conflict resolutions

| File | Status from `git merge` | Resolution |
|---|---|---|
| `docs/Data-Cleaning-Guide.md` | both modified (binary-flagged due to encoding) | Manual content merge. Use develop's edits as base; layer in the DM-codes-persisted-in-cleaning addition from E2-2. |
| `docs/Data-Contract.md` | auto-merged with no markers | Verify our changes survived: `ProbabilisticMatches.veto_reason` Optional (E2-1), `_dm_LastNM`/`_dm_FirstNM` in `CleanedRecords` (E2-2), `ZIP_BASE`. Add `n_contradictions` + `decision` columns to whichever schema develop attaches them to (likely a new `NonMatches`-sibling or columns on `NonMatches`). |
| `src/config/config.py` | deleted by develop, modified by us | Accept develop's deletion. Port the one field develop dropped — `matches_model_dir: Path = _DATA / "matches_model"` — into the new flat `src/config.py`. Keep all of develop's new fields (`rejects_dir`, `log_dir`, `log_level`, `log_to_file`, `log_file`, `log_format`, `log_datefmt`, `deploy_blocking_method`, `deploy_blocking_sample`, `deploy_seed`). |

### 3. Import sweep

Single mechanical rewrite across our branch's additions. The rename is a `git mv` on develop so most paths flip together.

| Old import | New import |
|---|---|
| `from src.data.transformations import ...` | `from src.preprocessing.transformations import ...` |
| `from src.data.clean import load_cleaned` | `from src.preprocessing.clean import load_cleaned` |
| `from src.features.blocking import ...` | `from src.preprocessing.blocking import ...` |
| `from src.features.run_blocking import ...` | `from src.preprocessing.run_blocking import ...` |
| `from src.config.config import settings` | `from src.config import settings` |

Sweep targets (anything our branch added or modified in E2-0..E2-5):

- `models/experiments/fs_splink_enhanced_2/**.py`
- `models/experiments/fs_splink_enhanced/**.py`
- `models/experiments/fs_splink_baseline/**.py`
- `models/common/fs_base.py`, `models/common/synthetic_data.py`, `models/common/versioning.py`, `models/common/eval_schema.py`
- `scripts/build_synthetic_records.py`, `scripts/sweep_enhanced_2_thresholds.py`, `scripts/_inject_section_10.py`, `scripts/audit_synthetic_coverage.py`
- `tests/conftest.py`, every `tests/**/test_*.py` that imports from `src.data.*` / `src.features.*` / `src.config.config`

Tool: `git grep -l 'src\.data\.\|src\.features\.\|src\.config\.config'` after the merge, then `sed -i` (or Edit calls) to flip each match.

### 4. Pipeline orchestrator (`src/pipeline.py`)

Both sides modified — develop refactored stage imports, we added Stage 4 wiring. Reconcile by:

- Taking develop's Stage 1–3 wiring verbatim (preprocessing paths).
- Re-attaching our Stage 4 call (`run_fs_enhanced(...)` → `data/matches_model/<run_id>.parquet`).
- **Open question:** whether to ALSO wire `classify_non_matches` between Stage 3 and Stage 4 in this commit. Default in this phase: **yes if develop already wired it; otherwise defer to a follow-up phase.** Resolves on conflict inspection.

### 5. Verification gate (must all pass before commit)

```bash
pytest tests/unit -q                                                # imports + contracts
pytest tests/integration -q                                          # stage wiring
pytest tests/regression -q                                           # known-pairs
pytest -q                                                            # full suite
python -m models.experiments.fs_splink_enhanced_2.run_synthetic_enhanced_2
```

The synthetic sandbox is the canary for silent column renames — it exercises every active comparison and validates `ProbabilisticMatches`.

### 6. Documentation updates (same commit)

| Doc | Change |
|---|---|
| `docs/Fellegi-Sunter-Enhanced_2.md` | Replace "Required upstream deterministic vetoes" section with "Upstream contradiction filter (`classify_non_matches`)" — explains the team's chosen pattern, points at `src/models/deterministic_rules.classify_non_matches` and `docs/Deterministic-Rules-Guide.md`. Remove all E2-5-fix5 / corroboration content. Update "Known limitations": Stage 4 now sees only `review` pairs (no contradictions); anti-evidence over-scoring should reduce materially but remains directionally present. |
| `docs/Data-Contract.md` | Add `n_contradictions` + `decision` columns to the relevant Stage-3 output schema (location TBD by reading develop's contract changes). |
| `CLAUDE.md` (local only — **never staged**) | Rewrite §"Stage 1" and §"Stage 2" path references (`src/preprocessing/` instead of `src/data/`+`src/features/`). Rewrite §"Stage 3" to mention `classify_non_matches`. Update §"Stage 4" to note vetoes are dropped and upstream contradiction-filter is the team's chosen pattern. Confirm via `git status` before commit that `CLAUDE.md` is unstaged. |

### 7. Commit

```
Phase E2-5b: integrate develop (preprocessing rename, classify_non_matches)
```

## Out of scope (deferred)

- **E2-5-fix5 corroboration gate** — dropped permanently in favor of upstream `classify_non_matches`.
- **Wiring `classify_non_matches` into `src/pipeline.py`** if conflict-reconciliation shows it requires non-trivial contract changes — becomes E2-5c.
- **Re-running real-data VM enhanced_2** to measure new auto_merge rate against the contradiction-filtered pool — happens next VM session after E2-5b lands. Headline numbers go into `docs/Fellegi-Sunter-Enhanced_2.md` "How to run" in a follow-up doc-only commit, not this one.
- **E2-6 validation notebook §11 head-to-head** — proceeds as planned in the original FS Enhanced_2 plan once E2-5b is on the branch.

## Risks

| Risk | Mitigation |
|---|---|
| `classify_non_matches` already wired into `src/pipeline.py` on develop creates a non-trivial pipeline conflict | Inspect develop's `src/pipeline.py` before resolving; if wired, take develop's wiring and bolt our Stage 4 onto its `review` output |
| Develop renamed a cleaning column we depend on (`SexAtBirthDSC_clean`, `_dm_*`) — Pandera catches missing-column but renamed-and-still-present columns slip through | Run synthetic sandbox post-merge; it touches every active comparison |
| `matches_model_dir` re-add to `src/config.py` uses wrong name/type → orchestrator + enhanced runners stop finding their output dir | Field name and `Path` type must match the deleted version verbatim; integration test asserts the directory resolves |
| Tests on develop assume new logging configuration; our tests don't call `configure_logging` | Add `configure_logging()` invocation to `tests/conftest.py` if develop's session-scope fixtures require it |
| Synthetic data CSVs (`data/synthetic/*.csv`) were committed by us in E2-0 via a `.gitignore` whitelist — develop's `.gitignore` may have changed | `git check-ignore data/raw/MDM_Population.csv data/synthetic/synthetic_test_v3.csv` post-merge; PHI must stay ignored, synthetic must stay tracked |

## Acceptance criteria

1. `git merge develop` produces one merge commit on `feature/fs-baseline-splink`.
2. `pytest -q` exits 0 across unit + integration + regression suites.
3. `python -m models.experiments.fs_splink_enhanced_2.run_synthetic_enhanced_2` round-trips through `ProbabilisticMatches` validation.
4. `git grep 'src\.data\.\|src\.features\.\|src\.config\.config'` returns zero hits.
5. `git check-ignore CLAUDE.md` exits 0 (still gitignored) and `CLAUDE.md` is not staged.
6. `docs/Fellegi-Sunter-Enhanced_2.md` no longer references E2-5-fix5 or corroboration gates; the upstream-contradiction-filter section is present.

## Why this design (vs alternatives)

- **Merge over rebase:** preserves E2-0..E2-5 commit boundaries; conflicts resolved once instead of replayed across each commit.
- **Single phase over two-phase split:** the changes are mechanical (renames + import rewrites + small config port). Splitting would commit a known-broken intermediate state.
- **Drop fix5 instead of layering it on top of `classify_non_matches`:** the team has chosen contradiction-filtering as the architectural pattern; a downstream corroboration gate would duplicate intent. If real-data measurement post-merge shows residual over-scoring, corroboration is still available as a future phase — this decision is reversible.
