# Develop Integration (E2-5b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `origin/develop` into `feature/fs-baseline-splink` as a single commit, reconciling the `src/data/` + `src/features/` → `src/preprocessing/` rename, the `src/config/config.py` → `src/config.py` flattening, and the new `classify_non_matches` upstream contradiction filter — without losing any of the FS Enhanced_2 work (E2-0..E2-5).

**Architecture:** One `git merge develop` produces a merge commit on the feature branch. Conflicts resolved manually (one doc, one config, one pipeline). All FS-side imports rewritten in a single mechanical sweep. Verification gate is the existing pytest suite plus the synthetic enhanced_2 sandbox. The E2-5-fix5 corroboration gate is dropped — superseded by develop's `classify_non_matches`.

**Tech Stack:** Git (merge commit, no rebase), Python 3.11, pytest, Splink 4.x / DuckDB (for sandbox verification only).

**Spec:** `docs/superpowers/specs/2026-06-22-develop-integration-design.md`

---

## Pre-flight: confirm starting state

### Task 0: Verify clean working tree on the feature branch

**Files:** none

- [ ] **Step 1: Confirm branch + clean tree**

Run:
```bash
git status && git rev-parse --abbrev-ref HEAD
```
Expected:
```
On branch feature/fs-baseline-splink
nothing to commit, working tree clean
feature/fs-baseline-splink
```

- [ ] **Step 2: Confirm develop is fetched and not behind**

Run:
```bash
git fetch origin && git log --oneline HEAD..origin/develop | head
```
Expected: three commits listed (`14a8800`, `a892bc4`, `04d5414`). If empty, develop is already merged — stop and report.

---

## Task 1: Execute the merge

**Files:** none yet — surfaces conflicts.

- [ ] **Step 1: Start the merge**

Run:
```bash
git merge develop
```
Expected output (paraphrased):
```
warning: Cannot merge binary files: docs/Data-Cleaning-Guide.md (HEAD vs. develop)
CONFLICT (content): Merge conflict in docs/Data-Cleaning-Guide.md
CONFLICT (modify/delete): src/config/config.py deleted in develop and modified in HEAD.
Automatic merge failed; fix conflicts and then commit the result.
```

Auto-merged files include `docs/Data-Contract.md`, `src/contracts.py`, `src/pipeline.py`, `src/preprocessing/blocking.py`, `tests/conftest.py`. These have no conflict markers but their content must be inspected later.

- [ ] **Step 2: List conflict files**

Run:
```bash
git diff --name-only --diff-filter=U
```
Expected:
```
docs/Data-Cleaning-Guide.md
src/config/config.py
```

If anything else appears, stop and report — the merge produced unexpected conflicts.

---

## Task 2: Resolve `src/config/config.py` (delete on develop, port `matches_model_dir`)

**Files:**
- Delete: `src/config/config.py` (and the empty `src/config/` dir if it ends up empty)
- Modify: `src/config.py` (add `matches_model_dir` field)

- [ ] **Step 1: Capture our branch's `matches_model_dir` line**

Run:
```bash
git show HEAD:src/config/config.py | grep -n matches_model_dir
```
Expected: one line containing `matches_model_dir: Path = _DATA / "matches_model"` (the field name and default).

- [ ] **Step 2: Accept develop's deletion of the old file**

Run:
```bash
git rm src/config/config.py
```
Expected:
```
rm 'src/config/config.py'
```

- [ ] **Step 3: Add `matches_model_dir` into the new flat config**

Read `src/config.py` to find the block of `*_dir: Path = _DATA / "<name>"` fields (look around the line containing `rejects_dir`). Insert the new field directly after `non_matches_dir` to preserve ordering with our prior config:

```python
    matches_model_dir: Path = _DATA / "matches_model"
```

Verify with:
```bash
grep -n matches_model_dir src/config.py
```
Expected: one line in the `Settings` class body.

- [ ] **Step 4: Confirm the `src/config/` directory is gone**

Run:
```bash
ls src/config 2>/dev/null && echo "EXISTS" || echo "removed"
```
Expected: `removed`. If `EXISTS`, run `rmdir src/config`.

- [ ] **Step 5: Stage the config changes**

Run:
```bash
git add src/config.py
git status
```
Expected: `deleted: src/config/config.py` and `modified: src/config.py` (or `new file` if develop tracked it fresh) both staged. `src/config/config.py` no longer in unmerged list.

---

## Task 3: Resolve `docs/Data-Cleaning-Guide.md` (content conflict)

**Files:**
- Modify: `docs/Data-Cleaning-Guide.md`

- [ ] **Step 1: Inspect the conflict markers**

Run:
```bash
grep -nE '^(<<<<<<<|=======|>>>>>>>)' docs/Data-Cleaning-Guide.md
```
Expected: at least one set of `<<<<<<< HEAD` / `=======` / `>>>>>>> develop` markers.

- [ ] **Step 2: Read each conflict region and choose**

For each conflicted region: read the HEAD section (our changes — the E2-2 phonetic-code-persistence section in particular) and the develop section. The resolution rule:

- If develop's region is a doc cleanup of an unchanged-by-us section → take develop.
- If our region documents `_dm_LastNM` / `_dm_FirstNM` being persisted in cleaning output (E2-2 addition) → keep ours, layered on develop's surrounding text.
- If both edited the same paragraph → manually combine: take develop's prose as base, splice our DM-codes sentence(s) in.

Remove all `<<<<<<<`, `=======`, `>>>>>>>` markers as you resolve.

- [ ] **Step 3: Verify all markers are gone**

Run:
```bash
grep -nE '^(<<<<<<<|=======|>>>>>>>)' docs/Data-Cleaning-Guide.md && echo "MARKERS REMAIN" || echo "clean"
```
Expected: `clean`.

- [ ] **Step 4: Stage**

Run:
```bash
git add docs/Data-Cleaning-Guide.md
```

---

## Task 4: Verify auto-merged files survived our prior changes

**Files (verify only, edit if a regression):**
- `docs/Data-Contract.md`
- `src/contracts.py`
- `src/pipeline.py`
- `tests/conftest.py`

- [ ] **Step 1: Verify `src/contracts.py` retains our E2-1/E2-2 additions**

Run:
```bash
grep -n 'veto_reason\|_dm_LastNM\|_dm_FirstNM\|ZIP_BASE' src/contracts.py
```
Expected: all four tokens present. `veto_reason` should be in a context indicating it is `Optional`.

If any are missing → open `src/contracts.py`, find what auto-merge lost, restore it.

- [ ] **Step 2: Verify `src/pipeline.py` imports the new preprocessing module AND has our Stage 4 hook (if our branch added one)**

Run:
```bash
grep -nE 'from src\.preprocessing|run_fs_enhanced|classify_non_matches' src/pipeline.py
```
Expected:
- `from src.preprocessing.clean` and `from src.preprocessing.blocking` imports present (from develop)
- `classify_non_matches` imported from `src.models.deterministic_rules` (from develop)
- `run_fs_enhanced` imported and invoked **if our branch wired Stage 4 in**

Inspect the file by reading it. If our Stage-4 invocation block was dropped by auto-merge, re-add it: the block should accept the `non_matches` frame (or the `review` slice after `classify_non_matches`), call `run_fs_enhanced(...)`, validate against `ProbabilisticMatches`, and write the result to `settings.matches_model_dir / f"matches_model_{run_id}.parquet"`. See the pre-merge `src/pipeline.py` on `HEAD~1` for the exact form:

```bash
git show HEAD:src/pipeline.py | grep -nE 'run_fs_enhanced|matches_model' || echo "no Stage 4 on our branch"
```

If the grep returns no matches, our branch never wired Stage 4 — nothing to restore.

- [ ] **Step 3: Verify `docs/Data-Contract.md` retains our E2-1/E2-2 contract changes**

Run:
```bash
grep -nE 'veto_reason|_dm_LastNM|_dm_FirstNM|n_contradictions|decision' docs/Data-Contract.md
```
Expected: all five tokens present (our three + develop's two on `NonMatches`). If any are missing, manually re-add by reading `git show HEAD:docs/Data-Contract.md` for the missing section and splicing it in.

- [ ] **Step 4: Verify `tests/conftest.py` retains our session-scoped fixtures**

Run:
```bash
grep -nE 'fs_df_clean|fs_candidate_pairs|fs_classified' tests/conftest.py
```
Expected: all three fixture names present.

- [ ] **Step 5: If you re-edited any file in this task, stage it**

Run:
```bash
git status
git add <any files you edited>
```

---

## Task 5: Import sweep — rewrite old paths in our branch's additions

**Files (modify):** all files identified by `git grep` in step 1. Expected list (from inspection on `feature/fs-baseline-splink`):

- `README.md`
- `docs/Fellegi-Sunter-Enhanced_2.md`
- `models/common/synthetic_data.py`
- `models/experiments/fs_splink_baseline/fellegi_sunter_baseline.py`
- `models/experiments/fs_splink_baseline/run_synthetic_baseline.py`
- `models/experiments/fs_splink_enhanced/fellegi_sunter_enhanced.py`
- `models/experiments/fs_splink_enhanced/run_real_enhanced.py`
- `models/experiments/fs_splink_enhanced/run_synthetic_enhanced.py`
- `models/experiments/fs_splink_enhanced_2/fs_enhanced_2.py`
- `notebooks/eda/transformation_validation_eda.ipynb`
- `notebooks/fellegi_sunter/fellegi_sunter_validation.ipynb`
- `notebooks/test_blocking_synthetic.ipynb`
- `scripts/audit_synthetic_coverage.py`
- `scripts/verify_pipeline.py`
- `tests/integration/test_batch_blocking.py`
- `tests/integration/test_blocking_index.py`
- `tests/integration/test_inference_blocking.py`
- `tests/integration/test_serialization_roundtrip.py`
- `tests/regression/test_dm_codes_persisted.py`
- `tests/regression/test_known_pairs.py`
- `tests/unit/test_fellegi_sunter_baseline.py`
- `tests/unit/test_key_generation.py`
- `tests/unit/test_pair_canonicalization.py`
- `tests/unit/test_phone_parser.py`
- `tests/unit/test_phonetic_helpers.py`
- `tests/unit/test_transformations.py`

Tracked-by-develop files (do NOT touch — develop already rewrote their imports):
`src/contracts.py`, `src/evaluation/rule_eval.py`, `src/pipeline.py`, `src/models/deterministic_rules.py`, `src/models/run_rules.py`, `tests/conftest.py`.

- [ ] **Step 1: Generate the working list**

Run:
```bash
git grep -lE 'from src\.data\.|from src\.features\.|from src\.config\.config|src\.config\.config import' -- '*.py' '*.md' '*.ipynb'
```
Use the resulting file list as the authoritative target — diverges from the list above only if develop touched something we didn't anticipate.

- [ ] **Step 2: Apply the mechanical rewrites**

For each Python file in the working list, run:
```bash
python3 -c "
import sys, pathlib
for p in sys.argv[1:]:
    path = pathlib.Path(p)
    text = path.read_text()
    new = (text
        .replace('from src.data.transformations', 'from src.preprocessing.transformations')
        .replace('from src.data.clean', 'from src.preprocessing.clean')
        .replace('from src.features.blocking', 'from src.preprocessing.blocking')
        .replace('from src.features.run_blocking', 'from src.preprocessing.run_blocking')
        .replace('from src.config.config import', 'from src.config import')
        .replace('src.config.config import', 'src.config import')
        .replace('from src.data import', 'from src.preprocessing import')
        .replace('from src.features import', 'from src.preprocessing import')
    )
    if new != text:
        path.write_text(new)
        print('rewrote', p)
" $(git grep -lE 'from src\.data\.|from src\.features\.|from src\.config\.config|src\.config\.config import' -- '*.py')
```

For `.md` and `.ipynb` files, apply the same rewrites via the same script with the relevant extensions. Notebooks contain JSON-encoded source — the same `.replace` string matches.

- [ ] **Step 3: Verify zero stale imports remain in our code**

Run:
```bash
git grep -nE 'from src\.data\.|from src\.features\.|from src\.config\.config' -- '*.py' '*.ipynb' '*.md'
```
Expected: zero hits. If any remain, inspect each manually — likely a `src.data.something_else` we don't actually own, or a code comment that should be updated.

- [ ] **Step 4: Stage the rewrites**

Run:
```bash
git add -u
git status
```

Expected: all rewritten files staged under "Changes to be committed" with no "Unmerged paths" remaining.

---

## Task 6: Verification gate — full pytest + synthetic sandbox

**Files:** none — runs the existing suites.

- [ ] **Step 1: Unit tests**

Run:
```bash
pytest tests/unit -q 2>&1 | tail -30
```
Expected: all tests pass. Common failure modes if Task 5 missed a file:
- `ModuleNotFoundError: No module named 'src.data'` → re-run the rewrite script
- `ImportError: cannot import name 'X' from 'src.config'` → a symbol moved during develop's flattening; read `src/config.py` and update the import name

- [ ] **Step 2: Integration tests**

Run:
```bash
pytest tests/integration -q 2>&1 | tail -30
```
Expected: pass.

- [ ] **Step 3: Regression tests**

Run:
```bash
pytest tests/regression -q 2>&1 | tail -30
```
Expected: pass.

- [ ] **Step 4: Full suite**

Run:
```bash
pytest -q 2>&1 | tail -20
```
Expected: green. If any test fails for a reason unrelated to import paths or `matches_model_dir`, STOP and report — don't paper over a real regression.

- [ ] **Step 5: Synthetic enhanced_2 sandbox smoke**

Run:
```bash
python -m models.experiments.fs_splink_enhanced_2.run_synthetic_enhanced_2 2>&1 | tail -30
```
Expected log lines (paraphrased):
```
INFO ... Records: 100000 | Train labels: 40000 ...
INFO ... Tier breakdown: {'no_match': ..., 'human_review': ..., 'auto_merge': ...}
INFO ... ✓ ProbabilisticMatches validation passed (10000 rows)
INFO ... Sandbox smoke complete.
```

Critical assertion: `ProbabilisticMatches validation passed`. If validation fails, a column was renamed by develop in cleaning — inspect with `python -c "import pandas as pd; print(pd.read_csv('data/synthetic/synthetic_records_v3.csv', nrows=1).columns.tolist())"` and compare against the comparison registry.

---

## Task 7: Documentation updates (same commit)

**Files:**
- Modify: `docs/Fellegi-Sunter-Enhanced_2.md`
- Modify: `docs/Data-Contract.md`
- Modify locally only (NEVER stage): `CLAUDE.md`

- [ ] **Step 1: Rewrite the "Required upstream deterministic vetoes" section of `docs/Fellegi-Sunter-Enhanced_2.md`**

Find the section by:
```bash
grep -n 'Required upstream deterministic vetoes\|upstream vetoes\|veto' docs/Fellegi-Sunter-Enhanced_2.md | head -20
```

Replace the entire section (heading + body) with the following (verbatim):

```markdown
## Upstream contradiction filter (`classify_non_matches`)

Stage 3 (`src/models/deterministic_rules.classify_non_matches`) is the team's
chosen pattern for keeping anti-evidence pairs out of Stage 4. It performs a
three-way split on every candidate pair:

1. **Confirmed** — at least one deterministic rule fires → `data/matches/`
2. **Reject** — ≥ `DEFAULT_REJECT_MIN_CONTRADICTIONS` strong identifiers
   strictly disagree → dropped from the pipeline entirely
3. **Review** — neither confirmed nor rejected → flows into Stage 4 as
   `non_matches`

Stage 4 (this module) therefore receives only the "review" survivors —
ambiguous pairs without enough contradicting evidence to discard, but without
enough corroborating evidence to confirm. Anti-evidence comparison levels
inside the FS model (`Household_discount`, `Sex_positive[M↔F]`,
`SSN[9-digit mismatch]`) cannot be learned by supervised m-training (positives
never fire them) and will still inflate scores when they fire, but the
upstream filter removes most of the population on which that inflation
matters.

**Design precedent:** see `docs/Deterministic-Rules-Guide.md` — the team's
deterministic engine has consistently chosen "filter on contradictions /
corroborate by rule" over "veto by single signal".

**Reversibility:** if post-deployment measurement shows residual over-scoring
on review-tier pairs, a downstream corroboration gate (a positive-predicate
requirement on auto_merge promotion, mirroring the Stage 3 `RULES` registry)
remains available as a follow-up. It was scoped as Phase E2-5-fix5 and
deliberately dropped here because the upstream contradiction filter addresses
the same failure mode at the architecturally correct layer.
```

If the file already has a "Known limitations" section, update its first item to read:

```markdown
- **Anti-evidence over-scoring is mitigated, not eliminated.** Stage 4 sees
  only review-tier pairs from `classify_non_matches`, so the household-FP
  population that inflated raw scores in pre-merge runs is now mostly
  dropped before scoring. Residual inflation is possible on borderline
  review pairs where one anti-evidence level fires; revisit if real-cohort
  measurement shows auto_merge precision below the threshold gate.
```

- [ ] **Step 2: Update `docs/Data-Contract.md`**

Confirm `n_contradictions` and `decision` are documented on the `NonMatches`-related table (develop introduced these). If absent, inspect develop's `classify_non_matches` return columns:
```bash
grep -nA 3 'n_contradictions\|decision' src/models/deterministic_rules.py | head -30
```
Add a documentation row matching the function's return shape.

- [ ] **Step 3: Update `CLAUDE.md` locally — do NOT stage**

Edit `CLAUDE.md` to:
- Replace `src/data/` references with `src/preprocessing/`
- Replace `src/features/` references with `src/preprocessing/`
- Replace `from src.config.config import settings` with `from src.config import settings`
- Add `classify_non_matches` to the Stage 3 description as the three-way split
- Remove any references to Phase E2-5-fix5 and corroboration gates
- Add a one-line entry under "Stage 4" noting it now consumes only `review`-tier pairs

After editing, verify it's still gitignored:
```bash
git check-ignore CLAUDE.md && echo "still ignored" || echo "WARNING: tracked"
git status | grep CLAUDE.md && echo "STAGED — UNSTAGE NOW" || echo "unstaged"
```
Expected: `still ignored` and `unstaged`.

- [ ] **Step 4: Stage the doc changes**

Run:
```bash
git add docs/Fellegi-Sunter-Enhanced_2.md docs/Data-Contract.md
git status
```
Expected: no `CLAUDE.md` in the staged list.

---

## Task 8: Final pre-commit checks

**Files:** none — confirms the staged state.

- [ ] **Step 1: Confirm `CLAUDE.md` is not staged**

Run:
```bash
git diff --cached --name-only | grep -E '^CLAUDE\.md$' && echo "STAGED — REMOVE" || echo "ok"
```
Expected: `ok`. If `STAGED — REMOVE`, run `git reset HEAD CLAUDE.md`.

- [ ] **Step 2: Confirm zero stale imports**

Run:
```bash
git grep -nE 'from src\.data\.|from src\.features\.|from src\.config\.config' -- '*.py'
```
Expected: zero hits.

- [ ] **Step 3: Confirm `.gitignore` invariants survived**

Run:
```bash
git check-ignore data/raw/MDM_Population.csv && echo "raw ignored: ok"
git check-ignore data/synthetic/synthetic_test_v3.csv 2>&1 | grep -q . && echo "WARNING: synthetic ignored — bad" || echo "synthetic tracked: ok"
```
Expected both lines: `ok`.

- [ ] **Step 4: Re-run the full suite one more time**

Run:
```bash
pytest -q 2>&1 | tail -10
```
Expected: green.

---

## Task 9: Commit

**Files:** none — finalizes.

- [ ] **Step 1: Inspect final staged set**

Run:
```bash
git diff --cached --stat | tail -30
```
Expected: a mix of (a) develop-side files now merged in, (b) our import-rewritten files from Task 5, (c) the resolved config + docs from Tasks 2/3/7.

- [ ] **Step 2: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
Phase E2-5b: integrate develop (preprocessing rename, classify_non_matches)

Merges origin/develop into feature/fs-baseline-splink. Resolves the
src/data/+src/features/ -> src/preprocessing/ rename and the
src/config/config.py -> src/config.py flattening across the FS Enhanced_2
work (E2-0..E2-5).

Drops the planned E2-5-fix5 corroboration gate in favor of develop's
new `classify_non_matches` upstream contradiction filter, the team's
chosen pattern for keeping anti-evidence pairs out of Stage 4.
EOF
)"
```
Expected: one merge commit on `feature/fs-baseline-splink`.

- [ ] **Step 3: Confirm commit landed and tree is clean**

Run:
```bash
git log --oneline -3 && git status
```
Expected: top commit is the merge commit just created, then `Phase E2-5b: ...`, working tree clean.

- [ ] **Step 4: Update the in-session task tracker**

Mark task #9 (E2-5b: develop-integration phase) as completed via TaskUpdate.

---

## Out of scope (deferred, not in this commit)

- **Real-data VM run of `run_real_enhanced_2.py`** against the new contradiction-filtered non_matches pool — happens in your next VM session. Headline numbers (tier counts, score quantiles, auto_merge rate vs the pre-merge 62%) go into `docs/Fellegi-Sunter-Enhanced_2.md` "How to run" in a follow-up doc-only commit.
- **E2-6 validation notebook §11** (3-way head-to-head) — proceeds on the next phase per the original FS Enhanced_2 plan.
- **A standalone E2-5c phase** wiring `classify_non_matches` differently in `src/pipeline.py` — only needed if Task 4 Step 2 reveals develop's wiring is incompatible with our Stage 4 hook. If it works as-is, no follow-up phase needed.
