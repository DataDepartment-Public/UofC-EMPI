# FS Refactor Design — Phase E3

**Date:** 2026-06-23
**Branch:** `feature/fs-baseline-splink`
**Plan cross-reference:** `/Users/sachinpatel/.claude/plans/i-want-to-plan-goofy-lerdorf.md` (Phase E3 section)

---

## 1. TL;DR

Refactor `fs_splink_baseline/` and `fs_splink_enhanced/` onto the shared `models/common/fs_base.py` OO scaffold that `fs_splink_enhanced_2/` already consumes. Each experiment becomes a thin `FSModel` subclass. No new base machinery is required — `fs_base.py` is already complete; this refactor is consumption-only.

---

## 2. Why

### The problem: two duplicated functional modules

`fs_splink_baseline/fellegi_sunter_baseline.py` and `fs_splink_enhanced/fellegi_sunter_enhanced.py` were written before the OO scaffold existed. They are ~98% duplicated: both modules implement `build_settings`, `build_linker`, `_register_candidate_pairs`, `train_model`, `predict_pairs`, `classify_pairs`, `_apply_n_blocks_bump`, `to_evaluation_schema`, `to_probabilistic_matches`, and `run_fs_*` in parallel, with minor threshold and comparison-vector differences being the only genuine variation.

Phase E2-1 introduced `models/common/fs_base.py` with `FSModel`, `TrainingStrategy` (`EMTraining`, `SupervisedTraining`), `ComparisonSpec`, `ComparisonRegistry`, and `ClassificationConfig` — and `fs_splink_enhanced_2/fs_enhanced_2.py` proved the scaffold works cleanly. The baseline and enhanced functional modules were not migrated then ("out of scope … separate follow-up after enhanced_2 ships" — E2 plan). That follow-up is Phase E3.

### The goal

- **Eliminate duplication.** Every shared mechanic lives once in `fs_base.py`; subclasses supply only what genuinely varies: the comparison registry, the classification config, and — for enhanced — the veto + manual-prior overrides.
- **Converge on one API.** Three experiments, one interface: `FSModel.run(cp_df, df_clean, ...)`. Downstream code (runners, tests, pipeline) no longer imports from the old functional modules.
- **Reduce maintenance surface.** Future comparison changes (e.g., adding a new FS_Enhanced_3 level) require editing one `ComparisonSpec`, not duplicating a 30-line Splink boilerplate block.

---

## 3. Architecture

Everything needed is already in `models/common/fs_base.py`. No new base machinery is introduced in Phase E3.

### 3.1 `FSModel` ABC

Abstract base class. Public entry point: `run(cp_df, df_clean, full_output=False, u_max_pairs=1e6)`. Subclasses override:

- `prepare_model_input(df_clean) -> pd.DataFrame` — derive DOB-string, phonetic codes, phones array, shim columns.
- `build_settings() -> dict` — call `self.registry.build_all()` + Splink boilerplate. ~10 lines per subclass.

`FSModel` provides (no override needed): `build_linker`, `train`, `predict`, `classify`, `_apply_n_blocks_bump`, `_register_candidate_pairs`, `_estimate_u_with_guard`, `_warn_on_weight_inversions`, `to_evaluation_schema`, `to_probabilistic_matches`.

### 3.2 `TrainingStrategy` — `EMTraining` and `SupervisedTraining`

`TrainingStrategy` is the strategy object injected at construction time. It defines `fit(linker, u_max_pairs, seed)`.

- `EMTraining(em_blocking_rules, prior_rules, recall)` — the baseline + enhanced pattern: random-sample u, three EM sessions, match-prevalence prior.
- `SupervisedTraining(labels_path, labels_records_df, ...)` — the enhanced_2 pattern: supervised m from labeled pairs, u from random sampling.

### 3.3 `ComparisonSpec` and `ComparisonRegistry`

`ComparisonSpec` wraps one Splink comparison builder. `ComparisonRegistry` is an ordered collection of specs with `build_all()`, `with_added()`, `with_removed()`, and `names()`. Each experiment defines its own registry constant in `comparisons.py` and passes it to the subclass.

### 3.4 `ClassificationConfig`

Immutable dataclass: `auto_merge_threshold`, `review_floor`, `n_blocks_bump_threshold`, `n_blocks_bump_max_bits`. Subclasses set these at class definition time.

---

## 4. What changes per module

| Module | Action | Key files |
|---|---|---|
| **`fs_splink_baseline`** | Refactor | DELETE `fellegi_sunter_baseline.py`; CREATE `fs_baseline.py` (`FSBaseline(FSModel)`) + `comparisons.py` (`BASELINE_REGISTRY`); EDIT `run_real_baseline.py`, `run_synthetic_baseline.py`, `__init__.py` |
| **`fs_splink_enhanced`** | Refactor | DELETE `fellegi_sunter_enhanced.py`; CREATE `fs_enhanced.py` (`FSEnhanced(FSModel)`) + `comparisons.py` (`ENHANCED_REGISTRY`); KEEP `deterministic_vetoes.py` + `manual_priors.py` as standalone helpers wired via `FSEnhanced.classify()` + `FSEnhanced.train()` overrides; EDIT runners + `__init__.py` |
| **`fs_splink_enhanced_2`** | No changes | Already on the scaffold; reference shape. |

### 4.1 `FSBaseline`

```python
class FSBaseline(FSModel):
    model_name = "fs_splink_baseline"
    registry   = BASELINE_REGISTRY          # 7 comparisons: First, Last, DOB, SSN, Email, Phones, ZIP
    classification_config = ClassificationConfig(
        auto_merge_threshold=0.90,
        review_floor=0.50,
    )
    training = EMTraining(
        em_blocking_rules=[SSN_BLOCK, EMAIL_BLOCK, SOUNDEX_BLOCK],
        prior_rules=[SSN_BLOCK, EMAIL_BLOCK, DM_LAST_DOB_BLOCK],
        recall=0.80,
    )

    def prepare_model_input(self, df_clean): ...   # ~20 lines (Phones_array, shim cols)
    def build_settings(self):                      # ~10 lines
        return self.registry.build_all() + splink_boilerplate(...)
```

Thresholds `0.90 / 0.50` are the **baseline-tuned values**, not enhanced's `0.95 / 0.40`.

### 4.2 `FSEnhanced`

```python
class FSEnhanced(FSModel):
    model_name = "fs_splink_enhanced"
    classification_config = ClassificationConfig(
        auto_merge_threshold=0.95,
        review_floor=0.40,
        n_blocks_bump_threshold=2,
        n_blocks_bump_max_bits=4.0,
    )

    def __init__(self, include_address=True, u_max_pairs=1e6):
        self.registry = ENHANCED_REGISTRY if include_address else _BASE_REGISTRY
        self.training = EMTraining(...)

    def prepare_model_input(self, df_clean): ...   # identical to baseline + Address shim
    def build_settings(self):                      # calls registry.build_all() + priors
        settings = self.registry.build_all() + splink_boilerplate(...)
        apply_manual_priors(settings)              # from manual_priors.py
        return settings

    def train(self, linker, df_clean=None):
        super().train(linker, df_clean)
        # manual priors are locked at build_settings() time; no extra step needed here

    def classify(self, df_predictions, df_clean):
        out = super().classify(df_predictions)     # n_blocks bump + thresholds
        out = apply_vetoes(out, df_clean)          # from deterministic_vetoes.py
        vetoed = out["veto_reason"].notna()
        out.loc[vetoed, "classification_tier"] = "no_match"
        return out
```

`deterministic_vetoes.py` and `manual_priors.py` stay as-is. `FSEnhanced` wires them in via overrides — no hooks added to `FSModel`.

### 4.3 Comparison registries

**Baseline** (`BASELINE_REGISTRY`): 7 `ComparisonSpec` entries — `FirstNM`, `LastNM`, `BirthDT`, `SSN` (4-level), `Email` (4-level), `Phones_array` (array-intersect), `ZIP` (4-level). Order is preserved from the existing `build_settings()`.

**Enhanced** (`ENHANCED_REGISTRY`): all 7 baseline specs, plus `Household_discount` (new), `Address` (4-level, optional via `include_address`), plus E4 additions: JW<0.5 explicit-mismatch levels inserted into `FirstNM` + `LastNM`, SSN full-mismatch level inserted into `SSN`. The `with_added()` / `with_removed()` registry API composes these cleanly.

---

## 5. Locked decisions

These were decided before Phase E3 execution began. Do not relitigate during implementation.

**(a) Aggressive notebook trim.** Sections §16–20 (2-way rules + enhanced head-to-head, superseded by §21–26 3-way) and §8 (libpostal coverage audit, R2-specific) and §27 (per-comparison m/u diagnostic for enhanced_2) are removed in Phase E3-5. §21–26 (the 3-way head-to-head) is the canonical comparison artifact; §8 data is not needed post-R2; §27 content moves to `docs/Fellegi-Sunter-Enhanced_2.md`.

**(b) Tests rewritten against `FSModel` API.** No shim layer (`run_fs_baseline = lambda: ...`). The existing test files — `test_fellegi_sunter_baseline.py`, `test_fs_enhanced_e4_levels.py`, `test_fs_enhanced_e5_thresholds_bump.py`, `test_fs_enhanced_manual_priors.py`, plus their integration and regression counterparts — are rewritten (same phase that retires their target functional API), renamed to `test_fs_baseline.py` / `test_fs_enhanced_*.py`.

**(c) Enhanced extras stay as subclass overrides.** `FSModel` gains no `post_classify_hook` / `post_train_hook` / `post_predict_hook`. `FSEnhanced.classify()` calls `super().classify()` then `apply_vetoes()`; any future hook would live in the subclass. This keeps `fs_base.py` minimal and well-bounded.

**(d) Dead-code removal scope.** Deleted in Phase E3-0: `models/common/eval_schema.py` (0 Python importers — constants duplicated inline in runners) and the u-sweep diagnostic script in `models/experiments/fs_splink_baseline/` (one-off u-budget diagnostic, not in the production path). `models/common/synthetic_data.py` is kept (used by sandbox runners and the session-scoped `conftest.py` fixture).

**(e) No changes to `fs_splink_enhanced_2/`.** It is the reference shape. Touching it would risk regressions in the most recently validated model. It is only modified if Phase E3 surfaces a genuine base-class bug.

---

## 6. Acceptance criteria

All of the following must hold at the end of Phase E3-4 (last implementation phase before notebook trim + docs):

1. **All tests green.** `pytest -q` on the full suite, no skips beyond pre-existing ones.

2. **Eval-schema parquets byte-equivalent.** For both baseline and enhanced, run the synthetic sandbox runner pre- and post-refactor and diff on `(PATID_A, PATID_B, model_name, predicted_tier)` — must be empty. `score` may shift by at most `1e-6` due to floating-point reordering (same EM seed, same DuckDB version).

3. **No legacy module references in Python source.**

   ```bash
   rg "fellegi_sunter_baseline|fellegi_sunter_enhanced" --type py
   ```

   Must return zero hits (excluding `__pycache__` and any `.pyc` files).

4. **Contract validation passes on runner output.** Both `run_synthetic_baseline.py` and `run_synthetic_enhanced.py` complete without pandera errors and write a valid `ProbabilisticMatches` parquet.

5. **`FSEnhanced` veto coverage unchanged.** The regression test `test_fs_enhanced_vetoes.py` continues to pass: an SSN-conflict pair is forced to `no_match` regardless of `match_probability`.

---

## 7. Out of scope

The following are explicitly deferred and must not be addressed during Phase E3:

- **FS_Enhanced_3 scoping.** The next model is undefined; Phase E3 does not introduce hook points for it.
- **Stage 5 clustering.** Union of deterministic + probabilistic edges into `cluster_id`s. The `Edges` and `ClusterAssignments` contracts in `src/contracts.py` are not touched.
- **Base-class hook points for vetoes/priors.** The `post_classify_hook` / `post_train_hook` pattern is explicitly rejected (see §5c). `fs_base.py` is closed to extension via hooks.
- **Orchestrator changes (`src/pipeline.py`).** Stage 4 is wired to `run_fs_enhanced`; after Phase E3-3 it calls `FSEnhanced().run(...)` instead, but the pipeline boundary contracts (`NonMatches` in, `ProbabilisticMatches` out) are unchanged. No behavioral change to the orchestrator beyond the import swap.
- **Calibration / prior re-labeling.** `manual_priors.py` values are not revised in Phase E3. If post-refactor VM numbers diverge from pre-refactor, that is a regression to investigate — not a trigger for re-labeling.
- **Notebook §11 (3-way head-to-head injector).** Shipped in Phase E2-6; not re-touched in E3 unless the refactor changes runner output paths (which it should not).
