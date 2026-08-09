# empi-model-training

Azure ML training pipeline for the eMPI record-matching system's three
classifier models — the Splink-based Fellegi-Sunter (FS) matcher, the
LightGBM v5 confident-match classifier, and the non-match gate. Lives inside
`UofC-EMPI/` and is tracked/pushed as part of that same repo, but is a
**logically independent** codebase — see "Independent by design" below.

**Scope:** training + experiment tracking + model registry only. Serving is
unchanged — the `empi-service` backend still loads a model artifact from its
`models/fs` / `models/ml` / `models/nonmatch_gate` directories (or the
`empi-models` Azure Files share in Azure) and scores in-process. There are
no Azure ML managed online endpoints here.

## Runs fully locally — Azure ML is optional

All three training scripts are plain Python CLIs with no Azure dependency in
their core logic:

```bash
uv sync

uv run python -m empi_model_training.training.fs_train \
    --cleaned-index path/to/cleaned.parquet \
    --silver-labels path/to/silver_labels.csv \
    --promote

uv run python -m empi_model_training.training.lightgbm_train \
    --cleaned-index path/to/cleaned.parquet \
    --gold-labels path/to/gold_labels.csv \
    --promote

uv run python -m empi_model_training.training.nonmatch_gate_train \
    --cleaned-index path/to/cleaned.parquet \
    --gold-labels path/to/gold_labels.csv \
    --promote
```

MLflow tracking works locally too (writes to `./mlruns/`, no server needed)
— model *registry* registration is skipped locally and only happens when
actually running under Azure ML (detected via the tracking URI scheme), since
a plain local run has no database-backed registry to register into.

`uv run pytest` runs all three scripts end-to-end against small synthetic
fixtures (`tests/conftest.py`) — a real Splink fit and two real LightGBM
fits, not mocked, so an API-usage mistake in any training path fails a test
locally before it ever reaches Azure ML.

The `submit.py` / `components/` / `utils/` modules (below) are strictly
additive on top of this — submitting to Azure ML, registering environments,
etc. None of it is required to train or test locally.

## Independent by design

`empi-service`'s own `src/models/fs_matcher/train.py` and
`src/models/ml_matcher/train.py` (the latter a stub) are **left untouched**,
and this repo does **not** import or call into `empi-service`'s code — a
deliberate project decision (see `CLAUDE.md`), not an oversight. Concretely:

- `training/fs_train.py` is an **independent reimplementation** of the
  comparison structure and supervised training procedure documented in
  `empi-service/docs/FS-Matcher-Production-Guide.md` — faithful enough that
  the resulting model JSON loads fine in empi-service's own
  `FSMatcher.load_settings()`/`score()`, but the code itself is maintained
  separately in each repo.
- `training/lightgbm_train.py` is an **independent reimplementation** of
  both the feature engineering (`empi-service/src/models/ml_matcher/
  lightgbm_v5.py`'s `FeatureBuilderV5`) and the training loop (previously
  only a notebook, `empi-service/notebooks/ml_model/confident_match/
  pair_classifier_lightgbm_confident_match_v5.ipynb`). Trains on the whole
  gold-labeled pool (no population filter) with class 1 = confident match,
  matching `DirectMatchAdapter`'s pass-through serving semantics — the v5
  generation of this model; see `empi-service/docs/ML-Model-LightGBM-v5.md`
  for what changed from the earlier v3/v4 generation and why.
- `training/nonmatch_gate_train.py` is an **independent reimplementation**
  of the pipeline's Stage-4.25 non-match gate, documented in
  `empi-service/docs/Nonmatch-Gate-Guide.md` and previously only a notebook
  (`empi-service/notebooks/ml_model/confident_nonmatch/
  pair_classifier_lightgbm_nonmatch_gate_v1.ipynb`). Shares
  `lightgbm_train.py`'s feature engineering (duplicated, not imported across
  the two training scripts — same reasoning as the service's own choice to
  duplicate `FeatureBuilderV5` usage rather than share code across model
  packages) but trains a **separate model** on a different target: class 1
  = plausible (match ∪ ambiguous), serialized as a raw fitted classifier
  with no adapter wrapper. Has its own model store (`models/nonmatch_gate/`)
  and is never confused with the ML matcher in `registry_utils.py`'s
  deploy-gate (`gate_pr` reads a different metric shape than
  `auto_merge_pr` — see that module's docstring).
- `training/registry_utils.py` is an independent, smaller copy of the
  active-model/deploy-gate pattern in `empi-service/src/models/
  {fs_matcher,ml_matcher,nonmatch_gate}/registry.py` — same file-naming
  convention (`*_model_<ts>.json|.pkl` / `nonmatch_gate_<ts>.pkl`,
  `.meta.json`, `active.json`) so a promoted artifact drops into the serving
  share unchanged, but not a shared import.

If something in `empi-service` changes in a way that should be reflected
here (a new field in the cleaned-records contract, a threshold default),
that's a manual, deliberate update in this repo — never an automatic one.

## Connecting to Azure ML

Set these from `terraform output` in `UofC-EMPI/terraform`:

```bash
export AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
export AZURE_RESOURCE_GROUP=$(terraform -chdir=../terraform output -raw resource_group_name)
export AZURE_ML_WORKSPACE_NAME=$(terraform -chdir=../terraform output -raw ml_workspace_name)
```

### Data movement (read before submitting a job)

Azure ML's compute uses its own managed network (isolated from the app's
VNet — see `terraform/ml_workspace.tf` in `UofC-EMPI`), so it has no direct
network path to the app's private storage. Moving training inputs in and
the promoted model artifact out is a **deliberate, explicit step**, not
automatic — arguably the correct posture anyway once PHI is involved: data
crossing a trust boundary should be auditable, not silent.

1. Stage the training inputs (cleaned records, silver/gold labels, and
   optionally `reviewer_labels/*.csv` — see below) locally — e.g. via the
   same VNet-connectivity path already documented for `az webapp ssh` in
   `UofC-EMPI/terraform/README.md`.
2. Register them as versioned Data assets:
   ```bash
   uv run python -m empi_model_training.utils.register_dataset \
       --name empi-cleaned-records --version 2026.07.25 \
       --path ./staging/cleaned.parquet --type uri_file
   ```
3. Register (or update) the training environment once:
   ```bash
   uv run python -m empi_model_training.utils.register_environment
   ```
4. Submit (`--job` is one of `fs`, `lightgbm`, `gate`):
   ```bash
   uv run python -m empi_model_training.submit --job fs \
       --cleaned-index azureml:empi-cleaned-records:2026.07.25 \
       --labels azureml:empi-silver-labels:2026.07.25 \
       --compute cpu-cluster --promote
   ```
   `submit.py` runs preflight checks first (workspace/compute/environment/
   dataset all resolve) and refuses to submit otherwise — see
   `utils/preflight.py`.
5. Once a run's metrics have been reviewed and promoted to champion (see
   `UofC-EMPI/.github/workflows/promote-model.yml` — this repo has no
   `.github/` of its own; GitHub only reads workflows from the outer repo's
   root), copy the resulting artifact into the serving share the same way —
   an explicit step, not automatic.

### Reviewer-confirmed labels (optional third input)

`training/fs_train.py` and `training/lightgbm_train.py` both take an
optional `--reviewer-labels PATH` — labels derived from reviewer actions in
the dashboard (`UofC-EMPI/empi-service/scripts/export_reviewer_labels.py`),
higher-trust than silver labels or gold labels since every row traces back
to a live human decision, not a proxy or a separate labeling pass. Wins
over the primary label source for any pair present in both. Stage and
register it as a versioned Data asset exactly like silver/gold labels
(steps 1-2 above), then pass it through when invoking the script directly:

```bash
uv run python -m empi_model_training.training.fs_train \
    --cleaned-index ./staging/cleaned.parquet \
    --silver-labels ./staging/silver_labels.csv \
    --reviewer-labels ./staging/reviewer_labels.csv \
    --model-dir models/fs
```

**Not yet wired into `submit.py`** — that entry point takes a single
`--labels` reference, so `--reviewer-labels` currently only works for local
or direct-CLI training runs, not ones submitted through `submit.py` to
Azure ML. Extending `submit.py` and the AML component definitions
(`components/fs_train_component.py`, `components/lightgbm_train_component.py`,
`components/nonmatch_gate_train_component.py`) to accept a second labels
input is a natural follow-up, not done here.

## Commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Layout

```
src/empi_model_training/
  submit.py               # unified Azure ML job submission entry point
  training/                # the actual training logic -- runs with zero Azure dependency
    fs_train.py
    lightgbm_train.py
    nonmatch_gate_train.py
    registry_utils.py
  components/               # Azure ML component definitions (wrap training/ CLIs unchanged)
    fs_train_component.py
    lightgbm_train_component.py
    nonmatch_gate_train_component.py
  utils/                     # Azure ML plumbing: connecting, registering, preflight checks
    azure_client.py
    preflight.py
    register_environment.py
    register_dataset.py
environment.yml              # conda spec for the Azure ML training environment
tests/
  conftest.py                 # synthetic (non-PHI) fixtures for real end-to-end smoke tests
  test_fs_train.py
  test_lightgbm_train.py
  test_nonmatch_gate_train.py
```
