"""End-to-end eMPI pipeline orchestrator — the single production entry point.

Runs the stages **in process**, passing DataFrames stage-to-stage rather than
re-resolving "the latest file in the directory":

    raw ─► clean ─► blocking ─► deterministic rules
                                    ├─► matches (auto-merge) ─────────────┐
                                    └─► non-matches                       │
                                          │                               │
                                          ▼                               │
                                  Stage 4: FS matcher (if a model         │
                                  is active) — scores non-matches,        │
                                  emits ProbabilisticMatches (audit) +    │
                                  FSFeatures (candidates). AUDIT-ONLY.     │
                                          │                               │
                                          ▼                               │
                                  Stage 4.25: ML non-match gate — drops   │
                                  confident non-matches; only plausible   │
                                  pairs continue.                          │
                                          │                               │
                                          ▼                               │
                                  Stage 4.5: ML matcher (if a model       │
                                  is active) — pluggable model/features.  │
                                  Emits ClassificationResults              │
                                  (audit) + MLFeatures (candidates).       │
                                          │                               │
                                          ▼                               ▼
                                  Each classifier's auto_merge-tier   clustering (terminal)
                                  edges union into clustering ONLY       └─► cluster assignments
                                  when its `*_feeds_clustering` toggle
                                  is on (both default OFF — rules'
                                  matches always feed clustering).

Because one in-memory cleaned frame feeds every stage, the lineage mismatch the
standalone CLIs are vulnerable to cannot occur here. Every boundary is validated
against the contracts in `src/contracts.py`, and a single `run_id` ties all
artifacts together via a `RunManifest` written to `data/runs/`.

USAGE:
    python -m src.pipeline                         # uses settings.raw_input
    python -m src.pipeline --input data/raw/MDM_Population.csv
    python -m src.pipeline --run-id 20260603T120000Z   # override the run id

Stage 4 (the Fellegi-Sunter matcher, `src/models/fs_matcher/`) and Stage 4.5
(the pluggable ML matcher, `src/models/ml_matcher/`) each load a pre-trained
model artifact and score the deterministic rules' `non_matches` pool — both
are candidate + feature generators, structurally identical
(`src.models.base.PairClassifier`). Their output feeds clustering only when
`settings.fs_feeds_clustering` / `settings.ml_feeds_clustering` is explicitly
turned on (both default `False`, so out of the box clustering still runs on
the deterministic auto-merge edges only). If no active model is resolvable
for a stage, that stage is skipped (train + promote one with
`python -m src.models.fs_matcher.train --promote` /
`python -m src.models.ml_matcher.train --promote`).

Stage 4.25 (`src/models/nonmatch_gate/`) is the **non-match gate**: it decides
which of the `non_matches` are plausible enough to reach Stage 4.5 and discards
the rest as confident non-matches. That role used to belong to Stage 4's FS
`no_match` tier; FS is now audit-only. Setting `settings.gate_supersedes_fs`
to False restores the legacy FS gate, which is also the automatic fallback
when no gate model is active.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    Settings,
    configure_logging,
    settings as default_settings,
)
from src.contracts import (  # noqa: E402
    ArtifactRef,
    CandidatePairs,
    ClassificationResults,
    CleanedRecords,
    ClusterAssignments,
    Matches,
    NonMatches,
    ProbabilisticMatches,
    Rejects,
    RunManifest,
    TIER_HUMAN_REVIEW,
    TIER_NO_MATCH,
    assert_patid_coverage,
    validate,
    validate_fs_features,
    validate_ml_features,
    validate_pair_explanations,
)
from src.preprocessing.clean import _load as _read_raw, write_cleaned  # noqa: E402
from src.preprocessing.transformations import transform_dataframe  # noqa: E402
from src.preprocessing.stacked_blocking import run_stacked_blocking  # noqa: E402
from src.models.base import to_edges  # noqa: E402
from src.models.clustering import assign_clusters, build_cluster_assignments  # noqa: E402
from src.models.deterministic_rules import (  # noqa: E402
    apply_rules,
    classify_non_matches,
    get_match_stats,
)
# Every registry's resolve_active_model imports no heavy deps (json/pathlib
# only); the FS matcher (which pulls splink), the non-match gate and the ML
# matcher (lightgbm/joblib) are lazy-imported inside their stages.
from src.models.fs_matcher.registry import resolve_active_model as resolve_active_fs_model  # noqa: E402
from src.models.ml_matcher.registry import resolve_active_model as resolve_active_ml_model  # noqa: E402
from src.models.nonmatch_gate.registry import resolve_active_model as resolve_active_gate_model  # noqa: E402

logger = logging.getLogger("eMPI.pipeline")


# ── Small utilities ─────────────────────────────────────────────────────────
def _new_run_id() -> str:
    """A sortable, collision-resistant id for one pipeline run."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _artifact_ref(path: Path, df: pd.DataFrame, root: Path) -> ArtifactRef:
    return ArtifactRef(path=_rel(path, root), rows=len(df), sha256=_file_sha256(path))


def _write_explanations(
    explanations: pd.DataFrame | None,
    path: Path,
    settings: Settings,
    *,
    stage: str,
) -> Path | None:
    """Validate and persist a stage's per-pair SHAP frame; return its path.

    Returns None (with a log line, not an exception) when the stage produced
    nothing — a BYOM model that can't emit contributions is a legitimate
    state, and a missing explanation must never fail a run whose scores are
    perfectly good.
    """
    if explanations is None:
        logger.info(
            "%s — no explanations (model does not expose SHAP contributions)", stage,
        )
        return None
    validate_pair_explanations(explanations)
    explanations.to_parquet(path, index=False)
    logger.info(
        "%s — explanations for %d pairs → %s",
        stage, len(explanations), _rel(path, settings.project_root),
    )
    return path


def _fs_plausible_pool(non_matches: pd.DataFrame, eval_frame_fs: pd.DataFrame) -> pd.DataFrame:
    """LEGACY gate — keep only the `non_matches` rows FS did NOT rank
    `no_match` (i.e. FS `match_probability >= settings.fs_review_floor`).

    The FS matcher used to be the pipeline's non-match gate; Stage 4.25's
    `nonmatch_gate` model replaced it. This path still runs as the fallback
    when no gate model is active, or when `settings.gate_supersedes_fs` is
    turned off for a comparison run.

    The survivors are the *plausible* pairs — the same population the ML
    matcher was trained on (match ∪ ambiguous) — which the ML matcher then
    classifies as confident match (`auto_merge`) vs ambiguous
    (`human_review`). Passthrough columns (`source_blocks`/`n_blocks`) are
    preserved because the result comes from `non_matches`, not the FS frame.
    """
    plausible = eval_frame_fs.loc[
        eval_frame_fs["predicted_tier"] != TIER_NO_MATCH, ["PATID_A", "PATID_B"]
    ]
    return non_matches.merge(
        plausible, on=["PATID_A", "PATID_B"], how="inner"
    ).reset_index(drop=True)


# ── Orchestration ────────────────────────────────────────────────────────────
def run_pipeline(
    raw_input: Path | None = None,
    settings: Settings = default_settings,
    run_id: str | None = None,
    cleaned_input: Path | None = None,
) -> RunManifest:
    """Run clean → block → rules for one input and return the run manifest.

    `cleaned_input` skips Stage 1 and starts from an already-cleaned Parquet
    frame instead. That is not a shortcut for production runs — it exists so an
    evaluation harness can push a record set that is *already* in cleaned form
    (the synthetic label file's `*_l`/`*_r` fields, say) through the real
    Stages 2-5 rather than a reimplementation of them. Re-running the cleaning
    rules over already-cleaned values would be lossy, and reimplementing the
    stages would mean evaluating a copy of the pipeline instead of the pipeline.
    The frame still has to satisfy `CleanedRecords`, and the manifest records it
    as the run's input so lineage stays intact.
    """
    configure_logging(settings)
    run_id = run_id or _new_run_id()
    from_cleaned = cleaned_input is not None
    if from_cleaned:
        raw_input = Path(cleaned_input)
    else:
        raw_input = Path(raw_input) if raw_input is not None else settings.raw_input
    settings.ensure_dirs()
    started = datetime.utcnow()

    logger.info("=" * 64)
    logger.info("eMPI pipeline run %s starting", run_id)
    logger.info("Input: %s%s", raw_input, " (pre-cleaned)" if from_cleaned else "")
    logger.info("=" * 64)

    if not raw_input.exists():
        raise FileNotFoundError(f"Raw input not found: {raw_input}")

    # ── Stage 1: clean ────────────────────────────────────────────────────
    if from_cleaned:
        cleaned = pd.read_parquet(raw_input)
        validate(cleaned, CleanedRecords)
        n_valid = int(cleaned["valid_record"].sum())
        logger.info(
            "[1/7] CLEAN — skipped, starting from a pre-cleaned frame: "
            "%d rows (%d valid)", len(cleaned), n_valid,
        )
    else:
        raw_df = _read_raw(raw_input)
        cleaned = transform_dataframe(raw_df)
        validate(cleaned, CleanedRecords)
        n_valid = int(cleaned["valid_record"].sum())
        logger.info(
            "[1/7] CLEAN — %d raw → %d cleaned rows (%d valid)",
            len(raw_df), len(cleaned), n_valid,
        )
    cleaned_path = settings.processed_dir / f"{settings.cleaned_stem}_{run_id}.parquet"
    write_cleaned(cleaned, cleaned_path)

    # ── Stage 2: block (stacked: 8-block ∪ q-gram → CNP/ARCS prune) ────────
    candidate_pairs = run_stacked_blocking(cleaned)
    validate(candidate_pairs, CandidatePairs)
    logger.info("[2/7] BLOCK — %d candidate pairs", len(candidate_pairs))
    pairs_path = settings.blocking_dir / f"candidate_pairs_{run_id}.parquet"
    candidate_pairs.to_parquet(pairs_path, index=False)

    # ── Stage 3: deterministic rules ──────────────────────────────────────
    assert_patid_coverage(candidate_pairs, cleaned)  # guaranteed in-process; guard anyway
    confirmed = apply_rules(
        candidate_pairs, cleaned, ssn_fanout_threshold=settings.ssn_fanout_threshold
    )
    # Every rule is auto-merge — a rule fires only when it will confirm the pair
    # outright, so there is no tier split here. Pairs no rule confirmed are
    # decided by `classify_non_matches` below.
    matches = confirmed.reset_index(drop=True)
    if not matches.empty:
        clusters = assign_clusters(matches)
        matches = matches.copy()
        matches["cluster_id"] = matches["PATID_A"].map(clusters)
    validate(matches, Matches)
    # Three-way split. `confirmed` is removed from the contradiction split — a
    # rule-confirmed pair is already decided and is never reject-scored.
    decided = classify_non_matches(candidate_pairs, confirmed, cleaned)
    pair_cols = list(candidate_pairs.columns)
    non_matches = (
        decided[decided["decision"] == TIER_HUMAN_REVIEW][pair_cols]
        .reset_index(drop=True)
    )
    rejects = decided[decided["decision"] == TIER_NO_MATCH].reset_index(drop=True)
    validate(non_matches, NonMatches)
    validate(rejects, Rejects)
    stats = get_match_stats(matches, n_records=len(cleaned), decided=decided)
    logger.info(
        "[3/7] RULES — %d auto-merge, %d review, %d reject, %d clusters",
        len(matches), len(non_matches), len(rejects),
        stats.get("n_clusters", 0),
    )
    matches_path = settings.auto_merge_dir / f"matches_{run_id}.parquet"
    matches.to_parquet(matches_path, index=False)
    non_matches_path = settings.non_matches_dir / f"non_matches_{run_id}.parquet"
    non_matches.to_parquet(non_matches_path, index=False)
    rejects_path = settings.no_match_dir / f"rejects_{run_id}.parquet"
    rejects.to_parquet(rejects_path, index=False)

    # ── Stage 4: Fellegi-Sunter matcher (candidate/feature generator) ──────
    # Scores the rules' non_matches pool with the pre-trained active FS model.
    # Emits a full ProbabilisticMatches AUDIT frame and the candidate-filtered
    # FSFeatures parquet, plus the uniform ClassificationResults frame shared
    # with every classifier stage. Feeds clustering only when
    # settings.fs_feeds_clustering is on (Stage 5 below). Skipped (with a
    # clear log) when no active model is resolvable or the non_matches pool
    # is empty — so a pre-deployment pipeline still runs.
    matches_model_path: Path | None = None
    fs_features_path: Path | None = None
    eval_frame_fs: pd.DataFrame | None = None
    active_fs_model = resolve_active_fs_model(settings)
    if active_fs_model is None:
        logger.info(
            "[4/7] MODEL(FS) — skipped (no active FS model; train + promote one "
            "with `python -m src.models.fs_matcher.train --promote`)"
        )
    elif non_matches.empty:
        logger.info("[4/7] MODEL(FS) — skipped (no non-match pairs to score)")
    else:
        logger.info(
            "[4/7] MODEL(FS) — scoring %d non-match pairs with FS model %s...",
            len(non_matches), active_fs_model.name,
        )
        # Lazy import keeps splink/duckdb out of the import path when Stage 4 is
        # skipped (registry.resolve_active_model above pulls no heavy deps).
        from src.models.fs_matcher.matcher import (
            FSMatcher,
            classification_config_from_settings,
        )
        _model = FSMatcher(classification_config=classification_config_from_settings(settings))
        _trained = FSMatcher.load_settings(active_fs_model)
        classified = _model.score(non_matches, cleaned, _trained)

        # Uniform 5-col shape (src.contracts.ClassificationResults), shared with
        # deterministic rules and the ML matcher — feeds the optional
        # fs_feeds_clustering union in Stage 5 below.
        eval_frame_fs = _model.to_evaluation_schema(classified)
        validate(eval_frame_fs, ClassificationResults)

        prob_matches = _model.to_probabilistic_matches(classified)
        validate(prob_matches, ProbabilisticMatches)
        matches_model_path = settings.fs_output_dir / f"matches_model_{run_id}.parquet"
        prob_matches.to_parquet(matches_model_path, index=False)

        fs_features = _model.to_fs_features(classified, candidates_only=True)
        validate_fs_features(fs_features)
        fs_features_path = settings.fs_output_dir / f"fs_features_{run_id}.parquet"
        fs_features.to_parquet(fs_features_path, index=False)

        tier_counts = {k: int(v) for k, v in
                       prob_matches["classification_tier"].value_counts().items()}
        logger.info(
            "[4/7] MODEL(FS) — tiers %s → %d candidates → %s",
            tier_counts, len(fs_features), _rel(fs_features_path, settings.project_root),
        )

    # ── Stage 4.25: ML non-match gate ─────────────────────────────────────
    # The pipeline's confident-non-match gate: scores the rules' non_matches
    # pool with P(plausible) and discards everything below
    # settings.gate_threshold. Only the plausible survivors reach the ML
    # matcher. This replaced the FS matcher's no_match tier as the gate; set
    # settings.gate_supersedes_fs=False to fall back to the legacy FS gate.
    # With neither available the pool passes through ungated (with a warning).
    gate_results_path: Path | None = None
    gate_explanations_path: Path | None = None
    eval_frame_gate: pd.DataFrame | None = None
    gate_explanations: pd.DataFrame | None = None
    ml_input_pool: pd.DataFrame = non_matches
    active_gate_model = (
        resolve_active_gate_model(settings) if settings.gate_supersedes_fs else None
    )
    if non_matches.empty:
        logger.info("[5/7] GATE — skipped (no non-match pairs to gate)")
    elif active_gate_model is None:
        if not settings.gate_supersedes_fs:
            logger.info("[5/7] GATE — disabled (gate_supersedes_fs=False)")
        else:
            logger.info(
                "[5/7] GATE — no active gate model (train + promote one from "
                "notebooks/ml_model/confident_nonmatch/)"
            )
        if eval_frame_fs is not None:
            ml_input_pool = _fs_plausible_pool(non_matches, eval_frame_fs)
            logger.info(
                "[5/7] GATE — falling back to the FS gate: %d/%d pairs plausible, "
                "dropped %d confident non-matches",
                len(ml_input_pool), len(non_matches),
                len(non_matches) - len(ml_input_pool),
            )
        else:
            logger.warning(
                "[5/7] GATE — no gate available (no gate model, no FS model); the ML "
                "matcher will score the full non_matches pool, which still contains "
                "true non-matches"
            )
    else:
        logger.info(
            "[5/7] GATE — gating %d non-match pairs with gate model %s...",
            len(non_matches), active_gate_model.name,
        )
        # Lazy import mirrors Stages 4/6 — keeps lightgbm/joblib off the import
        # path when the gate is skipped (the registry pulls no heavy deps).
        from src.models.nonmatch_gate.gate import NonMatchGate
        from src.models.nonmatch_gate.registry import load_model_artifact as load_gate_artifact

        _gate = NonMatchGate(
            model=load_gate_artifact(active_gate_model),
            threshold=settings.gate_threshold,
        )
        ml_input_pool, eval_frame_gate, gate_explanations = _gate.apply(
            non_matches, cleaned, explain=settings.explanations_enabled,
        )
        validate(eval_frame_gate, ClassificationResults)
        gate_results_path = settings.gate_output_dir / f"gate_results_{run_id}.parquet"
        eval_frame_gate.to_parquet(gate_results_path, index=False)
        logger.info(
            "[5/7] GATE — %d/%d pairs plausible, dropped %d confident non-matches → %s",
            len(ml_input_pool), len(non_matches), len(non_matches) - len(ml_input_pool),
            _rel(gate_results_path, settings.project_root),
        )
        if gate_explanations is not None:
            # Stamp the serving artifact onto the frame so it is fully
            # self-describing — the endpoint can say which model explained
            # a decision without consulting a registry that may have moved on.
            gate_explanations["model_file"] = active_gate_model.name
        gate_explanations_path = _write_explanations(
            gate_explanations, settings.gate_output_dir / f"gate_explanations_{run_id}.parquet",
            settings, stage="[5/7] GATE",
        )

    # ── Stage 4.5: pluggable ML matcher (candidate/feature generator) ──────
    # Scores the gated pool (ml_input_pool = the pairs the gate deemed
    # plausible), optionally enriched with Stage 4's FSFeatures. With the gate
    # removing confident non-matches upstream, this stage is a 2-tier
    # classifier — confident match (auto_merge) vs ambiguous (human_review).
    # It emits no no_match tier by construction: discarding is the gate's job,
    # and the gate is the only stage that records its drops. Feeds
    # clustering only when settings.ml_feeds_clustering is on. Skipped when no
    # active model resolves or the pool is empty.
    matches_ml_path: Path | None = None
    ml_features_path: Path | None = None
    ml_explanations_path: Path | None = None
    eval_frame_ml: pd.DataFrame | None = None
    ml_explanations: pd.DataFrame | None = None
    active_ml_model = resolve_active_ml_model(settings)
    if active_ml_model is None:
        logger.info(
            "[6/7] MODEL(ML) — skipped (no active ML model; train + promote one "
            "with `python -m src.models.ml_matcher.train --promote`)"
        )
    elif ml_input_pool.empty:
        logger.info("[6/7] MODEL(ML) — skipped (no pairs to score)")
    else:
        logger.info(
            "[6/7] MODEL(ML) — scoring %d plausible pairs with ML model %s...",
            len(ml_input_pool), active_ml_model.name,
        )
        # Lazy import mirrors Stage 4 — keeps whatever the implementer's model
        # framework needs (torch/xgboost/...) out of the import path when
        # Stage 4.5 is skipped.
        from src.models.ml_matcher.base import MLClassificationConfig
        from src.models.ml_matcher.matcher import MLMatcher
        # BYOM artifact loading is the implementer's extension point — see
        # docs/ML-Matcher-Integration-Guide.md. load_model_artifact deserializes
        # whatever active.json points at; FeatureBuilderV5 is the matching BYOF
        # implementation for the served LightGBM v5 model (the same 12 features
        # v3/v4 used, so the gate shares the builder). Both are imported lazily
        # so Stage 4.5's deps stay off the import path when it's skipped.
        from src.models.ml_matcher.lightgbm_v5 import FeatureBuilderV5
        from src.models.ml_matcher.registry import load_model_artifact

        _ml_model = MLMatcher(
            model=load_model_artifact(active_ml_model),
            feature_builder=FeatureBuilderV5(),
            classification_config=MLClassificationConfig(
                auto_merge_threshold=settings.ml_auto_merge_threshold,
            ),
        )
        classified_ml = _ml_model.score(
            ml_input_pool, cleaned,
            fs_features=fs_features if fs_features_path is not None else None,
        )

        # Uniform 5-col shape, shared with deterministic rules and the FS
        # matcher — feeds the optional ml_feeds_clustering union in Stage 5.
        eval_frame_ml = _ml_model.to_evaluation_schema(classified_ml)
        validate(eval_frame_ml, ClassificationResults)
        matches_ml_path = settings.ml_output_dir / f"matches_ml_{run_id}.parquet"
        eval_frame_ml.to_parquet(matches_ml_path, index=False)

        ml_features = _ml_model.to_ml_features(classified_ml)
        validate_ml_features(ml_features)
        ml_features_path = settings.ml_output_dir / f"ml_features_{run_id}.parquet"
        ml_features.to_parquet(ml_features_path, index=False)

        tier_counts_ml = {k: int(v) for k, v in
                          eval_frame_ml["predicted_tier"].value_counts().items()}
        logger.info(
            "[6/7] MODEL(ML) — tiers %s → %d candidates → %s",
            tier_counts_ml, len(ml_features), _rel(ml_features_path, settings.project_root),
        )
        if settings.explanations_enabled:
            ml_explanations = _ml_model.explain(classified_ml)
            if ml_explanations is not None:
                ml_explanations["model_file"] = active_ml_model.name
            ml_explanations_path = _write_explanations(
                ml_explanations,
                settings.ml_output_dir / f"ml_explanations_{run_id}.parquet",
                settings, stage="[6/7] MODEL(ML)",
            )

    # ── Stage 5: cluster (terminal) — connected components over the deterministic
    #    auto-merge edges, plus any classifier stage's auto_merge-tier edges
    #    whose `*_feeds_clustering` toggle is on. Both toggles default off, so
    #    out of the box this is byte-identical to clustering on `matches` alone.
    cluster_edges = [matches[["PATID_A", "PATID_B"]]]
    if settings.fs_feeds_clustering and eval_frame_fs is not None:
        cluster_edges.append(to_edges(eval_frame_fs, match_source="model")[["PATID_A", "PATID_B"]])
    if settings.ml_feeds_clustering and eval_frame_ml is not None:
        cluster_edges.append(to_edges(eval_frame_ml, match_source="ml")[["PATID_A", "PATID_B"]])
    clustering_input = (
        pd.concat(cluster_edges, ignore_index=True).drop_duplicates()
        if len(cluster_edges) > 1 else cluster_edges[0]
    )
    cluster_assignments = build_cluster_assignments(clustering_input, cleaned)
    validate(cluster_assignments, ClusterAssignments)
    n_total_clusters = int(cluster_assignments["cluster_id"].nunique())
    logger.info(
        "[7/7] CLUSTER — %d records → %d clusters (incl. singletons)",
        len(cluster_assignments), n_total_clusters,
    )
    clusters_path = settings.clusters_dir / f"cluster_assignments_{run_id}.parquet"
    cluster_assignments.to_parquet(clusters_path, index=False)

    # ── Manifest ──────────────────────────────────────────────────────────
    root = settings.project_root
    manifest = RunManifest(
        run_id=run_id,
        created_utc=started.isoformat() + "Z",
        git_sha=_current_git_sha(),
        # When starting pre-cleaned, the input file *is* the cleaned frame, so
        # its row count is the cleaned row count — there is no raw stage to
        # count. The path + sha still pin the run's lineage to a real file.
        raw_input=_artifact_ref(raw_input, cleaned if from_cleaned else raw_df, root),
        cleaned=_artifact_ref(cleaned_path, cleaned, root),
        candidate_pairs=_artifact_ref(pairs_path, candidate_pairs, root),
        matches=_artifact_ref(matches_path, matches, root),
        non_matches=_artifact_ref(non_matches_path, non_matches, root),
        rejects=_artifact_ref(rejects_path, rejects, root),
        clusters=_artifact_ref(clusters_path, cluster_assignments, root),
        matches_model=(
            _artifact_ref(matches_model_path, prob_matches, root)
            if matches_model_path is not None else None
        ),
        fs_features=(
            _artifact_ref(fs_features_path, fs_features, root)
            if fs_features_path is not None else None
        ),
        gate_results=(
            _artifact_ref(gate_results_path, eval_frame_gate, root)
            if gate_results_path is not None else None
        ),
        gate_explanations=(
            _artifact_ref(gate_explanations_path, gate_explanations, root)
            if gate_explanations_path is not None else None
        ),
        matches_ml=(
            _artifact_ref(matches_ml_path, eval_frame_ml, root)
            if matches_ml_path is not None else None
        ),
        ml_features=(
            _artifact_ref(ml_features_path, ml_features, root)
            if ml_features_path is not None else None
        ),
        ml_explanations=(
            _artifact_ref(ml_explanations_path, ml_explanations, root)
            if ml_explanations_path is not None else None
        ),
        counts={
            "raw_rows": len(cleaned) if from_cleaned else len(raw_df),
            "cleaned_rows": len(cleaned),
            "valid_records": n_valid,
            "candidate_pairs": len(candidate_pairs),
            "matches": len(matches),
            "non_matches": len(non_matches),
            "gate_plausible": len(ml_input_pool),
            "gate_dropped": len(non_matches) - len(ml_input_pool),
            "rejects": len(rejects),
            "clusters": int(stats.get("n_clusters", 0)),
            "total_clusters": n_total_clusters,
        },
    )
    manifest_path = settings.runs_dir / f"run_{run_id}.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    elapsed = (datetime.utcnow() - started).total_seconds()
    logger.info("=" * 64)
    logger.info("Run %s complete in %.1fs", run_id, elapsed)
    logger.info("Manifest: %s", _rel(manifest_path, root))
    logger.info("=" * 64)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="eMPI end-to-end pipeline")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Raw CSV/XLSX to process (default: settings.raw_input).",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Override the generated run id (default: UTC timestamp).",
    )
    parser.add_argument(
        "--cleaned", type=Path, default=None,
        help="Start from an already-cleaned Parquet frame, skipping Stage 1. "
             "For evaluation harnesses feeding pre-cleaned records (see "
             "src/evaluation/synthetic.py); mutually exclusive with --input.",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override EMPI_LOG_LEVEL for this run (DEBUG/INFO/WARNING/...).",
    )
    args = parser.parse_args()
    if args.cleaned is not None and args.input is not None:
        parser.error("--input and --cleaned are mutually exclusive.")
    configure_logging(level=args.log_level)
    run_pipeline(raw_input=args.input, run_id=args.run_id, cleaned_input=args.cleaned)


if __name__ == "__main__":
    main()
