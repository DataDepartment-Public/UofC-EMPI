"""Evaluation execution script — one report per pipeline run/deployment.

Runs the full pipeline (clean → block → rules) and then both evaluation suites,
bundling everything an operator needs to judge a release into a **single JSON
artifact** plus a human-readable text summary:

  * **config**        — the effective :class:`~src.config.Settings`
                        (thresholds, paths, knobs) for this run.
  * **environment**   — Python version, platform, and key package versions, so
                        the run is reproducible.
  * **manifest**      — the :class:`~src.contracts.RunManifest` lineage record
                        (input hash, per-stage row counts, artifact checksums).
  * **rule_eval**     — deterministic-rule precision proxy, agreement profile,
                        confidence calibration, suspicious typology, and the
                        false-negative estimate (see ``rule_eval``).
  * **blocking_eval** — blocking recall against rule-confirmed matches and the
                        per-rule miss breakdown (see ``blocking_eval``).

PHI: the artifact is aggregate-only. We deliberately drop the SSN/email
fan-out *value* lists and the missed/adjudicated *example rows* — those carry
identifiers — keeping only counts and rates.

LOCATION:
    src/evaluation/run_eval.py

USAGE:
    From the project root (UofC-EMPI/):
        # Run the pipeline AND evaluate, writing data/runs/eval_<run_id>.json
        python src/evaluation/run_eval.py --input data/raw/MDM_Population.csv

        # Evaluate an existing run without re-running the pipeline
        python src/evaluation/run_eval.py --run-id <run_id> --evaluate-only

        # Faster: skip the (slow) blocking-recall eval
        python src/evaluation/run_eval.py --skip-blocking-eval

    Also importable / runnable as a module:
        python -m src.evaluation.run_eval --input data/raw/MDM_Population.csv

OUTPUT:
    eval_<run_id>.json  — the full machine-readable report
    eval_<run_id>.txt   — a human-readable summary
    both written to settings.runs_dir (data/runs/).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# ── Path resolution ───────────────────────────────────────────────────────────
# This file lives at src/evaluation/run_eval.py; the project root is two levels
# up. Adding it to sys.path lets `from src...` resolve when invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    Settings,
    configure_logging,
    settings as default_settings,
)
from src.contracts import RunManifest  # noqa: E402
from src.evaluation.blocking_eval import (  # noqa: E402
    BlockingEvalReport,
    evaluate_blocking,
)
from src.evaluation.rule_eval import (  # noqa: E402
    RuleEvalReport,
    evaluate as evaluate_rules,
)
from src.models.deterministic_rules import RULES  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

#: Packages whose versions are worth pinning in the evaluation record.
_TRACKED_PACKAGES = (
    "pandas", "numpy", "pandera", "pydantic", "pydantic-settings",
    "jellyfish", "phonetics", "pyarrow", "python-stdnum",
)


# ── Serialization helpers ─────────────────────────────────────────────────────
def _records(df: pd.DataFrame | None) -> list[dict]:
    """A DataFrame as JSON-friendly records (``[]`` when empty/None)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    return df.to_dict(orient="records")


def _aggregate_fanout(fanout: dict) -> dict:
    """Fan-out summary without the ``top`` value list (which contains PHI)."""
    return {k: v for k, v in fanout.items() if k != "top"}


def _serialize_rule_eval(report: RuleEvalReport) -> dict:
    """RuleEvalReport → JSON-safe dict (aggregates only, no example rows)."""
    return {
        "headline": report.headline,
        "precision": _records(report.precision),
        "agreement": _records(report.agreement),
        "calibration": _records(report.calibration),
        "overlap": (
            report.overlap.to_dict(orient="index")
            if not report.overlap.empty else {}
        ),
        "suspicious": report.suspicious,
        "false_negatives": report.false_negatives,
        "ssn_fanout": _aggregate_fanout(report.ssn_fanout),
        "email_fanout": _aggregate_fanout(report.email_fanout),
        "clusters": report.clusters,
    }


def _serialize_blocking_eval(report: BlockingEvalReport) -> dict:
    """BlockingEvalReport → JSON-safe dict (no missed-pair example rows)."""
    return {
        "headline": report.headline,
        "recall_pct": report.recall,
        "missed_by_rule": report.missed_by_rule,
        "hits_by_block": report.hits_by_block,
    }


def _environment_info() -> dict:
    """Python, platform, and tracked-package versions for reproducibility."""
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


# ── Report ────────────────────────────────────────────────────────────────────
@dataclass
class EvalReport:
    """Everything describing one pipeline run/deployment, ready to serialize."""

    run_id: str
    created_utc: str
    git_sha: str | None
    config: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    rule_eval: dict = field(default_factory=dict)
    blocking_eval: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_text(self) -> str:
        L = ["=" * 64, "  eMPI EVALUATION REPORT", "=" * 64]
        L.append(f"  run_id        {self.run_id}")
        L.append(f"  created_utc   {self.created_utc}")
        L.append(f"  git_sha       {self.git_sha or 'n/a'}")
        L.append(f"  python        {self.environment.get('python', 'n/a')}")
        L += ["", "  STAGE COUNTS"]
        L += [f"    {k:<20}{v}" for k, v in self.counts.items()]
        if self.rule_eval:
            L += ["", "  RULES — HEADLINE"]
            L += [f"    {k:<20}{v}"
                  for k, v in self.rule_eval.get("headline", {}).items()]
            prec = self.rule_eval.get("precision", [])
            if prec:
                L += ["", "  RULES — PRECISION PROXY (adjudicator, not ground truth)"]
                for row in prec:
                    L.append(
                        f"    {row.get('rule', '?'):<18}"
                        f"precision={row.get('precision_pct')}% "
                        f"(n={row.get('decided')})"
                    )
            fn = self.rule_eval.get("false_negatives", {})
            if fn:
                L.append(
                    f"    missed-match rate (rejected sample): "
                    f"{fn.get('missed_match_rate_pct')}%"
                )
        if self.blocking_eval:
            L += ["", "  BLOCKING — RECALL (rules as ground truth)"]
            L.append(f"    recall              {self.blocking_eval.get('recall_pct')}%")
            mbr = self.blocking_eval.get("missed_by_rule", {})
            if mbr:
                L += ["    missed by rule:"]
                L += [f"      {r:<18}{c}" for r, c in mbr.items()]
        L += ["", "  NOTE: precision/recall are vs the rule-based proxy, not "
              "labeled truth.", "=" * 64]
        return "\n".join(L)


# ── Loading / orchestration ───────────────────────────────────────────────────
def _load_artifacts(
    run_id: str, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load the cleaned / matches / non-matches frames for a finished run."""
    cleaned = pd.read_parquet(
        settings.processed_dir / f"{settings.cleaned_stem}_{run_id}.parquet"
    )
    matches = pd.read_parquet(settings.matches_dir / f"matches_{run_id}.parquet")
    nm_path = settings.non_matches_dir / f"non_matches_{run_id}.parquet"
    non_matches = pd.read_parquet(nm_path) if nm_path.exists() else None
    return cleaned, matches, non_matches


def build_eval_report(
    manifest: RunManifest,
    settings: Settings = default_settings,
    *,
    skip_rule_eval: bool = False,
    skip_blocking_eval: bool = False,
    blocking_method: str | None = None,
    blocking_sample: int | None = None,
    adj_sample: int | None = None,
    seed: int | None = None,
) -> EvalReport:
    """Assemble an :class:`EvalReport` for an already-finished run.

    Loads the run's artifacts (by ``manifest.run_id``) and runs the rule and
    blocking evaluations, honoring the ``EMPI_DEPLOY_*`` / ``EMPI_ADJ_*``
    defaults unless overridden by the keyword arguments.
    """
    blocking_method = blocking_method or settings.deploy_blocking_method
    blocking_sample = blocking_sample or settings.deploy_blocking_sample
    adj_sample = adj_sample or settings.adj_sample_size
    seed = seed if seed is not None else settings.deploy_seed

    cleaned, matches, non_matches = _load_artifacts(manifest.run_id, settings)
    rule_conf = {r.name: r.confidence for r in RULES}

    rule_eval: dict = {}
    if not skip_rule_eval:
        logger.info("Evaluating deterministic rules (adj sample=%d)...", adj_sample)
        rule_report = evaluate_rules(
            matches, cleaned, n_sample=adj_sample, seed=seed,
            non_matches=non_matches, rule_confidence=rule_conf,
        )
        rule_eval = _serialize_rule_eval(rule_report)

    blocking_eval: dict = {}
    if not skip_blocking_eval:
        logger.info("Evaluating blocking recall (method=%s)...", blocking_method)
        blocking_report = evaluate_blocking(
            cleaned, method=blocking_method,
            sample_size=blocking_sample, seed=seed,
        )
        blocking_eval = _serialize_blocking_eval(blocking_report)

    return EvalReport(
        run_id=manifest.run_id,
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=manifest.git_sha,
        config=settings.model_dump(mode="json"),
        environment=_environment_info(),
        counts=dict(manifest.counts),
        manifest=manifest.model_dump(mode="json"),
        rule_eval=rule_eval,
        blocking_eval=blocking_eval,
    )


def write_eval_report(
    report: EvalReport, settings: Settings = default_settings
) -> tuple[Path, Path]:
    """Persist the report as JSON + a text summary under ``runs_dir``."""
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = settings.runs_dir / f"eval_{report.run_id}.json"
    text_path = settings.runs_dir / f"eval_{report.run_id}.txt"
    json_path.write_text(report.to_json())
    text_path.write_text(report.to_text())
    return json_path, text_path


def run_eval(
    raw_input: Path | None = None,
    settings: Settings = default_settings,
    run_id: str | None = None,
    *,
    evaluate_only: bool = False,
    skip_rule_eval: bool = False,
    skip_blocking_eval: bool = False,
    blocking_method: str | None = None,
    blocking_sample: int | None = None,
    adj_sample: int | None = None,
    seed: int | None = None,
) -> EvalReport:
    """Run the pipeline (unless ``evaluate_only``) then build + persist the report."""
    configure_logging(settings)
    if evaluate_only:
        if run_id is None:
            raise ValueError("--run-id is required with --evaluate-only")
        manifest_path = settings.runs_dir / f"run_{run_id}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest for run {run_id}: {manifest_path}")
        manifest = RunManifest.model_validate_json(manifest_path.read_text())
    else:
        manifest = run_pipeline(raw_input=raw_input, settings=settings, run_id=run_id)

    report = build_eval_report(
        manifest, settings,
        skip_rule_eval=skip_rule_eval,
        skip_blocking_eval=skip_blocking_eval,
        blocking_method=blocking_method,
        blocking_sample=blocking_sample,
        adj_sample=adj_sample,
        seed=seed,
    )
    json_path, text_path = write_eval_report(report, settings)
    logger.info("Evaluation report written: %s", json_path)
    logger.info("Summary written: %s", text_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the eMPI pipeline and emit an evaluation report"
    )
    ap.add_argument("--input", type=Path, default=None,
                    help="Raw CSV/XLSX to process (default: settings.raw_input).")
    ap.add_argument("--run-id", type=str, default=None,
                    help="Run id (generated if omitted; required with "
                         "--evaluate-only).")
    ap.add_argument("--evaluate-only", action="store_true",
                    help="Skip the pipeline; evaluate an existing run.")
    ap.add_argument("--skip-rule-eval", action="store_true",
                    help="Skip the deterministic-rule evaluation suite.")
    ap.add_argument("--skip-blocking-eval", action="store_true",
                    help="Skip the (slower) blocking-recall evaluation.")
    ap.add_argument("--blocking-method", choices=("loose", "sample"), default=None,
                    help=f"Wide-candidate strategy "
                         f"(default: {default_settings.deploy_blocking_method}).")
    ap.add_argument("--blocking-n", type=int, default=None,
                    help="Sample size for --blocking-method sample.")
    ap.add_argument("--adj-n", type=int, default=None,
                    help="Adjudication sample size per rule.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log-level", type=str, default=None,
                    help="Override EMPI_LOG_LEVEL (DEBUG/INFO/WARNING/...).")
    args = ap.parse_args()

    configure_logging(level=args.log_level)
    report = run_eval(
        raw_input=args.input,
        run_id=args.run_id,
        evaluate_only=args.evaluate_only,
        skip_rule_eval=args.skip_rule_eval,
        skip_blocking_eval=args.skip_blocking_eval,
        blocking_method=args.blocking_method,
        blocking_sample=args.blocking_n,
        adj_sample=args.adj_n,
        seed=args.seed,
    )
    print(report.to_text())


if __name__ == "__main__":
    main()
