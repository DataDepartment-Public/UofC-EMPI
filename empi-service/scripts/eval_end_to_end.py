"""Score a completed pipeline run end to end against a pairwise label set.

The existing eval scripts stop at the deterministic rules
(`eval_against_labels.py`, `eval_standardized_metrics.py`). This one starts
from a `RunManifest` and scores the thing the pipeline actually ships — the
Stage-5 cluster assignments — plus every stage boundary in between, so a bad
number can be attributed rather than guessed at. See
`docs/End-to-End-Evaluation-Guide.md`.

Gold labels are the default source and the default holdout is **strict**,
because the Stage-4.25 gate and the served Stage-4.5 matcher were both trained
on the gold file — see `src/evaluation/holdout.py`. `--holdout none` is
available for comparison but its ML-stage numbers are memorization.

**For the usual case, use `scripts/evaluate_all.py` instead** — it runs the
pipeline and scores gold (both holdouts) and synthetic in one session. This
script is the granular tool: one existing run, one label file, one holdout.

Usage:
    # headline, leakage-safe (gold)
    python scripts/eval_end_to_end.py --run-id 20260728T101500Z

    # the same run over every gold pair, for comparison
    python scripts/eval_end_to_end.py --run-id 20260728T101500Z --holdout none

    # a generically-shaped label file (e.g. silver)
    python scripts/eval_end_to_end.py --run-id <id> \
        --labels data/silver_labels/silver_labels_v1_2026_06_21.csv \
        --label-col silver_label --source silver --holdout none
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from src.config import configure_logging, settings  # noqa: E402
from src.evaluation.holdout import (  # noqa: E402
    DEFAULT_GOLD_LABELS,
    GOLD_AMBIGUOUS_COL,
    GOLD_LABEL_COL,
    holdout_keys,
    load_labels,
    verify_model_provenance,
)
from src.evaluation.pipeline_eval import (  # noqa: E402
    evaluate_run,
    load_manifest,
    write_report,
)

logger = logging.getLogger(__name__)


def _warn_if_models_predate_the_holdout() -> None:
    """Surface any served model not trained under the current holdout spec.
    A mismatch makes `--holdout strict` a fiction for that model — the "safe"
    pairs may be its training data — so it is logged loudly rather than
    silently trusted."""
    from src.models.ml_matcher import registry as ml_registry
    from src.models.nonmatch_gate import registry as gate_registry

    metas = {}
    for name, reg in (("ml_matcher", ml_registry), ("nonmatch_gate", gate_registry)):
        try:
            active = reg.resolve_active_model(settings)
            metas[name] = reg.load_model_meta(active) if active is not None else None
        except Exception:  # noqa: BLE001 - provenance must never break an eval
            metas[name] = None
    for problem in verify_model_provenance(metas):
        logger.warning("HOLDOUT PROVENANCE — %s", problem)

DEFAULT_GOLD = DEFAULT_GOLD_LABELS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="Run to score (needs a manifest).")
    ap.add_argument("--labels", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--label-col", default=GOLD_LABEL_COL)
    ap.add_argument("--source", default=None,
                    help="Name for the label source in the report (default: "
                         "inferred from the filename).")
    ap.add_argument("--holdout", default="strict",
                    choices=["strict", "none"],
                    help="'strict' (the default) restricts to the shared "
                         "holdout both models were held out on — computed by "
                         "src.evaluation.holdout, the same function the "
                         "training notebooks call. The only leakage-safe choice "
                         "for gold. 'none' scores every labeled pair and its "
                         "ML-stage numbers are memorization. Ignored for "
                         "non-gold label files.")
    ap.add_argument("--session-id", default=None,
                    help="Group this report with others on one timeline point "
                         "(default: the run id). scripts/evaluate_all.py sets "
                         "this so the real and synthetic runs share a point.")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON output path (a .txt sibling is written too). "
                         "Defaults into settings.evaluations_dir.")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    if not args.labels.exists():
        raise SystemExit(
            f"Label file not found: {args.labels}\n"
            "Gold and silver labels are VM-only PHI and are gitignored — this "
            "script must run where the data lives."
        )

    manifest = load_manifest(args.run_id, settings)
    try:
        labels = load_labels(args.labels, args.label_col)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source = args.source or args.labels.stem

    # Gold carries the extra columns that let the gate and the matcher be
    # scored against *their own* targets rather than the match label.
    is_gold = GOLD_LABEL_COL in labels.columns and GOLD_AMBIGUOUS_COL in labels.columns
    plausible_col = confident_col = None
    if is_gold:
        labels = labels.copy()
        labels["plausible"] = labels[GOLD_LABEL_COL] | labels[GOLD_AMBIGUOUS_COL]
        labels["confident_match"] = labels[GOLD_LABEL_COL] & ~labels[GOLD_AMBIGUOUS_COL]
        plausible_col, confident_col = "plausible", "confident_match"

    holdout = None
    holdout_name = "none"
    if args.holdout != "none":
        if not is_gold:
            logger.warning(
                "--holdout %s ignored: %s is not the gold label file, so the "
                "notebooks' folds are not defined over it.", args.holdout, args.labels.name,
            )
        else:
            holdout = holdout_keys(labels)
            holdout_name = args.holdout
            _warn_if_models_predate_the_holdout()

    report = evaluate_run(
        manifest, labels, args.label_col,
        settings=settings,
        label_source=source,
        holdout=holdout,
        holdout_name=holdout_name,
        session_id=args.session_id,
        plausible_col=plausible_col,
        confident_match_col=confident_col,
        ambiguous_col=GOLD_AMBIGUOUS_COL if is_gold else None,
    )

    print(report.to_text())

    out = write_report(report, settings, out=args.out)
    print(f"\nWrote {out}\n      {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
