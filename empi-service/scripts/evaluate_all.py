"""Run the whole system and score it — the single command for "how good is it right now?".

One invocation = one **evaluation session**, which does everything end to end:

    1. run the pipeline over the real input      -> real run_id
    2. score that run against the gold labels    -> holdout `none` AND `strict`
    3. run the pipeline over the synthetic set   -> synthetic run_id
    4. score that run against the entity truth

and writes every report into `settings.evaluations_dir` under one `session_id`.

**Why a session rather than a run id.** The real-data run and the synthetic run
are necessarily different pipeline runs with different ids, but they are one
measurement of one state of the system. Keying the reports on a session is what
lets both land on the same point of a trend chart instead of drifting apart on
the x-axis.

**Why gold is evaluated at both holdouts.** The Stage-4.25 gate and the
Stage-4.5 matcher were trained on the gold labels, so their per-stage numbers
are only honest under `strict`.

Whether the *headline* also needs `strict` depends on configuration. With
`ml_feeds_clustering` ON (the default) the gold-trained matcher forms merge
edges, so the clusters are partly memorized and **`strict` is the only honest
headline**. Turn both feed toggles off and clustering unions deterministic-rule
edges alone — never fit on gold — and `none` becomes the better headline, with
8x more labeled positives for a tighter estimate. Each report's `leakage` block
records which case it was written under, so the number is never read without it.
Both holdouts run either way.

Usage:
    # the whole thing
    python scripts/evaluate_all.py

    # reuse an existing real-data run instead of re-running the pipeline
    python scripts/evaluate_all.py --reuse-real-run 20260728T191705Z

    # name the session so it reads well on the trend chart
    python scripts/evaluate_all.py --session-id gate_v2_baseline

    # one half only
    python scripts/evaluate_all.py --skip-synthetic
    python scripts/evaluate_all.py --skip-real

For scoring an arbitrary run against arbitrary labels (silver, a single
holdout, a label file in a non-standard place), use `scripts/eval_end_to_end.py`
instead — this script is the batteries-included path.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from src.config import configure_logging, settings  # noqa: E402
from src.evaluation.holdout import (  # noqa: E402
    GOLD_AMBIGUOUS_COL,
    GOLD_LABEL_COL,
    holdout_keys,
    load_gold_labels,
    verify_model_provenance,
)
from src.evaluation.pipeline_eval import (  # noqa: E402
    evaluate_run,
    load_manifest,
    write_report,
)
from src.evaluation.synthetic import (  # noqa: E402
    load_synthetic_pairs,
    run_synthetic_pipeline,
    synthetic_records,
)

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GOLD = _ROOT / "data" / "gold_labels" / "final_gold_labels_v1_2026_07_05.csv"
DEFAULT_SYNTHETIC = _ROOT / "data" / "synthetic_data" / "synthetic_test_v3.csv"


def _new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n  {text}\n{'=' * 72}", flush=True)


# ── the real-data half ───────────────────────────────────────────────────────
def evaluate_real(session_id: str, args) -> list[Path]:
    from src.pipeline import run_pipeline  # lazy: heavy transitive imports

    if not args.gold.exists():
        print(f"! Gold labels not found at {args.gold} — skipping the real-data "
              "half. They are VM-only PHI; run this where the data lives.")
        return []

    if args.reuse_real_run:
        _banner(f"REAL DATA — reusing run {args.reuse_real_run}")
        manifest = load_manifest(args.reuse_real_run, settings)
    else:
        _banner(f"REAL DATA — running the pipeline over {args.input.name}")
        manifest = run_pipeline(raw_input=args.input, settings=settings,
                                run_id=f"{session_id}_real")

    labels = load_gold_labels(args.gold)
    # Gold's extra `ambiguous_pair` column lets the gate and the matcher be
    # scored against their OWN targets rather than the match label — keeping an
    # ambiguous non-match is correct behavior for the gate.
    labels["plausible"] = labels[GOLD_LABEL_COL] | labels[GOLD_AMBIGUOUS_COL]
    labels["confident_match"] = labels[GOLD_LABEL_COL] & ~labels[GOLD_AMBIGUOUS_COL]
    holdout_pairs = holdout_keys(labels)
    _warn_if_models_predate_the_holdout()

    written = []
    for holdout in ("none", "strict"):
        _banner(f"REAL DATA — scoring vs gold, holdout={holdout}")
        report = evaluate_run(
            manifest, labels, GOLD_LABEL_COL,
            settings=settings,
            label_source="gold",
            holdout=None if holdout == "none" else holdout_pairs,
            holdout_name=holdout,
            plausible_col="plausible",
            confident_match_col="confident_match",
            ambiguous_col=GOLD_AMBIGUOUS_COL,
            session_id=session_id,
        )
        print(report.to_text())
        written.append(write_report(report, settings))
    return written


# ── the synthetic half ───────────────────────────────────────────────────────
def evaluate_synthetic(session_id: str, args) -> list[Path]:
    if not args.synthetic.exists():
        print(f"! Synthetic labels not found at {args.synthetic} — skipping.")
        return []

    _banner("SYNTHETIC — running the pipeline over the reconstructed records")
    syn = load_synthetic_pairs(args.synthetic, args.synthetic_label_col)
    records, truth = synthetic_records(syn)
    manifest = run_synthetic_pipeline(records, f"{session_id}_synthetic", settings)

    _banner("SYNTHETIC — scoring vs entity ground truth")
    report = evaluate_run(
        manifest, syn, args.synthetic_label_col,
        settings=settings,
        label_source="synthetic",
        holdout_name="n/a",
        leakage_note="No holdout needed — the gate and the ML matcher were "
                     "trained on gold, never on this set, so every stage here "
                     "is leakage-free.",
        truth_partition=truth,
        session_id=session_id,
    )
    print(report.to_text())
    return [write_report(report, settings)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-id", default=None,
                    help="Names this evaluation point (default: UTC timestamp). "
                         "Use something memorable when it is a baseline.")
    ap.add_argument("--input", type=Path, default=settings.raw_input,
                    help="Raw CSV for the real-data pipeline run.")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    ap.add_argument("--synthetic-label-col", default="label")
    ap.add_argument("--reuse-real-run", default=None,
                    help="Score an existing real-data run instead of running "
                         "the pipeline again (it is the slow part).")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    session_id = args.session_id or _new_session_id()
    _banner(f"EVALUATION SESSION {session_id}")

    written: list[Path] = []
    failures: list[str] = []

    # Each half runs independently: a missing gold file or a broken real run
    # must not cost you the synthetic result (and vice versa). Anything that
    # fails is reported at the end rather than aborting the session.
    for name, skip, fn in (("real", args.skip_real, evaluate_real),
                           ("synthetic", args.skip_synthetic, evaluate_synthetic)):
        if skip:
            print(f"\n(skipping the {name} half — --skip-{name})")
            continue
        try:
            written.extend(fn(session_id, args))
        except Exception as exc:  # noqa: BLE001 - one half must not sink the other
            failures.append(f"{name}: {exc}")
            logger.error("The %s half failed:\n%s", name, traceback.format_exc())

    _banner(f"SESSION {session_id} COMPLETE")
    for path in written:
        print(f"  wrote {path}")
    if not written:
        print("  no reports written.")
    for failure in failures:
        print(f"  FAILED — {failure}")
    print("\nCompare sessions in notebooks/evaluation/end_to_end_eval.ipynb")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
