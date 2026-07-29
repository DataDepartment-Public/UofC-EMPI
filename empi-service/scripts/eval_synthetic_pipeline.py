"""Run the real pipeline over the synthetic label set and score it end to end.

The synthetic set is the project's only label source with **entity-level**
ground truth (`entity_id_a` / `entity_id_b`), which makes it the only one where
cluster metrics are exact rather than inferred from a transitive closure. It is
also the only one that is leakage-free for the ML stages: the Stage-4.25 gate
and the Stage-4.5 matcher were trained on gold, never on this.

The catch is its shape. `data/synthetic labels/synthetic_test_v3.csv` is a
*pair* file carrying already-cleaned `*_l` / `*_r` attributes, not raw records —
so it cannot be fed to `python -m src.pipeline --input`. This script:

  1. reconstructs a record frame from both sides of every pair
     (`synthetic_records`), in `CleanedRecords` shape;
  2. writes it as a cleaned Parquet and runs the **actual** pipeline over it via
     `run_pipeline(cleaned_input=...)` — Stages 2-5 exactly as production runs
     them, not a reimplementation;
  3. scores the resulting clusters against the entity ids.

What it measures and what it doesn't: Stage 1 (cleaning) is skipped by
construction, since the inputs are already cleaned — the corruptions in this
set were planted at the cleaned level. Blocking recall here is real and
meaningful (the positives were built independently of blocking), unlike with
gold/silver where the label universe *is* a blocking output.

Usage:
    python scripts/eval_synthetic_pipeline.py
    python scripts/eval_synthetic_pipeline.py --synthetic "data/synthetic labels/synthetic_test_v3.csv"
    python scripts/eval_synthetic_pipeline.py --reuse-run 20260728T101500Z   # score only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from src.config import configure_logging, settings  # noqa: E402
from src.evaluation.pipeline_eval import evaluate_run, load_manifest  # noqa: E402
from src.preprocessing.clean import write_cleaned  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SYNTHETIC = _ROOT / "data" / "synthetic_data" / "synthetic_test_v3.csv"

#: Columns stored space-delimited in the CSV that the pipeline expects as
#: collections (`contracts._is_listlike_or_null`).
_LIST_COLS = ("Phones_set", "full_name_tokens")


def _to_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [tok for tok in str(value).replace(",", " ").split() if tok]


def _side(syn: pd.DataFrame, side: str) -> pd.DataFrame:
    """Pull one side (`_l` or `_r`) of every pair into a record frame."""
    suffix = f"_{side}"
    id_col = "PATID_A" if side == "l" else "PATID_B"
    entity_col = "entity_id_a" if side == "l" else "entity_id_b"
    rename = {c: c[: -len(suffix)] for c in syn.columns if c.endswith(suffix)}
    out = syn[[id_col, entity_col, *rename]].rename(
        columns={id_col: "PATID", entity_col: "entity_id", **rename}
    )
    return out


def synthetic_records(syn: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Reconstruct the record population + its entity ground truth.

    Both sides of every pair become records. Each PATID appears on exactly one
    side in this file, but we de-duplicate anyway so the function stays correct
    for a future set that reuses a record across pairs.
    """
    records = pd.concat([_side(syn, "l"), _side(syn, "r")], ignore_index=True)
    records = records.drop_duplicates(subset="PATID").reset_index(drop=True)

    truth = {str(p): str(e) for p, e in zip(records["PATID"], records["entity_id"])}
    records = records.drop(columns=["entity_id"])

    for col in _LIST_COLS:
        if col in records.columns:
            records[col] = records[col].map(_to_list)

    # Every synthetic record is well-formed by construction — the corruptions
    # are realistic typos, not the malformed rows Stage 1's validity filter
    # exists to catch. Without this, blocking would drop the whole population.
    records["valid_record"] = True
    return records, truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    ap.add_argument("--run-id", default=None,
                    help="Run id for the pipeline run this script triggers.")
    ap.add_argument("--reuse-run", default=None,
                    help="Skip the pipeline and score an existing synthetic run "
                         "by id (must have been produced by this script).")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    configure_logging(level=args.log_level)

    if not args.synthetic.exists():
        raise SystemExit(f"Synthetic label file not found: {args.synthetic}")

    syn = pd.read_csv(args.synthetic, dtype=str, keep_default_na=True, na_values=[""])
    for col in ("PATID_A", "PATID_B", "entity_id_a", "entity_id_b"):
        if col not in syn.columns:
            raise SystemExit(
                f"{args.synthetic.name} has no {col!r} column — this harness needs "
                "the entity-level ground truth that only the v3 synthetic set carries."
            )
    syn[args.label_col] = syn[args.label_col].astype(int).astype(bool)

    records, truth = synthetic_records(syn)
    print(f"Synthetic: {len(syn)} labeled pairs ({int(syn[args.label_col].sum())} positive) "
          f"→ {len(records)} records over {len(set(truth.values()))} true entities")

    if args.reuse_run:
        manifest = load_manifest(args.reuse_run, settings)
    else:
        run_id = args.run_id or f"synthetic_{args.synthetic.stem}"
        cleaned_path = settings.processed_dir / f"{settings.cleaned_stem}_{run_id}.parquet"
        settings.ensure_dirs()
        write_cleaned(records, cleaned_path)
        print(f"Wrote reconstructed cleaned frame → {cleaned_path}")
        manifest = run_pipeline(cleaned_input=cleaned_path, run_id=run_id, settings=settings)

    report = evaluate_run(
        manifest, syn, args.label_col,
        settings=settings,
        label_source="synthetic_v3",
        holdout_name="n/a (models were trained on gold, not synthetic)",
        truth_partition=truth,
    )
    text = report.to_text()
    print(text)

    out = args.out or settings.runs_dir / f"eval_end_to_end_{manifest.run_id}_synthetic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2, default=str))
    out.with_suffix(".txt").write_text(text)
    print(f"\nWrote {out}\n      {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
