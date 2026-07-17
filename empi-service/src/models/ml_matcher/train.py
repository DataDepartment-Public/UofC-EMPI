"""ml-train — train + persist an ML matcher model (SCAFFOLD, not yet usable).

    python -m src.models.ml_matcher.train [--promote]

STATUS: scaffolding only. This CLI wires up the surrounding plumbing (input
resolution from the latest `RunManifest`, model persistence, the deploy-gate)
so that plugging in a real `FeatureBuilder` + `MLModel` and implementing
`MLMatcher.train()` is the only step left. Running it today reaches
`MLMatcher.train()` and stops there with a clear `NotImplementedError` — see
`docs/ML-Matcher-Integration-Guide.md` for the extension contract.

Mirrors `src/models/fs_matcher/train.py`'s shape (RunManifest-based input
resolution, `--promote`/`--force-promote`, a `.meta.json` sidecar),
simplified: no Splink-specific labeled-pairs handling, since BYOF/BYOM means
the label format and feature representation are the implementer's choice.

Flow (once `MLMatcher.train()` is implemented):
  1. Resolve cleaned records + candidate pairs from the latest RunManifest.
  2. Load a labels file (format is the implementer's choice).
  3. `model.train(candidate_pairs, df_clean, labels)`; persist the fitted
     model artifact + a `.meta.json` provenance/metrics sidecar into
     `settings.ml_model_dir`.
  4. With `--promote`, run the deploy-gate and (if it passes) point
     `active.json` at the new model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import settings as default_settings, configure_logging  # noqa: E402
from src.contracts import RunManifest  # noqa: E402
from src.models.ml_matcher.matcher import MLMatcher  # noqa: E402

logger = logging.getLogger("eMPI.ml_train")


# ── Input resolution (mirrors fs_matcher/train.py's manifest-based lookup) ────
def _resolve_from_manifest(
    settings, run_id: str | None = None,
) -> tuple[Path, Path]:
    """Resolve (cleaned, candidate_pairs) paths from a `RunManifest` — same-run
    lineage guarantee, same convention `fs_matcher/train.py` uses."""
    runs_dir = Path(settings.runs_dir)
    if run_id is not None:
        manifest_path = runs_dir / f"run_{run_id}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest for --run-id={run_id!r} at {manifest_path}.")
    else:
        candidates = sorted(runs_dir.glob("run_*.json"))
        if not candidates:
            raise FileNotFoundError(f"No run manifests found in {runs_dir}.")
        manifest_path = candidates[-1]

    manifest = RunManifest.model_validate_json(manifest_path.read_text())
    cleaned_path = settings.project_root / manifest.cleaned.path
    pool_path = settings.project_root / manifest.candidate_pairs.path
    for label, p in (("cleaned", cleaned_path), ("candidate_pairs", pool_path)):
        if not p.exists():
            raise FileNotFoundError(
                f"Manifest {manifest_path.name} references {label}={p}, which no "
                "longer exists on disk (likely cleaned up) — lineage broken. Pass "
                "--cleaned-index/--candidate-pairs explicitly, or a different --run-id."
            )
    logger.info(
        "Resolved inputs from manifest %s (run_id=%s)", manifest_path.name, manifest.run_id
    )
    return cleaned_path, pool_path


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    s = default_settings
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cleaned-index", type=Path, default=None,
                   help="Cleaned records parquet. Default: resolved from the latest "
                        f"RunManifest in {s.runs_dir}.")
    p.add_argument("--candidate-pairs", type=Path, default=None,
                   help="Full candidate-pairs parquet. Default: resolved from the latest "
                        f"RunManifest in {s.runs_dir}.")
    p.add_argument("--run-id", default=None,
                   help="Resolve inputs from this specific run's manifest "
                        "(data/runs/run_<run-id>.json) instead of the latest run.")
    p.add_argument("--labels", type=Path, default=None,
                   help="Labels file (format is the implementer's choice — see "
                        "docs/ML-Matcher-Integration-Guide.md).")
    p.add_argument("--promote", action="store_true",
                   help="After training, run the deploy-gate and point active.json at the new model.")
    p.add_argument("--force-promote", action="store_true",
                   help="Promote even if the deploy-gate fails (use with care).")
    p.add_argument("--log-level", default=None, help="Override EMPI_LOG_LEVEL.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = default_settings
    configure_logging(settings, level=args.log_level)
    settings.ensure_dirs()

    cleaned_path = args.cleaned_index
    pool_path = args.candidate_pairs
    if cleaned_path is None or pool_path is None:
        manifest_cleaned, manifest_pool = _resolve_from_manifest(settings, run_id=args.run_id)
        cleaned_path = cleaned_path or manifest_cleaned
        pool_path = pool_path or manifest_pool
    logger.info("cleaned=%s pool=%s labels=%s", cleaned_path, pool_path, args.labels)

    df_clean = pd.read_parquet(cleaned_path)
    candidate_pairs = pd.read_parquet(pool_path)
    labels = pd.read_csv(args.labels) if args.labels is not None else pd.DataFrame()

    # Plug in a real FeatureBuilder + (unfitted) MLModel here once one exists —
    # see docs/ML-Matcher-Integration-Guide.md.
    model = MLMatcher()
    model.train(candidate_pairs, df_clean, labels)  # STUB: raises NotImplementedError

    # Reached only once MLMatcher.train() is implemented: persist the fitted
    # model artifact + a .meta.json sidecar into settings.ml_model_dir (mirror
    # fs_matcher/train.py's model_path/meta write), then optionally
    # `from src.models.ml_matcher import registry; registry.promote(...)`
    # when --promote/--force-promote is set.


if __name__ == "__main__":
    main()
