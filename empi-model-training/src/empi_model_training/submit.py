"""Unified Azure ML job submission entry point for both training jobs.

    uv run python -m empi_model_training.submit --job fs \\
        --cleaned-index azureml:empi-cleaned-records:2026.07.25 \\
        --labels azureml:empi-silver-labels:2026.07.25 \\
        --compute cpu-cluster [--promote]

    uv run python -m empi_model_training.submit --job lightgbm \\
        --cleaned-index azureml:empi-cleaned-records:2026.07.25 \\
        --labels azureml:empi-gold-labels:2026.07.25 \\
        --compute cpu-cluster [--promote]

Runs `preflight.run_preflight()` first — refuses to submit against a
workspace/compute/environment/dataset that isn't actually ready, rather than
letting a job queue and fail minutes later for a reason that was checkable
up front.
"""

from __future__ import annotations

import argparse
import logging
import sys

from azure.ai.ml import Input

from empi_model_training.components.fs_train_component import build_fs_train_component
from empi_model_training.components.lightgbm_train_component import build_lightgbm_train_component
from empi_model_training.utils.azure_client import get_ml_client
from empi_model_training.utils.preflight import PreflightError, run_preflight
from empi_model_training.utils.register_environment import ENVIRONMENT_NAME

logger = logging.getLogger("empi_model_training.submit")

_EXPERIMENT_NAMES = {"fs": "empi-fs-matcher", "lightgbm": "empi-ml-matcher"}


def _resolve_environment(environment_arg: str | None) -> str:
    """`environment_arg` may be a bare name (resolved to its latest
    registered version) or an explicit `name:version` -- either way, returns
    an `azureml:name:version` reference `command()` accepts."""
    if environment_arg is None:
        environment_arg = ENVIRONMENT_NAME
    if ":" in environment_arg:
        return f"azureml:{environment_arg}"

    client = get_ml_client()
    versions = sorted(client.environments.list(name=environment_arg), key=lambda e: e.version)
    if not versions:
        raise RuntimeError(
            f"No registered versions of environment {environment_arg!r} -- run "
            "`uv run python -m empi_model_training.utils.register_environment` first."
        )
    return f"azureml:{environment_arg}:{versions[-1].version}"


def submit_job(
    job: str,
    cleaned_index_ref: str,
    labels_ref: str,
    compute: str,
    *,
    environment: str | None = None,
    promote: bool = False,
    skip_preflight: bool = False,
) -> str:
    """Returns the submitted job's name (also its Studio run ID)."""
    if job not in _EXPERIMENT_NAMES:
        raise ValueError(f"job must be one of {sorted(_EXPERIMENT_NAMES)}, got {job!r}")

    resolved_environment = _resolve_environment(environment)

    if not skip_preflight:
        problems = run_preflight(
            compute, resolved_environment.split(":")[1], [cleaned_index_ref, labels_ref]
        )
        if problems:
            for p in problems:
                logger.error("PREFLIGHT FAILED: %s", p)
            raise PreflightError(f"{len(problems)} preflight check(s) failed; not submitting.")

    client = get_ml_client()

    if job == "fs":
        component = build_fs_train_component(resolved_environment)
        node = component(
            cleaned_index=Input(path=cleaned_index_ref),
            silver_labels=Input(path=labels_ref),
            promote=promote or None,
        )
    else:
        component = build_lightgbm_train_component(resolved_environment)
        node = component(
            cleaned_index=Input(path=cleaned_index_ref),
            gold_labels=Input(path=labels_ref),
            promote=promote or None,
        )

    node.compute = compute
    node.experiment_name = _EXPERIMENT_NAMES[job]

    submitted = client.jobs.create_or_update(node)
    logger.info(
        "Submitted job %s (experiment=%s) -- %s",
        submitted.name,
        submitted.experiment_name,
        submitted.studio_url,
    )
    name: str = submitted.name
    return name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--job", required=True, choices=sorted(_EXPERIMENT_NAMES))
    p.add_argument(
        "--cleaned-index", required=True, dest="cleaned_index_ref", help="azureml:name:version"
    )
    p.add_argument("--labels", required=True, dest="labels_ref", help="azureml:name:version")
    p.add_argument("--compute", required=True)
    p.add_argument(
        "--environment", default=None, help="Bare name (latest version) or name:version."
    )
    p.add_argument("--promote", action="store_true")
    p.add_argument(
        "--skip-preflight", action="store_true", help="Not recommended -- see module docstring."
    )
    return p.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    args = parse_args()
    try:
        submit_job(
            args.job,
            args.cleaned_index_ref,
            args.labels_ref,
            args.compute,
            environment=args.environment,
            promote=args.promote,
            skip_preflight=args.skip_preflight,
        )
    except PreflightError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
