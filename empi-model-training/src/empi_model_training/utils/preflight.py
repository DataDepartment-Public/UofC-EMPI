"""Pre-submission checks — fail fast with an actionable message instead of
a job silently queuing against a broken workspace, a compute cluster that
doesn't exist, or an unregistered environment/dataset.

    uv run python -m empi_model_training.utils.preflight \
        --compute cpu-cluster --environment empi-model-training-env

`submit.py` calls `run_preflight()` before every submission; this is also
runnable standalone for a quick "is everything set up" sanity check.
"""

from __future__ import annotations

import argparse
import logging

from empi_model_training.utils.azure_client import MissingConfigError, get_ml_client
from empi_model_training.utils.register_environment import ENVIRONMENT_NAME

logger = logging.getLogger("empi_model_training.preflight")


class PreflightError(RuntimeError):
    """Raised by `main()` when one or more checks fail. Carries the full list
    of problems (see `run_preflight`), not just the first one."""


def run_preflight(
    compute_name: str,
    environment_name: str = ENVIRONMENT_NAME,
    data_refs: list[str] | None = None,
) -> list[str]:
    """Return a list of problems found (empty list = everything checks out).

    Never raises for a *missing resource* (compute/environment/dataset) —
    those are reported as problems, not exceptions, so a caller can decide
    whether to just log a warning or hard-fail. Only raises for
    infrastructure the rest of this function can't proceed without
    (credentials/workspace connectivity).
    """
    problems: list[str] = []

    try:
        client = get_ml_client()
    except MissingConfigError as exc:
        return [str(exc)]

    try:
        client.workspaces.get(client.workspace_name)
    except Exception as exc:  # noqa: BLE001 — surfaced as a preflight problem, not a crash
        return [f"Cannot reach workspace {client.workspace_name!r}: {exc}"]

    try:
        compute = client.compute.get(compute_name)
        if getattr(compute, "provisioning_state", "").lower() not in ("succeeded", ""):
            problems.append(
                f"Compute {compute_name!r} exists but provisioning_state="
                f"{compute.provisioning_state!r} (expected Succeeded)."
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Compute {compute_name!r} not found or unreachable: {exc}")

    try:
        versions = list(client.environments.list(name=environment_name))
        if not versions:
            problems.append(
                f"Environment {environment_name!r} has no registered versions -- "
                "run `uv run python -m empi_model_training.utils.register_environment` first."
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Environment {environment_name!r} not found or unreachable: {exc}")

    for ref in data_refs or []:
        name, _, version = ref.partition(":")
        try:
            client.data.get(name=name, version=version)
        except Exception as exc:  # noqa: BLE001
            problems.append(
                f"Data asset {ref!r} not found -- register it first with register_dataset.py: {exc}"
            )

    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--compute",
        required=True,
        help="Compute cluster name (see terraform output ml_compute_cluster_name).",
    )
    p.add_argument("--environment", default=ENVIRONMENT_NAME)
    p.add_argument("--data", action="append", default=[], help="name:version to check, repeatable.")
    return p.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    problems = run_preflight(args.compute, args.environment, args.data)
    if problems:
        for p in problems:
            logger.error("PREFLIGHT FAILED: %s", p)
        raise PreflightError(f"{len(problems)} preflight check(s) failed.")
    logger.info(
        "Preflight OK: workspace, compute, environment, and any requested data assets all resolve."
    )


if __name__ == "__main__":
    main()
