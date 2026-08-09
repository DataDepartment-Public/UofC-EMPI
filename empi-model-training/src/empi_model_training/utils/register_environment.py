"""Register (or update) the Azure ML Environment both training components run
in. There is no Terraform resource for AML environments — environments are a
workspace-internal, frequently-revised asset managed through the SDK/CLI,
not infrastructure — so this script is that asset's source of truth.

    uv run python -m empi_model_training.utils.register_environment
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from azure.ai.ml.entities import Environment

from empi_model_training.utils.azure_client import get_ml_client

logger = logging.getLogger("empi_model_training.register_environment")

ENVIRONMENT_NAME = "empi-model-training-env"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENVIRONMENT_YML = _REPO_ROOT / "environment.yml"
_BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"


def register_environment(*, description: str | None = None) -> str:
    """Create a new version of `ENVIRONMENT_NAME` from `environment.yml`.

    Azure ML environment versions are immutable and content-addressed in
    practice — always safe to call, it just creates a new version if
    `environment.yml` changed since the last registration. Returns the
    registered version string.
    """
    if not _ENVIRONMENT_YML.exists():
        raise FileNotFoundError(
            f"{_ENVIRONMENT_YML} not found -- expected the conda spec at the repo root."
        )

    client = get_ml_client()
    env = Environment(
        name=ENVIRONMENT_NAME,
        description=description or "empi-model-training: FS matcher + LightGBM v3 training.",
        conda_file=str(_ENVIRONMENT_YML),
        image=_BASE_IMAGE,
    )
    registered = client.environments.create_or_update(env)
    logger.info("Registered %s:%s", registered.name, registered.version)
    version: str = registered.version
    return version


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--description", default=None)
    return p.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    version = register_environment(description=args.description)
    print(f"{ENVIRONMENT_NAME}:{version}")


if __name__ == "__main__":
    main()
