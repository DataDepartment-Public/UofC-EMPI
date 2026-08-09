"""Shared Azure ML workspace client construction.

Every script that talks to Azure ML (preflight, environment/dataset
registration, job submission) goes through `get_ml_client()` so the
connection details live in exactly one place. Values come from environment
variables — set locally via `az` (see README), or already present in CI/an
Azure ML compute context.
"""

from __future__ import annotations

import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

_REQUIRED_ENV_VARS = (
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE_NAME",
)


class MissingConfigError(RuntimeError):
    """Raised when a required Azure ML connection env var is unset."""


def get_ml_client() -> MLClient:
    """Build an `MLClient` for the eMPI training workspace.

    `AZURE_SUBSCRIPTION_ID` / `AZURE_RESOURCE_GROUP` / `AZURE_ML_WORKSPACE_NAME`
    come from the `terraform output` values in `UofC-EMPI/terraform`
    (`ml_workspace_name`, plus the subscription/resource-group the workspace
    lives in) — see README.md "Connecting to Azure ML".
    """
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise MissingConfigError(
            f"Missing required env var(s): {', '.join(missing)}. Set them from "
            "`terraform output` in UofC-EMPI/terraform (see README.md)."
        )
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE_NAME"],
    )


__all__ = ["get_ml_client", "MissingConfigError"]
