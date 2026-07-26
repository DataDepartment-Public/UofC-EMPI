"""Register a versioned Azure ML Data asset from a local file/folder.

Training jobs reference data by `azureml:<name>:<version>` (see
`components/`), never a raw path — that's what gives training runs
reproducible data lineage. There is no Terraform resource for AML data
assets (same reasoning as environments — see `register_environment.py`).

This is the explicit "data in" half of the data-movement step described in
README.md: Azure ML's compute has no network path to the app's private
storage (`UofC-EMPI/terraform/ml_workspace.tf`), so training inputs are
staged locally first (e.g. via the same VNet-connectivity workaround already
documented for `az webapp ssh` in that repo) and registered from there.

    uv run python -m empi_model_training.utils.register_dataset \\
        --name empi-cleaned-records --version 2026.07.25 \\
        --path /local/staging/cleaned.parquet --type uri_file
"""

from __future__ import annotations

import argparse
import logging

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data

from empi_model_training.utils.azure_client import get_ml_client

logger = logging.getLogger("empi_model_training.register_dataset")

_ASSET_TYPES = {"uri_file": AssetTypes.URI_FILE, "uri_folder": AssetTypes.URI_FOLDER}


def register_dataset(
    name: str, version: str, path: str, asset_type: str, description: str = ""
) -> str:
    if asset_type not in _ASSET_TYPES:
        raise ValueError(f"--type must be one of {sorted(_ASSET_TYPES)}, got {asset_type!r}")

    client = get_ml_client()
    data = Data(
        name=name,
        version=version,
        path=path,
        type=_ASSET_TYPES[asset_type],
        description=description or f"Registered from local path {path}",
    )
    registered = client.data.create_or_update(data)
    logger.info("Registered %s:%s -> %s", registered.name, registered.version, registered.path)
    reference: str = f"azureml:{registered.name}:{registered.version}"
    return reference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--name", required=True, help="Data asset name, e.g. empi-cleaned-records.")
    p.add_argument(
        "--version",
        required=True,
        help="Version tag -- a date or the source data's own version tag.",
    )
    p.add_argument("--path", required=True, help="Local file or folder to upload and register.")
    p.add_argument("--type", dest="asset_type", choices=sorted(_ASSET_TYPES), default="uri_file")
    p.add_argument("--description", default="")
    return p.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args()
    reference = register_dataset(
        args.name, args.version, args.path, args.asset_type, args.description
    )
    print(reference)


if __name__ == "__main__":
    main()
