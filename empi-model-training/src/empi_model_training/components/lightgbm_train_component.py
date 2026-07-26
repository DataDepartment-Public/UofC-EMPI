"""Azure ML component definition for LightGBM v3 pair-classifier training.

Wraps `empi_model_training.training.lightgbm_train`'s CLI unchanged — see
`fs_train_component.py`'s docstring for why (never a second copy of the
training logic).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import CommandComponent

COMPONENT_NAME = "lightgbm_v3_train"
_SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src


def build_lightgbm_train_component(
    environment: str, compute: str | None = None
) -> CommandComponent:
    built = command(
        name=COMPONENT_NAME,
        display_name="Train LightGBM v3 pair classifier",
        description=(
            "Independent implementation (see empi-model-training/CLAUDE.md) of the "
            "match-vs-ambiguous classifier documented in "
            "empi-service/docs/ML-Model-LightGBM-v3.md."
        ),
        inputs={
            "cleaned_index": Input(type="uri_file"),
            "gold_labels": Input(type="uri_file"),
            "auto_merge_threshold": Input(type="number", default=0.70),
            "promote": Input(type="boolean", optional=True),
        },
        outputs={"model_dir": Output(type="uri_folder", mode="rw_mount")},
        code=str(_SRC_ROOT),
        environment=environment,
        compute=compute,
        command=(
            "python -m empi_model_training.training.lightgbm_train "
            "--cleaned-index ${{inputs.cleaned_index}} "
            "--gold-labels ${{inputs.gold_labels}} "
            "--auto-merge-threshold ${{inputs.auto_merge_threshold}} "
            "--model-dir ${{outputs.model_dir}} "
            "$[[--promote]]"
        ),
    )
    # See fs_train_component.py's matching comment -- .component is always a
    # real CommandComponent here, the SDK's str union covers a different case.
    return cast(CommandComponent, built.component)


__all__ = ["COMPONENT_NAME", "build_lightgbm_train_component"]
