"""Azure ML component definition for non-match gate training.

Wraps `empi_model_training.training.nonmatch_gate_train`'s CLI unchanged —
see `fs_train_component.py`'s docstring for why (never a second copy of the
training logic).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import CommandComponent

COMPONENT_NAME = "nonmatch_gate_train"
_SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src


def build_nonmatch_gate_train_component(
    environment: str, compute: str | None = None
) -> CommandComponent:
    built = command(
        name=COMPONENT_NAME,
        display_name="Train non-match gate (Stage 4.25)",
        description=(
            "Independent implementation (see empi-model-training/CLAUDE.md) of the "
            "confident-non-match gate documented in "
            "empi-service/docs/Nonmatch-Gate-Guide.md. NOT the Stage-4.5 ml_matcher "
            "(see lightgbm_train_component.py) -- separate model store, separate registry."
        ),
        inputs={
            "cleaned_index": Input(type="uri_file"),
            "gold_labels": Input(type="uri_file"),
            "gate_threshold": Input(type="number", default=0.30),
            "promote": Input(type="boolean", optional=True),
        },
        outputs={"model_dir": Output(type="uri_folder", mode="rw_mount")},
        code=str(_SRC_ROOT),
        environment=environment,
        compute=compute,
        command=(
            "python -m empi_model_training.training.nonmatch_gate_train "
            "--cleaned-index ${{inputs.cleaned_index}} "
            "--gold-labels ${{inputs.gold_labels}} "
            "--gate-threshold ${{inputs.gate_threshold}} "
            "--model-dir ${{outputs.model_dir}} "
            "$[[--promote]]"
        ),
    )
    # See fs_train_component.py's matching comment -- .component is always a
    # real CommandComponent here, the SDK's str union covers a different case.
    return cast(CommandComponent, built.component)


__all__ = ["COMPONENT_NAME", "build_nonmatch_gate_train_component"]
