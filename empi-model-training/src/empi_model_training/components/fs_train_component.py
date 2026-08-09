"""Azure ML component definition for FS matcher training.

Wraps `empi_model_training.fs_train`'s CLI unchanged — the component's
`command` invokes the exact same entry point a local run would, so there is
never a second copy of the training logic to keep in sync. See
`utils/register_environment.py` for the environment this runs in.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from azure.ai.ml import Input, Output, command
from azure.ai.ml.entities import CommandComponent

COMPONENT_NAME = "fs_matcher_train"
_SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src


def build_fs_train_component(environment: str, compute: str | None = None) -> CommandComponent:
    """`environment` is an `azureml:<name>:<version>` reference (see
    `register_environment.py`). `compute` is optional here — usually set at
    job-submission time instead (see `submit.py`), but can be pinned on the
    component itself if it should never run anywhere else.
    """
    built = command(
        name=COMPONENT_NAME,
        display_name="Train FS (Fellegi-Sunter) matcher",
        description=(
            "Independent implementation (see empi-model-training/CLAUDE.md) of the "
            "Splink-based Fellegi-Sunter matcher training procedure documented in "
            "empi-service/docs/FS-Matcher-Production-Guide.md."
        ),
        inputs={
            "cleaned_index": Input(type="uri_file"),
            "silver_labels": Input(type="uri_file"),
            "label_col": Input(type="string", default="silver_label"),
            "u_max_pairs": Input(type="number", default=1e6),
            "auto_merge_threshold": Input(type="number", default=0.95),
            "review_floor": Input(type="number", default=0.40),
            # Optional boolean flag -- $[[...]] includes the bracketed segment
            # only when this input is explicitly provided.
            "promote": Input(type="boolean", optional=True),
        },
        outputs={"model_dir": Output(type="uri_folder", mode="rw_mount")},
        code=str(_SRC_ROOT),
        environment=environment,
        compute=compute,
        command=(
            "python -m empi_model_training.training.fs_train "
            "--cleaned-index ${{inputs.cleaned_index}} "
            "--silver-labels ${{inputs.silver_labels}} "
            "--label-col ${{inputs.label_col}} "
            "--u-max-pairs ${{inputs.u_max_pairs}} "
            "--auto-merge-threshold ${{inputs.auto_merge_threshold}} "
            "--review-floor ${{inputs.review_floor}} "
            "--model-dir ${{outputs.model_dir}} "
            "$[[--promote]]"
        ),
    )
    # .component is typed str | CommandComponent (the SDK allows an
    # unresolved "azureml:name:version" reference there too) -- it's always a
    # real CommandComponent immediately after build(), before any
    # serialization round-trip.
    return cast(CommandComponent, built.component)


__all__ = ["COMPONENT_NAME", "build_fs_train_component"]
