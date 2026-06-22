"""fs_splink_enhanced_2 — third Fellegi-Sunter experiment (Phase E2-3).

Supervised m-training, no vetoes, OO scaffold over models/common/fs_base.py.
See docs/Fellegi-Sunter-Enhanced_2.md for the build guide.
"""

from models.experiments.fs_splink_enhanced_2.fs_enhanced_2 import (
    FSEnhanced2,
    MODEL_NAME,
    run_fs_enhanced_2,
)

__all__ = ["FSEnhanced2", "MODEL_NAME", "run_fs_enhanced_2"]
