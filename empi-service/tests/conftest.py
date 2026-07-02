"""
Shared pytest configuration, path setup, and fixtures used across all
test layers (unit, integration, regression).

PATH SETUP:
    Adds the project root (EMPI_latest/) to sys.path so that imports
    like `from src.preprocessing.blocking import ...` resolve correctly
    regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# This file lives at tests/conftest.py. The project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))