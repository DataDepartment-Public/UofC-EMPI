"""Shared helpers for resolving the highest-versioned parquet on disk.

The cleaning / blocking / matching pipeline stages each write files of the form
``<stem>_v<N>_<YYYY_MM_DD>.parquet``. Downstream consumers (runners, notebooks)
need to auto-resolve the newest file rather than hard-coding date stamps that
go stale on every refresh.

Use ``latest_versioned(dir_, pattern)`` from anywhere that needs that lookup.
"""

from __future__ import annotations

import re
from pathlib import Path

_VERSION_RE = re.compile(r"_v(\d+)_")


def latest_versioned(dir_: Path, pattern: str) -> Path:
    """Return the path in ``dir_`` matching ``pattern`` with the highest ``_v<N>_`` token.

    Parameters
    ----------
    dir_ : Path
        Directory to search. Must exist.
    pattern : str
        Glob pattern (e.g. ``"MDM_Population_cleaned_v*_*.parquet"``). Files
        without a ``_v<N>_`` token in their name are ignored.

    Returns
    -------
    Path
        Path to the matching file with the largest integer ``N``.

    Raises
    ------
    FileNotFoundError
        If the directory is missing or no files match the pattern.
    """
    if not dir_.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_}")
    candidates: list[tuple[int, Path]] = []
    for p in dir_.glob(pattern):
        m = _VERSION_RE.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(
            f"No files matching {pattern!r} in {dir_}. "
            "Confirm the upstream pipeline stage has been run."
        )
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def version_tag_from_filename(path: Path) -> str:
    """Extract the ``v<N>_<YYYY_MM_DD>`` tag from a versioned filename.

    Returns ``"unversioned"`` if no tag is present (e.g. a renamed file).
    """
    m = re.search(r"(v\d+_\d{4}_\d{2}_\d{2})", path.name)
    return m.group(1) if m else "unversioned"
