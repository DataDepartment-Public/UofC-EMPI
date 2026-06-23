"""Idempotent trim script: removes dead / superseded sections from the validation notebook.

Sections to remove (identified by their markdown header text prefix):

    "## 7.1 Libpostal coverage audit (R2 prerequisite)"
        — R2 Address comparison was dropped in commit ed79b02; diagnostic is dead.

    "## 10. Head-to-head — baseline vs combined system (rules + enhanced FS)"
        — 2-way comparison superseded by the 3-way §11 head-to-head.

    "### 11.5 Per-comparison m/u weights — enhanced_2 (anti-evidence inspection)"
        — Insight absorbed into docs/Fellegi-Sunter-Enhanced_2.md "Known limitations" in E3-4.

Usage
-----
    python scripts/_trim_notebook.py             # trim in-place
    python scripts/_trim_notebook.py --dry-run   # show what would be removed, no file write

The script is idempotent: if a target section is already absent, it is silently skipped.

A "section" is defined as the opening markdown cell whose first line matches the given prefix,
plus all following cells up to (but not including) the next markdown cell at the same heading
depth or shallower.  Heading depth is derived from the number of leading # characters:
  ## foo  → depth 2
  ### foo → depth 3

Stop condition when dropping a depth-D section: the next markdown header at depth ≤ D ends the
dropped block.  This correctly drops both the opener and all ### sub-sections of a ## section.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NB_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "fellegi_sunter"
    / "fellegi_sunter_validation.ipynb"
)

# ---------------------------------------------------------------------------
# Sections to remove — one entry per line for clarity.
# Each tuple: (display_label, header_prefix_to_match).
# header_prefix_to_match is compared with str.startswith against the first
# line of every markdown cell.
# ---------------------------------------------------------------------------
SECTIONS_TO_DROP = [
    (
        "§8 (Libpostal coverage audit)",
        "## 7.1 Libpostal coverage audit",
    ),
    (
        "§16-20 (2-way head-to-head §10)",
        "## 10. Head-to-head",
    ),
    (
        "§27 (enhanced_2 m/u inspection §11.5)",
        "### 11.5 Per-comparison m/u weights",
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_line(cell: nbformat.NotebookNode) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src.split("\n", 1)[0].strip()


def _header_depth(header_prefix: str) -> int:
    """Return Markdown heading depth of a header prefix (## → 2, ### → 3)."""
    stripped = header_prefix.lstrip("#")
    return len(header_prefix) - len(stripped)


def _cell_header_depth(cell: nbformat.NotebookNode) -> int | None:
    """If this markdown cell opens a heading, return its depth; else None."""
    if cell.get("cell_type") != "markdown":
        return None
    fl = _first_line(cell)
    if not fl.startswith("#"):
        return None
    stripped = fl.lstrip("#")
    return len(fl) - len(stripped)


def _drop_section_pass(
    cells: list[nbformat.NotebookNode],
    header_prefix: str,
) -> tuple[list[nbformat.NotebookNode], bool]:
    """Single-pass removal of one section.

    Returns (new_cells, dropped_any).
    Dropping begins at the cell whose first line starts with *header_prefix*
    and ends (exclusive) at the next markdown header at depth ≤ section depth.
    """
    stop_depth = _header_depth(header_prefix)
    new_cells: list[nbformat.NotebookNode] = []
    in_drop = False
    dropped_any = False

    for cell in cells:
        fl = _first_line(cell)

        if not in_drop:
            if cell.get("cell_type") == "markdown" and fl.startswith(header_prefix):
                in_drop = True
                dropped_any = True
                # Drop this opening cell — don't append.
                continue
            new_cells.append(cell)
            continue

        # in_drop == True: keep dropping until we hit the end-of-section marker.
        depth = _cell_header_depth(cell)
        if depth is not None and depth <= stop_depth:
            # This header closes the dropped section — keep it and stop dropping.
            in_drop = False
            new_cells.append(cell)
        # else: still inside the removed section — drop.

    return new_cells, dropped_any


# ---------------------------------------------------------------------------
# Main trim logic
# ---------------------------------------------------------------------------

def trim_notebook(dry_run: bool = False) -> None:
    nb = nbformat.read(str(NB_PATH), as_version=4)
    original_count = len(nb.cells)

    removed_labels: list[str] = []
    cells = list(nb.cells)

    for label, header_prefix in SECTIONS_TO_DROP:
        cells, dropped = _drop_section_pass(cells, header_prefix)
        if dropped:
            removed_labels.append(label)

    final_count = len(cells)
    cells_removed = original_count - final_count

    if not dry_run:
        nb.cells = cells
        nbformat.write(nb, str(NB_PATH))

    # Summary line
    if removed_labels:
        print(f"removed sections: {removed_labels}")
    else:
        print("removed sections: []  (all target sections already absent — nothing to do)")

    mode = " [DRY RUN — notebook NOT modified]" if dry_run else ""
    print(
        f"{'would keep' if dry_run else 'kept'} {final_count} cells; "
        f"was {original_count} cells; "
        f"{'would remove' if dry_run else 'removed'} {cells_removed} cells"
        f"{mode}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without modifying the notebook.",
    )
    args = parser.parse_args()
    trim_notebook(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
