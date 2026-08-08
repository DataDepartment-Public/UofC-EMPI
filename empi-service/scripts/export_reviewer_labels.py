"""Derive retraining labels from reviewer-confirmed audit_log actions.

Reviewer actions in the dashboard (`POST /audit/merge`, `/unmerge`,
`/dismiss`) were not used for retraining before this script existed --
`audit_log` only ever backed record-locking and the audit trail. This
turns those same actions into `(PATID_A, PATID_B, reviewer_label)` pairs,
in the same PHI-safe shape as `data/silver_labels/*.csv`.

Derivation, per action type (see `audit_log.related_patids`' DDL comment
in `src/api/backends/sql_backend.py` for what that column captures):

  * merge   -> positive pairs: every combination across `patids` (newly
              added) and `related_patids` (the entity's members right
              before this merge) -- everyone confirmed to be the same
              person.
  * unmerge -> negative pairs: the removed patid (`patids`) against each
              patid that stayed behind (`related_patids`).
  * dismiss -> negative pair: the two patids in `patids` directly (already
              a candidate-pair shape, no entity involved).

Unmerge rows from before `related_patids` existed have it as NULL and are
skipped -- not guessed at from entity_member's current-state-only table,
which can't reliably tell you who a patid was separated from after the
fact. The skipped count is logged so it's visible how much history is
unusable.

A pair confirmed by more than one event keeps its most recent label
(last-write-wins by `ts_utc`) -- a later reviewer action overrides an
earlier one for the same pair.

Usage:
    python scripts/export_reviewer_labels.py
    python scripts/export_reviewer_labels.py --out data/reviewer_labels/custom.csv
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from src.api.backends.index_backend import build_index_backend  # noqa: E402
from src.config import configure_logging, settings as default_settings  # noqa: E402

logger = logging.getLogger(__name__)

LABEL_COLUMNS = ("PATID_A", "PATID_B", "reviewer_label")

# list_audit_log's `limit` is sized for a UI feed (default 100) -- this
# script wants full history, so it passes a limit far beyond any real
# audit_log's row count rather than adding pagination for a one-shot export.
_EXPORT_LIMIT = 10_000_000


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Sort so (P1, P2) and (P2, P1) dedup to the same key."""
    return (a, b) if a < b else (b, a)


def _split_patids(value: str | None) -> list[str]:
    if not value:
        return []
    return [p for p in value.split(",") if p]


def derive_labeled_pairs(rows: list[dict]) -> tuple[pd.DataFrame, int]:
    """`rows`: raw dicts from `IndexBackend.list_audit_log` (newest-first or
    any order -- last-write-wins is resolved by `ts_utc`, not list order).

    Returns `(labels_df, skipped_unmerge_count)`.
    """
    latest: dict[tuple[str, str], tuple[str, int]] = {}
    skipped_unmerge = 0

    for row in rows:
        action = row["action"]
        ts = row["ts_utc"]
        patids = _split_patids(row["patids"])
        related = _split_patids(row.get("related_patids"))

        if action == "merge":
            pairs = [(_canonical_pair(a, b), 1) for a, b in itertools.combinations(patids + related, 2)]
        elif action == "unmerge":
            if not related:
                skipped_unmerge += 1
                continue
            if not patids:
                logger.warning("Skipping malformed unmerge row (no patids): %r", row)
                continue
            removed = patids[0]
            pairs = [(_canonical_pair(removed, other), 0) for other in related]
        elif action == "dismiss":
            if len(patids) != 2:
                logger.warning("Skipping malformed dismiss row (patids=%r)", row["patids"])
                continue
            pairs = [(_canonical_pair(patids[0], patids[1]), 0)]
        else:
            # "split" is declared in AuditLogRow's Literal but never emitted
            # by audit.py today -- nothing to derive from it.
            continue

        for pair, label in pairs:
            if pair not in latest or ts >= latest[pair][0]:
                latest[pair] = (ts, label)

    records = [
        {"PATID_A": a, "PATID_B": b, "reviewer_label": label}
        for (a, b), (_ts, label) in latest.items()
    ]
    return pd.DataFrame(records, columns=list(LABEL_COLUMNS)), skipped_unmerge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path (default: data/reviewer_labels/reviewer_labels_<today>.csv)",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure_logging(level=args.log_level)

    backend = build_index_backend(default_settings)
    try:
        rows = backend.list_audit_log(limit=_EXPORT_LIMIT)
    finally:
        backend.close()

    labels_df, skipped_unmerge = derive_labeled_pairs(rows)

    out_path = args.out or (
        default_settings.project_root / "data" / "reviewer_labels"
        / f"reviewer_labels_{datetime.date.today().isoformat()}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(out_path, index=False)

    logger.info(
        "Wrote %d labeled pairs to %s (%d pre-migration unmerge rows skipped, no related_patids)",
        len(labels_df), out_path, skipped_unmerge,
    )


if __name__ == "__main__":
    main()
