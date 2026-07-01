"""Publish step: one pipeline run's Parquet output → `empi.db` (SQLite).

Loads a `RunManifest`, groups the run's `ClusterAssignments` into entities,
and upserts them into the resolved-output DB. Never touches the Parquet
artifacts — they stay the immutable record of "what the algorithm produced".

Reconciliation (docs/API-Design.md §2, §6 open decision 1 — "sticky unmerge"):
a PATID `store.locked_patids` reports as reviewer-touched is **never**
repointed to a new `mid` here. Its would-be new grouping is written to
`entity_suggestion` instead — visible for a future admin/reviewer view, not
auto-applied. Untouched PATIDs upsert normally: reuse an existing `mid` if any
unlocked member of the new cluster already has one (ties broken by the
smallest existing `mid`), else mint a fresh one.

`confidence` on a multi-member entity is the max confidence among the
deterministic matches connecting its *unlocked* members — the pipeline's
evidence for the grouping actually being written.

PERFORMANCE: a real run clusters ~160k records into ~140k entities (see
`to-do.md` / the full-population smoke run). The whole publish runs as ONE
SQLite transaction, so it must be fast — issuing one `execute()` (and worse,
one lookup `SELECT`) per record made an early version of this function take
minutes and hold a write lock the whole time, causing "database is locked"
for any concurrent reader. Everything below is planned in pure Python/pandas
first (dict lookups, no DB round-trips per row), then written with four
`executemany` calls total.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import Settings
from src.contracts import (
    ADDRESS1,
    BIRTH_DT,
    EMAIL,
    FIRST_NM,
    LAST_NM,
    PATID,
    PATID_A,
    PATID_B,
    RunManifest,
    SEX,
    SSN_LAST4,
    ZIP_BASE,
)
from src.api import store

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest(run_id: str, settings: Settings) -> RunManifest:
    path = settings.runs_dir / f"run_{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest for run_id={run_id!r} at {path}")
    return RunManifest.model_validate_json(path.read_text())


def _resolve(manifest_path: str, settings: Settings) -> Path:
    return settings.project_root / manifest_path


def _clean_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def _confidence_by_patid(matches: pd.DataFrame) -> dict[str, list[float]]:
    """Every PATID's confidences from the matches it participates in (either side)."""
    out: dict[str, list[float]] = {}
    if matches.empty:
        return out
    for a, b, conf in zip(matches[PATID_A], matches[PATID_B], matches["confidence"]):
        out.setdefault(a, []).append(float(conf))
        out.setdefault(b, []).append(float(conf))
    return out


def _attrs_index(cleaned: pd.DataFrame) -> dict[str, dict]:
    """`PATID -> {column: value}` for the display columns, one-time vectorized
    conversion — avoids a pandas `.loc[patid]` call per record."""
    attrs = cleaned.drop_duplicates(subset=PATID, keep="first").set_index(PATID)
    birth_date_str = attrs[BIRTH_DT].dt.strftime("%Y-%m-%d")
    attrs = attrs.assign(**{BIRTH_DT: birth_date_str})
    return attrs.to_dict(orient="index")


def _attrs_row(patid: str, attrs_by_patid: dict[str, dict], run_id: str) -> tuple | None:
    row = attrs_by_patid.get(patid)
    if row is None:
        return None
    birth_date = row.get(BIRTH_DT)
    return (
        patid,
        _clean_str(row.get(FIRST_NM)),
        _clean_str(row.get(LAST_NM)),
        birth_date if isinstance(birth_date, str) else None,
        _clean_str(row.get(SSN_LAST4)),
        _clean_str(row.get(EMAIL)),
        _clean_str(row.get(ZIP_BASE)),
        _clean_str(row.get(ADDRESS1)),
        _clean_str(row.get(SEX)),
        run_id,
    )


def publish_run(conn, run_id: str, settings: Settings) -> dict:
    """Publish one run's final output into `empi.db`. Returns summary counts."""
    manifest = _load_manifest(run_id, settings)

    clusters = pd.read_parquet(_resolve(manifest.clusters.path, settings))
    matches = pd.read_parquet(_resolve(manifest.matches.path, settings))
    cleaned = pd.read_parquet(_resolve(manifest.cleaned.path, settings))

    attrs_by_patid = _attrs_index(cleaned)
    conf_by_patid = _confidence_by_patid(matches)
    locked = store.locked_patids(conn)
    existing_mid_by_patid = store.all_entity_member_mids(conn)
    next_seq = store.max_mid_sequence(conn) + 1
    now = _now()

    counts = {
        "clusters_seen": 0,
        "entities_upserted": 0,
        "members_upserted": 0,
        "locked_skipped": 0,
        "suggestions_written": 0,
    }

    entity_rows: list[tuple] = []
    member_rows: list[tuple] = []
    attrs_rows: list[tuple] = []
    suggestion_rows: list[tuple] = []

    for cluster_id, group in clusters.groupby("cluster_id"):
        counts["clusters_seen"] += 1
        group_patids: list[str] = sorted(group[PATID].astype(str))

        unlocked = [p for p in group_patids if p not in locked]
        skipped = [p for p in group_patids if p in locked]
        counts["locked_skipped"] += len(skipped)

        for patid in skipped:
            suggestion_rows.append(
                (patid, run_id, f"SUGGESTED-{run_id}-{int(cluster_id)}", now)
            )
        counts["suggestions_written"] += len(skipped)

        if unlocked:
            existing_mids = {
                existing_mid_by_patid[p]
                for p in unlocked
                if p in existing_mid_by_patid
            }
            if existing_mids:
                target_mid = min(existing_mids)
            else:
                target_mid = f"M-{next_seq:06d}"
                next_seq += 1

            confidences = [c for p in unlocked for c in conf_by_patid.get(p, [])]
            confidence = max(confidences) if confidences else None
            is_merged = len(unlocked) > 1
            origin = "deterministic" if is_merged else "none"

            entity_rows.append(
                (target_mid, run_id, origin, int(is_merged), confidence, now)
            )
            counts["entities_upserted"] += 1

            primary = unlocked[0]  # lexicographically smallest PATID
            for patid in unlocked:
                member_rows.append(
                    (patid, target_mid, int(patid == primary), "pipeline", now)
                )
                existing_mid_by_patid[patid] = target_mid
            counts["members_upserted"] += len(unlocked)

        for patid in group_patids:
            row = _attrs_row(patid, attrs_by_patid, run_id)
            if row is not None:
                attrs_rows.append(row)

    conn.execute("BEGIN")
    try:
        store.upsert_entities_bulk(conn, entity_rows)
        store.upsert_entity_members_bulk(conn, member_rows)
        store.upsert_record_attrs_bulk(conn, attrs_rows)
        store.upsert_suggestions_bulk(conn, suggestion_rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    logger.info(
        "Published run %s: %d clusters, %d entities, %d members, "
        "%d locked-skipped, %d suggestions",
        run_id, counts["clusters_seen"], counts["entities_upserted"],
        counts["members_upserted"], counts["locked_skipped"],
        counts["suggestions_written"],
    )
    return counts


__all__ = ["publish_run"]
