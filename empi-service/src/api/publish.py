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

`confidence`/`match_rule`/`evidence` on a multi-member entity come from the
highest-confidence deterministic pair connecting its *unlocked* members — the
pipeline's evidence for the grouping actually being written.

Review-tier data (FR-7/8/19/20/21 in the Dashboard FR doc): `non_matches`
(the full review queue — review-tier rule-confirmed pairs + uncertain pairs)
and the companion `review_evidence` artifact (rule provenance for the
rule-confirmed subset — see `src/pipeline.py`) become `review_candidate` rows.
A singleton entity whose sole member appears in the review queue gets
`origin='review'` instead of `'none'`.

Raw fields (FR-24, "View Raw Data" drawer): every `*_raw` passthrough column
from the cleaned dataset is denormalized into `record_raw` as one JSON blob
per PATID.

PERFORMANCE: a real run clusters ~160k records into ~140k entities (see
`to-do.md` / the full-population smoke run). The whole publish runs as ONE
SQLite transaction, so it must be fast — issuing one `execute()` (and worse,
one lookup `SELECT`) per record made an early version of this function take
minutes and hold a write lock the whole time, causing "database is locked"
for any concurrent reader. Everything below is planned in pure Python/pandas
first (dict lookups, no DB round-trips per row), then written with a handful
of `executemany` calls total.
"""

from __future__ import annotations

import json
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

#: `*_raw` passthrough columns on the cleaned dataset (see
#: `src/preprocessing/clean.py` / `docs/Data-Cleaning-Guide.md`) — the
#: un-scrubbed source fields for the raw-data drawer. Kept as a module
#: constant rather than importing from `contracts` since these are
#: passthrough-only columns the pipeline's stage boundaries don't depend on.
RAW_COLUMNS: tuple[str, ...] = (
    "FirstNM_raw", "LastNM_raw", "MiddleNM_raw", "SuffixNM_raw", "BirthDT_raw",
    "SSN_raw", "AddressLine1_raw", "AddressLine2_raw", "CityNM_raw", "ZipCD_raw",
    "StateCD_raw", "CountryNM", "PrimaryPhoneNBR_raw", "Phone01NBR_raw",
    "Phone02NBR_raw", "Phone03NBR_raw", "Email_raw", "SexAtBirthDSC_raw",
)

#: Not in contracts.py's canonical columns (only the parsed multi-phone
#: PHONES/Phones_set is) — the single primary cleaned phone, for the
#: NAME_DOB_PHONE rule's "key matching feature" display.
PHONE_CLEAN = "PrimaryPhoneNBR_clean"


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


def _json_safe(value: object) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    return value


def _pair_evidence_index(
    matches: pd.DataFrame,
) -> dict[frozenset, tuple[float, str, str]]:
    """`{a, b} -> (confidence, match_rule, rules_fired)` for every deterministic
    pair, keyed by the unordered PATID pair — used to find the founding pair's
    evidence for a merged entity without relying on `matches.cluster_id`
    numbering staying aligned with `ClusterAssignments.cluster_id`."""
    out: dict[frozenset, tuple[float, str, str]] = {}
    if matches.empty:
        return out
    for a, b, conf, rule, fired in zip(
        matches[PATID_A], matches[PATID_B], matches["confidence"],
        matches["match_rule"], matches["rules_fired"],
    ):
        out[frozenset((a, b))] = (float(conf), rule, fired)
    return out


def _best_pair_evidence(
    unlocked: list[str], pair_evidence: dict[frozenset, tuple[float, str, str]]
) -> tuple[float, str, str] | None:
    """Highest-confidence deterministic pair with both endpoints in `unlocked`.
    Clusters are small in practice (max observed size 7 on the real dataset —
    see the full-run validation), so the O(k^2) pairwise scan is cheap."""
    best = None
    for i, a in enumerate(unlocked):
        for b in unlocked[i + 1:]:
            found = pair_evidence.get(frozenset((a, b)))
            if found and (best is None or found[0] > best[0]):
                best = found
    return best


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
        _clean_str(row.get(PHONE_CLEAN)),
        run_id,
    )


def _raw_index(cleaned: pd.DataFrame) -> dict[str, dict]:
    present = [c for c in RAW_COLUMNS if c in cleaned.columns]
    raw = cleaned.drop_duplicates(subset=PATID, keep="first").set_index(PATID)[present]
    return raw.to_dict(orient="index")


def _raw_row(patid: str, raw_by_patid: dict[str, dict], run_id: str) -> tuple | None:
    row = raw_by_patid.get(patid)
    if row is None:
        return None
    payload = {k: _json_safe(v) for k, v in row.items()}
    return (patid, json.dumps(payload), run_id)


def _review_candidate_rows(
    non_matches: pd.DataFrame,
    review_evidence: pd.DataFrame | None,
    run_id: str,
    now: str,
) -> tuple[list[tuple], set[str]]:
    """`non_matches` is the full review queue (review-tier rule-confirmed +
    uncertain pairs); `review_evidence` (may be `None` for pre-existing
    manifests without it) carries `match_rule`/`rules_fired` for the
    rule-confirmed subset. Returns (rows, {every PATID in the queue})."""
    if non_matches.empty:
        return [], set()

    evidence_by_pair: dict[frozenset, tuple[str, float, str]] = {}
    if review_evidence is not None and not review_evidence.empty:
        for a, b, rule, conf, fired in zip(
            review_evidence[PATID_A], review_evidence[PATID_B],
            review_evidence["match_rule"], review_evidence["confidence"],
            review_evidence["rules_fired"],
        ):
            evidence_by_pair[frozenset((a, b))] = (rule, float(conf), fired)

    rows: list[tuple] = []
    patids: set[str] = set()
    for a, b, blocks in zip(
        non_matches[PATID_A], non_matches[PATID_B], non_matches["source_blocks"]
    ):
        patids.add(a)
        patids.add(b)
        rule, conf, fired = evidence_by_pair.get(frozenset((a, b)), (None, None, None))
        rows.append((a, b, rule, conf, fired, blocks, run_id, now))
    return rows, patids


def publish_run(conn, run_id: str, settings: Settings) -> dict:
    """Publish one run's final output into `empi.db`. Returns summary counts."""
    manifest = _load_manifest(run_id, settings)

    clusters = pd.read_parquet(_resolve(manifest.clusters.path, settings))
    matches = pd.read_parquet(_resolve(manifest.matches.path, settings))
    non_matches = pd.read_parquet(_resolve(manifest.non_matches.path, settings))
    cleaned = pd.read_parquet(_resolve(manifest.cleaned.path, settings))
    review_evidence = (
        pd.read_parquet(_resolve(manifest.review_evidence.path, settings))
        if manifest.review_evidence is not None
        else None
    )

    attrs_by_patid = _attrs_index(cleaned)
    raw_by_patid = _raw_index(cleaned)
    conf_by_patid = _confidence_by_patid(matches)
    pair_evidence = _pair_evidence_index(matches)
    locked = store.locked_patids(conn)
    existing_mid_by_patid = store.all_entity_member_mids(conn)
    next_seq = store.max_mid_sequence(conn) + 1
    now = _now()

    review_candidate_rows, review_patids = _review_candidate_rows(
        non_matches, review_evidence, run_id, now
    )

    counts = {
        "clusters_seen": 0,
        "entities_upserted": 0,
        "members_upserted": 0,
        "locked_skipped": 0,
        "suggestions_written": 0,
        "review_candidates": len(review_candidate_rows),
    }

    entity_rows: list[tuple] = []
    member_rows: list[tuple] = []
    attrs_rows: list[tuple] = []
    raw_rows: list[tuple] = []
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

            if is_merged:
                origin = "deterministic"
                match_rule = evidence = None
                best = _best_pair_evidence(unlocked, pair_evidence)
                if best is not None:
                    confidence, match_rule, evidence = best
            elif unlocked[0] in review_patids:
                origin, match_rule, evidence = "review", None, None
            else:
                origin, match_rule, evidence = "none", None, None

            entity_rows.append(
                (target_mid, run_id, origin, int(is_merged), confidence,
                 match_rule, evidence, now)
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
            raw_row = _raw_row(patid, raw_by_patid, run_id)
            if raw_row is not None:
                raw_rows.append(raw_row)

    conn.execute("BEGIN")
    try:
        store.upsert_entities_bulk(conn, entity_rows)
        store.upsert_entity_members_bulk(conn, member_rows)
        store.upsert_record_attrs_bulk(conn, attrs_rows)
        store.upsert_record_raw_bulk(conn, raw_rows)
        store.upsert_suggestions_bulk(conn, suggestion_rows)
        store.replace_review_candidates_for_run(conn, run_id, review_candidate_rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    logger.info(
        "Published run %s: %d clusters, %d entities, %d members, "
        "%d locked-skipped, %d suggestions, %d review candidates",
        run_id, counts["clusters_seen"], counts["entities_upserted"],
        counts["members_upserted"], counts["locked_skipped"],
        counts["suggestions_written"], counts["review_candidates"],
    )
    return counts


__all__ = ["publish_run"]
