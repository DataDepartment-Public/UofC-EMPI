"""Publish step: one pipeline run's Parquet output → the resolved-output
index (`empi.db` / SQLite, or the local Parquet index — see
docs/Data-Contract.md Stage 6).

Loads a `RunManifest`, groups the run's `ClusterAssignments` into entities,
and upserts them via `src.api.backends.index_backend.IndexBackend` — never
`sql_backend` directly, and never a raw connection, exactly like
`src/api/ingest/incremental.py` (see that module's docstring for why the
storage layer is backend-agnostic).
Never touches the Parquet pipeline artifacts themselves — they stay the
immutable record of "what the algorithm produced".

Reconciliation (docs/API-Design.md §2, §6 open decision 1 — "sticky unmerge"):
a PATID `backend.locked_patids` reports as reviewer-touched is **never**
repointed to a new `mid` here. Its would-be new grouping is written to
`entity_suggestion` instead — visible for a future admin/reviewer view, not
auto-applied. Untouched PATIDs upsert normally: reuse an existing `mid` if any
unlocked member of the new cluster already has one (ties broken by the
smallest existing `mid`), else mint a fresh one.

`confidence`/`match_rule`/`evidence` on a multi-member entity come from the
highest-confidence deterministic pair connecting its *unlocked* members — the
pipeline's evidence for the grouping actually being written.

Review-tier data (FR-7/8/19/20/21 in the Dashboard FR doc): `non_matches`
minus whatever the Stage-4.25 gate (`gate_results`) confidently dropped as
`TIER_NO_MATCH` is the review queue — those dropped pairs already carry a
decisive automated non-match decision and resolve straight to `origin='none'`
instead of surfacing for a human. The companion `review_evidence` artifact
(rule provenance for the rule-confirmed subset — see `src/pipeline.py`)
supplies `match_rule`/`rules_fired` for that subset. `gate_results` is `None`
on manifests with no gate model (legacy FS-gate fallback, or no gate
available at all — Stage 4.25 in `src/pipeline.py`), in which case every
`non_matches` pair is still review-eligible, matching pre-gate behavior.
A singleton entity whose sole member appears in the review queue gets
`origin='review'` instead of `'none'`.

Raw fields (FR-24, "View Raw Data" drawer): every `*_raw` passthrough column
from the cleaned dataset is denormalized into `record_raw` as one JSON blob
per PATID.

Incremental-scoring index (`block_key` / `cleaned_attrs` — see
`src/api/ingest/incremental.py`): every full publish also rebuilds these two tables
wholesale from `cleaned` — `block_key` is the on-disk equivalent of
`blocking.BlockingIndex` (a persisted posting list so a later single-record
score doesn't need to reload the whole population to find candidates), and
`cleaned_attrs` is a queryable mirror of the `CleanedRecords` contract (so
rule evaluation for a handful of candidates doesn't need a Parquet read
either). Both are the self-healing anchor point for indexes `incremental.py`
otherwise appends to a record at a time.

PERFORMANCE: a real run clusters ~160k records into ~140k entities (see
`to-do.md` / the full-population smoke run). The whole publish runs as ONE
backend transaction, so it must be fast — issuing one call (and worse, one
lookup) per record made an early version of this function take minutes and
hold a write lock the whole time, causing "database is locked" for any
concurrent reader (on the SQLite backend). Everything below is planned in
pure Python/pandas first (dict lookups, no backend round-trips per row), then
written with a handful of `*_bulk`/`replace_*` backend calls total.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.api.backends.index_backend import IndexBackend
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
    PHONES,
    RunManifest,
    SEX,
    SSN,
    SSN_LAST4,
    TIER_NO_MATCH,
    VALID_RECORD,
    ZIP_BASE,
)
from src.preprocessing.blocking import (
    HASHED_BLOCKS,
    _parse_phone_set,
    build_blocking_index,
    hash_block_key,
)

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
    gate_results: pd.DataFrame | None,
    ml_features: pd.DataFrame | None,
    run_id: str,
    now: str,
) -> tuple[list[tuple], set[str]]:
    """The review queue is `non_matches` (rule-uncertain pairs) minus
    whatever the Stage-4.25 gate confidently dropped as `TIER_NO_MATCH` in
    `gate_results` — those pairs already have a decisive automated
    non-match decision (see `src/models/nonmatch_gate/gate.py`) and belong
    in `origin='none'`, not the human queue. `gate_results` is `None` when
    no gate model scored this run (legacy FS-gate fallback or no gate
    available — Stage 4.25 in `src/pipeline.py`); every `non_matches` pair
    stays review-eligible in that case. `review_evidence` (may be `None`
    for pre-existing manifests without it) carries `match_rule`/
    `rules_fired` for the rule-confirmed subset. `ml_features` (may be
    `None` when no ML model scored this run) carries the Stage 4.5 ML
    matcher's `match_probability`/`classification_tier` — most queue pairs
    have no rule confidence at all, so without this the list/sort/filter
    have nothing but insertion order to fall back on for the ML-scored
    majority (see `sql_backend.list_review_candidates`'s `COALESCE`).
    Returns (rows, {every PATID in the queue})."""
    if non_matches.empty:
        return [], set()

    if gate_results is not None and not gate_results.empty:
        dropped = gate_results.loc[
            gate_results["predicted_tier"] == TIER_NO_MATCH, [PATID_A, PATID_B]
        ]
        if not dropped.empty:
            non_matches = non_matches.merge(
                dropped.assign(_gate_dropped=True), on=[PATID_A, PATID_B], how="left"
            )
            non_matches = non_matches[non_matches["_gate_dropped"].isna()].drop(
                columns="_gate_dropped"
            )
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

    ml_by_pair: dict[frozenset, tuple[float, str]] = {}
    if ml_features is not None and not ml_features.empty:
        for a, b, proba, tier in zip(
            ml_features[PATID_A], ml_features[PATID_B],
            ml_features["match_probability"], ml_features["classification_tier"],
        ):
            ml_by_pair[frozenset((a, b))] = (float(proba), tier)

    rows: list[tuple] = []
    patids: set[str] = set()
    for a, b, blocks in zip(
        non_matches[PATID_A], non_matches[PATID_B], non_matches["source_blocks"]
    ):
        patids.add(a)
        patids.add(b)
        rule, conf, fired = evidence_by_pair.get(frozenset((a, b)), (None, None, None))
        ml_proba, ml_tier = ml_by_pair.get(frozenset((a, b)), (None, None))
        rows.append((a, b, rule, conf, fired, blocks, run_id, now, ml_proba, ml_tier))
    return rows, patids


def _cleaned_attrs_rows(cleaned: pd.DataFrame, run_id: str) -> list[tuple]:
    """Full-population rows for `sql_backend.replace_cleaned_attrs` — restricted to
    `valid_record` rows, matching what `build_blocking_index` itself includes
    (an invalid record is never a blocking/rule-evaluation candidate).
    Column order matches `sql_backend._CLEANED_ATTRS_COLUMNS`."""
    valid = cleaned[cleaned[VALID_RECORD].astype(bool)]
    if valid.empty:
        return []
    dedup = valid.drop_duplicates(subset=PATID, keep="first")
    birth_dt_str = dedup[BIRTH_DT].dt.strftime("%Y-%m-%d")
    rows: list[tuple] = []
    for patid, first, last, birth, ssn, ssn4, email, zip_base, addr, sex, phones in zip(
        dedup[PATID], dedup[FIRST_NM], dedup[LAST_NM], birth_dt_str,
        dedup[SSN], dedup[SSN_LAST4], dedup[EMAIL], dedup[ZIP_BASE],
        dedup[ADDRESS1], dedup[SEX], dedup[PHONES],
    ):
        rows.append((
            patid, _clean_str(first), _clean_str(last),
            birth if isinstance(birth, str) else None,
            _clean_str(ssn), _clean_str(ssn4), _clean_str(email),
            _clean_str(zip_base), _clean_str(addr), _clean_str(sex),
            json.dumps(sorted(_parse_phone_set(phones))),
            run_id,
        ))
    return rows


def _block_key_rows(blocking_index) -> list[tuple]:
    """Invert a `blocking.BlockingIndex`'s posting lists into `(block_id,
    key_value, patid)` rows for `backend.replace_block_keys`. B1/B6 key
    values are hashed (`hash_block_key`) since they are direct identifiers
    (full SSN, email) — every other block's key is already a lossy derived
    signal (phonetic code, zip, birth year, name prefix)."""
    rows: list[tuple] = []
    scalar_indexes = {
        "B1": blocking_index.b1_index, "B3": blocking_index.b3_index,
        "B4": blocking_index.b4_index, "B6": blocking_index.b6_index,
        "B7": blocking_index.b7_index, "B8": blocking_index.b8_index,
        "B9": blocking_index.b9_index,
    }
    for block_id, index in scalar_indexes.items():
        hashed = block_id in HASHED_BLOCKS
        for key_value, patids in index.items():
            stored_key = hash_block_key(str(key_value)) if hashed else str(key_value)
            for patid in patids:
                rows.append((block_id, stored_key, patid))
    for phone, patids in blocking_index.b5_phone_index.items():
        for patid in patids:
            rows.append(("B5", phone, patid))
    return rows


def publish_run(backend: IndexBackend, run_id: str, settings: Settings) -> dict:
    """Publish one run's final output via `backend` (SQLite or Parquet local
    mode — see `src/api/backends/index_backend.py`). Returns summary counts."""
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
    gate_results = (
        pd.read_parquet(_resolve(manifest.gate_results.path, settings))
        if manifest.gate_results is not None
        else None
    )
    ml_features = (
        pd.read_parquet(_resolve(manifest.ml_features.path, settings))
        if manifest.ml_features is not None
        else None
    )

    attrs_by_patid = _attrs_index(cleaned)
    raw_by_patid = _raw_index(cleaned)
    conf_by_patid = _confidence_by_patid(matches)
    pair_evidence = _pair_evidence_index(matches)
    locked = backend.locked_patids()
    existing_mid_by_patid = backend.all_entity_member_mids()
    next_seq = backend.max_mid_sequence() + 1
    now = _now()

    review_candidate_rows, review_patids = _review_candidate_rows(
        non_matches, review_evidence, gate_results, ml_features, run_id, now
    )

    # Incremental-scoring index rebuild — pure Python/pandas, no DB round-trips
    # (see module docstring's PERFORMANCE note; same principle applies here).
    cleaned_attrs_rows = _cleaned_attrs_rows(cleaned, run_id)
    block_key_rows = _block_key_rows(build_blocking_index(cleaned))

    counts = {
        "clusters_seen": 0,
        "entities_upserted": 0,
        "members_upserted": 0,
        "locked_skipped": 0,
        "suggestions_written": 0,
        "review_candidates": len(review_candidate_rows),
        "cleaned_attrs_indexed": len(cleaned_attrs_rows),
        "block_keys_indexed": len(block_key_rows),
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

    backend.begin()
    try:
        backend.upsert_entities_bulk(entity_rows)
        backend.upsert_entity_members_bulk(member_rows)
        backend.upsert_record_attrs_bulk(attrs_rows)
        backend.upsert_record_raw_bulk(raw_rows)
        backend.upsert_suggestions_bulk(suggestion_rows)
        backend.replace_review_candidates_for_run(run_id, review_candidate_rows)
        backend.replace_cleaned_attrs(cleaned_attrs_rows)
        backend.replace_block_keys(block_key_rows)
        backend.commit()
    except Exception:
        backend.rollback()
        raise

    logger.info(
        "Published run %s: %d clusters, %d entities, %d members, "
        "%d locked-skipped, %d suggestions, %d review candidates, "
        "%d cleaned_attrs indexed, %d block_keys indexed",
        run_id, counts["clusters_seen"], counts["entities_upserted"],
        counts["members_upserted"], counts["locked_skipped"],
        counts["suggestions_written"], counts["review_candidates"],
        counts["cleaned_attrs_indexed"], counts["block_keys_indexed"],
    )
    return counts


__all__ = ["publish_run"]
