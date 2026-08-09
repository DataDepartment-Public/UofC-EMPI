"""Incremental single/batch record scoring — `POST /records/score`.

Scores one or more new raw records against the *existing* published
population and writes the outcome straight into `empi.db`, without
re-running `clean -> block -> rules -> cluster` over the whole dataset. The
gap this closes: `src/pipeline.py`'s `run_stacked_blocking` always computes
candidates over the full cleaned frame, and `deterministic_rules.apply_rules`
needs the full `*_clean` attributes for both sides of a pair. Neither of
those is available (or affordable) for a single incoming record without the
persisted index this module reads and maintains:

    raw record -> transform_dataframe (Stage 1, unchanged)
                -> backend.lookup_block_candidates (point lookups against the
                   persisted block-key index, the on-disk mirror of
                   blocking.BlockingIndex that a full publish rebuilds)
                -> backend.get_cleaned_attrs (persisted mirror of the
                   CleanedRecords contract for just the handful of matched
                   candidates)
                -> deterministic_rules.apply_rules / classify_non_matches
                   (Stage 3, unchanged, run over a tiny in-memory frame)
                -> auto-merge tier: join/bridge existing entities directly
                   (bypassing clustering.assign_clusters, which needs the
                   full matches set)
                -> review tier: review_candidate rows, optionally scored by
                   the active FS model (Stage 4, unchanged)

Storage is behind `src.api.backends.index_backend.IndexBackend` — every function here
takes a `backend`, never a raw connection, and never imports `store`
directly. Two implementations exist: `SqlIndexBackend` (the live FastAPI
service, `empi.db`) and `ParquetIndexBackend` (local-mode CLI,
`src/api/ingest/local_score.py`, or the API via `EMPI_INDEX_BACKEND=parquet`) — see
`index_backend.py` for the swap point.

Processes records one at a time and appends each one's own block-key /
cleaned-attrs rows immediately after scoring it — so two near-duplicate
records submitted in the same batch correctly match each other (the second
one's candidate lookup finds the first), without any special-casing.

Sticky-unmerge (docs/API-Design.md §2, §6): a candidate that is
reviewer-locked (`backend.locked_patids`) is never auto-merged into by an
incremental record; the would-be grouping is written to `entity_suggestion`
instead, exactly like `publish.py`'s handling of a locked cluster member.
(`ParquetIndexBackend.locked_patids` always returns empty — local mode has no
reviewer merge/unmerge UI to lock anything.)

KNOWN LIMITATION: `high_fanout_ssn` (an audit/review flag, not a routing
decision) is computed by `apply_rules` over the tiny per-call frame, not the
full population, so it under-counts relative to a batch run. Not worth a
bespoke aggregate query for a flag that never changes match/reject outcome.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from src.config import Settings
from src.contracts import (
    ADDRESS1,
    BIRTH_DT,
    CLEANED_REQUIRED_COLUMNS,
    EMAIL,
    FIRST_NM,
    LAST_NM,
    PATID,
    PHONES,
    SEX,
    SSN,
    SSN_LAST4,
    TIER_HUMAN_REVIEW,
    TIER_NO_MATCH,
    VALID_RECORD,
    ZIP_BASE,
)
from src.api.backends.index_backend import IndexBackend
from src.models import model_cache
from src.models.deterministic_rules import (
    AUTO_MERGE_RULES,
    apply_rules,
    classify_non_matches,
)
from src.models.fs_matcher.registry import resolve_active_model
from src.preprocessing.blocking import (
    HASHED_BLOCKS,
    compute_new_record_keys,
    hash_block_key,
)
from src.preprocessing.transformations import transform_dataframe

logger = logging.getLogger(__name__)

_SCALAR_BLOCK_IDS: tuple[str, ...] = ("B1", "B3", "B4", "B6", "B7", "B8", "B9")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


# ── Stage 1: clean one record, same normalization the batch CSV ingest uses ─
def _clean_one_record(raw: dict) -> pd.Series:
    """Run one raw record through `transform_dataframe`. `raw` must use the
    pipeline's raw column names (`PATID` + `transformations.RENAME_TO_RAW`:
    FirstNM, LastNM, BirthDT, SSN, AddressLine1, Email, ...), not the cleaned
    `*_clean` names."""
    df = pd.DataFrame([raw])
    cleaned = transform_dataframe(df)
    return cleaned.iloc[0]


# ── Persisted-index helpers (own record) ────────────────────────────────────
def _lookup_keys(raw_keys: dict[str, object]) -> dict[str, str | None]:
    """`compute_new_record_keys` output, hashed for B1/B6 — the shape
    `backend.lookup_block_candidates` / `backend.add_block_keys` expect."""
    return {
        block_id: (
            hash_block_key(str(raw_keys[block_id]))
            if block_id in HASHED_BLOCKS and raw_keys[block_id]
            else raw_keys[block_id]
        )
        for block_id in _SCALAR_BLOCK_IDS
    }


def _own_block_key_rows(
    lookup_keys: dict[str, str | None], phones: set[str], patid: str
) -> list[tuple[str, str, str]]:
    rows = [
        (block_id, key_value, patid)
        for block_id, key_value in lookup_keys.items()
        if key_value
    ]
    rows.extend(("B5", phone, patid) for phone in phones if phone)
    return rows


def _own_cleaned_attrs_row(cleaned_row: pd.Series, run_id: str) -> tuple:
    birth_dt = cleaned_row.get(BIRTH_DT)
    birth_str = birth_dt.strftime("%Y-%m-%d") if pd.notna(birth_dt) else None
    phones = cleaned_row.get(PHONES)
    phones_list = sorted(phones) if isinstance(phones, (set, frozenset)) else []
    return (
        str(cleaned_row[PATID]),
        _clean_str(cleaned_row.get(FIRST_NM)),
        _clean_str(cleaned_row.get(LAST_NM)),
        birth_str,
        _clean_str(cleaned_row.get(SSN)),
        _clean_str(cleaned_row.get(SSN_LAST4)),
        _clean_str(cleaned_row.get(EMAIL)),
        _clean_str(cleaned_row.get(ZIP_BASE)),
        _clean_str(cleaned_row.get(ADDRESS1)),
        _clean_str(cleaned_row.get(SEX)),
        json.dumps(phones_list),
        run_id,
    )


# ── Reconstructing a tiny CleanedRecords-shaped frame for rule evaluation ───
def _candidate_row_to_contract(row: dict) -> dict:
    birth_dt = pd.to_datetime(row["birth_dt"]) if row["birth_dt"] else pd.NaT
    phones = set(json.loads(row["phones_json"])) if row["phones_json"] else set()
    return {
        PATID: row["patid"], FIRST_NM: row["first_nm"], LAST_NM: row["last_nm"],
        BIRTH_DT: birth_dt, SSN: row["ssn"], SSN_LAST4: row["ssn_last4"],
        EMAIL: row["email"], ZIP_BASE: row["zip_base"], ADDRESS1: row["address1"],
        SEX: row["sex"], PHONES: phones, VALID_RECORD: True,
    }


def _mini_cleaned_frame(cleaned_row: pd.Series, candidate_attrs: list[dict]) -> pd.DataFrame:
    new_row = {col: cleaned_row.get(col) for col in CLEANED_REQUIRED_COLUMNS}
    rows = [new_row, *(_candidate_row_to_contract(r) for r in candidate_attrs)]
    return pd.DataFrame(rows, columns=list(CLEANED_REQUIRED_COLUMNS))


# ── One record ───────────────────────────────────────────────────────────────
def _score_one_record(
    backend: IndexBackend, settings: Settings, raw: dict, run_id: str
) -> dict:
    patid = raw.get("PATID")
    if not patid:
        raise ValueError("Each record must include a non-empty PATID.")
    patid = str(patid)
    now = _now()

    cleaned_row = _clean_one_record(raw)
    outcome: dict = {
        "patid": patid, "tier": "invalid", "mid": None, "match_rule": None,
        "confidence": None, "matched_patids": [],
        "fs_match_probability": None, "fs_classification_tier": None,
    }
    if not bool(cleaned_row.get(VALID_RECORD, False)):
        logger.info("Incremental score: PATID %s failed cleaning validity checks", patid)
        return outcome

    raw_keys = compute_new_record_keys(cleaned_row)
    lookup_keys = _lookup_keys(raw_keys)
    phones = raw_keys["phones"]
    threshold = settings.governance_threshold

    candidates = backend.lookup_block_candidates(lookup_keys, phones, threshold)
    # A caller resubmitting/updating an already-published PATID must never
    # match itself (its own prior block keys are already in the index).
    candidates.pop(patid, None)

    matches = pd.DataFrame(columns=["PATID_A", "PATID_B", "match_rule", "confidence", "rules_fired"])
    non_matches = pd.DataFrame(columns=["PATID_A", "PATID_B", "source_blocks", "n_blocks"])
    review_confirmed = matches
    mini_clean: pd.DataFrame | None = None

    if candidates:
        candidate_pairs = pd.DataFrame({
            "PATID_A": patid,
            "PATID_B": list(candidates.keys()),
            "source_blocks": ["|".join(sorted(v)) for v in candidates.values()],
            "n_blocks": [len(v) for v in candidates.values()],
        })
        candidate_attrs = backend.get_cleaned_attrs(list(candidates.keys()))
        mini_clean = _mini_cleaned_frame(cleaned_row, candidate_attrs)

        confirmed = apply_rules(
            candidate_pairs, mini_clean,
            ssn_fanout_threshold=settings.ssn_fanout_threshold,
        )
        is_auto = confirmed["match_rule"].isin(AUTO_MERGE_RULES)
        matches = confirmed[is_auto].reset_index(drop=True)
        review_confirmed = confirmed[~is_auto].reset_index(drop=True)

        decided = classify_non_matches(candidate_pairs, confirmed, mini_clean)
        pair_cols = ["PATID_A", "PATID_B", "source_blocks", "n_blocks"]
        non_matches = pd.concat(
            [review_confirmed[pair_cols], decided[decided["decision"] == TIER_HUMAN_REVIEW][pair_cols]]
        ).reset_index(drop=True)

    target_mid = _resolve_auto_merge(backend, patid, matches, run_id, now, outcome)
    has_review = not non_matches.empty
    if has_review:
        _persist_review_candidates(
            backend, settings, non_matches, review_confirmed, mini_clean, run_id, now,
        )

    if target_mid is None:
        target_mid = backend.next_mid()
        origin = "review" if has_review else "none"
        backend.upsert_entity(
            target_mid, run_id, origin, is_merged=False,
            confidence=None, updated_utc=now,
        )
        backend.upsert_entity_member(
            patid, target_mid, is_primary=True, added_by="pipeline", updated_utc=now,
        )
        outcome["tier"] = TIER_HUMAN_REVIEW if has_review else TIER_NO_MATCH

    outcome["mid"] = target_mid

    # Index this record's own keys/attrs LAST — after candidate lookup, so it
    # never matches itself; other records later in the same batch now can.
    backend.add_block_keys(_own_block_key_rows(lookup_keys, phones, patid))
    backend.upsert_cleaned_attrs(_own_cleaned_attrs_row(cleaned_row, run_id))

    return outcome


def _resolve_auto_merge(
    backend: IndexBackend, patid: str, matches: pd.DataFrame, run_id: str,
    now: str, outcome: dict,
) -> str | None:
    """Join/bridge existing entities for every unlocked auto-merge candidate.
    Returns the resulting `mid`, or None if there were none (or all were
    reviewer-locked, in which case a suggestion is written instead)."""
    if matches.empty:
        return None

    locked = backend.locked_patids()
    auto_patids = sorted(set(matches["PATID_B"]))
    unlocked = [p for p in auto_patids if p not in locked]
    locked_matched = [p for p in auto_patids if p in locked]

    if locked_matched:
        best = matches[matches["PATID_B"].isin(locked_matched)]
        best_row = best.loc[best["confidence"].idxmax()]
        suggested_mid = backend.get_entity_mid_for_patid(best_row["PATID_B"])
        if suggested_mid:
            backend.upsert_suggestion(patid, run_id, suggested_mid, now)

    if not unlocked:
        return None

    existing_mids = sorted({
        mid for mid in (backend.get_entity_mid_for_patid(p) for p in unlocked)
        if mid is not None
    })
    target_mid = existing_mids[0] if existing_mids else backend.next_mid()
    if len(existing_mids) > 1:
        backend.reassign_entity_members(existing_mids[1:], target_mid, now)

    own_best = matches.loc[matches["confidence"].idxmax()]
    confidence, match_rule, evidence = (
        float(own_best["confidence"]), own_best["match_rule"], own_best["rules_fired"],
    )
    prior = backend.get_entity(target_mid) if existing_mids else None
    if prior is not None and prior["entity"]["confidence"] is not None \
            and prior["entity"]["confidence"] > confidence:
        confidence = prior["entity"]["confidence"]
        match_rule = prior["entity"]["match_rule"]
        evidence = prior["entity"]["evidence"]

    backend.upsert_entity(
        target_mid, run_id, "deterministic", is_merged=True,
        confidence=confidence, updated_utc=now,
        match_rule=match_rule, evidence=evidence,
    )
    backend.upsert_entity_member(
        patid, target_mid, is_primary=False, added_by="pipeline", updated_utc=now,
    )
    outcome.update(tier="auto_merge", match_rule=match_rule, confidence=confidence,
                    matched_patids=unlocked)
    return target_mid


def _persist_review_candidates(
    backend: IndexBackend, settings: Settings, non_matches: pd.DataFrame,
    review_confirmed: pd.DataFrame, mini_clean: pd.DataFrame | None,
    run_id: str, now: str,
) -> None:
    fs_scores: dict[tuple[str, str], tuple[float, str]] = {}
    active_model = resolve_active_model(settings)
    if active_model is not None and mini_clean is not None:
        # Lazy import — keeps splink/duckdb out of the path when no model is
        # active or there is nothing to score, mirroring src/pipeline.py Stage 4.
        from src.models.fs_matcher.matcher import (
            FSMatcher,
            classification_config_from_settings,
        )

        model = FSMatcher(classification_config=classification_config_from_settings(settings))
        trained = model_cache.get_or_load("fs_matcher", active_model, FSMatcher.load_settings)
        classified = model.score(non_matches, mini_clean, trained)
        prob_matches = model.to_probabilistic_matches(classified)
        for a, b, score, tier in zip(
            prob_matches["PATID_A"], prob_matches["PATID_B"],
            prob_matches["score"], prob_matches["classification_tier"],
        ):
            fs_scores[(a, b)] = (float(score), str(tier))

    evidence_by_pair: dict[tuple[str, str], tuple[str, float, str]] = {}
    if not review_confirmed.empty:
        for a, b, rule, conf, fired in zip(
            review_confirmed["PATID_A"], review_confirmed["PATID_B"],
            review_confirmed["match_rule"], review_confirmed["confidence"],
            review_confirmed["rules_fired"],
        ):
            evidence_by_pair[(a, b)] = (rule, float(conf), fired)

    rows = []
    for a, b, blocks in zip(
        non_matches["PATID_A"], non_matches["PATID_B"], non_matches["source_blocks"]
    ):
        rule, conf, fired = evidence_by_pair.get((a, b), (None, None, None))
        fs_prob, fs_tier = fs_scores.get((a, b), (None, None))
        rows.append((a, b, rule, conf, fired, blocks, run_id, now, fs_prob, fs_tier))

    backend.insert_review_candidates(rows)


# ── Public entry point ───────────────────────────────────────────────────────
def score_records(
    backend: IndexBackend, settings: Settings, records: list[dict], run_id: str
) -> list[dict]:
    """Score every raw record in `records`, one at a time (see module
    docstring for why sequential processing is required), each in its own
    `backend` transaction. Returns one outcome dict per input record, in
    order: `{"patid", "tier", "mid", "match_rule", "confidence",
    "matched_patids", "fs_match_probability", "fs_classification_tier"}`;
    `tier` is one of `"auto_merge" | "human_review" | "no_match" | "invalid"`.
    """
    outcomes = []
    for raw in records:
        backend.begin()
        try:
            outcome = _score_one_record(backend, settings, raw, run_id)
            backend.commit()
        except Exception:
            backend.rollback()
            raise
        outcomes.append(outcome)
        logger.info(
            "Incremental score: PATID %s -> tier=%s mid=%s",
            outcome["patid"], outcome["tier"], outcome["mid"],
        )
    return outcomes


__all__ = ["score_records"]
