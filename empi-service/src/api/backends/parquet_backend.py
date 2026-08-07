"""`ParquetIndexBackend` — local-mode `IndexBackend` (see
`src/api/backends/index_backend.py`) backed by plain Parquet files instead of
`empi.db`. Nine files under a local-index directory mirror the SQLite
tables the resolved-output store uses (see docs/Data-Contract.md Stage 6):

    block_key.parquet          block_key
    cleaned_attrs.parquet       cleaned_attrs
    entity.parquet              entity
    entity_member.parquet       entity_member
    review_candidate.parquet    review_candidate
    entity_suggestion.parquet   entity_suggestion
    record_attrs.parquet        record_attrs
    record_raw.parquet          record_raw
    audit_log.parquet           audit_log

Loaded into memory once at construction, mutated in place, and written back
to disk on `commit()` — "maintained as if they were a database": no
concurrent-writer story, no partial-row locking, just read-modify-write per
transaction (one `score_records()` call, one `publish_run()` call, or one
`/audit/*` request). That's the right tradeoff for the intended use (local
dev/CLI, or a single-process local FastAPI deployment — see
`src/api/ingest/local_score.py` and `src/api/deps.get_backend`'s process-local lock),
not a general-purpose database replacement.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

_SCHEMAS: dict[str, list[str]] = {
    "block_key": ["block_id", "key_value", "patid"],
    "cleaned_attrs": [
        "patid", "first_nm", "last_nm", "birth_dt", "ssn", "ssn_last4",
        "email", "zip_base", "address1", "sex", "phones_json", "run_id",
    ],
    "entity": [
        "mid", "run_id", "origin", "is_merged", "confidence",
        "match_rule", "evidence", "updated_utc",
    ],
    "entity_member": ["patid", "mid", "is_primary", "added_by", "updated_utc"],
    "review_candidate": [
        "patid_a", "patid_b", "match_rule", "confidence", "evidence",
        "source_blocks", "run_id", "created_utc",
        "fs_match_probability", "fs_classification_tier",
    ],
    "entity_suggestion": ["patid", "run_id", "suggested_mid", "created_utc"],
    "record_attrs": [
        "patid", "first_name", "last_name", "birth_date", "ssn_last4",
        "email", "zip_code", "address1", "sex", "phone", "run_id",
    ],
    "record_raw": ["patid", "raw_json", "run_id"],
    "audit_log": [
        "id", "ts_utc", "user", "action", "patids", "mid",
        "prev_state", "next_state", "run_id", "prev_mid", "undo_of",
    ],
}

#: The 8 columns `publish.py`'s batch replace passes to
#: `replace_review_candidates_for_run` — no `fs_match_probability` /
#: `fs_classification_tier`, unlike `insert_review_candidates`'ish 10-column
#: incremental-append rows (batch publish never FS-scores a run; only
#: incremental scoring does). Matches `sql_backend._REVIEW_CANDIDATE_INSERT_SQL`'s
#: column list (as opposed to the ``_FULL_SQL`` variant).
_REVIEW_CANDIDATE_BATCH_COLUMNS: list[str] = [
    "patid_a", "patid_b", "match_rule", "confidence", "evidence",
    "source_blocks", "run_id", "created_utc",
]

#: `record_attrs` columns the dashboard read routes join in for display —
#: `run_id` deliberately excluded, matching `sql_backend.get_entity`'s SQL SELECT
#: (which also never pulls `record_attrs.run_id` into a member/candidate row).
_DISPLAY_ATTR_COLUMNS: list[str] = [
    "patid", "first_name", "last_name", "birth_date", "ssn_last4",
    "email", "zip_code", "address1", "sex", "phone",
]


def _row_to_dict(row: pd.Series) -> dict:
    """`pd.Series.to_dict()`, but NaN -> None — a value never written to a
    column comes back from Parquet as float NaN, not Python None, which
    would silently break `incremental.py`'s `is None` / truthiness checks
    (NaN is truthy)."""
    return {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}


class ParquetIndexBackend:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._tables: dict[str, pd.DataFrame] = {
            name: self._load(name) for name in _SCHEMAS
        }
        self._snapshot: dict[str, pd.DataFrame] | None = None

    def _path(self, name: str) -> Path:
        return self.data_dir / f"{name}.parquet"

    def _load(self, name: str) -> pd.DataFrame:
        path = self._path(name)
        if not path.exists():
            return pd.DataFrame(columns=_SCHEMAS[name])
        df = pd.read_parquet(path)
        # A column added to `_SCHEMAS[name]` after this file was last written
        # (e.g. `audit_log.prev_mid`/`undo_of`) won't exist on disk yet —
        # backfill it as all-missing rather than KeyError on first access,
        # mirroring `sql_backend._ensure_columns`'s ALTER TABLE migration.
        for col in _SCHEMAS[name]:
            if col not in df.columns:
                df[col] = None
        return df

    # ── Transaction lifecycle ────────────────────────────────────────────────
    def begin(self) -> None:
        self._snapshot = {name: df.copy(deep=True) for name, df in self._tables.items()}

    def commit(self) -> None:
        for name, df in self._tables.items():
            df.to_parquet(self._path(name), index=False)
        self._snapshot = None

    def rollback(self) -> None:
        if self._snapshot is not None:
            self._tables = self._snapshot
            self._snapshot = None

    def close(self) -> None:
        pass  # nothing held open between commits — writes happen in commit()

    # ── Locking (none in local mode — see module docstring) ─────────────────
    def locked_patids(self) -> set[str]:
        """Every PATID a reviewer has ever directly acted on (any
        `audit_log.patids`) — mirrors `sql_backend.locked_patids`. A later
        `publish_run` must never move these; see the sticky-unmerge
        invariant (docs/Data-Contract.md Stage 6)."""
        locked: set[str] = set()
        for patids in self._tables["audit_log"]["patids"]:
            locked.update(p for p in str(patids).split(",") if p)
        return locked

    # ── Persisted blocking index ────────────────────────────────────────────
    def lookup_block_candidates(
        self, keys: dict[str, str | None], phones: Iterable[str], threshold: int
    ) -> dict[str, set[str]]:
        bk = self._tables["block_key"]
        candidates: dict[str, set[str]] = {}

        def _add(block_id: str, key_value: str) -> None:
            hits = bk.loc[(bk["block_id"] == block_id) & (bk["key_value"] == key_value), "patid"]
            for patid in hits.head(threshold):
                candidates.setdefault(patid, set()).add(block_id)

        for block_id, key_value in keys.items():
            if key_value:
                _add(block_id, key_value)
        for phone in phones:
            if phone:
                _add("B5", phone)
        return candidates

    def add_block_keys(self, rows: list[tuple[str, str, str]]) -> None:
        if not rows:
            return
        new = pd.DataFrame(rows, columns=_SCHEMAS["block_key"])
        combined = pd.concat([self._tables["block_key"], new], ignore_index=True)
        self._tables["block_key"] = combined.drop_duplicates(
            subset=["block_id", "key_value", "patid"]
        ).reset_index(drop=True)

    # ── Persisted cleaned-attribute mirror ──────────────────────────────────
    def get_cleaned_attrs(self, patids: list[str]) -> list[dict]:
        if not patids:
            return []
        df = self._tables["cleaned_attrs"]
        hits = df[df["patid"].isin(patids)]
        return [_row_to_dict(row) for _, row in hits.iterrows()]

    def upsert_cleaned_attrs(self, row: tuple) -> None:
        cols = _SCHEMAS["cleaned_attrs"]
        df = self._tables["cleaned_attrs"]
        patid = row[0]
        df = df[df["patid"] != patid]
        new = pd.DataFrame([row], columns=cols)
        self._tables["cleaned_attrs"] = pd.concat([df, new], ignore_index=True)

    # ── Entities / membership ───────────────────────────────────────────────
    def get_entity_mid_for_patid(self, patid: str) -> str | None:
        df = self._tables["entity_member"]
        hit = df.loc[df["patid"] == patid, "mid"]
        return None if hit.empty else str(hit.iloc[0])

    def max_mid_sequence(self) -> int:
        """Highest numeric suffix among existing `M-<n>` mids, or 0 if none —
        scanned once per bulk publish rather than via a `next_mid()` call per
        entity (see `sql_backend.max_mid_sequence`'s equivalent SQL role)."""
        best = 0
        for mid in self._tables["entity"]["mid"]:
            if isinstance(mid, str) and mid.startswith("M-") and mid[2:].isdigit():
                best = max(best, int(mid[2:]))
        return best

    def next_mid(self) -> str:
        return f"M-{self.max_mid_sequence() + 1:06d}"

    def all_entity_member_mids(self) -> dict[str, str]:
        df = self._tables["entity_member"]
        return dict(zip(df["patid"], df["mid"]))

    def get_entity(self, mid: str) -> dict | None:
        entity_df = self._tables["entity"]
        hit = entity_df[entity_df["mid"] == mid]
        if hit.empty:
            return None
        members = self._tables["entity_member"]
        members = members[members["mid"] == mid].sort_values("patid")
        attrs = self._tables["record_attrs"][_DISPLAY_ATTR_COLUMNS]
        joined = members.merge(attrs, on="patid", how="left")
        return {
            "entity": _row_to_dict(hit.iloc[0]),
            "members": [_row_to_dict(row) for _, row in joined.iterrows()],
        }

    def upsert_entity(
        self, mid, run_id, origin, is_merged, confidence, updated_utc,
        match_rule=None, evidence=None,
    ) -> None:
        df = self._tables["entity"]
        df = df[df["mid"] != mid]
        new = pd.DataFrame([{
            "mid": mid, "run_id": run_id, "origin": origin,
            "is_merged": int(is_merged), "confidence": confidence,
            "match_rule": match_rule, "evidence": evidence, "updated_utc": updated_utc,
        }], columns=_SCHEMAS["entity"])
        self._tables["entity"] = pd.concat([df, new], ignore_index=True)

    def upsert_entity_member(self, patid, mid, is_primary, added_by, updated_utc) -> None:
        df = self._tables["entity_member"]
        df = df[df["patid"] != patid]
        new = pd.DataFrame([{
            "patid": patid, "mid": mid, "is_primary": int(is_primary),
            "added_by": added_by, "updated_utc": updated_utc,
        }], columns=_SCHEMAS["entity_member"])
        self._tables["entity_member"] = pd.concat([df, new], ignore_index=True)

    def reassign_entity_members(self, from_mids: list[str], to_mid: str, updated_utc: str) -> None:
        from_mids = [m for m in from_mids if m != to_mid]
        if not from_mids:
            return
        members = self._tables["entity_member"]
        mask = members["mid"].isin(from_mids)
        members.loc[mask, "mid"] = to_mid
        members.loc[mask, "updated_utc"] = updated_utc
        entities = self._tables["entity"]
        self._tables["entity"] = entities[~entities["mid"].isin(from_mids)].reset_index(drop=True)

    # ── Suggestions / review candidates ─────────────────────────────────────
    def upsert_suggestion(self, patid, run_id, suggested_mid, created_utc) -> None:
        df = self._tables["entity_suggestion"]
        df = df[df["patid"] != patid]
        new = pd.DataFrame([{
            "patid": patid, "run_id": run_id, "suggested_mid": suggested_mid,
            "created_utc": created_utc,
        }], columns=_SCHEMAS["entity_suggestion"])
        self._tables["entity_suggestion"] = pd.concat([df, new], ignore_index=True)

    def insert_review_candidates(self, rows: list[tuple]) -> None:
        if not rows:
            return
        new = pd.DataFrame(rows, columns=_SCHEMAS["review_candidate"])
        combined = pd.concat([self._tables["review_candidate"], new], ignore_index=True)
        self._tables["review_candidate"] = combined.drop_duplicates(
            subset=["patid_a", "patid_b", "run_id"], keep="last"
        ).reset_index(drop=True)

    # ── Bulk operations (batch publish — src/api/ingest/publish.py) ────────────────
    # Each is a batched filter-out-existing-then-concat (upsert) or a
    # wholesale delete-then-insert (replace), never a per-row loop — a real
    # publish touches every entity in a run at once, same performance
    # reasoning as sql_backend.py's own bulk functions (see that module's
    # PERFORMANCE note).
    def _upsert_bulk(self, table: str, key_col: str, rows: list[tuple]) -> None:
        if not rows:
            return
        cols = _SCHEMAS[table]
        new = pd.DataFrame(rows, columns=cols)
        df = self._tables[table]
        df = df[~df[key_col].isin(set(new[key_col]))]
        self._tables[table] = pd.concat([df, new], ignore_index=True)

    def upsert_entities_bulk(self, rows: list[tuple]) -> None:
        self._upsert_bulk("entity", "mid", rows)

    def upsert_entity_members_bulk(self, rows: list[tuple]) -> None:
        self._upsert_bulk("entity_member", "patid", rows)

    def upsert_record_attrs_bulk(self, rows: list[tuple]) -> None:
        self._upsert_bulk("record_attrs", "patid", rows)

    def upsert_record_raw_bulk(self, rows: list[tuple]) -> None:
        self._upsert_bulk("record_raw", "patid", rows)

    def upsert_suggestions_bulk(self, rows: list[tuple]) -> None:
        self._upsert_bulk("entity_suggestion", "patid", rows)

    def replace_review_candidates_for_run(self, run_id: str, rows: list[tuple]) -> None:
        """Full per-run replace, mirroring `sql_backend.replace_review_candidates_for_run`
        — a pair no longer in the latest run's `non_matches` disappears rather
        than lingering. `rows` are the 8-column batch-publish tuples (no FS
        score columns — batch publish never FS-scores a run, only incremental
        scoring does via `insert_review_candidates`'s 10-column rows); the two
        FS columns are set to `None` here, matching the `NULL` SQLite leaves
        for the columns its own 8-column INSERT omits."""
        df = self._tables["review_candidate"]
        df = df[df["run_id"] != run_id]
        if rows:
            new = pd.DataFrame(rows, columns=_REVIEW_CANDIDATE_BATCH_COLUMNS)
            new["fs_match_probability"] = None
            new["fs_classification_tier"] = None
            df = pd.concat([df, new[_SCHEMAS["review_candidate"]]], ignore_index=True)
        self._tables["review_candidate"] = df.reset_index(drop=True)

    def replace_cleaned_attrs(self, rows: list[tuple]) -> None:
        cols = _SCHEMAS["cleaned_attrs"]
        self._tables["cleaned_attrs"] = (
            pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        )

    def replace_block_keys(self, rows: list[tuple[str, str, str]]) -> None:
        cols = _SCHEMAS["block_key"]
        self._tables["block_key"] = (
            pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        )

    # ── Dashboard read routes (src/api/routers/records.py, dashboard.py) ────
    def list_entities(
        self,
        *,
        search: str | None = None,
        origin: str | None = None,
        is_merged: bool | None = None,
        birth_date: str | None = None,
        ssn_last4: str | None = None,
        updated_after: str | None = None,
        updated_before: str | None = None,
        confidence_min: float | None = None,
        confidence_max: float | None = None,
        sort: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict], int]:
        """Pandas equivalent of `sql_backend.list_entities` — each filter narrows
        the entity set the same way the SQL version's WHERE clauses do:
        `search`/`birth_date`/`ssn_last4` are member-level (computed as "the
        set of mids with >=1 matching member", then ANDed together and with
        the scalar entity-level filters). `confidence_min`/`confidence_max`
        exclude NaN confidence the same way SQL NULL comparisons do — pandas'
        comparison operators against NaN are already False, so no explicit
        `.notna()` guard is needed."""
        entities = self._tables["entity"]
        members = self._tables["entity_member"]
        attrs = self._tables["record_attrs"][
            ["patid", "first_name", "last_name", "birth_date", "ssn_last4"]
        ]
        joined_members = members.merge(attrs, on="patid", how="left")

        mask = pd.Series(True, index=entities.index)
        if origin is not None:
            mask &= entities["origin"].isin(origin.split(","))
        if is_merged is not None:
            mask &= entities["is_merged"] == int(is_merged)
        if updated_after:
            mask &= entities["updated_utc"] >= updated_after
        if updated_before:
            mask &= entities["updated_utc"] <= updated_before
        if confidence_min is not None:
            mask &= entities["confidence"] >= confidence_min
        if confidence_max is not None:
            mask &= entities["confidence"] <= confidence_max

        if search:
            like = search.lower()
            member_hit_mids = set(joined_members.loc[
                joined_members["patid"].str.lower().str.contains(like, na=False)
                | joined_members["first_name"].str.lower().str.contains(like, na=False)
                | joined_members["last_name"].str.lower().str.contains(like, na=False),
                "mid",
            ])
            mid_hit = entities["mid"].str.lower().str.contains(like, na=False)
            mask &= mid_hit | entities["mid"].isin(member_hit_mids)

        if birth_date:
            bd_mids = set(joined_members.loc[joined_members["birth_date"] == birth_date, "mid"])
            mask &= entities["mid"].isin(bd_mids)

        if ssn_last4:
            ssn_mids = set(joined_members.loc[joined_members["ssn_last4"] == ssn_last4, "mid"])
            mask &= entities["mid"].isin(ssn_mids)

        filtered = entities[mask]
        total = len(filtered)

        member_counts = members.groupby("mid").size().rename("member_count")
        filtered = filtered.merge(member_counts, on="mid", how="left")
        filtered["member_count"] = filtered["member_count"].fillna(0).astype(int)

        if sort == "confidence":
            filtered = filtered.sort_values(
                ["confidence", "mid"], ascending=[False, True], na_position="last"
            )
        elif sort == "name":
            primary_names = joined_members.loc[
                joined_members["is_primary"].astype(bool),
                ["mid", "first_name", "last_name"],
            ]
            filtered = filtered.merge(primary_names, on="mid", how="left")
            filtered = filtered.sort_values(
                ["last_name", "first_name", "mid"],
                ascending=True,
                na_position="last",
                key=lambda s: s.str.lower() if s.dtype == object else s,
            )
            filtered = filtered.drop(columns=["first_name", "last_name"])
        else:
            filtered = filtered.sort_values(["updated_utc", "mid"], ascending=[False, True])

        start = max(page - 1, 0) * page_size
        page_rows = filtered.iloc[start : start + page_size]
        return [_row_to_dict(row) for _, row in page_rows.iterrows()], total

    def dashboard_summary(self) -> dict:
        """Pandas equivalent of `sql_backend.dashboard_summary`."""
        entities = self._tables["entity"]
        members = self._tables["entity_member"]
        audit_action_counts = {
            str(k): int(v)
            for k, v in self._tables["audit_log"]["action"].value_counts().items()
        }

        total_records = len(members)
        origin_counts = {
            str(k): int(v) for k, v in entities["origin"].value_counts().items()
        }

        merged_mids = set(entities.loc[entities["is_merged"] == 1, "mid"])
        matched_members = int(members["mid"].isin(merged_mids).sum())
        duplicate_clusters = int((entities["is_merged"] == 1).sum())

        review_mids = set(entities.loc[entities["origin"] == "review", "mid"])
        needs_review_members = int(members["mid"].isin(review_mids).sum())

        none_mids = set(entities.loc[entities["origin"] == "none", "mid"])
        no_match_members = int(members["mid"].isin(none_mids).sum())

        auto_mids = set(
            entities.loc[entities["origin"].isin(["deterministic", "merge"]), "mid"]
        )
        auto_match_members = int(members["mid"].isin(auto_mids).sum())

        return {
            "total_records": total_records,
            "duplicate_clusters": duplicate_clusters,
            "matched_records": matched_members,
            "unmerged_records": total_records - matched_members,
            "needs_review_records": needs_review_members,
            "no_match_records": no_match_members,
            "auto_match_records": auto_match_members,
            "origin_counts": origin_counts,
            "manual_merge_actions": audit_action_counts.get("merge", 0),
            "manual_unmerge_actions": audit_action_counts.get("unmerge", 0),
        }

    def get_record_raw(self, patid: str) -> str | None:
        df = self._tables["record_raw"]
        hit = df.loc[df["patid"] == patid, "raw_json"]
        return None if hit.empty else str(hit.iloc[0])

    def _deduped_review_candidates(self) -> pd.DataFrame:
        """A pair can exist under more than one `run_id` — nothing deletes
        an OLDER run's `review_candidate` rows for a pair that's still
        unresolved when a later publish writes a fresh row for it (only
        `replace_review_candidates_for_run` clears rows for its OWN run_id
        first). Keep just the most-recently-created row per `(patid_a,
        patid_b)` so a pair is never listed twice — mirrors
        `sql_backend._DEDUPED_REVIEW_CANDIDATE`."""
        rc = self._tables["review_candidate"]
        return (
            rc.sort_values("created_utc", ascending=False)
            .drop_duplicates(subset=["patid_a", "patid_b"], keep="first")
        )

    def review_candidates_for_patid(self, patid: str) -> list[dict]:
        """Pandas equivalent of `sql_backend.review_candidates_for_patid` — joins
        `record_attrs` on both sides of the pair so the UI has the other
        PATID's display fields, not just its id. Ordered by confidence DESC
        (nulls last, pandas' default); `review_candidate` has no surrogate
        `id` column in Parquet mode (see its schema note), so `patid_a`/
        `patid_b` substitute as the deterministic tiebreak SQL gets from `id`."""
        rc = self._deduped_review_candidates()
        hits = rc[(rc["patid_a"] == patid) | (rc["patid_b"] == patid)]
        attrs = self._tables["record_attrs"][_DISPLAY_ATTR_COLUMNS]
        a_attrs = attrs.add_prefix("a_").rename(columns={"a_patid": "patid_a"})
        b_attrs = attrs.add_prefix("b_").rename(columns={"b_patid": "patid_b"})
        joined = (
            hits.merge(a_attrs, on="patid_a", how="left")
            .merge(b_attrs, on="patid_b", how="left")
            .sort_values(["confidence", "patid_a", "patid_b"], ascending=[False, True, True])
        )
        return [_row_to_dict(row) for _, row in joined.iterrows()]

    def list_review_candidates(
        self,
        *,
        confidence_min: float | None = None,
        confidence_max: float | None = None,
        reviewed: bool | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict], int]:
        """Pandas equivalent of `sql_backend.list_review_candidates` — candidate-grain
        (one row per pair, not per cluster). "Reviewed" mirrors the SQL
        version: both sides sharing a `mid` today (merged), or a prior
        `dismiss` audit_log entry for the exact pair."""
        rc = self._deduped_review_candidates()
        members = self._tables["entity_member"][["patid", "mid"]]
        attrs = self._tables["record_attrs"][_DISPLAY_ATTR_COLUMNS]

        mid_a = members.rename(columns={"patid": "patid_a", "mid": "mid_a"})
        mid_b = members.rename(columns={"patid": "patid_b", "mid": "mid_b"})
        a_attrs = attrs.add_prefix("a_").rename(columns={"a_patid": "patid_a"})
        b_attrs = attrs.add_prefix("b_").rename(columns={"b_patid": "patid_b"})

        joined = (
            rc.merge(mid_a, on="patid_a", how="inner")
            .merge(mid_b, on="patid_b", how="inner")
            .merge(a_attrs, on="patid_a", how="left")
            .merge(b_attrs, on="patid_b", how="left")
        )

        member_counts = members.groupby("mid").size()
        joined["member_count_a"] = joined["mid_a"].map(member_counts).fillna(0).astype(int)
        joined["member_count_b"] = joined["mid_b"].map(member_counts).fillna(0).astype(int)

        dismissed_pairs = set()
        audit = self._tables["audit_log"]
        for _, row in audit.loc[audit["action"] == "dismiss", "patids"].items():
            dismissed_pairs.add(row)
        joined["reviewed"] = (joined["mid_a"] == joined["mid_b"]) | (
            joined["patid_a"] + "," + joined["patid_b"]
        ).isin(dismissed_pairs)

        mask = pd.Series(True, index=joined.index)
        if confidence_min is not None:
            mask &= joined["confidence"] >= confidence_min
        if confidence_max is not None:
            mask &= joined["confidence"] <= confidence_max
        if search:
            like = search.lower()
            mask &= (
                joined["a_first_name"].str.lower().str.contains(like, na=False)
                | joined["a_last_name"].str.lower().str.contains(like, na=False)
                | joined["b_first_name"].str.lower().str.contains(like, na=False)
                | joined["b_last_name"].str.lower().str.contains(like, na=False)
                | joined["patid_a"].str.lower().str.contains(like, na=False)
                | joined["patid_b"].str.lower().str.contains(like, na=False)
            )
        if reviewed is True:
            mask &= joined["reviewed"]
        elif reviewed is False:
            mask &= ~joined["reviewed"]

        filtered = joined[mask].sort_values(
            ["confidence", "patid_a", "patid_b"], ascending=[False, True, True]
        )
        total = len(filtered)
        start = max(page - 1, 0) * page_size
        page_rows = filtered.iloc[start : start + page_size]
        return [_row_to_dict(row) for _, row in page_rows.iterrows()], total

    # ── Reviewer audit log (src/api/routers/audit.py) ───────────────────────
    def insert_audit_log(
        self,
        *,
        ts_utc: str,
        user: str,
        action: str,
        patids: str,
        mid: str,
        prev_state: str,
        next_state: str,
        run_id: str | None,
        prev_mid: str | None = None,
        undo_of: int | None = None,
    ) -> int:
        """Append-only insert. `id` is a synthetic auto-increment (scan
        existing max + 1) — same convention as `next_mid()` — since Parquet
        has no autoincrement primitive."""
        df = self._tables["audit_log"]
        next_id = int(df["id"].max()) + 1 if not df.empty else 1
        new = pd.DataFrame([{
            "id": next_id, "ts_utc": ts_utc, "user": user, "action": action,
            "patids": patids, "mid": mid, "prev_state": prev_state,
            "next_state": next_state, "run_id": run_id,
            "prev_mid": prev_mid, "undo_of": undo_of,
        }], columns=_SCHEMAS["audit_log"])
        self._tables["audit_log"] = pd.concat([df, new], ignore_index=True)
        return next_id

    def _undone_ids(self) -> set[int]:
        """Every audit_log id some other row's `undo_of` points at — `undone`
        is derived, never stored (see `get_audit_log_row`)."""
        undo_of = self._tables["audit_log"]["undo_of"].dropna()
        return {int(v) for v in undo_of}

    def list_audit_log(self, *, limit: int = 100, since: str | None = None) -> list[dict]:
        df = self._tables["audit_log"]
        if since:
            df = df[df["ts_utc"] > since]
        df = df.sort_values("id", ascending=False).head(limit)
        undone_ids = self._undone_ids()
        rows = []
        for _, row in df.iterrows():
            d = _row_to_dict(row)
            d["undone"] = int(d["id"]) in undone_ids
            rows.append(d)
        return rows

    def get_audit_log_row(self, audit_id: int) -> dict | None:
        """One audit_log row plus its derived `undone` flag — the reversal
        target of `POST /audit/{audit_id}/undo` (`src/api/routers/audit.py`)."""
        df = self._tables["audit_log"]
        hit = df[df["id"] == audit_id]
        if hit.empty:
            return None
        row = _row_to_dict(hit.iloc[0])
        row["undone"] = audit_id in self._undone_ids()
        return row


__all__ = ["ParquetIndexBackend"]
