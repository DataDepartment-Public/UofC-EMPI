"""Unit tests for src/api/ingest/incremental.py — incremental single-record scoring
against a small, hand-seeded "existing population" published via the real
`publish.publish_run` (so the seeded `block_key`/`cleaned_attrs`/`entity`
state is exactly what a real run would produce, not a hand-rolled stand-in).

Coverage:
    - Auto-merge: a new record sharing SSN+DOB with an existing entity joins
      it (does not mint a new mid).
    - No candidates: a genuinely unique new record becomes its own singleton
      (origin='none').
    - Review tier: a NAME_DOB_SEX-only match writes a review_candidate row
      and upgrades the new record's singleton origin to 'review', without
      auto-merging.
    - Sticky-unmerge: a reviewer-locked candidate is never auto-merged into;
      an entity_suggestion is written instead and the new record gets its
      own singleton.
    - Bridging: a new record that auto-matches members of two previously
      separate entities unifies them under one mid.
"""

import sqlite3

import pandas as pd
import pytest

from src.api.ingest import incremental, publish
from src.api.backends import sql_backend
from src.api.backends.index_backend import SqlIndexBackend
from src.config import Settings
from src.contracts import ArtifactRef, RunManifest


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    sql_backend.init_db(c)
    yield c
    c.close()


@pytest.fixture
def settings(tmp_path):
    s = Settings(project_root=tmp_path)
    s.runs_dir = tmp_path / "data" / "runs"
    s.clusters_dir = tmp_path / "data" / "clusters"
    s.matches_dir = tmp_path / "data" / "matches"
    s.non_matches_dir = tmp_path / "data" / "non_matches"
    s.processed_dir = tmp_path / "data" / "processed"
    # No FS model on disk -> resolve_active_model() returns None cleanly;
    # the FS stage is exercised separately by the fs_matcher test suite.
    s.fs_model_dir = tmp_path / "data" / "fs_model_empty"
    s.ensure_dirs()
    return s


def _seed_population(settings: Settings, conn) -> None:
    """Publish a small population: P1<->P2 auto-merged (SSN_DOB), P3 a true
    singleton, P4<->P5 a review-tier (NAME_DOB_SEX) candidate pair."""
    cleaned = pd.DataFrame({
        "PATID": ["P1", "P2", "P3", "P4", "P5"],
        "FirstNM_clean": ["JANE", "JANE", "JOHN", "AMY", "AMY"],
        "LastNM_clean": ["DOE", "DOE", "SMITH", "LEE", "LEE"],
        "BirthDT_clean": pd.to_datetime(
            ["1990-01-01", "1990-01-01", "1985-05-05", "1975-03-03", "1975-03-03"]
        ),
        "SSN_clean": ["234567891", "234567891", None, None, None],
        "last_4_SSN": ["7891", "7891", None, None, None],
        "Email_clean": [None, None, None, None, None],
        "ZipCD_clean_base": ["60601", "60601", None, None, None],
        "AddressLine1_clean": [None, None, None, None, None],
        "SexAtBirthDSC_clean": ["FEMALE", "FEMALE", "MALE", "FEMALE", "FEMALE"],
        "Phones_set": [set(), set(), set(), set(), set()],
        "FirstNM_raw": ["JANE", "JANE", "JOHN", "AMY", "AMY"],
        "SSN_raw": ["234-56-7891", "234567891", None, None, None],
        "valid_record": [True, True, True, True, True],
    })
    cleaned_path = settings.processed_dir / "cleaned_r0.parquet"
    cleaned.to_parquet(cleaned_path, index=False)

    matches = pd.DataFrame({
        "PATID_A": ["P1"], "PATID_B": ["P2"],
        "match_rule": ["SSN_DOB"], "confidence": [1.0],
        "rules_fired": ["SSN_DOB"], "is_suspicious": [False],
        "high_fanout_ssn": [False], "cluster_id": [0],
        "source_blocks": ["B1"], "n_blocks": [1],
    })
    matches_path = settings.matches_dir / "matches_r0.parquet"
    matches.to_parquet(matches_path, index=False)

    non_matches = pd.DataFrame({
        "PATID_A": ["P4"], "PATID_B": ["P5"],
        "source_blocks": ["B3"], "n_blocks": [1],
    })
    non_matches_path = settings.non_matches_dir / "non_matches_r0.parquet"
    non_matches.to_parquet(non_matches_path, index=False)

    review_evidence = pd.DataFrame({
        "PATID_A": ["P4"], "PATID_B": ["P5"],
        "match_rule": ["NAME_DOB_SEX"], "confidence": [0.98],
        "rules_fired": ["NAME_DOB_SEX"], "is_suspicious": [False],
        "high_fanout_ssn": [False],
        "source_blocks": ["B3"], "n_blocks": [1],
    })
    review_evidence_path = settings.non_matches_dir / "review_evidence_r0.parquet"
    review_evidence.to_parquet(review_evidence_path, index=False)

    clusters = pd.DataFrame({
        "PATID": ["P1", "P2", "P3", "P4", "P5"], "cluster_id": [0, 0, 1, 2, 3],
    })
    clusters_path = settings.clusters_dir / "clusters_r0.parquet"
    clusters.to_parquet(clusters_path, index=False)

    def ref(path, rows):
        return ArtifactRef(
            path=str(path.relative_to(settings.project_root)), rows=rows, sha256="x"
        )

    manifest = RunManifest(
        run_id="r0", created_utc="2026-07-01T00:00:00Z",
        raw_input=ref(cleaned_path, 5), cleaned=ref(cleaned_path, 5),
        candidate_pairs=ref(matches_path, 1), matches=ref(matches_path, 1),
        non_matches=ref(non_matches_path, 1),
        review_evidence=ref(review_evidence_path, 1),
        clusters=ref(clusters_path, 5),
        counts={},
    )
    (settings.runs_dir / "run_r0.json").write_text(manifest.model_dump_json())
    publish.publish_run(SqlIndexBackend(conn), "r0", settings)


def _raw(patid, first, last, birth, ssn=None, email=None, sex=None,
         zip_cd=None, address1=None, phone=None):
    return {
        "PATID": patid, "FirstNM": first, "LastNM": last, "MiddleNM": None,
        "SuffixNM": None, "BirthDT": birth, "SSN": ssn,
        "AddressLine1": address1, "AddressLine2": None, "CityNM": None,
        "ZipCD": zip_cd, "StateCD": None,
        "PrimaryPhoneNBR": phone, "Phone01NBR": None, "Phone02NBR": None,
        "Phone03NBR": None, "Email": email, "SexAtBirthDSC": sex,
    }


class TestAutoMerge:
    def test_new_record_joins_existing_entity_via_ssn_dob(self, conn, settings):
        _seed_population(settings, conn)
        existing_mid = sql_backend.get_entity_mid_for_patid(conn, "P1")

        outcomes = incremental.score_records(
            SqlIndexBackend(conn), settings,
            [_raw("P6", "JANE", "DOE", "1990-01-01", ssn="234-56-7891")],
            run_id="r1",
        )

        assert outcomes[0]["tier"] == "auto_merge"
        assert outcomes[0]["mid"] == existing_mid
        assert sql_backend.get_entity_mid_for_patid(conn, "P6") == existing_mid
        entity = sql_backend.get_entity(conn, existing_mid)
        assert {m["patid"] for m in entity["members"]} == {"P1", "P2", "P6"}
        assert entity["entity"]["is_merged"] == 1

    def test_bridges_two_previously_separate_entities(self, conn, settings):
        _seed_population(settings, conn)
        mid_12 = sql_backend.get_entity_mid_for_patid(conn, "P1")
        mid_3 = sql_backend.get_entity_mid_for_patid(conn, "P3")
        assert mid_12 != mid_3

        # SSN_DOB only requires ssn+dob agreement (not name) — give P3 both
        # P1/P2's SSN *and* DOB so a single incoming record's SSN_DOB match
        # touches both entities and has to bridge them.
        conn.execute(
            "UPDATE cleaned_attrs SET ssn=?, ssn_last4=?, birth_dt=? WHERE patid='P3'",
            ("234567891", "7891", "1990-01-01"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO block_key (block_id, key_value, patid) "
            "VALUES ('B1', ?, 'P3')",
            (sql_backend.hash_block_key("234567891"),),
        )
        conn.commit()

        outcomes = incremental.score_records(
            SqlIndexBackend(conn), settings,
            [_raw("P7", "JANE", "DOE", "1990-01-01", ssn="234-56-7891")],
            run_id="r1",
        )
        assert outcomes[0]["tier"] == "auto_merge"
        target = outcomes[0]["mid"]
        assert target == min(mid_12, mid_3)
        entity = sql_backend.get_entity(conn, target)
        assert {m["patid"] for m in entity["members"]} == {"P1", "P2", "P3", "P7"}
        # The absorbed entity must no longer exist.
        absorbed = mid_3 if target == mid_12 else mid_12
        assert sql_backend.get_entity(conn, absorbed) is None


class TestNoMatch:
    def test_unique_record_becomes_its_own_singleton(self, conn, settings):
        _seed_population(settings, conn)
        outcomes = incremental.score_records(
            SqlIndexBackend(conn), settings,
            [_raw("P8", "ZZZZUNIQUE", "NOBODY", "1800-01-01", ssn="000000001")],
            run_id="r1",
        )
        assert outcomes[0]["tier"] == "no_match"
        mid = outcomes[0]["mid"]
        entity = sql_backend.get_entity(conn, mid)
        assert [m["patid"] for m in entity["members"]] == ["P8"]
        assert entity["entity"]["origin"] == "none"
        assert entity["entity"]["is_merged"] == 0


class TestReviewTier:
    def test_name_dob_sex_only_match_writes_review_candidate_not_auto_merge(
        self, conn, settings
    ):
        _seed_population(settings, conn)
        # Same name/DOB/sex as P4/P5 but no SSN/email/phone corroboration ->
        # NAME_DOB_SEX only, which is review-tier, not auto-merge.
        outcomes = incremental.score_records(
            SqlIndexBackend(conn), settings,
            [_raw("P9", "AMY", "LEE", "1975-03-03", sex="FEMALE")],
            run_id="r1",
        )
        outcome = outcomes[0]
        assert outcome["tier"] == "human_review"
        entity = sql_backend.get_entity(conn, outcome["mid"])
        assert entity["entity"]["origin"] == "review"
        assert entity["entity"]["is_merged"] == 0

        candidates = sql_backend.review_candidates_for_patid(conn, "P9")
        assert len(candidates) >= 1
        assert all(c["match_rule"] in (None, "NAME_DOB_SEX") for c in candidates)


class TestStickyUnmerge:
    def test_locked_candidate_is_not_auto_merged_into(self, conn, settings):
        _seed_population(settings, conn)
        mid_3 = sql_backend.get_entity_mid_for_patid(conn, "P3")

        # Give the P3 singleton a distinguishing SSN so a new record can
        # match *only* it via SSN_DOB, then lock P3.
        conn.execute(
            "UPDATE cleaned_attrs SET ssn=?, ssn_last4=? WHERE patid='P3'",
            ("512345678", "5678"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO block_key (block_id, key_value, patid) "
            "VALUES ('B1', ?, 'P3')",
            (sql_backend.hash_block_key("512345678"),),
        )
        conn.commit()
        sql_backend.insert_audit_log(
            conn, ts_utc="t0", user="reviewer1", action="unmerge",
            patids="P3", mid=mid_3, prev_state="Merged",
            next_state="Unmerged", run_id="r0",
        )
        conn.commit()

        outcomes = incremental.score_records(
            SqlIndexBackend(conn), settings,
            [_raw("P10", "JOHN", "SMITH", "1985-05-05", ssn="512-34-5678")],
            run_id="r1",
        )
        outcome = outcomes[0]
        assert outcome["tier"] == "no_match"
        # P10 must NOT have joined P3's (locked) entity.
        assert sql_backend.get_entity_mid_for_patid(conn, "P10") != mid_3
        entity = sql_backend.get_entity(conn, mid_3)
        assert "P10" not in {m["patid"] for m in entity["members"]}

        # A suggestion should record what P10 would have joined.
        row = conn.execute(
            "SELECT * FROM entity_suggestion WHERE patid = 'P10'"
        ).fetchone()
        assert row is not None
        assert row["suggested_mid"] == mid_3


class TestIndexMaintenance:
    def test_scored_record_is_discoverable_by_a_later_call_in_same_batch(
        self, conn, settings
    ):
        _seed_population(settings, conn)
        outcomes = incremental.score_records(
            SqlIndexBackend(conn), settings,
            [
                _raw("P11", "NEWFIRST", "NEWLAST", "2001-01-01", ssn="555667777"),
                _raw("P12", "NEWFIRST", "NEWLAST", "2001-01-01", ssn="555667777"),
            ],
            run_id="r1",
        )
        assert outcomes[0]["tier"] == "no_match"
        assert outcomes[1]["tier"] == "auto_merge"
        assert outcomes[1]["mid"] == outcomes[0]["mid"]
