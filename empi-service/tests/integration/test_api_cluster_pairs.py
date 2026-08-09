"""Integration tests for GET /clusters/{mid}/pairs
(src/api/routers/cluster_pairs.py).

The route joins the index (who is in the cluster now) against the run's
Parquet artifacts (what the pipeline decided about each pair), so these tests
build both: a published entity plus a hand-written manifest whose artifacts
place one pair on each rung of the verdict ladder.

Fixture pattern follows test_api.py / test_api_explanations.py — `main.py`'s
lifespan reads the `src.config.settings` singleton directly, so it is
monkeypatched in place rather than injected.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.backends import sql_backend
from src.api.main import app
from src.api.schemas import CLUSTER_PAIR_VERDICTS
from src.config import settings as real_settings
from src.contracts import ArtifactRef, RunManifest

RUN_ID = "20260801T000000Z"
MID = "M-000001"

#: Six members, chosen so every pair below is a distinct rung of the ladder.
MEMBERS = ["A1", "A2", "A3", "A4", "A5", "A6"]


@pytest.fixture
def test_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(real_settings, "project_root", tmp_path)
    for attr, sub in [
        ("runs_dir", "runs"),
        ("auto_merge_dir", "auto_merge"),
        ("no_match_dir", "no_match"),
        ("gate_output_dir", "gate_output"),
        ("ml_output_dir", "ml_output"),
        ("blocking_dir", "blocking"),
    ]:
        monkeypatch.setattr(real_settings, attr, tmp_path / "data" / sub)
    monkeypatch.setattr(real_settings, "db_path", tmp_path / "empi.db")
    real_settings.ensure_dirs()
    conn = sql_backend.get_connection(real_settings.db_path)
    try:
        sql_backend.init_db(conn)
    finally:
        conn.close()
    return real_settings


def _seed_entity(settings, *, members=MEMBERS, reviewer_member: str | None = None):
    """One entity with `members`, written straight to the index.

    Bypasses `publish.py` deliberately: what is under test is the join between
    *current* membership and the run's artifacts, and publishing would derive
    membership from the cluster artifact, hiding exactly the divergence the
    route is built to handle.
    """
    conn = sql_backend.get_connection(settings.db_path)
    try:
        conn.execute(
            "INSERT INTO entity (mid, run_id, origin, is_merged, confidence, "
            "match_rule, evidence, updated_utc) VALUES (?,?,?,?,?,?,?,?)",
            (MID, RUN_ID, "deterministic", 1, 1.0, "SSN_DOB", "SSN_DOB",
             "2026-08-01T00:00:00Z"),
        )
        for patid in members:
            conn.execute(
                "INSERT INTO entity_member (patid, mid, is_primary, added_by, "
                "updated_utc) VALUES (?,?,?,?,?)",
                (patid, MID, int(patid == members[0]),
                 "klkendall" if patid == reviewer_member else "pipeline",
                 "2026-08-01T00:00:00Z"),
            )
        conn.commit()
    finally:
        conn.close()


#: Records the artifacts compare a member against, each living in its own
#: cluster — the counterparts of `external_pairs`.
OUTSIDERS = {"Z1": ("Jane", "Roe"), "Z2": ("John", "Poe")}


def _seed_outsiders(settings):
    conn = sql_backend.get_connection(settings.db_path)
    try:
        for i, (patid, (first, last)) in enumerate(OUTSIDERS.items(), start=2):
            other_mid = f"M-00000{i}"
            conn.execute(
                "INSERT INTO entity (mid, run_id, origin, is_merged, confidence, "
                "match_rule, evidence, updated_utc) VALUES (?,?,?,?,?,?,?,?)",
                (other_mid, RUN_ID, "none", 0, None, None, None,
                 "2026-08-01T00:00:00Z"),
            )
            conn.execute(
                "INSERT INTO entity_member (patid, mid, is_primary, added_by, "
                "updated_utc) VALUES (?,?,1,'pipeline',?)",
                (patid, other_mid, "2026-08-01T00:00:00Z"),
            )
            conn.execute(
                "INSERT INTO record_attrs (patid, first_name, last_name, "
                "birth_date, ssn_last4, run_id) VALUES (?,?,?,?,?,?)",
                (patid, first, last, "1970-01-01", "1234", RUN_ID),
            )
        conn.commit()
    finally:
        conn.close()


def _write_run(settings, *, run_id: str = RUN_ID, omit: tuple[str, ...] = ()):
    """A manifest whose artifacts put one member pair on each verdict.

    A1-A2 auto_merge_rule · A1-A3 reject · A2-A3 ml_auto_merge
    A2-A4 ml_human_review · A3-A4 gate_dropped · A4-A5 blocked_undecided
    every pair involving A6 -> not_compared (it appears in no artifact).

    Two rows straddle the cluster boundary and drive `external_pairs`:
    A1-Z1 gate_dropped and A2-Z2 reject, where Z1/Z2 live in their own
    clusters (`_seed_outsiders`).

    `omit` drops artifacts from the manifest entirely, standing in for a run
    that skipped a stage.
    """
    frames = {
        "matches": (
            settings.auto_merge_dir / f"matches_{run_id}.parquet",
            pd.DataFrame({
                "PATID_A": ["A1"], "PATID_B": ["A2"],
                "match_rule": ["SSN_DOB"], "confidence": [1.0],
                "rules_fired": ["SSN_DOB|NAME_DOB_PHONE"], "is_suspicious": [False],
                "high_fanout_ssn": [False], "cluster_id": [0],
                "source_blocks": ["B1"], "n_blocks": [1],
            }),
        ),
        "rejects": (
            settings.no_match_dir / f"rejects_{run_id}.parquet",
            pd.DataFrame({
                "PATID_A": ["A1", "A2"], "PATID_B": ["A3", "Z2"],
                "source_blocks": ["B3", "B4"], "n_blocks": [1, 1],
                "n_contradictions": [3, 4], "decision": ["no_match", "no_match"],
                "reject_rule": ["STRONG_ID_CONFLICT", "STRONG_ID_CONFLICT"],
            }),
        ),
        "gate_results": (
            settings.gate_output_dir / f"gate_results_{run_id}.parquet",
            pd.DataFrame({
                "PATID_A": ["A1", "A2", "A2", "A3"],
                "PATID_B": ["Z1", "A3", "A4", "A4"],
                "model_name": "nonmatch_gate",
                "score": [0.03, 0.97, 0.81, 0.02],
                "predicted_tier": [
                    "no_match", "human_review", "human_review", "no_match",
                ],
            }),
        ),
        "matches_ml": (
            settings.ml_output_dir / f"matches_ml_{run_id}.parquet",
            pd.DataFrame({
                "PATID_A": ["A2", "A2"], "PATID_B": ["A3", "A4"],
                "model_name": "ml_matcher",
                "score": [0.95, 0.12],
                "predicted_tier": ["auto_merge", "human_review"],
            }),
        ),
        "candidate_pairs": (
            settings.blocking_dir / f"candidate_pairs_{run_id}.parquet",
            pd.DataFrame({
                "PATID_A": ["A1", "A1", "A1", "A2", "A2", "A2", "A3", "A4"],
                "PATID_B": ["A2", "A3", "Z1", "A3", "A4", "Z2", "A4", "A5"],
                "source_blocks": [
                    "B1", "B3", "B7", "B3|B4", "B3", "B4", "B3", "B8",
                ],
                "n_blocks": [1, 1, 1, 2, 1, 1, 1, 1],
            }),
        ),
    }

    refs = {}
    for field, (path, frame) in frames.items():
        if field in omit:
            continue
        frame.to_parquet(path, index=False)
        refs[field] = ArtifactRef(
            path=str(path.relative_to(settings.project_root)),
            rows=len(frame), sha256="x",
        )

    placeholder = ArtifactRef(path="x.parquet", rows=0, sha256="x")
    manifest = RunManifest(
        run_id=run_id,
        created_utc="2026-08-01T00:00:00Z",
        raw_input=placeholder,
        cleaned=placeholder,
        candidate_pairs=refs.get("candidate_pairs", placeholder),
        matches=refs.get("matches", placeholder),
        non_matches=placeholder,
        rejects=refs.get("rejects"),
        gate_results=refs.get("gate_results"),
        matches_ml=refs.get("matches_ml"),
    )
    (settings.runs_dir / f"run_{run_id}.json").write_text(manifest.model_dump_json())
    return manifest


@pytest.fixture
def client(test_settings):
    with TestClient(app) as c:
        yield c


def _verdicts(body: dict) -> dict[tuple[str, str], str]:
    return {(p["patid_a"], p["patid_b"]): p["verdict"] for p in body["pairs"]}


# ─── the verdict ladder ───────────────────────────────────────────────────────
def test_every_rung_of_the_ladder_is_reachable(client, test_settings):
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()
    v = _verdicts(body)

    assert v[("A1", "A2")] == "auto_merge_rule"
    assert v[("A1", "A3")] == "reject"
    assert v[("A2", "A3")] == "ml_auto_merge"
    assert v[("A2", "A4")] == "ml_human_review"
    assert v[("A3", "A4")] == "gate_dropped"
    assert v[("A4", "A5")] == "blocked_undecided"
    assert v[("A1", "A6")] == "not_compared"


def test_ml_outranks_the_gate_that_let_the_pair_through(client, test_settings):
    """A2-A3 has both a gate row and an ML row. The gate only decided whether
    the matcher would see it; the matcher decided whether an edge formed.
    Reporting `gate_dropped` — or even the gate's tier — would misattribute
    the decision to the wrong stage."""
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()
    pair = next(p for p in body["pairs"] if (p["patid_a"], p["patid_b"]) == ("A2", "A3"))

    assert pair["verdict"] == "ml_auto_merge"
    # Both stages' numbers survive on the payload for the trail UI.
    assert pair["gate_score"] == pytest.approx(0.97)
    assert pair["ml_score"] == pytest.approx(0.95)


def test_every_verdict_emitted_is_a_declared_one(client, test_settings):
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()
    assert {p["verdict"] for p in body["pairs"]} <= set(CLUSTER_PAIR_VERDICTS)


def test_pairs_cover_every_combination_canonically_ordered(client, test_settings):
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    assert len(body["pairs"]) == 15  # C(6, 2)
    assert all(p["patid_a"] < p["patid_b"] for p in body["pairs"])
    assert body["members"] == sorted(MEMBERS)


# ─── stage detail carried through ─────────────────────────────────────────────
def test_deterministic_rule_provenance_survives(client, test_settings):
    """`publish.py` keeps only the winning pair's rule on the entity row; the
    per-pair rule and its full `rules_fired` string exist nowhere else."""
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()
    pair = next(p for p in body["pairs"] if p["verdict"] == "auto_merge_rule")

    assert pair["match_rule"] == "SSN_DOB"
    assert pair["rules_fired"] == "SSN_DOB|NAME_DOB_PHONE"
    assert pair["confidence"] == pytest.approx(1.0)
    assert pair["source_blocks"] == "B1"


def test_gate_drops_are_reported_with_their_score(client, test_settings):
    """The gate artifact is the only record a dropped pair was ever compared —
    without this the pair is indistinguishable from one blocking never saw."""
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()
    pair = next(p for p in body["pairs"] if p["verdict"] == "gate_dropped")

    assert (pair["patid_a"], pair["patid_b"]) == ("A3", "A4")
    assert pair["gate_score"] == pytest.approx(0.02)
    assert pair["ml_score"] is None  # dropped pairs never reach the matcher
    assert pair["blocked"] is True


def test_reject_carries_its_rule_and_contradiction_count(client, test_settings):
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()
    pair = next(p for p in body["pairs"] if p["verdict"] == "reject")

    assert pair["reject_rule"] == "STRONG_ID_CONFLICT"
    assert pair["n_contradictions"] == 3


def test_transitive_pairs_are_marked_unblocked(client, test_settings):
    """A6 is in the cluster but shares no candidate pair with anyone — it got
    here through someone else's match. `blocked=False` is what lets the UI say
    so instead of implying the two were compared and agreed."""
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()
    a6 = [p for p in body["pairs"] if "A6" in (p["patid_a"], p["patid_b"])]

    assert len(a6) == 5
    assert all(p["verdict"] == "not_compared" for p in a6)
    assert all(p["blocked"] is False for p in a6)
    assert all(p["source_blocks"] is None for p in a6)


def test_thresholds_are_served_alongside_the_scores(client, test_settings):
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    assert body["thresholds"]["gate_threshold"] == test_settings.gate_threshold
    assert (
        body["thresholds"]["ml_auto_merge_threshold"]
        == test_settings.ml_auto_merge_threshold
    )


# ─── clustering (stage 6, the transitive-join trace) ─────────────────────────
def test_a_rejected_pair_shows_the_chain_that_merged_it_anyway(client, test_settings):
    """A1-A3 is `reject` — the rules found strong contradictions — yet both
    ends are in the cluster because A1-A2 auto-merged and A2-A3 was the ML
    matcher's confident match. `cluster_path` is the only place that survives
    for a reviewer to see."""
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()
    pair = next(p for p in body["pairs"] if (p["patid_a"], p["patid_b"]) == ("A1", "A3"))

    assert pair["verdict"] == "reject"
    assert pair["cluster_path"] == ["A1", "A2", "A3"]


def test_a_direct_merge_edge_has_no_cluster_path(client, test_settings):
    """A pair whose own verdict formed the edge needs no transitive
    explanation — showing one would be circular."""
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    for verdict in ("auto_merge_rule", "ml_auto_merge"):
        pair = next(p for p in body["pairs"] if p["verdict"] == verdict)
        assert pair["cluster_path"] is None


def test_a_pair_disconnected_in_the_merge_graph_has_no_cluster_path(client, test_settings):
    """A4 and A6 never appear on either end of a confirmed edge in this run's
    artifacts (`_write_run`'s A4/A5/A6 rows are all non-merging verdicts), so
    no chain connects them to anything — an honest `None`, not a fabricated
    guess."""
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    for a, b in (("A2", "A4"), ("A1", "A6")):
        pair = next(p for p in body["pairs"] if (p["patid_a"], p["patid_b"]) == (a, b))
        assert pair["cluster_path"] is None


def test_a_reviewer_merge_transitively_connects_other_pairs_too(client, test_settings):
    """A5 merged in by hand fans out to every other member (see
    `test_a_reviewer_merged_member_taints_every_pair_it_is_in`), so it also
    acts as a bridge: A2-A6 shares no artifact row at all, but both now
    reach A5 directly, so `cluster_path` should route through it."""
    _seed_entity(test_settings, reviewer_member="A5")
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()
    pair = next(p for p in body["pairs"] if (p["patid_a"], p["patid_b"]) == ("A2", "A6"))

    assert pair["verdict"] == "not_compared"
    assert pair["cluster_path"] == ["A2", "A5", "A6"]


# ─── reviewer provenance ──────────────────────────────────────────────────────
def test_pipeline_placed_members_read_as_pipeline_joined(client, test_settings):
    _seed_entity(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    assert all(p["joined_by"] == "pipeline" for p in body["pairs"])
    assert all(p["reviewer_id"] is None for p in body["pairs"])


def test_a_reviewer_merged_member_taints_every_pair_it_is_in(client, test_settings):
    """`entity_member.added_by` is the per-record answer to "who put this
    here", so a pair is reviewer-joined exactly when one of its sides was
    merged in by hand — no `audit_log` scan needed."""
    _seed_entity(test_settings, reviewer_member="A5")
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    touching_a5 = [p for p in body["pairs"] if "A5" in (p["patid_a"], p["patid_b"])]
    assert len(touching_a5) == 5
    assert all(p["joined_by"] == "reviewer" for p in touching_a5)
    assert all(p["reviewer_id"] == "klkendall" for p in touching_a5)
    assert all(
        p["joined_by"] == "pipeline"
        for p in body["pairs"]
        if "A5" not in (p["patid_a"], p["patid_b"])
    )


# ─── degradation, not failure ─────────────────────────────────────────────────
def test_a_skipped_stage_nulls_its_fields_rather_than_500ing(client, test_settings):
    """No active gate or ML model is a normal run, not a broken one."""
    _seed_entity(test_settings)
    _write_run(test_settings, omit=("gate_results", "matches_ml"))
    r = client.get(f"/clusters/{MID}/pairs")
    assert r.status_code == 200
    body = r.json()

    assert all(p["gate_score"] is None and p["ml_score"] is None for p in body["pairs"])
    # The deterministic verdicts are unaffected; what the models would have
    # said falls back to what blocking knows.
    v = _verdicts(body)
    assert v[("A1", "A2")] == "auto_merge_rule"
    assert v[("A2", "A3")] == "blocked_undecided"


def test_an_artifact_deleted_from_disk_degrades_the_same_way(client, test_settings):
    _seed_entity(test_settings)
    _write_run(test_settings)
    (test_settings.gate_output_dir / f"gate_results_{RUN_ID}.parquet").unlink()
    body = client.get(f"/clusters/{MID}/pairs").json()

    assert all(p["gate_score"] is None for p in body["pairs"])
    assert body["artifacts_available"] is True  # other artifacts still resolved


def test_a_run_with_no_artifacts_left_still_lists_the_pairs(client, test_settings):
    """Artifacts age out; the cluster does not. The reviewer must still be able
    to see and unmerge its members, with the trace honestly blank."""
    _seed_entity(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    assert body["artifacts_available"] is False
    assert len(body["pairs"]) == 15
    assert all(p["verdict"] == "not_compared" for p in body["pairs"])


def test_a_singleton_cluster_has_no_pairs_and_is_not_an_error(client, test_settings):
    _seed_entity(test_settings, members=["A1"])
    _write_run(test_settings)
    r = client.get(f"/clusters/{MID}/pairs")

    assert r.status_code == 200
    assert r.json()["pairs"] == []
    assert r.json()["members"] == ["A1"]


def test_unknown_cluster_404s(client, test_settings):
    _write_run(test_settings)
    r = client.get("/clusters/M-999999/pairs")
    assert r.status_code == 404
    assert "Unknown cluster" in r.json()["detail"]


def test_unknown_run_id_404s(client, test_settings):
    """An explicit `run_id` the caller got wrong is an error, unlike an entity
    whose own run has aged out — that one degrades."""
    _seed_entity(test_settings)
    _write_run(test_settings)
    r = client.get(f"/clusters/{MID}/pairs?run_id=nope")
    assert r.status_code == 404
    assert "Unknown run_id" in r.json()["detail"]


def test_an_explicit_run_id_is_honored_over_the_entitys_own(client, test_settings):
    _seed_entity(test_settings)
    _write_run(test_settings)
    other = _write_run(test_settings, run_id="20260901T000000Z", omit=("matches",))
    body = client.get(f"/clusters/{MID}/pairs?run_id={other.run_id}").json()

    assert body["run_id"] == other.run_id
    assert _verdicts(body)[("A1", "A2")] == "blocked_undecided"


# ─── external comparisons (the "why did it stop here" list) ───────────────────
def test_comparisons_that_straddle_the_cluster_boundary_are_listed_separately(
    client, test_settings
):
    _seed_entity(test_settings)
    _seed_outsiders(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    got = {(p["member_patid"], p["other_patid"]): p["verdict"]
           for p in body["external_pairs"]}
    assert got == {("A1", "Z1"): "gate_dropped", ("A2", "Z2"): "reject"}
    # ...and none of them leaked into the within-cluster list.
    assert all(
        "Z1" not in (p["patid_a"], p["patid_b"])
        and "Z2" not in (p["patid_a"], p["patid_b"])
        for p in body["pairs"]
    )


def test_member_and_other_are_identified_regardless_of_canonical_order(
    client, test_settings
):
    """`patid_a < patid_b` puts the outsider on either side depending on how
    the ids sort, so the UI can't infer which record is "ours" — the payload
    has to say."""
    _seed_entity(test_settings)
    _seed_outsiders(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    for p in body["external_pairs"]:
        assert p["member_patid"] in MEMBERS
        assert p["other_patid"] in OUTSIDERS
        assert {p["member_patid"], p["other_patid"]} == {p["patid_a"], p["patid_b"]}


def test_the_counterpart_is_named_not_just_a_patid(client, test_settings):
    """"PAT004821 was rejected" is not something a reviewer can act on."""
    _seed_entity(test_settings)
    _seed_outsiders(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    z1 = next(p for p in body["external_pairs"] if p["other_patid"] == "Z1")
    assert (z1["other_first_name"], z1["other_last_name"]) == ("Jane", "Roe")
    assert z1["other_birth_date"] == "1970-01-01"
    # Where that record lives *now* is part of the answer — it may since have
    # been merged elsewhere.
    assert z1["other_mid"] == "M-000002"


def test_external_list_never_contains_not_compared(client, test_settings):
    """It is built from artifact rows, so a pair only appears if some stage
    actually looked at it. Enumerating the thousands of records the pipeline
    never considered would bury the handful that matter."""
    _seed_entity(test_settings)
    _seed_outsiders(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    assert body["external_pairs"]
    assert all(p["verdict"] != "not_compared" for p in body["external_pairs"])


def test_a_singleton_has_no_internal_pairs_but_keeps_its_history(
    client, test_settings
):
    """The case the section exists for: a lone record whose only story is the
    comparisons that didn't merge it.

    Membership comes from the index, not the run, so shrinking the cluster to
    A1 alone reclassifies A2 and A3 as outsiders — which is the point. A1-A2
    then shows as `auto_merge_rule` in the *external* list: the pipeline
    merged that pair and the current index has them apart, which is exactly
    what a sticky unmerge looks like and precisely what a reviewer opening a
    singleton needs to see.
    """
    _seed_entity(test_settings, members=["A1"])
    _seed_outsiders(test_settings)
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    assert body["pairs"] == []
    got = {p["other_patid"]: p["verdict"] for p in body["external_pairs"]}
    assert got == {
        "A2": "auto_merge_rule",
        "A3": "reject",
        "Z1": "gate_dropped",
    }
    z1 = next(p for p in body["external_pairs"] if p["other_patid"] == "Z1")
    assert z1["gate_score"] == pytest.approx(0.03)


def test_an_unresolvable_counterpart_degrades_to_a_bare_patid(
    client, test_settings
):
    """Artifacts outlive the index: a record scored in this run may never have
    been published, or may have been removed since. The comparison still
    happened and must still be shown."""
    _seed_entity(test_settings)  # note: no _seed_outsiders
    _write_run(test_settings)
    body = client.get(f"/clusters/{MID}/pairs").json()

    z1 = next(p for p in body["external_pairs"] if p["other_patid"] == "Z1")
    assert z1["verdict"] == "gate_dropped"
    assert z1["other_mid"] is None
    assert z1["other_last_name"] is None


# ─── min_members, the filter this view pages on ───────────────────────────────
def test_min_members_is_not_the_same_question_as_is_merged(client, test_settings):
    """`is_merged` counts *unlocked* members at publish time, so a cluster a
    reviewer has touched can hold several records and still be 0. The registry
    pages on member count instead — this asserts the two really do differ."""
    _seed_entity(test_settings)
    conn = sql_backend.get_connection(test_settings.db_path)
    try:
        conn.execute("UPDATE entity SET is_merged = 0 WHERE mid = ?", (MID,))
        conn.commit()
    finally:
        conn.close()

    assert client.get("/records?is_merged=true").json()["total"] == 0
    assert client.get("/records?min_members=2").json()["total"] == 1
    assert client.get("/records?min_members=7").json()["total"] == 0
