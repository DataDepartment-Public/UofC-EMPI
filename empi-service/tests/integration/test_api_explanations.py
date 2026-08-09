"""Integration tests for GET /explanations/... (src/api/routers/explanations.py).

The route reads the run's Parquet artifact through the `RunManifest` rather
than the index backend, so these tests build a manifest + artifact on disk and
exercise the resolution path — including the failure modes that are *normal*
outcomes (a pair the model never scored) rather than errors.

Follows test_api.py's fixture pattern: `main.py`'s lifespan reads the
`src.config.settings` singleton directly, so it is monkeypatched in place.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import settings as real_settings
from src.contracts import ArtifactRef, RunManifest


@pytest.fixture
def test_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(real_settings, "project_root", tmp_path)
    monkeypatch.setattr(real_settings, "runs_dir", tmp_path / "data" / "runs")
    monkeypatch.setattr(real_settings, "gate_output_dir", tmp_path / "data" / "gate_output")
    monkeypatch.setattr(real_settings, "ml_output_dir", tmp_path / "data" / "ml_output")
    monkeypatch.setattr(real_settings, "db_path", tmp_path / "empi.db")
    real_settings.ensure_dirs()
    return real_settings


def _explanation_frame() -> pd.DataFrame:
    """Two pairs, hand-built so the expected geometry is obvious."""
    return pd.DataFrame(
        {
            "PATID_A": ["00001", "00003"],
            "PATID_B": ["00002", "00004"],
            "model_name": "nonmatch_gate",
            "score": [0.9241418, 0.0474259],
            "predicted_tier": ["human_review", "no_match"],
            "base_value": [-0.5, -0.5],
            "model_file": "nonmatch_gate_20260721T160717Z.pkl",
            "shap_sim_dob": [2.0, -1.5],
            "shap_sound_last": [1.0, -1.0],
            "feat_sim_dob": [1.0, 0.25],
            "feat_sound_last": ["same", "different"],
        }
    )


def _write_run(settings, run_id: str = "20260728T120000Z", *, with_gate: bool = True):
    artifact = settings.gate_output_dir / f"gate_explanations_{run_id}.parquet"
    frame = _explanation_frame()
    if with_gate:
        frame.to_parquet(artifact, index=False)
    ref = ArtifactRef(path=f"data/gate_output/{artifact.name}", rows=len(frame), sha256="x")
    manifest = RunManifest(
        run_id=run_id,
        created_utc="2026-07-28T12:00:00Z",
        raw_input=ArtifactRef(path="r.csv", rows=1, sha256="x"),
        cleaned=ArtifactRef(path="c.parquet", rows=1, sha256="x"),
        candidate_pairs=ArtifactRef(path="p.parquet", rows=1, sha256="x"),
        matches=ArtifactRef(path="m.parquet", rows=1, sha256="x"),
        non_matches=ArtifactRef(path="n.parquet", rows=1, sha256="x"),
        gate_explanations=ref if with_gate else None,
    )
    (settings.runs_dir / f"run_{run_id}.json").write_text(manifest.model_dump_json(indent=2))
    return run_id


@pytest.fixture
def client(test_settings):
    with TestClient(app) as c:
        yield c


# ─── happy path ───────────────────────────────────────────────────────────────
def test_returns_a_plot_ready_waterfall(client, test_settings):
    run_id = _write_run(test_settings)
    r = client.get(f"/explanations/nonmatch_gate/00001/00002?run_id={run_id}")
    assert r.status_code == 200
    body = r.json()

    assert body["model"] == "nonmatch_gate"
    assert body["run_id"] == run_id
    assert body["units"] == "log_odds"
    assert body["decision"]["tier"] == "human_review"
    assert body["decision"]["threshold"] == test_settings.gate_threshold
    assert body["base_value"] == pytest.approx(-0.5)
    # -0.5 + 2.0 + 1.0
    assert body["final_margin"] == pytest.approx(2.5)


def test_bars_are_connected_and_start_at_the_base_value(client, test_settings):
    run_id = _write_run(test_settings)
    body = client.get(f"/explanations/nonmatch_gate/00001/00002?run_id={run_id}").json()

    features = body["features"]
    assert features[0]["start"] == pytest.approx(body["base_value"])
    for prev, nxt in zip(features, features[1:]):
        assert nxt["start"] == pytest.approx(prev["end"])
    assert features[-1]["end"] == pytest.approx(body["final_margin"])


def test_features_are_ranked_and_labelled(client, test_settings):
    run_id = _write_run(test_settings)
    body = client.get(f"/explanations/nonmatch_gate/00001/00002?run_id={run_id}").json()

    assert [f["name"] for f in body["features"]] == ["sim_dob", "sound_last"]
    assert body["features"][0]["label"] == "Date of birth similarity"
    assert body["features"][0]["direction"] == "positive"


def test_categorical_feature_value_round_trips(client, test_settings):
    run_id = _write_run(test_settings)
    body = client.get(f"/explanations/nonmatch_gate/00003/00004?run_id={run_id}").json()
    sound = next(f for f in body["features"] if f["name"] == "sound_last")
    assert sound["value"] == "different"
    assert sound["direction"] == "negative"


def test_model_file_provenance_is_served(client, test_settings):
    run_id = _write_run(test_settings)
    body = client.get(f"/explanations/nonmatch_gate/00001/00002?run_id={run_id}").json()
    assert body["model_file"] == "nonmatch_gate_20260721T160717Z.pkl"


def test_pair_order_is_normalized(client, test_settings):
    """Pairs are canonicalized upstream (`PATID_A < PATID_B`); the UI should
    not have to know or care which way round it asks."""
    run_id = _write_run(test_settings)
    forward = client.get(f"/explanations/nonmatch_gate/00001/00002?run_id={run_id}")
    reverse = client.get(f"/explanations/nonmatch_gate/00002/00001?run_id={run_id}")
    assert reverse.status_code == 200
    assert reverse.json() == forward.json()


def test_run_id_defaults_to_the_latest_run_with_explanations(client, test_settings):
    _write_run(test_settings, "20260701T120000Z")
    newest = _write_run(test_settings, "20260728T120000Z")
    body = client.get("/explanations/nonmatch_gate/00001/00002").json()
    assert body["run_id"] == newest


def test_dropped_pairs_are_explained_too(client, test_settings):
    """The gate's drops are unrecoverable and this frame is their only record,
    so they must be explainable — that is the whole point of storing them."""
    run_id = _write_run(test_settings)
    body = client.get(f"/explanations/nonmatch_gate/00003/00004?run_id={run_id}").json()
    assert body["decision"]["tier"] == "no_match"
    assert body["decision"]["score"] == pytest.approx(0.0474259)


# ─── 404s that are normal outcomes, not errors ────────────────────────────────
def test_unknown_model_404s(client, test_settings):
    _write_run(test_settings)
    r = client.get("/explanations/not_a_model/00001/00002")
    assert r.status_code == 404
    assert "Unknown model" in r.json()["detail"]


def test_unscored_pair_404s(client, test_settings):
    """A pair the gate dropped never reaches the ML matcher; a
    deterministic auto-merge is never scored by either model. Both are normal."""
    run_id = _write_run(test_settings)
    r = client.get(f"/explanations/nonmatch_gate/99998/99999?run_id={run_id}")
    assert r.status_code == 404
    assert "not scored" in r.json()["detail"]


def test_unknown_run_404s(client, test_settings):
    _write_run(test_settings)
    r = client.get("/explanations/nonmatch_gate/00001/00002?run_id=nope")
    assert r.status_code == 404
    assert "Unknown run_id" in r.json()["detail"]


def test_run_without_explanations_404s(client, test_settings):
    """Explanations disabled, or the stage skipped — the scores are still
    valid, so this is a missing artifact, not a broken run."""
    run_id = _write_run(test_settings, with_gate=False)
    r = client.get(f"/explanations/nonmatch_gate/00001/00002?run_id={run_id}")
    assert r.status_code == 404
    assert "no explanations" in r.json()["detail"]


def test_no_runs_at_all_404s(client, test_settings):
    r = client.get("/explanations/ml_matcher/00001/00002")
    assert r.status_code == 404


def test_missing_artifact_on_disk_404s(client, test_settings):
    run_id = _write_run(test_settings)
    (test_settings.gate_output_dir / f"gate_explanations_{run_id}.parquet").unlink()
    r = client.get(f"/explanations/nonmatch_gate/00001/00002?run_id={run_id}")
    assert r.status_code == 404
    assert "missing on disk" in r.json()["detail"]
