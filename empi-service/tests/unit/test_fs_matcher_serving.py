"""Serving round-trip test for FSMatcher: train -> save -> load -> score.

Trains a tiny Splink model ONCE per module (small inline fixture — no
`models/common` dependency) and asserts the load-and-score path is
deterministic and does not retrain.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.contracts import ProbabilisticMatches, validate, validate_fs_features
from src.models.fs_matcher.base import ClassificationConfig
from src.models.fs_matcher.matcher import FSMatcher

_CFG = ClassificationConfig(auto_merge_threshold=0.95, review_floor=0.40)


def _rec(pid, fn, ln, dob, ssn, email, addr, zip_, sex, phone):
    return dict(
        PATID=pid, FirstNM_clean=fn, LastNM_clean=ln, BirthDT_clean=dob,
        SSN_clean=ssn, last_4_SSN=(ssn[-4:] if ssn else None), Email_clean=email,
        AddressLine1_clean=addr, ZipCD_clean_base=zip_, SexAtBirthDSC_clean=sex,
        Phones_set={phone}, valid_record=True,
    )


def _fixture_frames():
    rows = [
        _rec("P01", "JOHN", "SMITH", "1990-01-01", "111223333", "john@x.com", "1 A ST", "60601", "MALE", "3125551000"),
        _rec("P02", "JOHN", "SMITH", "1990-01-01", "111223333", "john@x.com", "1 A ST", "60601", "MALE", "3125551000"),
        _rec("P03", "MARY", "JONES", "1985-05-05", "222334444", "mary@x.com", "2 B AVE", "60602", "FEMALE", "3125552000"),
        _rec("P04", "MARY", "JONES", "1985-05-05", "222334444", "mary@x.com", "2 B AVE", "60602", "FEMALE", "3125552000"),
        _rec("P05", "ALICE", "BROWN", "1970-07-07", "333445555", "alice@x.com", "3 C RD", "60603", "FEMALE", "3125553000"),
        _rec("P06", "ALICE", "BROWN", "1970-07-07", "333445555", "alice@x.com", "3 C RD", "60603", "FEMALE", "3125553000"),
        _rec("P07", "BOB", "DAVIS", "1960-03-03", "444556666", "bob@x.com", "4 D LN", "60604", "MALE", "3125554000"),
        _rec("P08", "BOB", "DAVIS", "1960-03-03", "444556666", "bob@x.com", "4 D LN", "60604", "MALE", "3125554000"),
        _rec("P09", "CARL", "WILSON", "1995-09-09", "555667777", "carl@x.com", "5 E CT", "60605", "MALE", "3125555000"),
        _rec("P10", "DANA", "MOORE", "1988-08-08", "666778888", "dana@x.com", "6 F WAY", "60606", "FEMALE", "3125556000"),
    ]
    df_clean = pd.DataFrame(rows)
    cp = pd.DataFrame({
        "PATID_A": ["P01", "P03", "P05", "P07", "P01", "P09"],
        "PATID_B": ["P02", "P04", "P06", "P08", "P03", "P10"],
        "source_blocks": ["SSN"] * 6,
        "n_blocks": [2, 2, 2, 2, 1, 1],
    })
    labels = pd.DataFrame({
        "PATID_A": ["P01", "P03", "P05", "P07", "P01", "P09"],
        "PATID_B": ["P02", "P04", "P06", "P08", "P03", "P10"],
        "silver_label": [1, 1, 1, 1, 0, 0],
    })
    return df_clean, cp, labels


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train once, persist settings to disk, return the serving handles."""
    df_clean, cp, labels = _fixture_frames()
    model = FSMatcher(labels_df=labels, label_col="silver_label",
                      classification_config=_CFG, u_max_pairs=1e4)
    _classified, linker = model.run(cp, df_clean, full_output=True, return_linker=True)
    settings_json = linker.misc.save_model_to_json()
    model_path = tmp_path_factory.mktemp("fsmodel") / "fs_model.json"
    model_path.write_text(json.dumps(settings_json))
    return {"model_path": model_path, "df_clean": df_clean, "cp": cp, "labels": labels}


def test_serving_matcher_does_not_carry_a_training_strategy():
    """A serving matcher (no labels) has no training strategy — score() can't retrain."""
    assert FSMatcher(classification_config=_CFG).training is None


def test_score_is_deterministic_across_two_load_and_score_passes(trained):
    serve = FSMatcher(classification_config=_CFG)
    s1 = serve.score_with_model_path(trained["cp"], trained["df_clean"], trained["model_path"])
    s2 = serve.score_with_model_path(trained["cp"], trained["df_clean"], trained["model_path"])
    merged = s1.merge(s2, on=["PATID_A", "PATID_B"], suffixes=("_1", "_2"))
    assert len(merged) == len(s1)
    assert (abs(merged["match_probability_1"] - merged["match_probability_2"]) < 1e-9).all()
    assert (merged["classification_tier_1"] == merged["classification_tier_2"]).all()


def test_true_duplicates_auto_merge(trained):
    serve = FSMatcher(classification_config=_CFG)
    scored = serve.score_with_model_path(trained["cp"], trained["df_clean"], trained["model_path"])
    tier = dict(zip(zip(scored["PATID_A"], scored["PATID_B"]), scored["classification_tier"]))
    # the 4 exact-duplicate pairs should clear auto_merge
    for pair in [("P01", "P02"), ("P03", "P04"), ("P05", "P06"), ("P07", "P08")]:
        assert tier[pair] == "auto_merge", pair


def test_projections_validate(trained):
    serve = FSMatcher(classification_config=_CFG)
    scored = serve.score_with_model_path(trained["cp"], trained["df_clean"], trained["model_path"])
    pm = serve.to_probabilistic_matches(scored)
    validate(pm, ProbabilisticMatches)
    assert len(pm) == len(scored)  # full audit — every scored pair

    feats = serve.to_fs_features(scored, candidates_only=True)
    validate_fs_features(feats)
    assert (feats["match_probability"] >= _CFG.review_floor).all()
    assert any(c.startswith("gamma_") for c in feats.columns)
    assert any(c.startswith("bf_") for c in feats.columns)
