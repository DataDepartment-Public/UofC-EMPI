"""Real end-to-end smoke test — not mocked. Runs the actual feature
engineering + LightGBM fit + persistence + promotion against tiny synthetic
data, to catch API-usage mistakes a purely static review can't."""

from __future__ import annotations

import joblib
import pandas as pd

from empi_model_training.training import nonmatch_gate_train


def test_merge_reviewer_labels_none_is_a_no_op():
    pairs = pd.DataFrame(
        {
            "PATID_A": ["P1"],
            "PATID_B": ["P2"],
            "final_gold_label": [True],
            "ambiguous_pair": [False],
            "target_plausible": [1],
        }
    )
    out = nonmatch_gate_train.merge_reviewer_labels(pairs, None)
    pd.testing.assert_frame_equal(out, pairs)


def test_merge_reviewer_labels_always_plausible(tmp_path):
    """A reviewer only ever acts on a pair the gate already let through, so
    every reviewer-derived row is target_plausible=1 regardless of which way
    reviewer_label went."""
    pairs = pd.DataFrame(
        {
            "PATID_A": [],
            "PATID_B": [],
            "final_gold_label": [],
            "ambiguous_pair": [],
            "target_plausible": [],
        }
    )
    reviewer_path = tmp_path / "reviewer.csv"
    pd.DataFrame(
        {
            "PATID_A": ["P1", "P4"],
            "PATID_B": ["P2", "P5"],
            "reviewer_label": [1, 0],
        }
    ).to_csv(reviewer_path, index=False)

    out = nonmatch_gate_train.merge_reviewer_labels(pairs, reviewer_path)

    row_match = out[(out["PATID_A"] == "P1") & (out["PATID_B"] == "P2")].iloc[0]
    assert row_match["final_gold_label"]
    assert not row_match["ambiguous_pair"]
    assert row_match["target_plausible"] == 1

    row_no_match = out[(out["PATID_A"] == "P4") & (out["PATID_B"] == "P5")].iloc[0]
    assert not row_no_match["final_gold_label"]
    assert not row_no_match["ambiguous_pair"]
    assert row_no_match["target_plausible"] == 1


def test_merge_reviewer_labels_wins_on_conflict(tmp_path):
    pairs = pd.DataFrame(
        {
            "PATID_A": ["P1"],
            "PATID_B": ["P2"],
            "final_gold_label": [False],
            "ambiguous_pair": [False],
            "target_plausible": [0],
        }
    )
    reviewer_path = tmp_path / "reviewer.csv"
    pd.DataFrame({"PATID_A": ["P2"], "PATID_B": ["P1"], "reviewer_label": [1]}).to_csv(
        reviewer_path, index=False
    )

    out = nonmatch_gate_train.merge_reviewer_labels(pairs, reviewer_path)

    assert len(out) == 1
    assert bool(out.iloc[0]["final_gold_label"]) is True
    assert (out.iloc[0]["PATID_A"], out.iloc[0]["PATID_B"]) == ("P1", "P2")


def test_load_gold_pairs_keeps_whole_pool(synthetic_gold_labels_csv):
    """v5-generation gate training uses the whole pool, no filter. Three
    mutually-exclusive classes partition it: confident match and ambiguous
    are both target_plausible=1 (a real match or a genuine hard case worth a
    human's time); only a confident non-match is target_plausible=0."""
    raw = pd.read_csv(synthetic_gold_labels_csv)
    out = nonmatch_gate_train.load_gold_pairs(synthetic_gold_labels_csv)
    assert len(out) == len(raw)
    plausible = out["final_gold_label"] | out["ambiguous_pair"]
    assert (out["target_plausible"] == plausible.astype(int)).all()
    confident_non_match = ~out["final_gold_label"] & ~out["ambiguous_pair"]
    assert (out.loc[confident_non_match, "target_plausible"] == 0).all()


def test_build_features_shape(synthetic_cleaned_parquet, synthetic_gold_labels_csv):
    _, cleaned = synthetic_cleaned_parquet
    gold = pd.read_csv(synthetic_gold_labels_csv, dtype={"PATID_A": str, "PATID_B": str})
    feat = nonmatch_gate_train.build_features(gold[["PATID_A", "PATID_B"]], cleaned)

    assert list(feat.columns[:2]) == ["PATID_A", "PATID_B"]
    for col in nonmatch_gate_train.FEATURE_COLS:
        assert col in feat.columns
    assert len(feat) == len(gold)


def test_full_training_run(tmp_path, synthetic_cleaned_parquet, synthetic_gold_labels_csv):
    cleaned_path, _ = synthetic_cleaned_parquet
    model_dir = tmp_path / "models" / "nonmatch_gate"

    args = nonmatch_gate_train.parse_args(
        [
            "--cleaned-index",
            str(cleaned_path),
            "--gold-labels",
            str(synthetic_gold_labels_csv),
            "--model-dir",
            str(model_dir),
            "--promote",
        ]
    )
    result = nonmatch_gate_train.run(args)

    assert result.model_path.exists()
    assert (model_dir / "active.json").exists()
    meta = result.meta
    assert meta["model_name"] == "nonmatch_gate"
    assert "gate_at_threshold" in meta["test_metrics"]
    assert 0.0 <= meta["test_metrics"]["gate_at_threshold"]["plausible_precision"] <= 1.0
    assert 0.0 <= meta["test_metrics"]["gate_at_threshold"]["plausible_recall"] <= 1.0

    # The persisted artifact is the RAW fitted model (no adapter wrapper,
    # unlike the ML matcher) and round-trips through joblib.
    loaded = joblib.load(result.model_path)
    gold = pd.read_csv(synthetic_gold_labels_csv, dtype={"PATID_A": str, "PATID_B": str})
    cleaned = pd.read_parquet(cleaned_path)
    feat = nonmatch_gate_train.build_features(gold[["PATID_A", "PATID_B"]].head(5), cleaned)
    proba = loaded.predict_proba(feat[nonmatch_gate_train.FEATURE_COLS])
    assert proba.shape[1] == 2
