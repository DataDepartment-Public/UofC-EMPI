"""Real end-to-end smoke test — not mocked. Runs the actual feature
engineering + LightGBM fit + persistence + promotion against tiny synthetic
data, to catch API-usage mistakes a purely static review can't."""

from __future__ import annotations

import joblib
import pandas as pd

from empi_model_training.training import lightgbm_train


def test_merge_reviewer_labels_none_is_a_no_op():
    pairs = pd.DataFrame({
        "PATID_A": ["P1"], "PATID_B": ["P2"],
        "final_gold_label": [True], "ambiguous_pair": [False], "target_ambiguous": [0],
    })
    out = lightgbm_train.merge_reviewer_labels(pairs, None)
    pd.testing.assert_frame_equal(out, pairs)


def test_merge_reviewer_labels_maps_to_ambiguous_pool(tmp_path):
    """Every reviewer-derived row is treated as 'ambiguous' regardless of
    which way the label went -- by construction, a pair only reaches a
    reviewer action because something upstream already flagged it as
    uncertain, so it belongs in the plausible pool either way."""
    pairs = pd.DataFrame({
        "PATID_A": [], "PATID_B": [], "final_gold_label": [], "ambiguous_pair": [],
        "target_ambiguous": [],
    })
    reviewer_path = tmp_path / "reviewer.csv"
    pd.DataFrame({
        "PATID_A": ["P1", "P4"], "PATID_B": ["P2", "P5"], "reviewer_label": [1, 0],
    }).to_csv(reviewer_path, index=False)

    out = lightgbm_train.merge_reviewer_labels(pairs, reviewer_path)

    row_match = out[(out["PATID_A"] == "P1") & (out["PATID_B"] == "P2")].iloc[0]
    assert row_match["final_gold_label"]
    assert row_match["ambiguous_pair"]
    assert row_match["target_ambiguous"] == 1

    row_no_match = out[(out["PATID_A"] == "P4") & (out["PATID_B"] == "P5")].iloc[0]
    assert not row_no_match["final_gold_label"]
    assert row_no_match["ambiguous_pair"]
    assert row_no_match["target_ambiguous"] == 1


def test_merge_reviewer_labels_wins_on_conflict(tmp_path):
    pairs = pd.DataFrame({
        "PATID_A": ["P1"], "PATID_B": ["P2"],
        "final_gold_label": [False], "ambiguous_pair": [False], "target_ambiguous": [0],
    })
    reviewer_path = tmp_path / "reviewer.csv"
    pd.DataFrame({"PATID_A": ["P2"], "PATID_B": ["P1"], "reviewer_label": [1]}).to_csv(
        reviewer_path, index=False
    )

    out = lightgbm_train.merge_reviewer_labels(pairs, reviewer_path)

    assert len(out) == 1
    assert bool(out.iloc[0]["final_gold_label"]) is True
    assert (out.iloc[0]["PATID_A"], out.iloc[0]["PATID_B"]) == ("P1", "P2")


def test_build_features_shape(synthetic_cleaned_parquet, synthetic_gold_labels_csv):
    _, cleaned = synthetic_cleaned_parquet
    gold = pd.read_csv(synthetic_gold_labels_csv, dtype={"PATID_A": str, "PATID_B": str})
    feat = lightgbm_train.build_features(gold[["PATID_A", "PATID_B"]], cleaned)

    assert list(feat.columns[:2]) == ["PATID_A", "PATID_B"]
    for col in lightgbm_train.FEATURE_COLS:
        assert col in feat.columns
    assert len(feat) == len(gold)


def test_full_training_run(tmp_path, synthetic_cleaned_parquet, synthetic_gold_labels_csv):
    cleaned_path, _ = synthetic_cleaned_parquet
    model_dir = tmp_path / "models" / "ml"

    args = lightgbm_train.parse_args(
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
    result = lightgbm_train.run(args)

    assert result.model_path.exists()
    assert (model_dir / "active.json").exists()
    meta = result.meta
    assert meta["model_name"] == "ml_matcher"
    assert "metrics_auto_merge" in meta["test_metrics"]
    assert 0.0 <= meta["test_metrics"]["metrics_auto_merge"]["precision"] <= 1.0

    # The persisted artifact round-trips through joblib and still scores.
    loaded = joblib.load(result.model_path)
    gold = pd.read_csv(synthetic_gold_labels_csv, dtype={"PATID_A": str, "PATID_B": str})
    cleaned = pd.read_parquet(cleaned_path)
    feat = lightgbm_train.build_features(gold[["PATID_A", "PATID_B"]].head(5), cleaned)
    proba = loaded.predict_proba(feat[lightgbm_train.FEATURE_COLS])
    assert proba.shape[1] == 2
