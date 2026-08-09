"""Real end-to-end smoke test — not mocked. Exercises actual Splink
comparison building + m/u/lambda estimation + persistence + promotion
against tiny synthetic data, to catch Splink API-usage mistakes a purely
static review can't."""

from __future__ import annotations

import json

import pandas as pd

from empi_model_training.training import fs_train


def test_merge_reviewer_labels_none_is_a_no_op():
    silver = pd.DataFrame({"PATID_A": ["P1"], "PATID_B": ["P2"], "silver_label": [1]})
    out = fs_train._merge_reviewer_labels(silver, None, "silver_label")
    pd.testing.assert_frame_equal(out, silver)


def test_merge_reviewer_labels_wins_on_conflict(tmp_path):
    """Reviewer-confirmed labels (empi-service's export_reviewer_labels.py
    output) are higher-trust than the silver-label proxy -- should override
    it for any pair present in both."""
    silver = pd.DataFrame({"PATID_A": ["P1"], "PATID_B": ["P2"], "silver_label": [1]})
    reviewer_path = tmp_path / "reviewer.csv"
    pd.DataFrame({"PATID_A": ["P2"], "PATID_B": ["P1"], "reviewer_label": [0]}).to_csv(
        reviewer_path, index=False
    )

    out = fs_train._merge_reviewer_labels(silver, reviewer_path, "silver_label")

    assert len(out) == 1
    assert out.iloc[0]["silver_label"] == 0
    assert (out.iloc[0]["PATID_A"], out.iloc[0]["PATID_B"]) == ("P1", "P2")


def test_build_settings_has_seven_comparisons():
    settings = fs_train.build_settings(include_address=True)
    assert len(settings["comparisons"]) == 7

    settings_no_address = fs_train.build_settings(include_address=False)
    assert len(settings_no_address["comparisons"]) == 6


def test_prepare_model_input_adds_derived_columns(synthetic_cleaned_parquet):
    _, cleaned = synthetic_cleaned_parquet
    prepared = fs_train.prepare_model_input(cleaned)
    assert fs_train.COL_DOB_STR in prepared.columns
    assert fs_train.COL_PHONES_ARRAY in prepared.columns
    assert prepared[fs_train.COL_DOB_STR].str.match(r"\d{4}-\d{2}-\d{2}").all()


def test_full_training_run(tmp_path, synthetic_cleaned_parquet, synthetic_silver_labels_csv):
    cleaned_path, _ = synthetic_cleaned_parquet
    model_dir = tmp_path / "models" / "fs"

    args = fs_train.parse_args(
        [
            "--cleaned-index",
            str(cleaned_path),
            "--silver-labels",
            str(synthetic_silver_labels_csv),
            "--model-dir",
            str(model_dir),
            "--u-max-pairs",
            "1e4",
            "--promote",
        ]
    )
    result = fs_train.run(args)

    assert result.model_path.exists()
    assert (model_dir / "active.json").exists()
    active = json.loads((model_dir / "active.json").read_text())
    assert active["model_file"] == result.model_path.name

    meta = result.meta
    assert meta["model_name"] == "fs_matcher"
    assert "metrics_auto_merge" in meta["test_metrics"]

    # The trained settings JSON is loadable and re-scorable (mirrors how
    # empi-service's FSMatcher.load_settings()/score() would use it).
    trained_settings = json.loads(result.model_path.read_text())
    import pandas as pd

    df_clean = pd.read_parquet(cleaned_path)
    df_model = fs_train.prepare_model_input(df_clean)
    sample_pairs = pd.DataFrame(
        {
            "PATID_A": df_model["PATID"].iloc[:3].values,
            "PATID_B": df_model["PATID"].iloc[3:6].values,
        }
    )
    scored = fs_train.score_pairs(df_model, sample_pairs, trained_settings)
    assert "match_probability" in scored.columns
