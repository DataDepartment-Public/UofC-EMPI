"""tests/unit/test_fs_enhanced_3_real_runner.py

Unit tests for the PHI-free helpers in run_real_enhanced_3 (silver-label
split, pair canonicalization, binary metrics, label validation). The
Splink-dependent train/score path is VM-only and not exercised here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from models.experiments.fs_splink_enhanced_3 import run_real_enhanced_3 as r


# ── _canonicalize_pairs ───────────────────────────────────────────────────────
def test_canonicalize_orders_patids():
    df = pd.DataFrame({"PATID_A": ["b", "a"], "PATID_B": ["a", "z"], "label": [1, 0]})
    out = r._canonicalize_pairs(df)
    assert (out["PATID_A"] <= out["PATID_B"]).all()
    # label column preserved untouched
    assert list(out["label"]) == [1, 0]


# ── _stratified_split ─────────────────────────────────────────────────────────
def test_stratified_split_preserves_class_ratio():
    labels = pd.DataFrame(
        {
            "PATID_A": [str(i) for i in range(100)],
            "PATID_B": [str(i + 1000) for i in range(100)],
            "label": [1] * 60 + [0] * 40,
        }
    )
    train, test = r._stratified_split(labels, "label", test_size=0.2, seed=42)
    assert len(test) == 20 and len(train) == 80
    assert int((test["label"] == 1).sum()) == 12  # 20% of 60
    assert int((test["label"] == 0).sum()) == 8  # 20% of 40
    # No leakage between train and test indices in the source.
    assert len(train) + len(test) == len(labels)


def test_stratified_split_is_deterministic():
    labels = pd.DataFrame(
        {
            "PATID_A": [str(i) for i in range(50)],
            "PATID_B": [str(i + 500) for i in range(50)],
            "label": [1, 0] * 25,
        }
    )
    a1, b1 = r._stratified_split(labels, "label", 0.2, 7)
    a2, b2 = r._stratified_split(labels, "label", 0.2, 7)
    pd.testing.assert_frame_equal(b1, b2)


# ── _binary_metrics ───────────────────────────────────────────────────────────
def test_binary_metrics_perfect():
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([True, True, False, False])
    m = r._binary_metrics(y_true, y_pred)
    assert m == {"tp": 2, "fp": 0, "fn": 0, "tn": 2,
                 "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_binary_metrics_handles_empty_positive_prediction():
    y_true = pd.Series([1, 0])
    y_pred = pd.Series([False, False])
    m = r._binary_metrics(y_true, y_pred)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


# ── _load_silver_labels validation ────────────────────────────────────────────
def test_load_silver_labels_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"PATID_A": ["a"], "foo": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        r._load_silver_labels(bad, "label")


def test_load_silver_labels_accepts_valid(tmp_path):
    good = tmp_path / "good.csv"
    pd.DataFrame(
        {"PATID_A": ["a", "c"], "PATID_B": ["b", "d"], "label": [1, 0]}
    ).to_csv(good, index=False)
    df = r._load_silver_labels(good, "label")
    assert df["label"].dtype.kind == "i"
    assert list(df.columns) == ["PATID_A", "PATID_B", "label"]
