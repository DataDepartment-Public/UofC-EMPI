"""
tests/integration/test_fellegi_sunter_baseline.py

Full-pipeline integration tests for the Fellegi-Sunter Splink baseline
(implementation plan Sections 5.2 and 7). These train a real Splink model on
the synthetic dataset (via the shared `fs_classified` / `fs_candidate_pairs`
fixtures in tests/conftest.py) and exercise the end-to-end orchestration.

Covers:
  - run_fs_baseline's default return obeys the standardized 5-column
    evaluation contract (PATID_A, PATID_B, model_name, score, predicted_tier)
  - clean separation of the 8 synthetic truth pairs vs. the 2 known
    non-matches
  - full_output=True still yields the rich Phase A calibration frame
  - the EXISTS candidate-pairs blocking rule scores EXACTLY the candidate
    pairs (plan Section 5.2) — no fan-out, none dropped

These are slower than tests/unit (they train a Splink model, ~5-10s) but
still synthetic-data-only and safe to run in CI. The trained-parameter
sanity / m-u snapshot checks live in tests/regression instead.
"""

from models.experiments.fs_splink_baseline import fellegi_sunter_baseline as fs
from models.common import synthetic_data as sd


# ===========================================================================
# run_fs_baseline — standardized output contract
# ===========================================================================
def test_run_fs_baseline_default_schema(fs_df_clean, fs_candidate_pairs_path):
    """Default return must obey the standardized evaluation contract."""
    # Single-CPU sandbox cap (see FS_U_MAX_PAIRS_SANDBOX in tests/conftest.py
    # and NEW ISSUE A in fellegi_sunter_baseline.py); VM default is 1e6.
    out = fs.run_fs_baseline(
        fs_candidate_pairs_path, fs_df_clean, u_max_pairs=1e4
    )
    assert list(out.columns) == fs.EVAL_SCHEMA_COLUMNS
    assert (out["model_name"] == fs.MODEL_NAME).all()
    assert out["score"].between(0.0, 1.0).all()
    assert out["predicted_tier"].isin(
        {"auto_merge", "human_review", "no_match"}
    ).all()


def test_run_fs_baseline_separates_truth_from_nonmatch(fs_classified):
    """All 8 true-duplicate pairs > 0.5; both known non-matches < 0.5."""
    eval_df = fs.to_evaluation_schema(fs_classified)
    scored = {
        (r.PATID_A, r.PATID_B): r.score for r in eval_df.itertuples(index=False)
    }

    truth = sd.ground_truth_pairs()
    nonmatch = sd.ground_truth_nonmatches()

    # Every truth pair was scored and scored as a probable match.
    for pair in truth:
        assert pair in scored, f"truth pair {pair} missing from output"
        assert scored[pair] > 0.5, f"truth pair {pair} scored low: {scored[pair]}"

    # Every known non-match was scored and scored as a probable non-match.
    for pair in nonmatch:
        assert pair in scored, f"non-match {pair} missing from output"
        assert scored[pair] < 0.5, f"non-match {pair} scored high: {scored[pair]}"


def test_run_fs_baseline_full_output(fs_classified):
    """full_output=True yields the rich calibration frame (gamma_* + weight)."""
    gamma = [c for c in fs_classified.columns if c.startswith("gamma_")]
    # Post-R2: FirstNM, LastNM, DOB, SSN, Email, Phones, ZIP, Address (8 comparisons).
    assert len(gamma) == 8
    assert "match_weight" in fs_classified.columns
    assert "match_probability" in fs_classified.columns
    assert "classification_tier" in fs_classified.columns
    # No inert shim passthrough columns should leak into the output.
    leaked = [
        c for c in fs_classified.columns
        if c.startswith("PATID_A_") or c.startswith("PATID_B_")
    ]
    assert leaked == []


# ===========================================================================
# Blocking-rule validation (plan Section 5.2)
# ===========================================================================
def test_scored_pairs_match_candidate_pairs_exactly(fs_classified, fs_candidate_pairs):
    """
    The EXISTS blocking rule must restrict scoring to EXACTLY the candidate
    pairs — no fan-out (extra pairs) and no dropped pairs.
    """
    scored = {
        (a, b) for a, b in zip(fs_classified["PATID_A"], fs_classified["PATID_B"])
    }
    expected = {
        (a, b)
        for a, b in zip(fs_candidate_pairs["PATID_A"], fs_candidate_pairs["PATID_B"])
    }
    assert scored == expected, (
        f"extra={scored - expected}  dropped={expected - scored}"
    )
    # And the row count matches (no duplicate scoring of a pair).
    assert len(fs_classified) == len(fs_candidate_pairs)