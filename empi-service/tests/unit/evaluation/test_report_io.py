"""Tests for the read side of the stored end-to-end reports.

These frames feed a notebook, so the contract that matters is *robustness*: a
directory containing an unreadable file, a report from an older schema, or a run
where a stage was skipped must still produce a usable table.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import Settings
from src.evaluation.report_io import (
    cluster_frame,
    funnel_frame,
    load_reports,
    loss_frame,
    metric_history,
    stage_frame,
    summary_frame,
    transitivity_frame,
)


def _report(run_id="R1", source="gold", holdout="strict", evaluated="2026-07-29T10:00:00+00:00",
            precision=0.9, recall=0.4, session_id=None, **over) -> dict:
    report = {
        "session_id": session_id or run_id,
        "run_id": run_id,
        "evaluated_utc": evaluated,
        "git_sha": "abcdef1234567890",
        "label_source": source,
        "label_col": "final_gold_label",
        "leakage": {"restriction": holdout, "note": None},
        "universe": {"labeled_pairs": 100, "positives": 10},
        "clustering": {"precision": precision, "recall": recall, "f1": 0.55,
                       "TP": 4, "FP": 1, "FN": 6},
        "stage_pairwise": {
            "rules_auto_merge": {"precision": 0.99, "recall": 0.37, "f1": 0.54,
                                 "TP": 4, "FP": 1, "FN": 6, "target": "final_gold_label",
                                 "scored_pairs": 100, "labeled_pairs": 100},
            "gate_pass": {"precision": 0.8, "recall": 1.0, "f1": 0.89, "TP": 8,
                          "FP": 2, "FN": 0, "target": "plausible",
                          "scored_pairs": 20, "labeled_pairs": 100},
            "ml_auto_merge": {"skipped": True},
        },
        "funnel": {"blocked": {"positives": 10, "negatives": 90, "total": 100},
                   "clustered": {"positives": 4, "negatives": 1, "total": 5}},
        "loss_attribution": {"total_missed": 6,
                             "by_stage": {"not blocked": 1,
                                          "left in review (no auto_merge edge)": 5}},
        "transitivity": {"direct": {"n": 4, "TP": 4, "FP": 0, "precision": 1.0},
                         "transitive_only": {"n": 1, "TP": 0, "FP": 1, "precision": 0.0}},
        "cluster_level": {
            "bcubed": {"precision": 0.93, "recall": 0.9, "f1": 0.91, "n_records": 50},
            "recovery": {"exact_rate": 0.75,
                         "non_singleton": {"exact_rate": 0.35, "split": 12, "merged": 3}},
            "closure_diagnostics": {"n_contradicted": 2},
        },
    }
    report.update(over)
    return report


@pytest.fixture
def runs_dir(tmp_path):
    d = tmp_path / "evaluations"
    d.mkdir()
    return d


def _write(runs_dir, report, name=None):
    holdout = report["leakage"]["restriction"]
    name = name or f"eval_{report['session_id']}__{report['label_source']}__{holdout}.json"
    (runs_dir / name).write_text(json.dumps(report))


# ── loading ──────────────────────────────────────────────────────────────────
def test_loads_every_report_in_the_directory(runs_dir):
    _write(runs_dir, _report(run_id="R1"))
    _write(runs_dir, _report(run_id="R2"))
    assert len(load_reports(runs_dir)) == 2


def test_ignores_files_that_are_not_reports(runs_dir):
    _write(runs_dir, _report())
    (runs_dir / "run_R1.json").write_text('{"run_id": "R1"}')  # a pipeline manifest
    (runs_dir / "eval_R1__gold__strict.txt").write_text("text report")
    assert len(load_reports(runs_dir)) == 1


def test_one_corrupt_report_does_not_sink_the_rest(runs_dir):
    """A half-written file must not make the whole comparison unreadable."""
    _write(runs_dir, _report(run_id="R1"))
    (runs_dir / "eval_R2__gold__strict.json").write_text("{not json")
    reports = load_reports(runs_dir)
    assert len(reports) == 1 and reports[0]["run_id"] == "R1"


def test_missing_directory_returns_empty_not_an_error(tmp_path):
    assert load_reports(tmp_path / "nope") == []


def test_newest_evaluation_first(runs_dir):
    _write(runs_dir, _report(run_id="OLD", evaluated="2026-07-01T00:00:00+00:00"))
    _write(runs_dir, _report(run_id="NEW", evaluated="2026-07-29T00:00:00+00:00"))
    assert [r["run_id"] for r in load_reports(runs_dir)] == ["NEW", "OLD"]


def test_filters_use_report_fields_not_filenames(runs_dir):
    """A hand-renamed file must still be bucketed by what it actually contains."""
    _write(runs_dir, _report(run_id="R1", source="gold", holdout="strict"),
           name="eval_misleading_name__synthetic__none.json")
    assert len(load_reports(runs_dir, sources=["synthetic"])) == 0
    assert len(load_reports(runs_dir, sources=["gold"], holdouts=["strict"])) == 1


def test_settings_supply_the_default_directory(runs_dir):
    _write(runs_dir, _report())
    assert len(load_reports(settings=Settings(evaluations_dir=runs_dir))) == 1


def test_reports_from_before_sessions_existed_fall_back_to_run_id(runs_dir):
    """Old files must still plot rather than collapsing onto a '?' bucket."""
    legacy = _report(run_id="OLD")
    del legacy["session_id"]
    _write(runs_dir, legacy, name="eval_OLD__gold__strict.json")
    assert summary_frame(load_reports(runs_dir))["session_id"].iloc[0] == "OLD"


# ── frames ───────────────────────────────────────────────────────────────────
def test_summary_frame_is_one_row_per_report(runs_dir):
    _write(runs_dir, _report(run_id="R1"))
    _write(runs_dir, _report(run_id="R2", holdout="none"))
    df = summary_frame(load_reports(runs_dir))
    assert len(df) == 2
    assert set(df.columns) >= {"session_id", "run_id", "source", "holdout",
                               "precision", "recall", "f1"}


def test_summary_frame_truncates_the_git_sha(runs_dir):
    df = summary_frame([_report()])
    assert df["git_sha"].iloc[0] == "abcdef12"


def test_summary_frame_of_nothing_is_empty_not_an_error():
    assert summary_frame([]).empty


def test_stage_frame_keeps_skipped_stages_as_rows():
    """Dropping them would silently mis-align a comparison across reports."""
    df = stage_frame(_report())
    assert len(df) == 3
    assert df.loc[df["stage"] == "ml_auto_merge", "status"].iloc[0] == "not present in this run"


def test_stage_frame_carries_the_scored_population():
    df = stage_frame(_report()).set_index("stage")
    assert df.loc["gate_pass", "scored_pairs"] == 20
    assert df.loc["rules_auto_merge", "scored_pairs"] == 100


def test_funnel_and_transitivity_frames_shape():
    assert list(funnel_frame(_report())["stage"]) == ["blocked", "clustered"]
    assert list(transitivity_frame(_report())["merge_kind"]) == ["direct", "transitive_only"]


def test_loss_frame_is_sorted_by_impact_with_shares():
    df = loss_frame(_report())
    assert df["missed"].tolist() == [5, 1]
    assert df["share_pct"].tolist() == [83.3, 16.7]


def test_loss_frame_handles_a_run_that_missed_nothing():
    report = _report(loss_attribution={"total_missed": 0, "by_stage": {"not blocked": 0}})
    assert loss_frame(report)["share_pct"].isna().all()


def test_cluster_frame_keeps_the_closure_caveat_beside_the_metrics():
    """The contradiction count must travel with the B-cubed numbers, not live
    in a separate table where it gets dropped."""
    df = cluster_frame([_report()])
    assert df["closure_contradictions"].iloc[0] == 2
    assert df["bcubed_f1"].iloc[0] == 0.91
    assert df["nonsingleton_exact_rate"].iloc[0] == 0.35


def test_cluster_frame_labels_declared_truth_when_present():
    report = _report()
    report["cluster_level"]["closure_diagnostics"]["source"] = "declared ground truth"
    assert cluster_frame([report])["truth_source"].iloc[0] == "declared ground truth"


# ── history ──────────────────────────────────────────────────────────────────
def test_metric_history_is_long_format_one_row_per_series():
    reports = [_report(run_id="R1", recall=0.4), _report(run_id="R2", recall=0.5)]
    df = metric_history(reports, "recall")
    assert len(df) == 2
    assert df["value"].tolist() == [0.4, 0.5]
    assert df["series"].unique().tolist() == ["gold / strict"]


def test_metric_history_puts_both_halves_of_a_session_on_one_point():
    """The real-data and synthetic runs of one session are different pipeline
    runs but one measurement — they must share an x-position."""
    reports = [
        _report(session_id="S1", run_id="S1_real", source="gold", holdout="none"),
        _report(session_id="S1", run_id="S1_synthetic", source="synthetic",
                holdout="n/a"),
    ]
    df = metric_history(reports, "recall")
    assert df["session_id"].unique().tolist() == ["S1"]
    assert set(df["series"]) == {"gold / none", "synthetic / n/a"}


def test_metric_history_separates_source_and_holdout_into_series():
    reports = [_report(run_id="R1", holdout="strict"),
               _report(run_id="R1", holdout="none"),
               _report(run_id="R1", source="synthetic", holdout="n/a")]
    assert set(metric_history(reports, "recall")["series"]) == {
        "gold / strict", "gold / none", "synthetic / n/a"}


def test_metric_history_reads_any_stage():
    df = metric_history([_report()], "recall", stage="gate_pass")
    assert df["value"].iloc[0] == 1.0


def test_metric_history_skips_stages_a_run_did_not_have():
    assert metric_history([_report()], "recall", stage="ml_auto_merge").empty


def test_metric_history_of_nothing_is_an_empty_frame():
    assert isinstance(metric_history([], "recall"), pd.DataFrame)
    assert metric_history([], "recall").empty
