"""Unit tests for src/evaluation/run_eval serialization.

Focuses on the JSON-safety and PHI-stripping of the report assembly (the slow
pipeline/eval orchestration is exercised by integration paths, not here).
"""

import json

import pandas as pd

from src.evaluation.blocking_eval import BlockingEvalReport
from src.evaluation.run_eval import (
    EvalReport,
    _aggregate_fanout,
    _serialize_blocking_eval,
    _serialize_rule_eval,
)
from src.evaluation.rule_eval import RuleEvalReport


def test_aggregate_fanout_drops_value_list():
    fanout = {"distinct_shared": 3, "max_fanout": 7, "top": {"123456789": 7}}
    out = _aggregate_fanout(fanout)
    assert out == {"distinct_shared": 3, "max_fanout": 7}
    assert "top" not in out


def test_serialize_rule_eval_is_json_safe_and_phi_free():
    report = RuleEvalReport(
        headline={"confirmed_matches": 5},
        precision=pd.DataFrame([{"rule": "SSN_DOB", "precision_pct": 99.0}]),
        ssn_fanout={"distinct_shared": 1, "max_fanout": 9, "top": {"999999999": 9}},
        email_fanout={"distinct_shared": 0, "max_fanout": 0, "top": {}},
    )
    payload = _serialize_rule_eval(report)
    text = json.dumps(payload)  # must not raise
    assert "999999999" not in text  # SSN value stripped
    assert "top" not in payload["ssn_fanout"]
    assert payload["precision"][0]["rule"] == "SSN_DOB"


def test_serialize_blocking_eval_drops_example_rows():
    report = BlockingEvalReport(
        headline={"records_evaluated": 10},
        recall=98.0,
        missed_by_rule={"SSN_DOB": 2},
        missed_examples=pd.DataFrame([{"PATID_A": "X1", "PATID_B": "X2"}]),
    )
    payload = _serialize_blocking_eval(report)
    text = json.dumps(payload)
    assert "X1" not in text and "X2" not in text
    assert payload["recall_pct"] == 98.0
    assert payload["missed_by_rule"] == {"SSN_DOB": 2}


def test_eval_report_roundtrips_to_json():
    report = EvalReport(
        run_id="r1",
        created_utc="2026-06-21T00:00:00+00:00",
        git_sha="abc123",
        counts={"matches": 4},
        rule_eval={"headline": {"confirmed_matches": 4}, "precision": []},
        blocking_eval={"recall_pct": 97.9, "missed_by_rule": {}},
    )
    loaded = json.loads(report.to_json())
    assert loaded["run_id"] == "r1"
    assert loaded["counts"]["matches"] == 4
    assert "EVALUATION REPORT" in report.to_text()
