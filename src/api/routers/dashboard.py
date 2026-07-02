"""GET /dashboard/summary — KPIs and chart data for the Dashboard tab
(FR-4..FR-17 in the Dashboard FR doc / docs/Dashboard-Guide.md §2).

Combines two sources, matching how the doc frames the two storage tiers:
  * `empi.db` (via `store.dashboard_summary`) — live, reviewer-action-aware
    counts (matched/needs-review/no-match/manual-merge), recomputed on every
    call so FR-40's real-time refresh is just "call this again".
  * the latest `RunManifest` on disk — pipeline-level numbers `empi.db` never
    had in the first place (raw/invalid record counts, candidate pairs) plus
    the run history for the FR-14/15 trend chart. Per docs/Dashboard-Guide.md's
    own "Open items" note, that trend is auto-match-rate / review-rate only —
    no precision/recall, since there are no ground-truth labels to compute
    them against.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from src.api import store
from src.api.deps import get_db, get_settings
from src.api.schemas import DashboardSummary, MatchStatusCounts
from src.config import Settings
from src.contracts import RunManifest
from src.models.deterministic_rules import RULES

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

CONFIDENCE_THRESHOLDS = {rule.name: rule.confidence for rule in RULES}


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def _all_manifests(settings: Settings) -> list[RunManifest]:
    if not settings.runs_dir.exists():
        return []
    out = []
    for path in sorted(settings.runs_dir.glob("run_*.json")):
        try:
            out.append(RunManifest.model_validate_json(path.read_text()))
        except (json.JSONDecodeError, ValueError):
            continue
    return sorted(out, key=lambda m: m.created_utc)


@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(
    conn=Depends(get_db), settings: Settings = Depends(get_settings)
) -> DashboardSummary:
    live = store.dashboard_summary(conn)
    manifests = _all_manifests(settings)
    latest = manifests[-1] if manifests else None
    latest_counts = latest.counts if latest else {}

    total_raw_rows = latest_counts.get("raw_rows", live["total_records"])
    valid_records = latest_counts.get("valid_records", live["total_records"])
    invalid_records = max(total_raw_rows - valid_records, 0)
    candidate_pairs = latest_counts.get("candidate_pairs", 0)
    auto_matches = latest_counts.get("matches", live["auto_match_records"])

    manual_total = live["manual_merge_actions"] + live["manual_unmerge_actions"]

    history = [
        {
            "run_id": m.run_id,
            "created_utc": m.created_utc,
            "auto_match_rate": _pct(
                m.counts.get("matches", 0), m.counts.get("candidate_pairs", 1)
            ),
            "review_rate": _pct(
                m.counts.get("non_matches", 0), m.counts.get("candidate_pairs", 1)
            ),
        }
        for m in manifests
    ]

    return DashboardSummary(
        last_run_id=latest.run_id if latest else None,
        last_run_created_utc=latest.created_utc if latest else None,
        model_version=latest.git_sha if latest else None,
        total_raw_rows=total_raw_rows,
        total_records=live["total_records"],
        invalid_records=invalid_records,
        duplicate_clusters=live["duplicate_clusters"],
        matched_records=live["matched_records"],
        matched_pct=_pct(live["matched_records"], live["total_records"]),
        unmerged_pct=_pct(live["unmerged_records"], live["total_records"]),
        needs_review_records=live["needs_review_records"],
        manual_merge_actions=live["manual_merge_actions"],
        manual_unmerge_actions=live["manual_unmerge_actions"],
        manual_merge_pct=_pct(live["manual_merge_actions"], manual_total),
        manual_unmerge_pct=_pct(live["manual_unmerge_actions"], manual_total),
        auto_match_rate=_pct(auto_matches, candidate_pairs),
        status_counts=MatchStatusCounts(
            auto_match=live["auto_match_records"],
            needs_review=live["needs_review_records"],
            no_match=live["no_match_records"],
        ),
        confidence_thresholds=CONFIDENCE_THRESHOLDS,
        history=history,
    )


__all__ = ["router"]
