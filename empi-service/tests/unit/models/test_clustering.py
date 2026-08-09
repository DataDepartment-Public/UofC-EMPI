"""
Unit tests for the terminal clustering stage in src/models/clustering.py.

Coverage:
    - assign_clusters links transitive matches into connected components;
      cluster ids are contiguous and deterministic.
    - build_cluster_assignments gives every valid record a cluster id,
      including singletons no rule matched, and excludes invalid records.
"""

import pandas as pd
import pytest

from src.contracts import ClusterAssignments, validate
from src.models.clustering import assign_clusters, build_cluster_assignments


def _clean_df(patids: list[str], valid: list[bool] | None = None) -> pd.DataFrame:
    valid = [True] * len(patids) if valid is None else valid
    return pd.DataFrame({"PATID": patids, "valid_record": valid})


# ── assign_clusters ──────────────────────────────────────────────────────────
class TestAssignClusters:
    def test_transitive_link_one_cluster(self):
        clusters = assign_clusters(
            pd.DataFrame({"PATID_A": ["A", "B"], "PATID_B": ["B", "C"]})
        )
        assert clusters["A"] == clusters["B"] == clusters["C"]

    def test_separate_components_distinct_ids(self):
        clusters = assign_clusters(
            pd.DataFrame({"PATID_A": ["A", "C"], "PATID_B": ["B", "D"]})
        )
        assert clusters["A"] == clusters["B"]
        assert clusters["C"] == clusters["D"]
        assert clusters["A"] != clusters["C"]

    def test_cluster_ids_are_contiguous(self):
        clusters = assign_clusters(
            pd.DataFrame({"PATID_A": ["A", "C", "E"], "PATID_B": ["B", "D", "F"]})
        )
        assert sorted(set(clusters.values())) == [0, 1, 2]

    def test_empty_matches_returns_empty_dict(self):
        assert assign_clusters(pd.DataFrame(columns=["PATID_A", "PATID_B"])) == {}


# ── build_cluster_assignments ────────────────────────────────────────────────
class TestBuildClusterAssignments:
    def test_matched_records_share_a_cluster(self):
        matches = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["B"]})
        cleaned = _clean_df(["A", "B"])
        out = build_cluster_assignments(matches, cleaned)
        row_a = out.loc[out["PATID"] == "A", "cluster_id"].iloc[0]
        row_b = out.loc[out["PATID"] == "B", "cluster_id"].iloc[0]
        assert row_a == row_b

    def test_singletons_get_their_own_cluster(self):
        matches = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["B"]})
        cleaned = _clean_df(["A", "B", "C", "D"])
        out = build_cluster_assignments(matches, cleaned)
        assert set(out["PATID"]) == {"A", "B", "C", "D"}
        assert out["cluster_id"].nunique() == 3  # {A,B}, {C}, {D}

    def test_no_matches_every_record_is_a_singleton(self):
        matches = pd.DataFrame(columns=["PATID_A", "PATID_B"])
        cleaned = _clean_df(["A", "B", "C"])
        out = build_cluster_assignments(matches, cleaned)
        assert out["cluster_id"].nunique() == 3
        assert out["cluster_id"].is_unique

    def test_invalid_records_excluded(self):
        matches = pd.DataFrame(columns=["PATID_A", "PATID_B"])
        cleaned = _clean_df(["A", "B"], valid=[True, False])
        out = build_cluster_assignments(matches, cleaned)
        assert set(out["PATID"]) == {"A"}

    def test_cluster_ids_deterministic_across_calls(self):
        matches = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["B"]})
        cleaned = _clean_df(["A", "B", "C", "D"])
        out1 = build_cluster_assignments(matches, cleaned)
        out2 = build_cluster_assignments(matches, cleaned)
        pd.testing.assert_frame_equal(
            out1.sort_values("PATID").reset_index(drop=True),
            out2.sort_values("PATID").reset_index(drop=True),
        )

    def test_output_validates_against_contract(self):
        matches = pd.DataFrame({"PATID_A": ["A"], "PATID_B": ["B"]})
        cleaned = _clean_df(["A", "B", "C"])
        out = build_cluster_assignments(matches, cleaned)
        validate(out, ClusterAssignments, allow_empty=False)

    def test_empty_valid_population_returns_empty_frame(self):
        matches = pd.DataFrame(columns=["PATID_A", "PATID_B"])
        cleaned = _clean_df([], valid=[])
        out = build_cluster_assignments(matches, cleaned)
        assert out.empty
        assert list(out.columns) == ["PATID", "cluster_id"]
