import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import { ClusterComparisons } from "./ClusterComparisons";
import type { ClusterPairsResponse, Entity } from "@/lib/schemas";

const mockUseClusterPairs = vi.fn();
vi.mock("@/lib/hooks", () => ({
  useClusterPairs: (mid: string | null) => mockUseClusterPairs(mid),
}));

const entity = {
  mid: "M-000001",
  run_id: "r1",
  origin: "merge",
  is_merged: true,
  confidence: 0.9,
  updated_utc: "2026-08-09T12:00:00Z",
  members: [
    { patid: "P1", first_name: "Jane", last_name: "Doe" },
    { patid: "P2", first_name: "Jane", last_name: "Doe" },
  ],
} as unknown as Entity;

function trace(over: Partial<ClusterPairsResponse>): ClusterPairsResponse {
  return {
    mid: "M-000001",
    run_id: "r1",
    artifacts_available: true,
    unresolved_run_id: null,
    pairs_truncated: false,
    members: ["P1", "P2"],
    thresholds: { gate_threshold: 0.3, ml_auto_merge_threshold: 0.7 },
    pairs: [
      {
        patid_a: "P1",
        patid_b: "P2",
        verdict: "auto_merge_rule",
        blocked: true,
        joined_by: "pipeline",
      },
    ],
    external_pairs: [],
    ...over,
  } as unknown as ClusterPairsResponse;
}

function show(data: ClusterPairsResponse) {
  mockUseClusterPairs.mockReturnValue({ data, isLoading: false, isError: false });
  render(<ClusterComparisons entity={entity} />);
}

describe("ClusterComparisons", () => {
  it("renders the trace normally when the run resolved", () => {
    show(trace({}));
    expect(screen.queryByText(/no longer on disk/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/outside a full pipeline run/i)).not.toBeInTheDocument();
  });

  it("says the run kept no artifacts, not that the pipeline decided nothing", () => {
    // The entity was last touched by incremental scoring, so the API returns
    // an empty trace with `unresolved_run_id` rather than another run's data.
    show(
      trace({
        run_id: null,
        artifacts_available: false,
        unresolved_run_id: "score-20260809T120000Z",
      }),
    );
    expect(screen.getByText(/outside a full pipeline run/i)).toBeInTheDocument();
    expect(screen.getByText("score-20260809T120000Z")).toBeInTheDocument();
    expect(screen.queryByText(/no longer on disk/i)).not.toBeInTheDocument();
  });

  it("keeps the aged-out-artifacts wording when the run itself resolved", () => {
    show(trace({ artifacts_available: false }));
    expect(screen.getByText(/no longer on disk/i)).toBeInTheDocument();
  });

  it("explains an oversized cluster instead of showing an empty pair list", () => {
    show(
      trace({
        pairs_truncated: true,
        pairs: [],
        members: Array.from({ length: 700 }, (_, i) => `H${i}`),
      }),
    );
    expect(screen.getByText(/too many to trace/i)).toBeInTheDocument();
    expect(screen.getByText(/700 records/)).toBeInTheDocument();
    // Must not be mistaken for the singleton case.
    expect(screen.queryByText(/single-record cluster/i)).not.toBeInTheDocument();
  });
});
