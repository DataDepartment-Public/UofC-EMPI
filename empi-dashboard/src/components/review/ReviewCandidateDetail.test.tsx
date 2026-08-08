import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReviewCandidateDetail } from "./ReviewCandidateDetail";
import { api } from "@/lib/api-client";
import type { ReviewQueueItem } from "@/lib/schemas";

/** The Merge/Not-a-match/View-raw-data buttons are the primary way a
 * reviewer acts on a candidate pair — this pins that each one calls its
 * callback with the exact patid/mid arguments the parent page needs to
 * build the right request, not just that the button exists. */

vi.mock("@/lib/api-client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api-client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getRaw: vi.fn().mockResolvedValue(null),
      getExplanation: vi.fn().mockResolvedValue(null),
    },
  };
});
void api; // keep the mocked import referenced for clarity

const item: ReviewQueueItem = {
  patid_a: "P1",
  patid_b: "P2",
  mid_a: "M-000001",
  mid_b: "M-000002",
  member_count_a: 1,
  member_count_b: 1,
  match_rule: "NAME_DOB_EMAIL",
  confidence: 0.99,
  evidence: "NAME_DOB_EMAIL",
  source_blocks: "B3|B7",
  reviewed: false,
  patient_a: { patid: "P1", first_name: "Jane", last_name: "Doe" },
  patient_b: { patid: "P2", first_name: "Jane", last_name: "Doe" },
};

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe("ReviewCandidateDetail", () => {
  it("calls onMerge with mid_a and patid_b when Merge is clicked", async () => {
    const onMerge = vi.fn();
    const user = userEvent.setup();
    renderWithClient(
      <ReviewCandidateDetail
        item={item}
        onMerge={onMerge}
        onDismiss={vi.fn()}
        onViewRaw={vi.fn()}
        dismissPending={false}
      />,
    );

    await user.click(screen.getByRole("button", { name: /^merge$/i }));
    expect(onMerge).toHaveBeenCalledWith("M-000001", ["P2"]);
  });

  it("calls onDismiss with both patids when Not a match is clicked", async () => {
    const onDismiss = vi.fn();
    const user = userEvent.setup();
    renderWithClient(
      <ReviewCandidateDetail
        item={item}
        onMerge={vi.fn()}
        onDismiss={onDismiss}
        onViewRaw={vi.fn()}
        dismissPending={false}
      />,
    );

    await user.click(screen.getByRole("button", { name: /not a match/i }));
    expect(onDismiss).toHaveBeenCalledWith("P1", "P2");
  });

  it("calls onViewRaw with patid_a when View raw data is clicked", async () => {
    const onViewRaw = vi.fn();
    const user = userEvent.setup();
    renderWithClient(
      <ReviewCandidateDetail
        item={item}
        onMerge={vi.fn()}
        onDismiss={vi.fn()}
        onViewRaw={onViewRaw}
        dismissPending={false}
      />,
    );

    await user.click(screen.getByRole("button", { name: /view raw data/i }));
    expect(onViewRaw).toHaveBeenCalledWith("P1");
  });

  it("disables Not a match while a dismiss is already pending", () => {
    renderWithClient(
      <ReviewCandidateDetail
        item={item}
        onMerge={vi.fn()}
        onDismiss={vi.fn()}
        onViewRaw={vi.fn()}
        dismissPending={true}
      />,
    );

    expect(screen.getByRole("button", { name: /not a match/i })).toBeDisabled();
  });
});
