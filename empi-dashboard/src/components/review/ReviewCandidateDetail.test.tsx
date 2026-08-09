import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReviewCandidateDetail } from "./ReviewCandidateDetail";
import { api } from "@/lib/api-client";
import type { ReviewQueueItem } from "@/lib/schemas";

/** The Merge/Not-a-match buttons are the primary way a reviewer acts on a
 * candidate pair — this pins that each one calls its callback with the exact
 * patid/mid arguments the parent page needs to build the right request, not
 * just that the button exists. */

const getRaw = vi.fn();

vi.mock("@/lib/api-client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api-client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getRaw: (patid: string) => getRaw(patid),
      getExplanation: vi.fn().mockResolvedValue(null),
    },
  };
});
void api; // keep the mocked import referenced for clarity

beforeEach(() => {
  getRaw.mockReset();
  getRaw.mockImplementation((patid: string) =>
    Promise.resolve({
      patid,
      fields: {
        FirstNM_raw: "Jane",
        BirthDT_raw: patid === "P1" ? "1974-02-12 00:00:00" : "1974-02-12",
        Email_raw: patid === "P1" ? "jane@example.com" : "j.doe@example.com",
      },
    }),
  );
});

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
  verdict: "ml_human_review",
  bucket: "needs_review",
  reviewer_decision: null,
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
        dismissPending={false}
      />,
    );

    await user.click(screen.getByRole("button", { name: /not a match/i }));
    expect(onDismiss).toHaveBeenCalledWith("P1", "P2");
  });

  it("disables Not a match while a dismiss is already pending", () => {
    renderWithClient(
      <ReviewCandidateDetail
        item={item}
        onMerge={vi.fn()}
        onDismiss={vi.fn()}
        dismissPending={true}
      />,
    );

    expect(screen.getByRole("button", { name: /not a match/i })).toBeDisabled();
  });
});

/** The panel replaced a drawer that could only show one side of the pair, so
 * what matters is that it stays out of the way until asked and then renders
 * *both* records against each other — a regression to one side would
 * silently reinstate the view this panel exists to fix. */
describe("ReviewCandidateDetail raw comparison", () => {
  const renderDetail = () =>
    renderWithClient(
      <ReviewCandidateDetail
        item={item}
        onMerge={vi.fn()}
        onDismiss={vi.fn()}
        dismissPending={false}
      />,
    );

  it("stays collapsed until the reviewer asks for it", () => {
    renderDetail();
    expect(
      screen.getByRole("button", { name: /compare raw source data/i }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("FirstNM_raw")).not.toBeInTheDocument();
  });

  it("shows both records and compares them field by field", async () => {
    const user = userEvent.setup();
    renderDetail();

    await user.click(
      screen.getByRole("button", { name: /compare raw source data/i }),
    );

    expect(getRaw).toHaveBeenCalledWith("P1");
    expect(getRaw).toHaveBeenCalledWith("P2");

    const firstNameRow = (await screen.findByText("FirstNM_raw")).closest("tr")!;
    expect(within(firstNameRow).getByText("Exact match")).toBeInTheDocument();

    // Both sides present but unequal — the reviewer's actual signal.
    const emailRow = screen.getByText("Email_raw").closest("tr")!;
    expect(within(emailRow).getByText("jane@example.com")).toBeInTheDocument();
    expect(within(emailRow).getByText("j.doe@example.com")).toBeInTheDocument();
    expect(within(emailRow).getByText("Different")).toBeInTheDocument();

    // A midnight time on one side only must not read as a disagreement.
    const dobRow = screen.getByText("BirthDT_raw").closest("tr")!;
    expect(within(dobRow).getByText("Exact match")).toBeInTheDocument();
  });
});
