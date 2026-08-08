import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { UnmergeModal } from "./UnmergeModal";

describe("UnmergeModal", () => {
  it("names the patient, patid, and master record being split", () => {
    render(
      <UnmergeModal
        mid="M-000005"
        patid="P2"
        patientName="Jane Doe"
        pending={false}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText("Jane Doe")).toBeInTheDocument();
    expect(screen.getByText("P2")).toBeInTheDocument();
    expect(screen.getByText("M-000005")).toBeInTheDocument();
  });

  it("calls onConfirm when confirmed", async () => {
    const onConfirm = vi.fn();
    const user = userEvent.setup();
    render(
      <UnmergeModal
        mid="M-000005"
        patid="P2"
        patientName="Jane Doe"
        pending={false}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /confirm unmerge/i }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("disables both buttons while pending", () => {
    render(
      <UnmergeModal
        mid="M-000005"
        patid="P2"
        patientName="Jane Doe"
        pending={true}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: /unmerging/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /cancel/i })).toBeDisabled();
  });
});
