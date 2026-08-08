import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MergeModal } from "./MergeModal";

/** The confirmation step before a permanent, audited merge — this is the
 * one thing standing between a reviewer's click and two patient records
 * being combined, so it has to (a) actually name the records being merged,
 * not just show a generic "are you sure," and (b) never let a second click
 * fire a second request while the first is still pending. */

describe("MergeModal", () => {
  it("names the target mid and every patid being merged", () => {
    render(
      <MergeModal
        targetMid="M-000005"
        patids={["P2", "P3"]}
        pending={false}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText("M-000005")).toBeInTheDocument();
    expect(screen.getByText("P2")).toBeInTheDocument();
    expect(screen.getByText("P3")).toBeInTheDocument();
  });

  it("calls onConfirm when the confirm button is clicked", async () => {
    const onConfirm = vi.fn();
    const user = userEvent.setup();
    render(
      <MergeModal
        targetMid="M-000005"
        patids={["P2"]}
        pending={false}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /confirm merge/i }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("calls onCancel when the cancel button is clicked", async () => {
    const onCancel = vi.fn();
    const user = userEvent.setup();
    render(
      <MergeModal
        targetMid="M-000005"
        patids={["P2"]}
        pending={false}
        onConfirm={vi.fn()}
        onCancel={onCancel}
      />,
    );

    await user.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("disables both buttons while a merge is pending, so a second click can't fire", () => {
    render(
      <MergeModal
        targetMid="M-000005"
        patids={["P2"]}
        pending={true}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: /merging/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /cancel/i })).toBeDisabled();
  });
});
