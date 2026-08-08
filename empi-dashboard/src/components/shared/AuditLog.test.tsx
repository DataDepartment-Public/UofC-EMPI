import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuditLog } from "./AuditLog";
import { ApiError, api } from "@/lib/api-client";
import type { AuditLogRow } from "@/lib/schemas";

/**
 * The undo button's enable/disable logic is the exact behavior a real bug
 * hit in production: `undone` wasn't being threaded through the backend's
 * `insert_audit_log` on every backend, so an already-undone row kept
 * showing an active "Undo" button. These tests pin that a row's `undone`
 * flag (and its action type) is what the button responds to — not, e.g.,
 * whether a *different* row was just undone.
 */

vi.mock("@/lib/api-client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api-client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      listAudit: vi.fn(),
      undoAudit: vi.fn(),
    },
  };
});

const mockedApi = vi.mocked(api, true);

function row(overrides: Partial<AuditLogRow>): AuditLogRow {
  return {
    id: 1,
    ts_utc: "2026-08-05T12:00:00Z",
    user: "reviewer.jclark",
    action: "merge",
    patids: "P1,P2",
    mid: "M-000001",
    prev_state: "Needs review",
    next_state: "Merged",
    run_id: "20260805T000000Z-abc123",
    undone: false,
    ...overrides,
  };
}

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("AuditLog undo button", () => {
  it("shows a working Undo button for an undoable, not-yet-undone row", async () => {
    mockedApi.listAudit.mockResolvedValue([row({ id: 1, action: "merge", undone: false })]);
    renderWithClient(<AuditLog onFlash={vi.fn()} />);

    const button = await screen.findByRole("button", { name: /^undo$/i });
    expect(button).toBeEnabled();
    expect(screen.queryByText(/undone/i)).not.toBeInTheDocument();
  });

  it("shows 'Undone' with no button for a row that's already been undone", async () => {
    mockedApi.listAudit.mockResolvedValue([row({ id: 1, action: "merge", undone: true })]);
    renderWithClient(<AuditLog onFlash={vi.fn()} />);

    await screen.findByText(/undone/i);
    expect(screen.queryByRole("button", { name: /undo/i })).not.toBeInTheDocument();
  });

  it("shows no undo control for a non-undoable action (dismiss), even if undone is false", async () => {
    mockedApi.listAudit.mockResolvedValue([row({ id: 1, action: "dismiss", undone: false })]);
    renderWithClient(<AuditLog onFlash={vi.fn()} />);

    await screen.findByText("Dismissed");
    expect(screen.queryByRole("button", { name: /undo/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/^undone$/i)).not.toBeInTheDocument();
  });

  it("only disables the clicked row's button while its own undo is pending, not other rows'", async () => {
    mockedApi.listAudit.mockResolvedValue([
      row({ id: 1, action: "merge", undone: false }),
      row({ id: 2, action: "unmerge", undone: false, patids: "P3" }),
    ]);
    let resolveUndo: (v: Awaited<ReturnType<typeof api.undoAudit>>) => void = () => {};
    mockedApi.undoAudit.mockImplementation(
      () => new Promise((resolve) => (resolveUndo = resolve)),
    );
    const user = userEvent.setup();
    renderWithClient(<AuditLog onFlash={vi.fn()} />);

    const buttons = await screen.findAllByRole("button", { name: /^undo$/i });
    expect(buttons).toHaveLength(2);
    await user.click(buttons[0]);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /undoing/i })).toBeInTheDocument();
    });
    // Row 2's button is untouched by row 1's in-flight undo.
    expect(screen.getByRole("button", { name: /^undo$/i })).toBeEnabled();

    resolveUndo({ audit_id: 1, reversed_action: "merge", new_mids: [] });
  });

  it("calls undoAudit with the row's id and flashes a success message", async () => {
    mockedApi.listAudit.mockResolvedValue([row({ id: 42, action: "merge", undone: false })]);
    mockedApi.undoAudit.mockResolvedValue({
      audit_id: 42,
      reversed_action: "merge",
      new_mids: ["M-000010"],
    });
    const onFlash = vi.fn();
    const user = userEvent.setup();
    renderWithClient(<AuditLog onFlash={onFlash} />);

    await user.click(await screen.findByRole("button", { name: /^undo$/i }));

    await waitFor(() => expect(mockedApi.undoAudit).toHaveBeenCalledWith(42));
    await waitFor(() =>
      expect(onFlash).toHaveBeenCalledWith(expect.stringContaining("#42")),
    );
  });

  it("flashes the backend's error message when undo is rejected (e.g. already undone)", async () => {
    mockedApi.listAudit.mockResolvedValue([row({ id: 7, action: "merge", undone: false })]);
    mockedApi.undoAudit.mockRejectedValue(
      new ApiError(400, "Audit entry #7 has already been undone."),
    );
    const onFlash = vi.fn();
    const user = userEvent.setup();
    renderWithClient(<AuditLog onFlash={onFlash} />);

    await user.click(await screen.findByRole("button", { name: /^undo$/i }));

    await waitFor(() =>
      expect(onFlash).toHaveBeenCalledWith("Audit entry #7 has already been undone."),
    );
  });
});
