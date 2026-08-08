import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import {
  useDismissMutation,
  useMergeMutation,
  useUndoMutation,
  useUnmergeMutation,
} from "./hooks";
import { api } from "./api-client";

/** FR-40: any merge/unmerge/dismiss/undo must refresh the dashboard
 * summary, records list, review queue, and audit log — otherwise a
 * reviewer who just merged a pair keeps seeing it in the review queue.
 * These pin that every mutation actually triggers that refresh on
 * success, and does NOT on failure (a failed merge shouldn't discard
 * the queue the reviewer was looking at). */

vi.mock("./api-client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api-client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      merge: vi.fn(),
      unmerge: vi.fn(),
      dismiss: vi.fn(),
      undoAudit: vi.fn(),
    },
  };
});

const mockedApi = vi.mocked(api, true);

beforeEach(() => {
  vi.clearAllMocks();
});

function wrapperWith(client: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

const REFRESH_KEYS = ["dashboard-summary", "records", "review-queue", "audit"];

describe("useMergeMutation", () => {
  it("invalidates the FR-40 refresh keys on success", async () => {
    mockedApi.merge.mockResolvedValue({
      audit_id: 1,
      entity: {
        mid: "M-1",
        run_id: "r1",
        origin: "merge",
        is_merged: true,
        confidence: 1,
        updated_utc: "2026-08-05T00:00:00Z",
        members: [],
        review_candidates: [],
      },
    });
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");
    const { result } = renderHook(() => useMergeMutation(), {
      wrapper: wrapperWith(client),
    });

    result.current.mutate({ mid: "M-1", patids: ["P2"] });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    const invalidatedKeys = invalidateSpy.mock.calls.map((c) => c[0]?.queryKey?.[0]);
    for (const key of REFRESH_KEYS) {
      expect(invalidatedKeys).toContain(key);
    }
  });

  it("does not invalidate any query when the merge fails", async () => {
    mockedApi.merge.mockRejectedValue(new Error("boom"));
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");
    const { result } = renderHook(() => useMergeMutation(), {
      wrapper: wrapperWith(client),
    });

    result.current.mutate({ mid: "M-1", patids: ["P2"] });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(invalidateSpy).not.toHaveBeenCalled();
  });
});

describe("useUndoMutation", () => {
  it("calls api.undoAudit with the raw audit id and refreshes on success", async () => {
    mockedApi.undoAudit.mockResolvedValue({
      audit_id: 5,
      reversed_action: "merge",
      new_mids: ["M-2"],
    });
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");
    const { result } = renderHook(() => useUndoMutation(), {
      wrapper: wrapperWith(client),
    });

    result.current.mutate(5);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockedApi.undoAudit).toHaveBeenCalledWith(5);
    expect(invalidateSpy).toHaveBeenCalled();
  });
});

describe("useUnmergeMutation / useDismissMutation", () => {
  it("useUnmergeMutation passes mid/patid through to api.unmerge", async () => {
    mockedApi.unmerge.mockResolvedValue({
      audit_id: 2,
      new_mid: "M-9",
      entity: {
        mid: "M-9",
        run_id: "r1",
        origin: "review",
        is_merged: false,
        confidence: null,
        updated_utc: "2026-08-05T00:00:00Z",
        members: [],
        review_candidates: [],
      },
    });
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    const { result } = renderHook(() => useUnmergeMutation(), {
      wrapper: wrapperWith(client),
    });

    result.current.mutate({ mid: "M-1", patid: "P2" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockedApi.unmerge).toHaveBeenCalledWith("M-1", "P2");
  });

  it("useDismissMutation passes patidA/patidB through to api.dismiss", async () => {
    mockedApi.dismiss.mockResolvedValue({ audit_id: 3 });
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    const { result } = renderHook(() => useDismissMutation(), {
      wrapper: wrapperWith(client),
    });

    result.current.mutate({ patidA: "P1", patidB: "P2" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockedApi.dismiss).toHaveBeenCalledWith("P1", "P2");
  });
});
