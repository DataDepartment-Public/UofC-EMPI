import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./api-client";

/** These wrap the exact calls that mutate patient identity (merge/unmerge/
 * dismiss) and reverse them (undo) — the request shape (method, path, body)
 * has to match what the backend's Pydantic models expect exactly, and a
 * non-2xx response has to surface as a catchable ApiError with the
 * backend's actual detail message, not a generic parse crash, since the UI
 * shows that message directly to the reviewer (see review/page.tsx's
 * flash()). */

function jsonResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: async () => body,
  } as Response;
}

const baseEntity = {
  mid: "M-000001",
  run_id: "20260805T000000Z-abc123",
  origin: "merge",
  is_merged: true,
  confidence: 0.98,
  updated_utc: "2026-08-05T12:00:00Z",
  members: [],
  review_candidates: [],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api.merge", () => {
  it("POSTs mid/patids as JSON to /api/audit/merge", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ audit_id: 5, entity: baseEntity }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.merge("M-000001", ["P2", "P3"]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/api/audit/merge");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ mid: "M-000001", patids: ["P2", "P3"] });
    expect(result.audit_id).toBe(5);
  });

  it("throws ApiError with the backend's detail on failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "Unknown mid: X" }, 404)),
    );

    await expect(api.merge("X", ["P2"])).rejects.toMatchObject({
      status: 404,
      message: "Unknown mid: X",
    });
  });

  it("throws ApiError even when the error body is unparseable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => {
        throw new Error("not json");
      },
    } as unknown as Response));

    await expect(api.merge("M-1", ["P2"])).rejects.toBeInstanceOf(ApiError);
  });
});

describe("api.unmerge", () => {
  it("POSTs mid/patid as JSON to /api/audit/unmerge", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ audit_id: 6, new_mid: "M-000099", entity: baseEntity }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.unmerge("M-000001", "P2");

    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/api/audit/unmerge");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ mid: "M-000001", patid: "P2" });
  });
});

describe("api.dismiss", () => {
  it("POSTs patid_a/patid_b as JSON to /api/audit/dismiss", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ audit_id: 7 }));
    vi.stubGlobal("fetch", fetchMock);

    await api.dismiss("P1", "P2");

    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/api/audit/dismiss");
    expect(JSON.parse(init.body)).toEqual({ patid_a: "P1", patid_b: "P2" });
  });
});

describe("api.undoAudit", () => {
  it("POSTs to /api/audit/{id}/undo with no body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ audit_id: 5, reversed_action: "merge", new_mids: ["M-2"] }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.undoAudit(5);

    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/api/audit/5/undo");
    expect(init.method).toBe("POST");
    expect(result.reversed_action).toBe("merge");
  });

  it("rejects when the response fails the UndoResponse schema", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ audit_id: 5, reversed_action: "bogus" })),
    );

    await expect(api.undoAudit(5)).rejects.toThrow();
  });

  it("surfaces an already-undone 400 as ApiError with the backend detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: "Audit entry #5 has already been undone." }, 400),
      ),
    );

    await expect(api.undoAudit(5)).rejects.toMatchObject({
      status: 400,
      message: "Audit entry #5 has already been undone.",
    });
  });
});
