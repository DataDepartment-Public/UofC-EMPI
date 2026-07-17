/**
 * Browser-side fetch wrapper. Every call hits this app's own `/api/*` Route
 * Handlers (the BFF) — never FastAPI directly — and every response is
 * validated against the zod schemas in `lib/schemas.ts` so a backend
 * contract drift surfaces immediately instead of as a silent UI bug.
 */
import { z } from "zod";
import {
  AuditLogRowSchema,
  DashboardSummary,
  DashboardSummarySchema,
  DismissResponseSchema,
  Entity,
  EntitySchema,
  MergeResponseSchema,
  RawRecord,
  RawRecordSchema,
  RecordsPage,
  RecordsPageSchema,
  ReviewQueuePage,
  ReviewQueuePageSchema,
  RunCreateResponse,
  RunCreateResponseSchema,
  RunDetail,
  RunDetailSchema,
  RunSummary,
  RunSummarySchema,
  UnmergeResponseSchema,
} from "./schemas";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(
  schema: z.ZodType<T>,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(path, init);
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText;
    throw new ApiError(res.status, detail);
  }
  return schema.parse(body);
}

export interface RecordsFilters {
  search?: string;
  origin?: string;
  is_merged?: boolean;
  birth_date?: string;
  ssn_last4?: string;
  confidence_min?: number;
  confidence_max?: number;
  page?: number;
  page_size?: number;
}

export interface ReviewQueueFilters {
  search?: string;
  confidence_min?: number;
  confidence_max?: number;
  reviewed?: boolean;
  page?: number;
  page_size?: number;
}

function toQuery<T extends object>(params: T): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  }
  const qs = search.toString();
  return qs ? `?${qs}` : "";
}

export const api = {
  dashboardSummary: () =>
    call<DashboardSummary>(DashboardSummarySchema, "/api/dashboard/summary"),

  listRecords: (filters: RecordsFilters = {}) =>
    call<RecordsPage>(RecordsPageSchema, `/api/records${toQuery(filters)}`),

  listReviewQueue: (filters: ReviewQueueFilters = {}) =>
    call<ReviewQueuePage>(
      ReviewQueuePageSchema,
      `/api/review-queue${toQuery(filters)}`,
    ),

  getCluster: (mid: string) =>
    call<Entity>(EntitySchema, `/api/clusters/${encodeURIComponent(mid)}`),

  getRaw: (patid: string) =>
    call<RawRecord>(
      RawRecordSchema,
      `/api/records/${encodeURIComponent(patid)}/raw`,
    ),

  listRuns: () => call<RunSummary[]>(z.array(RunSummarySchema), "/api/runs"),

  getRun: (runId: string) =>
    call<RunDetail>(RunDetailSchema, `/api/runs/${encodeURIComponent(runId)}`),

  createRunFromPath: (inputPath: string) => {
    const form = new FormData();
    form.set("input_path", inputPath);
    return call<RunCreateResponse>(RunCreateResponseSchema, "/api/runs", {
      method: "POST",
      body: form,
    });
  },

  createRunFromFile: (file: File) => {
    const form = new FormData();
    form.set("file", file);
    return call<RunCreateResponse>(RunCreateResponseSchema, "/api/runs", {
      method: "POST",
      body: form,
    });
  },

  listAudit: (limit = 100) =>
    call(z.array(AuditLogRowSchema), `/api/audit?limit=${limit}`),

  merge: (mid: string, patids: string[]) =>
    call(MergeResponseSchema, "/api/audit/merge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mid, patids }),
    }),

  unmerge: (mid: string, patid: string) =>
    call(UnmergeResponseSchema, "/api/audit/unmerge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mid, patid }),
    }),

  dismiss: (patidA: string, patidB: string) =>
    call(DismissResponseSchema, "/api/audit/dismiss", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patid_a: patidA, patid_b: patidB }),
    }),
};

export { ApiError };
