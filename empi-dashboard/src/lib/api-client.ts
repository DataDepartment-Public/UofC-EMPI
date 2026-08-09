/**
 * Browser-side fetch wrapper. Every call hits this app's own `/api/*` Route
 * Handlers (the BFF) — never FastAPI directly — and every response is
 * validated against the zod schemas in `lib/schemas.ts` so a backend
 * contract drift surfaces immediately instead of as a silent UI bug.
 */
import { z } from "zod";
import {
  AuditLogRowSchema,
  CleanSsn,
  CleanSsnSchema,
  ClusterPairsResponse,
  ClusterPairsResponseSchema,
  DashboardSummary,
  DashboardSummarySchema,
  DismissResponseSchema,
  Entity,
  EntitySchema,
  MergeResponseSchema,
  PairExplanation,
  PairExplanationSchema,
  RawRecord,
  RawRecordSchema,
  ReadyResponse,
  ReadyResponseSchema,
  RecordsPage,
  RecordsPageSchema,
  ReviewBucket,
  ReviewQueuePage,
  ReviewQueuePageSchema,
  RunCreateResponse,
  RunCreateResponseSchema,
  RunDetail,
  RunDetailSchema,
  RunSummary,
  RunSummarySchema,
  ThresholdSettings,
  ThresholdSettingsSchema,
  UndoResponse,
  UndoResponseSchema,
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

/** Like `call`, but a "not ready" backend is a real, expected status to
 * render (see ReadyResponseSchema) rather than a thrown error — the BFF's
 * health route mirrors FastAPI's 503 for `not_ready`, so this reads the body
 * regardless of `res.ok`. */
async function callHealth(path: string): Promise<ReadyResponse> {
  const res = await fetch(path);
  const body = await res.json().catch(() => null);
  return ReadyResponseSchema.parse(body);
}

/** Like `call`, but a 404 means "not scored by this model" — a normal,
 * expected outcome (a purely deterministic-rule pair, or one the gate
 * dropped before it reached `ml_matcher`) — not an error, so it resolves to
 * `null` instead of throwing. */
async function callOr404Null<T>(
  schema: z.ZodType<T>,
  path: string,
  init?: RequestInit,
): Promise<T | null> {
  const res = await fetch(path, init);
  if (res.status === 404) return null;
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
  /** Clusters with at least this many members. The Patient Registry's
   * "something to compare" filter — deliberately not `is_merged`, which the
   * backend derives from the *unlocked* member count at publish time and so
   * misses any multi-record cluster a reviewer has already touched. */
  min_members?: number;
  sort?: "confidence" | "name" | "updated";
  page?: number;
  page_size?: number;
}

export interface ReviewQueueFilters {
  search?: string;
  confidence_min?: number;
  confidence_max?: number;
  /** Bounds on the Stage-4.25 gate's P(plausible). A separate axis from
   * `confidence_*`, which bounds the matcher side — a gate-dropped pair has
   * no matcher score at all and is reachable only here. `gate_score_max` is
   * exclusive (half-open bands); `confidence_max` is inclusive. */
  gate_score_min?: number;
  gate_score_max?: number;
  /** One of `src/api/pair_verdicts.py`'s verdicts. The only filter that
   * reaches pairs no model scored — a deterministic reject has no number for
   * any range to match. */
  verdict?: string;
  /** Which of the four sections to list. Unset returns every candidate; each
   * item carries its own `bucket` either way. */
  bucket?: ReviewBucket;
  /** Narrow to one exact pair (either order — the backend canonicalizes),
   * ignoring pagination and every other filter's meaning for that pair.
   * Used to deep-link from elsewhere in the UI (e.g. a cluster's comparison
   * history) straight to a specific candidate. Pass both or neither. */
  patid_a?: string;
  patid_b?: string;
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

  /** Every pair of a cluster's members and what the pipeline decided about
   * it. `runId` pins the trace to the run the reviewer is looking at; omitted,
   * the backend uses the run that published the entity. */
  getClusterPairs: (mid: string, runId?: string) =>
    call<ClusterPairsResponse>(
      ClusterPairsResponseSchema,
      `/api/clusters/${encodeURIComponent(mid)}/pairs${
        runId ? `?run_id=${encodeURIComponent(runId)}` : ""
      }`,
    ),

  getRaw: (patid: string) =>
    call<RawRecord>(
      RawRecordSchema,
      `/api/records/${encodeURIComponent(patid)}/raw`,
    ),

  getCleanSsn: (patid: string) =>
    call<CleanSsn>(
      CleanSsnSchema,
      `/api/records/${encodeURIComponent(patid)}/ssn-clean`,
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

  undoAudit: (auditId: number) =>
    call<UndoResponse>(
      UndoResponseSchema,
      `/api/audit/${encodeURIComponent(auditId)}/undo`,
      { method: "POST" },
    ),

  getHealth: () => callHealth("/api/health"),

  getThresholds: () =>
    call<ThresholdSettings>(ThresholdSettingsSchema, "/api/admin/thresholds"),

  updateThresholds: (values: ThresholdSettings) =>
    call<ThresholdSettings>(ThresholdSettingsSchema, "/api/admin/thresholds", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    }),

  getExplanation: (
    model: "nonmatch_gate" | "ml_matcher",
    patidA: string,
    patidB: string,
    runId?: string,
  ) =>
    callOr404Null<PairExplanation>(
      PairExplanationSchema,
      `/api/explanations/${model}/${encodeURIComponent(patidA)}/${encodeURIComponent(patidB)}${
        runId ? `?run_id=${encodeURIComponent(runId)}` : ""
      }`,
    ),
};

export { ApiError };
