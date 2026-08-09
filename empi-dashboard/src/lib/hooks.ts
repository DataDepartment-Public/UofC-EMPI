"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, RecordsFilters, ReviewQueueFilters } from "./api-client";
import type { ThresholdSettings } from "./schemas";

/** FR-40: any workflow action must refresh Dashboard KPIs, the status
 * chart, and the dataset list. Centralizing the query keys + this
 * invalidation set is what makes that "automatic" — every mutation below
 * invalidates all four. */
const REFRESH_KEYS = [
  ["dashboard-summary"],
  ["records"],
  ["review-queue"],
  ["audit"],
  // An unmerge changes which members a cluster has, so its pair trace is
  // stale the moment the mutation lands — the Patient Registry renders both
  // from the same selection.
  ["cluster-pairs"],
] as const;

export function useDashboardSummary() {
  return useQuery({
    queryKey: ["dashboard-summary"],
    queryFn: api.dashboardSummary,
    refetchInterval: 15_000,
  });
}

export function useRecords(filters: RecordsFilters) {
  return useQuery({
    queryKey: ["records", filters],
    queryFn: () => api.listRecords(filters),
  });
}

export function useReviewQueue(filters: ReviewQueueFilters) {
  return useQuery({
    queryKey: ["review-queue", filters],
    queryFn: () => api.listReviewQueue(filters),
  });
}

/** One exact review-queue candidate, regardless of the main list's current
 * filters/page — the deep-link a cluster's comparison history sends a
 * reviewer on. `data` resolves to `null` (not an error) once the query
 * settles and the pair isn't a review candidate — dismissed, merged away,
 * or from a run that's aged out. */
export function useReviewCandidate(
  patidA: string | null,
  patidB: string | null,
) {
  return useQuery({
    queryKey: ["review-queue", "pair", patidA, patidB],
    queryFn: async () => {
      const page = await api.listReviewQueue({
        patid_a: patidA as string,
        patid_b: patidB as string,
      });
      return page.items[0] ?? null;
    },
    enabled: patidA !== null && patidB !== null,
  });
}

export function useCluster(mid: string | null) {
  return useQuery({
    queryKey: ["cluster", mid],
    queryFn: () => api.getCluster(mid as string),
    enabled: mid !== null,
  });
}

/** The pairwise decision trace behind one cluster. `null` while nothing is
 * selected — same `enabled`-gating idiom as `useRawRecord` below, so callers
 * never have to conditionally call a hook. */
export function useClusterPairs(mid: string | null, runId?: string) {
  return useQuery({
    queryKey: ["cluster-pairs", mid, runId ?? null],
    queryFn: () => api.getClusterPairs(mid as string, runId),
    enabled: mid !== null,
  });
}

export function useRawRecord(patid: string | null) {
  return useQuery({
    queryKey: ["raw", patid],
    queryFn: () => api.getRaw(patid as string),
    enabled: patid !== null,
  });
}

export function useCleanSsn(patid: string | null) {
  return useQuery({
    queryKey: ["ssn-clean", patid],
    queryFn: () => api.getCleanSsn(patid as string),
    enabled: patid !== null,
  });
}

/** The ML matcher only ever scores a pair the gate let through, so a gate
 * drop has no `ml_matcher` explanation — try it first and fall back to the
 * gate's own explanation (the only record of *why* it was dropped). `null`
 * (neither model scored this pair — e.g. a purely deterministic-rule match)
 * is a normal, successful result, not a query error.
 *
 * `runId` pins the explanation to a specific run. Pass it wherever the score
 * beside the waterfall came from a known run (the Patient Registry's cluster
 * trace does); omitted, the backend falls back to the newest run that
 * explained that model, which can disagree with what's on screen. */
function useModelExplanation(
  model: "nonmatch_gate" | "ml_matcher",
  patidA: string | null,
  patidB: string | null,
  runId?: string,
) {
  return useQuery({
    queryKey: ["explanation", model, patidA, patidB, runId ?? null],
    queryFn: () =>
      api.getExplanation(model, patidA as string, patidB as string, runId),
    enabled: patidA !== null && patidB !== null,
  });
}

/** Both models' persisted explanations for one pair, fetched in parallel and
 * cached per model.
 *
 * The two are separate stages of the pipeline, not alternatives: the
 * Stage-4.25 gate scores `P(plausible)` and decides whether the Stage-4.5 ML
 * matcher ever sees the pair. Either can legitimately be absent — the gate
 * never scores a pair the deterministic rules resolved, and the matcher never
 * scores one the gate dropped — so the caller has to distinguish "didn't run"
 * from "scored", which is why this returns both rather than the first hit. */
export function usePairExplanations(
  patidA: string | null,
  patidB: string | null,
  runId?: string,
) {
  const gate = useModelExplanation("nonmatch_gate", patidA, patidB, runId);
  const ml = useModelExplanation("ml_matcher", patidA, patidB, runId);
  return { gate, ml, isLoading: gate.isLoading || ml.isLoading };
}

/** The one explanation that describes the pair's *decisive* model, for views
 * that show a single waterfall. The ML matcher wins when it ran, since it is
 * the stage that decided whether an edge formed; otherwise the gate, which is
 * then where the pair stopped. */
export function usePairExplanation(
  patidA: string | null,
  patidB: string | null,
  runId?: string,
) {
  const { gate, ml, isLoading } = usePairExplanations(patidA, patidB, runId);
  return {
    data: ml.data ?? gate.data ?? null,
    isLoading,
    isError: gate.isError || ml.isError,
  };
}

export function useRuns() {
  return useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: (query) => {
      const runs = query.state.data;
      const inFlight = runs?.some(
        (r) => r.status === "queued" || r.status === "running",
      );
      return inFlight ? 3_000 : false;
    },
  });
}

export function useAuditLog(limit = 100) {
  return useQuery({
    queryKey: ["audit", limit],
    queryFn: () => api.listAudit(limit),
  });
}

function useRefreshAll() {
  const qc = useQueryClient();
  return () =>
    REFRESH_KEYS.forEach((key) => qc.invalidateQueries({ queryKey: key }));
}

export function useMergeMutation() {
  const refresh = useRefreshAll();
  return useMutation({
    mutationFn: ({ mid, patids }: { mid: string; patids: string[] }) =>
      api.merge(mid, patids),
    onSuccess: refresh,
  });
}

export function useUnmergeMutation() {
  const refresh = useRefreshAll();
  return useMutation({
    mutationFn: ({ mid, patid }: { mid: string; patid: string }) =>
      api.unmerge(mid, patid),
    onSuccess: refresh,
  });
}

export function useDismissMutation() {
  const refresh = useRefreshAll();
  return useMutation({
    mutationFn: ({ patidA, patidB }: { patidA: string; patidB: string }) =>
      api.dismiss(patidA, patidB),
    onSuccess: refresh,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.getHealth,
    refetchInterval: 15_000,
  });
}

export function useThresholds() {
  return useQuery({
    queryKey: ["admin-thresholds"],
    queryFn: api.getThresholds,
  });
}

export function useUpdateThresholds() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (values: ThresholdSettings) => api.updateThresholds(values),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin-thresholds"] }),
  });
}

export function useUndoMutation() {
  const refresh = useRefreshAll();
  return useMutation({
    mutationFn: (auditId: number) => api.undoAudit(auditId),
    onSuccess: refresh,
  });
}

export function useCreateRun() {
  const refresh = useRefreshAll();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: { file: File } | { path: string }) =>
      "file" in input
        ? api.createRunFromFile(input.file)
        : api.createRunFromPath(input.path),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      refresh();
    },
  });
}
