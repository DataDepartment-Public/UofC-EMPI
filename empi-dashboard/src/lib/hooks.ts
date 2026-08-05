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

export function useCluster(mid: string | null) {
  return useQuery({
    queryKey: ["cluster", mid],
    queryFn: () => api.getCluster(mid as string),
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

/** The ML matcher only ever scores a pair the gate let through, so a gate
 * drop has no `ml_matcher` explanation — try it first and fall back to the
 * gate's own explanation (the only record of *why* it was dropped). `null`
 * (neither model scored this pair — e.g. a purely deterministic-rule match)
 * is a normal, successful result, not a query error. */
export function usePairExplanation(patidA: string | null, patidB: string | null) {
  return useQuery({
    queryKey: ["explanation", patidA, patidB],
    queryFn: async () => {
      const ml = await api.getExplanation("ml_matcher", patidA as string, patidB as string);
      if (ml) return ml;
      return api.getExplanation("nonmatch_gate", patidA as string, patidB as string);
    },
    enabled: patidA !== null && patidB !== null,
  });
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
