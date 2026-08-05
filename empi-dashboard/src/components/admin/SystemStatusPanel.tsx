"use client";

import { useDashboardSummary, useHealth } from "@/lib/hooks";
import { formatDate } from "@/lib/format";

type StatusKey = "ok" | "not_ready" | "unknown" | "checking";

const STATUS_STYLES: Record<StatusKey, { label: string; bg: string; fg: string }> = {
  ok: { label: "Healthy", bg: "#e9f6e9", fg: "#3f8a3f" },
  not_ready: { label: "Degraded", bg: "#fdeaea", fg: "#b3372c" },
  unknown: { label: "Unreachable", bg: "#f2f2f2", fg: "#7a7a7a" },
  checking: { label: "Checking…", bg: "#f2f2f2", fg: "#7a7a7a" },
};

/** Admin-tab operator summary: which model/run is live, when data last
 * refreshed, and whether the API backing it is actually reachable — pulled
 * from the same dashboard-summary + /health/ready endpoints the rest of the
 * app already polls, not duplicated state. */
export function SystemStatusPanel() {
  const { data: summary, isLoading: summaryLoading } = useDashboardSummary();
  const {
    data: health,
    isLoading: healthLoading,
    isFetching: healthFetching,
    isError: healthError,
    refetch,
  } = useHealth();

  const statusKey: StatusKey = healthLoading
    ? "checking"
    : healthError
      ? "unknown"
      : health?.status === "ok"
        ? "ok"
        : "not_ready";
  const style = STATUS_STYLES[statusKey];

  return (
    <div className="card w-full max-w-md p-5">
      <div className="mb-1 flex items-center justify-between">
        <h4 className="text-[15px] font-semibold text-ink-2">System status</h4>
        <button
          onClick={() => refetch()}
          disabled={healthFetching}
          className="text-[11px] font-bold text-brand-blue hover:opacity-80 disabled:opacity-50"
        >
          {healthFetching ? "Checking…" : "Recheck"}
        </button>
      </div>

      <Row
        k="API health"
        v={
          <span
            className="inline-block rounded-full px-2.5 py-0.5 text-[11px] font-bold"
            style={{ background: style.bg, color: style.fg }}
          >
            {style.label}
          </span>
        }
      />
      {health?.checks && (
        <>
          <Row k="Database" v={health.checks.db ? "OK" : "Failing"} compact />
          <Row
            k="Data directories"
            v={health.checks.data_dirs ? "OK" : "Failing"}
            compact
          />
        </>
      )}

      <Row
        k="Model version"
        v={
          summaryLoading
            ? "…"
            : summary?.model_version
              ? summary.model_version.slice(0, 12)
              : "—"
        }
        mono
      />
      <Row
        k="Last data refresh"
        v={summaryLoading ? "…" : formatDate(summary?.last_run_created_utc)}
      />
    </div>
  );
}

function Row({
  k,
  v,
  mono = false,
  compact = false,
}: {
  k: string;
  v: React.ReactNode;
  mono?: boolean;
  compact?: boolean;
}) {
  return (
    <div
      className={`flex items-center justify-between border-b border-line text-[13px] last:border-none ${
        compact ? "py-1.5" : "py-2.5"
      }`}
    >
      <span className="font-semibold text-gray">{k}</span>
      <span className={`font-bold text-ink-2 ${mono ? "font-mono text-xs" : ""}`}>
        {v}
      </span>
    </div>
  );
}
