import type { DashboardSummary } from "@/lib/schemas";

/** FR-15/16/17: model/run version + the exact confidence thresholds used to
 * classify records — pulled live from the deterministic rule definitions
 * (src/models/deterministic_rules.py RULES), not hardcoded. */
export function ModelInfoPanel({ summary }: { summary: DashboardSummary }) {
  return (
    <div className="card">
      <div className="px-5 pt-4 pb-1">
        <h4 className="text-[15px] font-semibold text-ink-2">Model info</h4>
      </div>
      <div className="px-5 pb-4">
        <Row k="Last run ID" v={summary.last_run_id ?? "—"} mono />
        <Row
          k="Run completed"
          v={
            summary.last_run_created_utc
              ? new Date(summary.last_run_created_utc).toLocaleString()
              : "—"
          }
        />
        <Row
          k="Model / git version"
          v={summary.model_version ? summary.model_version.slice(0, 12) : "—"}
          mono
        />
        <div className="pt-3">
          <div className="mb-2 text-[11px] font-bold tracking-wide text-gray uppercase">
            Match confidence thresholds
          </div>
          {Object.entries(summary.confidence_thresholds).map(([rule, conf]) => (
            <Row key={rule} k={rule} v={conf.toFixed(3)} mono compact />
          ))}
        </div>
      </div>
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
  v: string;
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
