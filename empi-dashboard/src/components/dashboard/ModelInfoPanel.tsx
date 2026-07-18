import type { DashboardSummary } from "@/lib/schemas";

/** FR-15/16/17: model/run version + the exact confidence thresholds used to
 * classify records — pulled live from the deterministic rule definitions
 * (src/models/deterministic_rules.py RULES), not hardcoded.
 *
 * Kept to a glance-level summary up top; the raw run ID, full git SHA, and
 * per-rule threshold table are real data reviewers rarely need at a glance —
 * they live behind a "Technical details" disclosure instead of always-on. */
export function ModelInfoPanel({ summary }: { summary: DashboardSummary }) {
  const thresholds = Object.entries(summary.confidence_thresholds);
  const thresholdRange =
    thresholds.length > 0
      ? [...thresholds.map(([, c]) => c)].sort((a, b) => a - b)
      : null;

  return (
    <div className="card">
      <div className="px-5 pt-4 pb-1">
        <h4 className="text-[15px] font-semibold text-ink-2">Model info</h4>
      </div>
      <div className="px-5 pb-4">
        <Row
          k="Run completed"
          v={
            summary.last_run_created_utc
              ? new Date(summary.last_run_created_utc).toLocaleString()
              : "—"
          }
        />
        <Row
          k="Model version"
          v={summary.model_version ? summary.model_version.slice(0, 12) : "—"}
          mono
        />
        <Row
          k="Rules active"
          v={
            thresholdRange
              ? `${thresholds.length} (confidence ${thresholdRange[0].toFixed(2)}–${thresholdRange[thresholdRange.length - 1].toFixed(2)})`
              : "—"
          }
        />

        <details className="group mt-3">
          <summary className="cursor-pointer list-none text-[11px] font-bold tracking-wide text-brand-blue uppercase">
            Technical details
          </summary>
          <div className="mt-2 border-t border-line pt-2">
            <Row k="Last run ID" v={summary.last_run_id ?? "—"} mono compact />
            <Row
              k="Model / git SHA"
              v={summary.model_version ?? "—"}
              mono
              compact
            />
            <div className="pt-2">
              <div className="mb-1.5 text-[10px] font-bold tracking-wide text-gray uppercase">
                Match confidence thresholds
              </div>
              {thresholds.map(([rule, conf]) => (
                <Row key={rule} k={rule} v={conf.toFixed(3)} mono compact />
              ))}
            </div>
          </div>
        </details>
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
