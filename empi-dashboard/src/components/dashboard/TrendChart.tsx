"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DashboardSummary } from "@/lib/schemas";

/** FR-14/15: auto-match-rate / review-rate trend across past runs.
 * docs/Dashboard-Guide.md's own "Open items" note: this is deliberately
 * limited to label-free metrics (auto-match rate, review rate) — no
 * precision/recall, since there's no ground-truth to compute them against.
 * Points are real, chronological pipeline runs (not synthesized monthly
 * buckets) — the x-axis label says so. */
export function TrendChart({
  history,
}: {
  history: DashboardSummary["history"];
}) {
  const data = history.map((h, i) => ({
    label: `Run ${i + 1}`,
    date: h.created_utc.slice(0, 10),
    "Auto-match rate": h.auto_match_rate,
    "Review rate": h.review_rate,
  }));

  return (
    <div className="card p-5">
      <h4 className="text-[15px] font-semibold text-ink-2">
        Performance over time
      </h4>
      <p className="mt-1 mb-2 text-xs text-gray">
        Auto-match rate and review rate across the last {data.length} pipeline
        run{data.length === 1 ? "" : "s"} (label-free metrics only — no
        ground-truth precision/recall available).
      </p>
      {data.length < 2 ? (
        <div className="flex h-[220px] items-center justify-center text-sm text-gray">
          Need at least two runs to chart a trend.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis
              dataKey="label"
              tickLine={false}
              axisLine={{ stroke: "var(--line)" }}
              tick={{ fontSize: 10, fill: "var(--gray)" }}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={36}
              unit="%"
              tick={{ fontSize: 10, fill: "var(--gray)" }}
            />
            <Tooltip
              contentStyle={{
                borderRadius: 8,
                borderColor: "var(--line)",
                fontSize: 12,
              }}
              labelFormatter={(label, payload) =>
                `${label} — ${payload?.[0]?.payload?.date ?? ""}`
              }
            />
            <Line
              type="monotone"
              dataKey="Auto-match rate"
              stroke="var(--status-auto)"
              strokeWidth={2.5}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="Review rate"
              stroke="var(--brand-teal)"
              strokeWidth={2.5}
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
      <div className="mt-3 flex flex-wrap gap-4">
        <span className="flex items-center gap-1.5 text-xs font-semibold text-gray-2">
          <i className="inline-block h-2.5 w-2.5 rounded-sm bg-status-auto" />
          Auto-match rate
        </span>
        <span className="flex items-center gap-1.5 text-xs font-semibold text-gray-2">
          <i className="inline-block h-2.5 w-2.5 rounded-sm bg-brand-teal" />
          Review rate
        </span>
      </div>
    </div>
  );
}
