"use client";

import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/** FR-11/12/13: Auto-match / Needs review / No match bar chart, fixed
 * green/yellow/red color coding. */
export function MatchStatusChart({
  autoMatch,
  needsReview,
  noMatch,
}: {
  autoMatch: number;
  needsReview: number;
  noMatch: number;
}) {
  const data = [
    { name: "Auto-match", value: autoMatch, color: "var(--status-auto)" },
    { name: "Needs review", value: needsReview, color: "var(--status-review)" },
    { name: "No match", value: noMatch, color: "var(--status-nomatch)" },
  ];

  return (
    <div className="card chart-card p-5">
      <h4 className="text-[15px] font-semibold text-ink-2">
        Match status distribution
      </h4>
      <p className="mt-1 mb-2 text-xs text-gray">
        Live counts across the current resolved dataset.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 20, right: 8, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="name"
            tickLine={false}
            axisLine={{ stroke: "var(--line)" }}
            tick={{ fontSize: 12, fill: "var(--gray-2)", fontWeight: 600 }}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            width={36}
            tick={{ fontSize: 10, fill: "var(--gray)" }}
          />
          <Tooltip
            cursor={{ fill: "rgba(0,0,0,0.03)" }}
            contentStyle={{
              borderRadius: 8,
              borderColor: "var(--line)",
              fontSize: 12,
            }}
          />
          <Bar dataKey="value" radius={[8, 8, 0, 0]} maxBarSize={90}>
            {data.map((entry) => (
              <Cell key={entry.name} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-3 flex flex-wrap gap-4">
        <Legend color="var(--status-auto)" label="Auto-match" />
        <Legend color="var(--status-review)" label="Needs review" />
        <Legend color="var(--status-nomatch)" label="No match" />
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-xs font-semibold text-gray-2">
      <i
        className="inline-block h-2.5 w-2.5 rounded-sm"
        style={{ background: color }}
      />
      {label}
    </span>
  );
}
