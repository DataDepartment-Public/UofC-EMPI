"use client";

import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PairExplanation } from "@/lib/schemas";

interface Props {
  explanation: PairExplanation;
  /** How many of the (already |shap|-descending sorted) features to show —
   * the backend's `top_n` is only a suggestion, never a truncation, so the
   * frontend does the actual slicing. */
  maxFeatures?: number;
}

/** A per-pair SHAP waterfall. Each bar's `start`/`end` (a cumulative
 * log-odds range) is precomputed server-side (`src/models/explanations.py`),
 * so this component does no arithmetic of its own beyond slicing to
 * `maxFeatures` — it just maps fields to a recharts floating/range bar. */
export function ShapWaterfall({ explanation, maxFeatures = 8 }: Props) {
  const shown = explanation.features.slice(0, maxFeatures);
  const data = shown.map((f) => ({
    label: f.label,
    range: [f.start, f.end] as [number, number],
    direction: f.direction,
    shap: f.shap,
    displayValue:
      f.display_value ?? (f.value != null ? String(f.value) : "—"),
  }));

  return (
    <div>
      <ResponsiveContainer
        width="100%"
        height={Math.max(160, shown.length * 36)}
      >
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
        >
          <XAxis
            type="number"
            domain={[explanation.axis.min, explanation.axis.max]}
            tick={{ fontSize: 10, fill: "var(--gray)" }}
            tickFormatter={(v: number) => v.toFixed(1)}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={150}
            tick={{ fontSize: 11, fill: "var(--ink-2)" }}
            tickLine={false}
          />
          <ReferenceLine
            x={explanation.base_value}
            stroke="var(--gray)"
            strokeDasharray="3 3"
          />
          <Tooltip
            contentStyle={{
              borderRadius: 8,
              borderColor: "var(--line)",
              fontSize: 12,
            }}
            formatter={(_value, _name, item) => {
              const p = item.payload as (typeof data)[number] | undefined;
              if (!p) return ["", ""];
              return [
                `${p.shap >= 0 ? "+" : ""}${p.shap.toFixed(3)} (value: ${p.displayValue})`,
                "Contribution",
              ];
            }}
          />
          <Bar dataKey="range" radius={2}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={
                  d.direction === "positive"
                    ? "var(--status-auto)"
                    : "var(--status-nomatch-display)"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-2 flex flex-wrap gap-4 text-xs text-gray-2">
        <span className="flex items-center gap-1.5 font-semibold">
          <i className="inline-block h-2.5 w-2.5 rounded-sm bg-status-auto" />
          Pushes toward match
        </span>
        <span className="flex items-center gap-1.5 font-semibold">
          <i className="inline-block h-2.5 w-2.5 rounded-sm bg-status-nomatch-display" />
          Pushes toward non-match
        </span>
      </div>
    </div>
  );
}
