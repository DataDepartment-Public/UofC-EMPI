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
import { toProbabilityWaterfall } from "@/lib/explain";
import type { WaterfallBar } from "@/lib/explain";
import type { PairExplanation } from "@/lib/schemas";

/** How far the arrow head juts past the body, in px. */
const TIP_PX = 7;
/** Below this pixel width there's no room for a head, so the bar degrades to
 * a plain sliver rather than collapsing into a pure triangle. */
const MIN_ARROW_PX = 4;

interface ArrowBarProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  fill?: string;
  payload?: WaterfallBar;
}

/** A waterfall bar drawn as an arrow: a pointed head at the leading edge
 * showing which way the confidence moved, and a matching notch cut into the
 * trailing edge so consecutive bars read as one flowing chain — the shape
 * the reference SHAP waterfall uses.
 *
 * The head points along the bar's *direction of travel*, which is the sign
 * of `deltaPct`, not the on-screen left-to-right order: a bar that lowers
 * confidence runs right-to-left and must point left. Recharts hands us an
 * already-normalized `x`/`width` (always left edge + positive width), so the
 * direction has to come from the payload.
 *
 * Geometry is clamped so a hairline bar can't invert: the head and notch
 * each take at most a quarter of the width. */
export function arrowBarPath({
  x,
  y,
  width,
  height,
  pointsRight,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  pointsRight: boolean;
}): string | null {
  const w = Math.abs(width);
  // Too narrow to shape — the caller falls back to a plain sliver so a
  // near-zero contribution is still visible as *something*.
  if (w < MIN_ARROW_PX) return null;

  const tip = Math.min(TIP_PX, w / 2);
  const midY = y + height / 2;
  const left = x;
  const right = x + w;
  const bottom = y + height;

  // Head on the leading edge only; the trailing edge stays flat. A notch cut
  // into the tail would pull the bar's visible start inward by `tip`, opening
  // a gap where it should butt up against the previous bar's tip — the chain
  // has to stay unbroken, since each bar begins exactly where the last ended.
  return pointsRight
    ? [
        `M ${left} ${y}`,
        `L ${right - tip} ${y}`,
        `L ${right} ${midY}`, // head
        `L ${right - tip} ${bottom}`,
        `L ${left} ${bottom}`,
        "Z",
      ].join(" ")
    : [
        `M ${right} ${y}`,
        `L ${left + tip} ${y}`,
        `L ${left} ${midY}`, // head
        `L ${left + tip} ${bottom}`,
        `L ${right} ${bottom}`,
        "Z",
      ].join(" ");
}

function ArrowBar({ x = 0, y = 0, width = 0, height = 0, fill, payload }: ArrowBarProps) {
  const w = Math.abs(width);
  const d = arrowBarPath({
    x,
    y,
    width: w,
    height,
    pointsRight: (payload?.deltaPct ?? 0) >= 0,
  });

  if (d === null) {
    return <rect x={x} y={y} width={Math.max(w, 1)} height={height} fill={fill} />;
  }
  return <path d={d} fill={fill} />;
}

interface Props {
  explanation: PairExplanation;
  /** How many features get their own bar — the backend's `top_n` is a
   * suggestion, never a truncation, so the frontend does the slicing. The
   * rest are collapsed into one "N other features" bar, not dropped. */
  maxFeatures?: number;
}

/** A per-pair feature-contribution waterfall drawn on a **confidence axis**:
 * the x-axis is the model's probability from 0-100%, so a reviewer reads
 * "this pair started at 12% and the matching birth date took it to 78%"
 * rather than a log-odds margin that means nothing at a glance.
 *
 * The geometry (including why a bar's width can't be converted on its own)
 * lives in `toProbabilityWaterfall`. */
export function ShapWaterfall({ explanation, maxFeatures = 8 }: Props) {
  const { bars, basePct, domain } = toProbabilityWaterfall(
    explanation,
    maxFeatures,
  );

  // Axis ticks: a near-certain pair can live inside a fraction of a percent,
  // where "78%" on every tick would read as one flat number; a wide span
  // doesn't need decimals at all. This is about tick spacing, so it keys off
  // the span.
  const span = domain[1] - domain[0];
  const tickDecimals = span < 1 ? 2 : span < 10 ? 1 : 0;
  const asTick = (v: number) => `${v.toFixed(tickDecimals)}%`;

  // A single confidence value keys off its own magnitude instead, so the
  // same number reads the same on every chart. Sharing the axis rule made
  // the base rate render as "0%" on a wide chart and "0.11%" on a zoomed
  // one — one model constant looking like a per-pair quantity.
  const asConfidence = (v: number) =>
    `${v.toFixed(v < 1 ? 2 : v < 10 ? 1 : 0)}%`;

  return (
    <div>
      <ResponsiveContainer width="100%" height={Math.max(160, bars.length * 36)}>
        <BarChart
          data={bars}
          layout="vertical"
          margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
        >
          <XAxis
            type="number"
            domain={domain}
            tick={{ fontSize: 10, fill: "var(--gray)" }}
            tickFormatter={asTick}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={150}
            tick={{ fontSize: 11, fill: "var(--ink-2)" }}
            tickLine={false}
          />
          <ReferenceLine x={basePct} stroke="var(--gray)" strokeDasharray="3 3" />
          <Tooltip
            contentStyle={{
              borderRadius: 8,
              borderColor: "var(--line)",
              fontSize: 12,
            }}
            formatter={(_value, _name, item) => {
              const p = item.payload as (typeof bars)[number] | undefined;
              if (!p) return ["", ""];
              const delta = `${p.deltaPct >= 0 ? "+" : ""}${p.deltaPct.toFixed(
                Math.abs(p.deltaPct) < 1 ? 2 : 1,
              )} pts`;
              return [
                p.isRemainder ? delta : `${delta} (value: ${p.displayValue})`,
                p.isRemainder ? "Combined" : "Confidence change",
              ];
            }}
          />
          <Bar dataKey="range" shape={<ArrowBar />}>
            {/* The remainder bar is coloured like any other: it moves the
                confidence the same way, and greying it out read as a
                separate kind of thing rather than "the rest of the
                features, net". Its label already names it. */}
            {bars.map((d, i) => (
              <Cell
                key={i}
                fill={
                  d.direction === "positive"
                    ? "var(--status-auto)"
                    : "var(--status-nomatch)"
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="mt-2 flex flex-wrap gap-4 text-xs text-gray-2">
        <span className="flex items-center gap-1.5 font-semibold">
          <i className="inline-block h-2.5 w-2.5 rounded-sm bg-status-auto" />
          Raises confidence
        </span>
        <span className="flex items-center gap-1.5 font-semibold">
          <i className="inline-block h-2.5 w-2.5 rounded-sm bg-status-nomatch" />
          Lowers confidence
        </span>
        <span className="flex items-center gap-1.5">
          Starting confidence {asConfidence(basePct)} (dashed)
        </span>
      </div>
    </div>
  );
}
