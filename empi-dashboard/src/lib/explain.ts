import type { PairExplanation, RecordAttrs } from "./schemas";

/**
 * The Model Explanation page (FR-31..FR-38) needs both compared records'
 * full display fields plus the deterministic rule's evidence. The Dataset
 * page already has all of this in hand (from `Entity`/`ReviewCandidate`) —
 * rather than adding a new backend "compare two arbitrary PATIDs" endpoint,
 * we pass it through the URL as one compact payload. That also means the
 * explanation page works correctly on a hard refresh or direct link, with
 * no extra fetch/loading state.
 */
export interface ExplainPatient extends RecordAttrs {
  patid: string;
}

export interface ExplainPayload {
  mid: string;
  patientA: ExplainPatient;
  patientB: ExplainPatient;
  rule: string | null;
  confidence: number | null;
  evidence: string | null;
  updated: string | null;
}

export function encodeExplainPayload(payload: ExplainPayload): string {
  return encodeURIComponent(JSON.stringify(payload));
}

export function decodeExplainPayload(raw: string | null): ExplainPayload | null {
  if (!raw) return null;
  try {
    return JSON.parse(decodeURIComponent(raw)) as ExplainPayload;
  } catch {
    return null;
  }
}

/** One bar of the waterfall, positioned on a 0–100 confidence axis. */
export interface WaterfallBar {
  label: string;
  /** [start, end] in percent — where the bar is drawn. */
  range: [number, number];
  direction: "positive" | "negative";
  /** Signed width in percentage points: `end - start`. */
  deltaPct: number;
  displayValue: string;
  /** True for the collapsed "N other features" bar. */
  isRemainder: boolean;
}

export interface ProbabilityWaterfall {
  bars: WaterfallBar[];
  /** The model's starting confidence before any feature, in percent. */
  basePct: number;
  /** The final confidence — equals `decision.score * 100`. */
  finalPct: number;
  /** Padded [min, max] for the axis, in percent, clamped to [0, 100]. */
  domain: [number, number];
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/**
 * Lay the explanation out on a confidence axis instead of the model's raw
 * log-odds margin.
 *
 * Reviewers read the x-axis as "how sure is it", and a log-odds scale
 * ("-3.4 to -0.1") answers a question nobody asked. The backend already
 * anticipates this: it ships `cumulative_prob` per step precisely so a
 * probability axis can be drawn — see `Explanations-Guide.md` §2, which also
 * warns not to reach the same place by adding `shap` values as if they were
 * probabilities. They are log-odds; summing them yields a number that is not
 * any probability. So each bar is drawn between two *converted* cumulative
 * points, never by converting a bar's width on its own.
 *
 * One consequence worth knowing: bar widths here are path-dependent. In
 * log-odds a feature's contribution is the same wherever it sits in the
 * order; after the sigmoid, an identical contribution is wide in the middle
 * of the curve and narrow out at the saturated ends. The order is the
 * backend's (|shap| descending), so the picture is stable and reproducible —
 * but a bar's width is "how much this feature moved the confidence, given
 * everything ahead of it", not a standalone property of the feature.
 *
 * Features past `maxFeatures` are collapsed into one remainder bar rather
 * than dropped, so the waterfall lands exactly on the confidence shown in
 * the header. Truncating instead would leave a chart whose bars visibly stop
 * short of the stated score, which on a percentage axis reads as a bug.
 */
export function toProbabilityWaterfall(
  explanation: PairExplanation,
  maxFeatures = 8,
): ProbabilityWaterfall {
  const basePct = sigmoid(explanation.base_value) * 100;
  const finalPct = explanation.decision.score * 100;

  const shown = explanation.features.slice(0, maxFeatures);
  const hidden = explanation.features.length - shown.length;

  const bars: WaterfallBar[] = [];
  let cursor = basePct;

  for (const f of shown) {
    const end = f.cumulative_prob * 100;
    bars.push({
      label: f.label,
      range: [cursor, end],
      direction: f.direction,
      deltaPct: end - cursor,
      displayValue: f.display_value ?? (f.value != null ? String(f.value) : "—"),
      isRemainder: false,
    });
    cursor = end;
  }

  if (hidden > 0) {
    bars.push({
      label: `${hidden} other feature${hidden === 1 ? "" : "s"}`,
      range: [cursor, finalPct],
      direction: finalPct >= cursor ? "positive" : "negative",
      deltaPct: finalPct - cursor,
      displayValue: "—",
      isRemainder: true,
    });
  }

  const points = [basePct, finalPct, ...bars.flatMap((b) => b.range)];
  const lo = Math.min(...points);
  const hi = Math.max(...points);
  // Pad so the extreme bars aren't flush against the frame. A confident pair
  // can span a fraction of a percent, so the pad is proportional with a
  // floor rather than a fixed slice of the full 0-100 range.
  const pad = Math.max(0.5, (hi - lo) * 0.08);

  return {
    bars,
    basePct,
    finalPct,
    domain: [Math.max(0, lo - pad), Math.min(100, hi + pad)],
  };
}
