import { describe, expect, it } from "vitest";
import { toProbabilityWaterfall } from "./explain";
import type { ExplanationFeature, PairExplanation } from "./schemas";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Build a payload the way the backend does: bars chain end-to-end in
 * log-odds from `base_value`, and each carries `sigmoid(end)`. */
function makeExplanation(base: number, shaps: number[]): PairExplanation {
  let cursor = base;
  const features: ExplanationFeature[] = shaps.map((shap, i) => {
    const start = cursor;
    const end = cursor + shap;
    cursor = end;
    return {
      name: `f${i}`,
      label: `Feature ${i}`,
      value: 0.5,
      display_value: "0.500",
      shap,
      start,
      end,
      direction: shap >= 0 ? "positive" : "negative",
      cumulative_prob: sigmoid(end),
    };
  });
  return {
    model: "ml_matcher",
    run_id: "r1",
    model_file: "m.pkl",
    patid_a: "A",
    patid_b: "B",
    decision: { score: sigmoid(cursor), tier: "auto_merge", threshold: 0.7 },
    base_value: base,
    final_margin: cursor,
    units: "log_odds",
    top_n: 8,
    axis: { min: 0, max: 1 },
    features,
  };
}

describe("toProbabilityWaterfall", () => {
  it("starts at the model's base rate as a percentage", () => {
    const w = toProbabilityWaterfall(makeExplanation(0, [1]));
    expect(w.basePct).toBeCloseTo(50, 6); // sigmoid(0) == 0.5
    expect(w.bars[0].range[0]).toBeCloseTo(50, 6);
  });

  it("chains bars end-to-end and lands exactly on the decision score", () => {
    const e = makeExplanation(-0.5, [1.5, -0.75, 0.25]);
    const w = toProbabilityWaterfall(e);
    for (let i = 1; i < w.bars.length; i++) {
      expect(w.bars[i].range[0]).toBeCloseTo(w.bars[i - 1].range[1], 10);
    }
    const last = w.bars[w.bars.length - 1].range[1];
    expect(last).toBeCloseTo(e.decision.score * 100, 10);
    expect(w.finalPct).toBeCloseTo(e.decision.score * 100, 10);
  });

  it("converts cumulative points, never a bar width on its own", () => {
    // The trap Explanations-Guide.md §2 names: a shap of +1.5 is not "+1.5
    // of probability" and not sigmoid(1.5) either. Its width depends on
    // where on the curve it lands.
    const w = toProbabilityWaterfall(makeExplanation(-0.5, [1.5]));
    const expected = (sigmoid(1.0) - sigmoid(-0.5)) * 100;
    expect(w.bars[0].deltaPct).toBeCloseTo(expected, 10);
    expect(w.bars[0].deltaPct).not.toBeCloseTo(sigmoid(1.5) * 100, 1);
    expect(w.bars[0].deltaPct).not.toBeCloseTo(1.5, 1);
  });

  it("gives an identical contribution a different width at the saturated end", () => {
    // Same +1.0 shap, once near the middle of the curve and once far out.
    const middle = toProbabilityWaterfall(makeExplanation(0, [1.0]));
    const saturated = toProbabilityWaterfall(makeExplanation(4, [1.0]));
    expect(middle.bars[0].deltaPct).toBeGreaterThan(
      saturated.bars[0].deltaPct * 5,
    );
  });

  it("collapses the overflow into one remainder bar that closes the gap", () => {
    const e = makeExplanation(0, [1, -0.9, 0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1]);
    const w = toProbabilityWaterfall(e, 8);
    expect(w.bars).toHaveLength(9);
    const remainder = w.bars[8];
    expect(remainder.isRemainder).toBe(true);
    expect(remainder.label).toBe("2 other features");
    expect(remainder.range[1]).toBeCloseTo(e.decision.score * 100, 10);
  });

  it("adds no remainder bar when every feature is shown", () => {
    const w = toProbabilityWaterfall(makeExplanation(0, [1, -0.5]), 8);
    expect(w.bars).toHaveLength(2);
    expect(w.bars.some((b) => b.isRemainder)).toBe(false);
  });

  it("singularizes a one-feature remainder", () => {
    const w = toProbabilityWaterfall(makeExplanation(0, [1, -0.5, 0.25]), 2);
    expect(w.bars[2].label).toBe("1 other feature");
  });

  it("keeps the axis inside 0-100 and around the data", () => {
    const w = toProbabilityWaterfall(makeExplanation(-6, [-2]));
    expect(w.domain[0]).toBeGreaterThanOrEqual(0);
    expect(w.domain[1]).toBeLessThanOrEqual(100);
    expect(w.domain[0]).toBeLessThanOrEqual(w.finalPct);
    expect(w.domain[1]).toBeGreaterThanOrEqual(w.basePct);
  });

  it("keeps a near-certain pair's tiny span visible rather than collapsing it", () => {
    // Everything here sits above 99.7%; the domain must not be a zero-width
    // sliver, or every bar renders on top of the axis line.
    const w = toProbabilityWaterfall(makeExplanation(6, [0.5, 0.25]));
    expect(w.domain[1] - w.domain[0]).toBeGreaterThan(0.5);
  });

  it("marks direction from the model, not from the bar's own arithmetic", () => {
    const w = toProbabilityWaterfall(makeExplanation(0, [1.2, -0.8]));
    expect(w.bars[0].direction).toBe("positive");
    expect(w.bars[1].direction).toBe("negative");
    expect(w.bars[0].deltaPct).toBeGreaterThan(0);
    expect(w.bars[1].deltaPct).toBeLessThan(0);
  });
});
