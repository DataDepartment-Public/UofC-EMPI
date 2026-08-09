import { describe, expect, it } from "vitest";
import {
  BAND_DEFS,
  bandFilter,
  bandFromFilters,
  bandRangeLabel,
  bucketBadge,
  signalFor,
  signalTooltip,
  verdictLabel,
  type LiveThresholds,
} from "./pair-signal";

/** .env sets EMPI_ML_AUTO_MERGE_THRESHOLD=0.90 and EMPI_GATE_THRESHOLD=0.30;
 * both are live-tunable, which is why nothing here hardcodes them. */
const T: LiveThresholds = { ml: 0.9, gate: 0.3 };

describe("signalFor — the match axis", () => {
  const scored = (p: number, verdict = "ml_human_review") =>
    signalFor({ confidence: null, ml_match_probability: p, verdict }).band;

  it("bands the presentational cuts", () => {
    expect(scored(0.55)).toBe("leans");
    expect(scored(0.4)).toBe("leans");
    expect(scored(0.39)).toBe("uncertain");
    expect(scored(0.2)).toBe("uncertain");
    expect(scored(0.01)).toBe("ambiguous");
  });

  it("calls a merged pair confident regardless of where its score falls", () => {
    // Thresholds are live-tunable and changes are forward-looking, so a
    // published row must be described by the decision it was given, not by
    // today's bar.
    expect(scored(0.85, "ml_auto_merge")).toBe("confident");
    expect(
      signalFor({
        confidence: 0.99,
        ml_match_probability: null,
        verdict: "auto_merge_rule",
      }).band,
    ).toBe("confident");
  });

  it("does not call an ambiguous pair confident just because it scores high", () => {
    // Published at 0.85 under a 0.90 bar. The row says human_review, so the
    // band must too — and it must still land somewhere, which is why
    // `leans` has no ceiling.
    expect(scored(0.85)).toBe("leans");
  });

  it("prefers a rule's confidence over the ML score, mirroring the backend COALESCE", () => {
    expect(
      signalFor({
        confidence: 0.67,
        ml_match_probability: 0.02,
        verdict: "ml_human_review",
      }),
    ).toEqual({ score: 0.67, source: "rule", band: "leans" });
  });
});

describe("signalFor — the gate axis", () => {
  const dropped = (gate: number | null) =>
    signalFor({
      confidence: null,
      ml_match_probability: null,
      gate_score: gate,
      verdict: "gate_dropped",
    });

  it("splits gate drops at 20% plausible", () => {
    expect(dropped(0.25).band).toBe("unlikely");
    expect(dropped(0.2).band).toBe("unlikely");
    expect(dropped(0.19).band).toBe("implausible");
    expect(dropped(0.001).band).toBe("implausible");
  });

  it("reports the gate as the score's source, not the matcher", () => {
    // Reading a gate-dropped pair on the matcher's axis is the same
    // category error as the 1% bug this scale exists to fix.
    expect(dropped(0.25).source).toBe("gate");
    expect(dropped(0.25).score).toBe(0.25);
  });

  it("degrades to the weaker band when the row predates gate_score", () => {
    // An un-backfilled publish has no score to split on; inventing a
    // distinction the data can't support would be worse than under-claiming.
    expect(dropped(null).band).toBe("implausible");
    expect(dropped(null).source).toBe("none");
  });

  it("puts a deterministic reject at the bottom on the verdict alone", () => {
    // Scored by nothing at all — no rule confidence, no gate score, no ML
    // score — so no numeric rule could place it.
    expect(
      signalFor({
        confidence: null,
        ml_match_probability: null,
        gate_score: null,
        verdict: "reject",
      }),
    ).toEqual({ score: null, source: "none", band: "implausible" });
  });

  it("never lets a matcher score leak onto a gate-dropped row", () => {
    // A stale ml_match_probability must not re-band a pair the matcher
    // never saw.
    expect(
      signalFor({
        confidence: null,
        ml_match_probability: 0.95,
        gate_score: 0.05,
        verdict: "gate_dropped",
      }).band,
    ).toBe("implausible");
  });
});

describe("bandFilter / bandFromFilters", () => {
  it("round-trips every band through the filter params", () => {
    for (const def of BAND_DEFS) {
      expect(bandFromFilters(bandFilter(def.band, T), T)).toBe(def.band);
    }
  });

  it("sends gate bands to the gate axis and match bands to the match axis", () => {
    expect(bandFilter("unlikely", T)).toMatchObject({
      gate_score_min: 0.2,
      gate_score_max: 0.3,
      confidence_min: undefined,
    });
    expect(bandFilter("uncertain", T)).toMatchObject({
      confidence_min: 0.2,
      confidence_max: 0.4,
      gate_score_min: undefined,
    });
  });

  it("takes the confident floor from the live threshold, not a constant", () => {
    expect(bandFilter("confident", T).confidence_min).toBe(0.9);
    expect(bandFilter("confident", { ml: 0.7, gate: 0.3 }).confidence_min).toBe(
      0.7,
    );
  });

  it("clears both axes when the band is cleared", () => {
    expect(bandFilter(null, T)).toEqual({
      confidence_min: undefined,
      confidence_max: undefined,
      gate_score_min: undefined,
      gate_score_max: undefined,
      verdict: undefined,
    });
  });

  it("reads no band out of filters that match none", () => {
    expect(bandFromFilters({ confidence_min: 0.123 }, T)).toBeNull();
  });

  it("classifies exactly the two reject bands onto the gate axis", () => {
    // `ReviewQueueList` switches the bucket tab to Auto-rejected off this
    // axis flag. A gate band left under Needs review can never match — a
    // gate-dropped pair is in Auto-rejected by construction — so a new band
    // added without an axis would silently return an empty list.
    expect(
      BAND_DEFS.filter((d) => d.axis === "gate").map((d) => d.band),
    ).toEqual(["unlikely", "implausible"]);
  });
});

describe("bandRangeLabel", () => {
  it("states the two threshold-derived edges from the live values", () => {
    expect(bandRangeLabel("confident", T)).toContain("≥90%");
    expect(bandRangeLabel("unlikely", T)).toContain("30%");
    expect(bandRangeLabel("confident", { ml: 0.7, gate: 0.3 })).toContain("≥70%");
  });

  it("degrades to wording with no number when thresholds haven't loaded", () => {
    expect(bandRangeLabel("confident", { ml: null, gate: null })).not.toMatch(/\d/);
    expect(bandRangeLabel("unlikely", { ml: null, gate: null })).not.toMatch(/\d/);
  });

  it("names the axis each band is measured on", () => {
    expect(bandRangeLabel("uncertain", T)).toBe(
      "20–40% probability of confident match",
    );
    expect(bandRangeLabel("ambiguous", T)).toBe(
      "<20% probability of confident match",
    );
    expect(bandRangeLabel("implausible", T)).toBe(
      "<20% probability plausible, or rejected by deterministic rules",
    );
  });

  it("keeps every line short enough to sit on one row of the legend", () => {
    for (const def of BAND_DEFS) {
      expect(bandRangeLabel(def.band, T).length).toBeLessThanOrEqual(70);
    }
  });
});

describe("signalTooltip", () => {
  it("keeps the exact percentage and names the question it answers", () => {
    const tip = signalTooltip({
      confidence: null,
      ml_match_probability: 0.014,
      verdict: "ml_human_review",
    });
    expect(tip).toContain("1.4%");
    expect(tip).toContain("P(confident match)");
    expect(tip).toContain('not "not a match"');
  });

  it("names the gate when the number came from the gate", () => {
    const tip = signalTooltip({
      gate_score: 0.05,
      verdict: "gate_dropped",
    });
    expect(tip).toContain("5.0%");
    expect(tip).toContain("P(plausible)");
  });

  it("marks a rule's number as a property of the rule, not of the pair", () => {
    const tip = signalTooltip({ confidence: 0.67, verdict: "ml_human_review" });
    expect(tip).toContain("67.0%");
    expect(tip).toContain("not of this pair");
  });
});

describe("bucketBadge", () => {
  it("names all four sections, matching the queue's own tab labels", () => {
    expect(bucketBadge({ bucket: "needs_review" }).label).toBe("Needs review");
    expect(bucketBadge({ bucket: "auto_merged" }).label).toBe("Auto-merged");
    expect(bucketBadge({ bucket: "auto_rejected" }).label).toBe("Auto-rejected");
  });

  it("reports what the reviewer concluded on a reviewed pair", () => {
    expect(
      bucketBadge({ bucket: "reviewed", reviewer_decision: "merged" }).label,
    ).toBe("Merged");
    expect(
      bucketBadge({ bucket: "reviewed", reviewer_decision: "not_a_match" })
        .label,
    ).toBe("Not a match");
  });

  it("distinguishes a reviewer merge from a pipeline merge", () => {
    // Both end with the records together, but only one is a review outcome
    // — conflating them is what once filed auto-merges under "reviewed".
    expect(
      bucketBadge({ bucket: "reviewed", reviewer_decision: "merged" }).label,
    ).not.toBe(bucketBadge({ bucket: "auto_merged" }).label);
  });

  it("falls back to open work on an unknown bucket rather than blanking", () => {
    expect(bucketBadge({ bucket: "something_new" }).label).toBe("Needs review");
  });
});

describe("verdictLabel", () => {
  it("names the deciding stage for each verdict in the pipeline vocabulary", () => {
    expect(verdictLabel("ml_human_review")).toBe("Model: ambiguous");
    expect(verdictLabel("ml_auto_merge")).toBe("Model: confident match");
    expect(verdictLabel("gate_dropped")).toBe("Gate: dropped as implausible");
    expect(verdictLabel("auto_merge_rule")).toBe("Rule: auto-merged");
    expect(verdictLabel("reject")).toBe("Rules: contradicted");
    expect(verdictLabel("undecided")).toBe("No stage decided");
  });

  it("labels a null verdict as incremental scoring, not as undecided", () => {
    // Null means the row came from `POST /records/score`, which runs no
    // model stage at all — a different thing from a stage declining to act.
    expect(verdictLabel(null)).toBe("Scored incrementally");
  });
});
