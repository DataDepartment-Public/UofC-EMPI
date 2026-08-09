import type { ReviewQueueFilters } from "@/lib/api-client";

/** The Review Queue's confidence scale: one ordered vocabulary describing
 * how far a candidate pair got through the pipeline, and how strongly.
 *
 * WHY A BAND AND NOT A PERCENTAGE
 * ------------------------------
 * The queue used to render `confidence ?? ml_match_probability` as a bare
 * percentage in the auto-merge green. Both halves of that were misleading:
 *
 *   * The two quantities answer different questions. `confidence` is a
 *     deterministic rule's *static* precision — a property of the rule, the
 *     same number for every pair it fires on. `ml_match_probability` is the
 *     Stage-4.5 v5 model's per-pair score.
 *   * The v5 model trains class 1 = *confident match*, so class 0 is
 *     `ambiguous ∪ confident non-match`. A 1% score means "not obviously a
 *     match" — it does **not** mean "probably a different person", because
 *     the model never separates those two cases. A reviewer reading "1%" on
 *     a scale whose top is a merge concludes exactly that, about a pair the
 *     gate may well have kept at 70% plausible.
 *
 * TWO AXES, ONE SCALE
 * -------------------
 * The bands run over two different models' scores, because the pipeline
 * asks two different questions and a pair only ever reaches one of them:
 *
 *   * `Confident` / `Leans match` / `Uncertain` / `Ambiguous` describe
 *     `P(confident match)` from the Stage-4.5 matcher — pairs that passed
 *     the gate and were scored for merging.
 *   * `Unlikely` / `Implausible` describe `P(plausible)` from the Stage-4.25
 *     gate — pairs the gate dropped, which the matcher never saw. Their only
 *     score is the gate's, which is why `gate_score` had to be published
 *     onto the candidate row before this scale could exist.
 *
 * Reading a gate-dropped pair on the matcher axis (or vice versa) is the
 * same category error as the original 1% bug, so the two never mix: which
 * axis a row is on is decided by `verdict`, not by which score is non-null.
 *
 * WHY THE TOP BAND READS `verdict` AND NOT THE SCORE
 * --------------------------------------------------
 * `confident` is *not* "score >= ml_auto_merge_threshold". Thresholds are
 * live-tunable (`GET/PUT /admin/thresholds`, `src/api/threshold_store.py`)
 * and a change there is explicitly forward-looking — it "never rewrites
 * tiers already computed and published by a prior run". Comparing a
 * published score against the *current* bar therefore mislabels every row
 * published under a different one: move the bar to 0.80 and a row recorded
 * at 0.85 still carries `human_review`, while a score-derived band would
 * call it confident, contradicting the verdict shown beside it.
 *
 * `verdict` is what the run actually decided, recorded at publish time and
 * immutable after. The live thresholds are still fetched (`useThresholds`)
 * for the two places that want today's numbers: the legend's stated bars and
 * the filter bounds.
 */

/** Mirrors `MERGE_VERDICTS` in `src/api/pair_verdicts.py`. */
const MERGE_VERDICTS = new Set(["auto_merge_rule", "ml_auto_merge"]);

/** The boundary between the two reject bands, on the gate's axis. Below the
 * gate threshold by construction (the gate dropped these), so it needs no
 * live value — unlike the band's upper edge, which *is* the live threshold. */
const IMPLAUSIBLE_CEILING = 0.2;

export type SignalBand =
  | "confident"
  | "leans"
  | "uncertain"
  | "ambiguous"
  | "unlikely"
  | "implausible";

/** Which model's score a band is expressed in. Also picks the filter
 * parameters: `match` bands bound `confidence_*`, `gate` bands bound
 * `gate_score_*`. */
export type SignalAxis = "match" | "gate";

export interface BandDef {
  band: SignalBand;
  label: string;
  axis: SignalAxis;
  /** Inclusive floor, exclusive ceiling, on that band's axis. `null` means
   * unbounded; the confident band's floor is the live threshold instead. */
  min: number | null;
  max: number | null;
  /** Tailwind text+bg tone. Per `globals.css`'s FR-13 note, display badges
   * use the `*-display` variants rather than the gold/red action colors. */
  tone: string;
}

/** Ordered strongest to weakest — the order the legend and the filter show.
 * The 40%/20% cuts are presentational and no stage acts on them; the 90%
 * and 30% edges are the two live thresholds. */
export const BAND_DEFS: BandDef[] = [
  {
    band: "confident",
    label: "Confident",
    axis: "match",
    min: null, // the live ml_auto_merge_threshold — see bandFilter
    max: null,
    tone: "text-status-auto bg-status-auto/15",
  },
  {
    band: "leans",
    label: "Leans match",
    axis: "match",
    // No ceiling. With `confident` decided by verdict, a pair left at
    // `human_review` can still score high — anything published under a
    // higher bar than today's. Capping this would leave it in no band.
    min: 0.4,
    max: null,
    tone: "text-status-review-display bg-status-review-display/15",
  },
  {
    band: "uncertain",
    label: "Uncertain",
    axis: "match",
    min: 0.2,
    max: 0.4,
    tone: "text-status-nomatch-display bg-status-nomatch-display/15",
  },
  {
    band: "ambiguous",
    label: "Ambiguous",
    axis: "match",
    min: null,
    max: 0.2,
    tone: "text-gray-2 bg-bg",
  },
  {
    band: "unlikely",
    label: "Unlikely",
    axis: "gate",
    min: IMPLAUSIBLE_CEILING,
    max: null, // the live gate_threshold — see bandFilter
    tone: "text-status-nomatch bg-status-nomatch/10",
  },
  {
    band: "implausible",
    label: "Implausible",
    axis: "gate",
    min: null,
    max: IMPLAUSIBLE_CEILING,
    tone: "text-status-nomatch bg-status-nomatch/10",
  },
];

const BY_BAND = new Map(BAND_DEFS.map((d) => [d.band, d]));

export function bandDef(band: SignalBand): BandDef {
  return BY_BAND.get(band) ?? BAND_DEFS[BAND_DEFS.length - 1];
}

export type SignalSource = "rule" | "model" | "gate" | "none";

export interface SignalInput {
  confidence?: number | null;
  ml_match_probability?: number | null;
  gate_score?: number | null;
  verdict?: string | null;
}

/** The band, the number behind it, and which model produced that number.
 *
 * `verdict` picks the axis first, so a gate-dropped pair is never described
 * by a matcher score and a matched pair is never described by the gate's. */
export function signalFor(item: SignalInput): {
  score: number | null;
  source: SignalSource;
  band: SignalBand;
} {
  const verdict = item.verdict ?? null;

  // ── Gate axis: pairs that never reached the matcher ──────────────────
  // A deterministic `reject` was scored by nothing at all; it belongs at
  // the bottom of this axis on the strength of the verdict alone.
  if (verdict === "reject") {
    return { score: null, source: "none", band: "implausible" };
  }
  if (verdict === "gate_dropped") {
    const gate = item.gate_score ?? null;
    return {
      score: gate,
      source: gate == null ? "none" : "gate",
      // An un-backfilled row (published before `gate_score` existed) has no
      // score to split on, so it reads as the weaker of the two rather than
      // inventing a distinction the data can't support.
      band:
        gate != null && gate >= IMPLAUSIBLE_CEILING ? "unlikely" : "implausible",
    };
  }

  // ── Match axis ───────────────────────────────────────────────────────
  const [score, source]: [number | null, SignalSource] =
    item.confidence != null
      ? [item.confidence, "rule"]
      : item.ml_match_probability != null
        ? [item.ml_match_probability, "model"]
        : [null, "none"];

  if (verdict != null && MERGE_VERDICTS.has(verdict)) {
    return { score, source, band: "confident" };
  }
  if (score == null) return { score, source, band: "ambiguous" };
  if (score >= 0.4) return { score, source, band: "leans" };
  if (score >= 0.2) return { score, source, band: "uncertain" };
  return { score, source, band: "ambiguous" };
}

/** The exact number on the chip's tooltip, named by the question it answers,
 * so nothing is lost by demoting the percentage off the row. */
export function signalTooltip(item: SignalInput): string {
  const { score, source } = signalFor(item);
  if (score == null || source === "none") {
    return "No score — this pair was decided before any model ran.";
  }
  const pct = `${(score * 100).toFixed(1)}%`;
  if (source === "rule") {
    return `Deterministic rule precision: ${pct}. A property of the rule, not of this pair.`;
  }
  if (source === "gate") {
    return `Non-match gate P(plausible): ${pct}. The matcher never scored this pair.`;
  }
  return `ML matcher P(confident match): ${pct}. Low means "not obviously a match", not "not a match".`;
}

export interface LiveThresholds {
  /** `ml_auto_merge_threshold` — the confident band's floor. */
  ml: number | null;
  /** `gate_threshold` — the unlikely band's ceiling. */
  gate: number | null;
}

/** The one-line range shown beside each band in the legend. */
export function bandRangeLabel(
  band: SignalBand,
  t: LiveThresholds,
): string {
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  switch (band) {
    case "confident":
      return t.ml != null
        ? `≥${pct(t.ml)} probability of confident match, or an auto-merge rule`
        : "At or above the auto-merge bar, or an auto-merge rule";
    case "leans":
      return "≥40% probability of confident match";
    case "uncertain":
      return "20–40% probability of confident match";
    case "ambiguous":
      return "<20% probability of confident match";
    case "unlikely":
      return t.gate != null
        ? `20–${pct(t.gate)} probability plausible — dropped by the non-match gate`
        : "Just below the gate's bar — dropped by the non-match gate";
    case "implausible":
      return "<20% probability plausible, or rejected by deterministic rules";
  }
}

/** Translate a band into the filter params the API takes.
 *
 * Which axis a band sits on decides which pair of bounds it uses. The two
 * verdict-decided edges come from the live thresholds rather than
 * `BAND_DEFS`: `confident`'s floor is `ml_auto_merge_threshold`, and
 * `unlikely`'s ceiling is `gate_threshold`.
 *
 * `implausible` additionally needs no bound to catch deterministic rejects
 * — they carry no score of any kind — so it can't be expressed as a range
 * alone. It filters by `verdict` instead, which necessarily narrows it to
 * gate drops; the rule rejects are reachable via `verdict=reject`. Being
 * approximate here is the cost of one dropdown instead of two. */
export function bandFilter(
  band: SignalBand | null,
  t: LiveThresholds,
): Partial<ReviewQueueFilters> {
  const cleared: Partial<ReviewQueueFilters> = {
    confidence_min: undefined,
    confidence_max: undefined,
    gate_score_min: undefined,
    gate_score_max: undefined,
    verdict: undefined,
  };
  if (band == null) return cleared;
  const def = bandDef(band);

  if (def.axis === "gate") {
    return {
      ...cleared,
      gate_score_min: band === "unlikely" ? IMPLAUSIBLE_CEILING : undefined,
      gate_score_max:
        band === "unlikely" ? (t.gate ?? undefined) : IMPLAUSIBLE_CEILING,
    };
  }
  return {
    ...cleared,
    confidence_min: band === "confident" ? (t.ml ?? undefined) : (def.min ?? undefined),
    confidence_max: def.max ?? undefined,
  };
}

/** Which band the current filters represent, or null for "Any". Derived from
 * the filters rather than held separately so a caller that sets bounds some
 * other way still renders a consistent control. */
export function bandFromFilters(
  filters: ReviewQueueFilters,
  t: LiveThresholds,
): SignalBand | null {
  for (const def of BAND_DEFS) {
    const f = bandFilter(def.band, t);
    if (
      f.confidence_min === filters.confidence_min &&
      f.confidence_max === filters.confidence_max &&
      f.gate_score_min === filters.gate_score_min &&
      f.gate_score_max === filters.gate_score_max
    ) {
      return def.band;
    }
  }
  return null;
}

/** Where the pair stands: which of the queue's four sections it's in, and
 * for a reviewed one, what the human concluded.
 *
 * Shared by the list rows and the detail header so the badge can never
 * disagree with the section tab the reviewer clicked to get there. Reads
 * `bucket` (and `reviewer_decision`) rather than inferring from
 * `mid_a === mid_b`: a pair can share a mid because the pipeline merged it,
 * because clustering unioned it transitively, or because a reviewer merged
 * it, and only the third is a review outcome. */
export function bucketBadge(item: {
  bucket: string;
  reviewer_decision?: string | null;
}): { label: string; tone: string } {
  switch (item.bucket) {
    case "reviewed":
      return {
        // A reviewer has ruled either way, so both read as resolved.
        label: item.reviewer_decision === "merged" ? "Merged" : "Not a match",
        tone: "text-status-auto bg-status-auto/15",
      };
    case "auto_merged":
      return { label: "Auto-merged", tone: "text-status-auto bg-status-auto/15" };
    case "auto_rejected":
      return {
        label: "Auto-rejected",
        tone: "text-status-nomatch bg-status-nomatch/15",
      };
    default:
      // Deliberately the gold `--status-review` rather than FR-13's
      // `*-display` variant: open work is the one state a reviewer should be
      // able to pick out of a list at a glance, and blue read as inert
      // beside the green/red resolved states.
      return { label: "Needs review", tone: "text-status-review bg-status-review/15" };
  }
}

/** Which stage decided this pair, in the row's own words.
 *
 * `verdict` is the pipeline's own vocabulary (`src/api/pair_verdicts.py`),
 * already carried on every queue row and already the authority
 * `PipelineStages.skipReason` reads. Surfacing it is what turns a row from
 * "here is a number" into "here is why you are looking at this". */
export function verdictLabel(verdict: string | null | undefined): string {
  switch (verdict) {
    case "auto_merge_rule":
      return "Rule: auto-merged";
    case "reject":
      return "Rules: contradicted";
    case "gate_dropped":
      return "Gate: dropped as implausible";
    case "ml_auto_merge":
      return "Model: confident match";
    case "ml_human_review":
      return "Model: ambiguous";
    case "undecided":
      return "No stage decided";
    default:
      // `verdict` is null exactly for pairs written by incremental scoring,
      // which runs no model stage (see `ReviewQueueItemSchema`).
      return "Scored incrementally";
  }
}

/** The one-line "what the pipeline did with this pair", named by the stage
 * that did it — `Deterministic rule: SSN_DOB`, `Gate: not a match`,
 * `Matcher: confident match`.
 *
 * This is `verdictLabel` plus the two things a reviewer needs alongside it:
 * the rule that fired (a rule verdict without its rule name says nothing
 * about *why*), and whether clustering put the two records together anyway.
 *
 * `cluster` is non-null only when the pair shares a `mid` without its own
 * merge verdict — i.e. clustering unioned it through some other record. A
 * pair that merged on its own edge already says so in `label`, so repeating
 * "same cluster" there would be noise. The Review Queue's candidate row
 * can't distinguish a transitive join from a reviewer's merge, so a reviewed
 * pair is left to its state badge rather than guessed at. */
export function pipelineOutcome(item: {
  verdict?: string | null;
  match_rule?: string | null;
  mid_a?: string | null;
  mid_b?: string | null;
  bucket?: string;
}): { label: string; cluster: string | null } {
  const rule = item.match_rule ?? null;
  const verdict = item.verdict ?? null;

  let label: string;
  switch (verdict) {
    case "auto_merge_rule":
      label = rule ? `Deterministic rule: ${rule}` : "Deterministic rule: merged";
      break;
    case "reject":
      label = "Deterministic rules: not a match";
      break;
    case "gate_dropped":
      label = "Gate: not a match";
      break;
    case "ml_auto_merge":
      label = "Matcher: confident match";
      break;
    case "ml_human_review":
      label = rule ? `Matcher: ambiguous · rule ${rule}` : "Matcher: ambiguous";
      break;
    case "undecided":
      label = "No stage decided";
      break;
    default:
      label = "Scored incrementally";
  }

  const sameCluster =
    item.mid_a != null && item.mid_b != null && item.mid_a === item.mid_b;
  const cluster =
    sameCluster && !MERGE_VERDICTS.has(verdict ?? "") && item.bucket !== "reviewed"
      ? "Cluster: joined transitively"
      : null;

  return { label, cluster };
}
