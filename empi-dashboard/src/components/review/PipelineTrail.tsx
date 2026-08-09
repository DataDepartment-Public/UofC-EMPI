"use client";

import clsx from "clsx";
import { useRawRecord, usePairExplanations } from "@/lib/hooks";
import type { ReviewQueueItem } from "@/lib/schemas";
import { formatRawDate, fullName } from "@/lib/format";

/** Value lineage for a candidate pair, one column per pipeline stage:
 * Raw -> Cleaned -> Deterministic rule -> Non-match gate -> ML matcher.
 *
 * Stages 1-3 show real data (raw via the same `/records/:patid/raw` endpoint
 * the raw-data drawer uses; cleaned via the already-fetched display fields;
 * rule from the candidate itself). Stages 4 and 5 read
 * `GET /explanations/{model}/{a}/{b}` — the two models' *persisted* scores,
 * computed at score time against the model and feature vector that actually
 * produced the recorded decision.
 *
 * **The gate and the matcher are separate columns because they answer
 * different questions and are scored on different scales.** The gate's number
 * is `P(plausible)` — it decides whether the matcher sees the pair at all;
 * the matcher's is `P(confident match)`. Collapsing them into one "ML signal"
 * column (as this did before) meant a reviewer looking at a percentage
 * couldn't tell which question it answered, and a gate-dropped pair looked
 * simply unscored rather than deliberately discarded with a number attached.
 *
 * Either stage can be legitimately absent, and `_skipReason` says which and
 * why rather than showing a bare "—": the gate never scores a pair the
 * deterministic rules already resolved, and the matcher never scores one the
 * gate dropped. Never a fabricated score.
 *
 * The FS matcher is intentionally not a stage here — it's audit-only in the
 * backend, kept for lineage, not a reviewer-facing decision signal. */
export function PipelineTrail({ item }: { item: ReviewQueueItem }) {
  const rawA = useRawRecord(item.patid_a);
  const rawB = useRawRecord(item.patid_b);
  const { gate, ml } = usePairExplanations(item.patid_a, item.patid_b);

  const rawField = (
    data: { fields: Record<string, unknown> } | undefined,
    key: string,
  ) => {
    const v = data?.fields?.[key];
    return v == null || v === "" ? "—" : String(v);
  };

  /** Raw birthdate, minus the midnight time real source exports carry. */
  const rawBirthDate = (data: { fields: Record<string, unknown> } | undefined) =>
    formatRawDate(data?.fields?.["BirthDT_raw"]);

  const gateDropped = gate.data?.decision.tier === "no_match";

  return (
    // The five stages stretch to fill whatever width the detail panel has; the
    // scroll container is only the floor for a genuinely narrow window, since
    // below ~800px the stage text stops being readable at all.
    <div className="overflow-x-auto pb-1">
      <div className="flex min-w-[800px]">
        <Stage num={1} label="Source" title="Raw">
          {rawA.isLoading || rawB.isLoading ? (
            <Row muted>Loading…</Row>
          ) : (
            <>
              <Row>
                {rawField(rawA.data, "FirstNM_raw")} {rawField(rawA.data, "LastNM_raw")}
                {" vs "}
                {rawField(rawB.data, "FirstNM_raw")} {rawField(rawB.data, "LastNM_raw")}
              </Row>
              <Row>
                {rawBirthDate(rawA.data)} vs {rawBirthDate(rawB.data)}
              </Row>
            </>
          )}
        </Stage>

        <Stage num={2} label="Normalized" title="Cleaned">
          <Row>
            {fullName(item.patient_a.first_name, item.patient_a.last_name)},{" "}
            {formatRawDate(item.patient_a.birth_date)}
          </Row>
          <Row>
            {fullName(item.patient_b.first_name, item.patient_b.last_name)},{" "}
            {formatRawDate(item.patient_b.birth_date)}
          </Row>
        </Stage>

        <Stage num={3} label="Deterministic" title="Rule">
          {item.match_rule ? (
            <>
              <Row ok>{item.match_rule} fired</Row>
              <Row>
                Fixed confidence{" "}
                {item.confidence != null ? `${Math.round(item.confidence * 100)}%` : "—"}
              </Row>
            </>
          ) : (
            <Row muted>No rule reached threshold</Row>
          )}
        </Stage>

        <Stage num={4} label="Non-match gate" title="P(plausible)">
          {gate.isLoading ? (
            <Row muted>Loading…</Row>
          ) : gate.data ? (
            <>
              <Score decision={gate.data.decision} />
              <Row ok={!gateDropped} bad={gateDropped}>
                {gateDropped
                  ? "Dropped — confident non-match"
                  : "Passed to the matcher"}
              </Row>
            </>
          ) : (
            <Row muted>{_skipReason(item.verdict, "gate")}</Row>
          )}
        </Stage>

        <Stage num={5} label="ML matcher" title="P(confident match)">
          {ml.isLoading ? (
            <Row muted>Loading…</Row>
          ) : ml.data ? (
            <>
              <Score decision={ml.data.decision} />
              <Row ok={ml.data.decision.tier === "auto_merge"}>
                {ml.data.decision.tier === "auto_merge"
                  ? "Confident match — auto-merged"
                  : "Ambiguous — sent to review"}
              </Row>
            </>
          ) : (
            <Row muted>
              {gateDropped
                ? "Didn't run — the gate dropped this pair"
                : _skipReason(item.verdict, "matcher")}
            </Row>
          )}
        </Stage>
      </div>
    </div>
  );
}

/** A model stage's score against the threshold it was actually judged by,
 * both taken from the pair's persisted decision.
 *
 * **Both stages report the probability of the pair being the same patient**,
 * never its complement — the gate's `P(plausible)`, the matcher's
 * `P(confident match)`. Two reasons this is worth preserving. The threshold
 * beside it is a `>=` bound on that same quantity, so score and bar compare
 * directly; showing `1 - score` would mean inverting the bar too and
 * reversing the comparison. And the SHAP waterfall further down the panel
 * explains the score in this orientation — contributions are normalized so
 * positive pushes toward the model's positive class (`Explanations-Guide.md`
 * §2), which for the gate *is* "plausible". A headline number that ran
 * opposite to the waterfall explaining it would read perfectly plausibly and
 * be exactly backwards, which is the failure mode the v5 matcher rewrite
 * exists to rule out.
 *
 * Showing the threshold is what makes a low score legible: bare "0%" reads as
 * "no signal", "0% · needs 30%" reads as the deliberate drop it was. */
function Score({
  decision,
}: {
  decision: { score: number; tier: string; threshold?: number | null };
}) {
  return (
    <Row>
      <span className="font-bold">{Math.round(decision.score * 100)}%</span>
      {decision.threshold != null && (
        <span className="text-gray">
          {" · needs "}
          {Math.round(decision.threshold * 100)}%
        </span>
      )}
    </Row>
  );
}

/** Why a model stage has no score, given what the pipeline concluded about
 * the pair. `verdict` is the authority here rather than the missing
 * explanation itself: the artifact being absent tells you the stage didn't
 * run, but only the verdict says which earlier stage ended the pair's
 * journey — and "no rule fired, no model scored it either" (an ungated run)
 * is a genuinely different state from "a rule decided it at stage 3". */
function _skipReason(
  verdict: string | null | undefined,
  stage: "gate" | "matcher",
): string {
  if (verdict === "auto_merge_rule") {
    return "Not scored — an auto-merge rule decided this at stage 3";
  }
  if (verdict === "reject") {
    return "Not scored — the reject rules decided this at stage 3";
  }
  return stage === "gate"
    ? "Not scored — no gate model ran for this run"
    : "Didn't run — no ML model scored this run";
}

function Stage({
  num,
  label,
  title,
  children,
}: {
  num: number;
  label: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className={clsx(
        "min-w-0 flex-1 basis-0 border border-line px-3 py-2.5",
        "first:rounded-l-md last:rounded-r-md [&:not(:last-child)]:border-r-0",
      )}
    >
      <div className="text-[10px] font-bold tracking-wide text-gray uppercase">
        {num} · {label}
      </div>
      <div className="mb-1.5 text-[12px] font-bold text-ink-2">{title}</div>
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}

function Row({
  ok = false,
  bad = false,
  muted = false,
  children,
}: {
  ok?: boolean;
  bad?: boolean;
  muted?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className={clsx(
        "text-[11px]",
        ok && "font-bold text-status-auto",
        bad && "font-bold text-status-nomatch",
        muted && "italic text-gray",
        !ok && !bad && !muted && "text-gray-2",
      )}
    >
      {children}
    </div>
  );
}
