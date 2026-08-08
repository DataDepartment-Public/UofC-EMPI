"use client";

import clsx from "clsx";
import { useRawRecord, usePairExplanation } from "@/lib/hooks";
import type { ReviewQueueItem } from "@/lib/schemas";
import { formatRawDate, fullName } from "@/lib/format";

/** Value lineage for a candidate pair: Raw -> Cleaned -> Deterministic rule
 * -> ML model. Stages 1-3 show real data (raw via the same
 * `/records/:patid/raw` endpoint the raw-data drawer uses; cleaned via the
 * already-fetched display fields; rule from the candidate itself). Stage 4
 * reads `GET /explanations/{model}/{a}/{b}` (the gate/ML matcher's
 * persisted scores) — `null` for a pair neither model ever scored (dropped
 * by the gate, or a purely deterministic-rule match) shows an honest "not
 * scored" state rather than a fabricated score. The FS matcher stage is
 * intentionally not shown here — it's audit-only in the backend, kept for
 * lineage, but not part of the reviewer-facing decision trail. */
export function PipelineTrail({ item }: { item: ReviewQueueItem }) {
  const rawA = useRawRecord(item.patid_a);
  const rawB = useRawRecord(item.patid_b);
  const explanation = usePairExplanation(item.patid_a, item.patid_b);

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

  return (
    <div className="overflow-x-auto pb-1">
      <div className="flex min-w-[760px]">
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

        <Stage
          num={4}
          label={explanation.data?.model === "nonmatch_gate" ? "Non-match gate" : "ML matcher"}
          title="ML signal"
        >
          {explanation.isLoading ? (
            <Row muted>Loading…</Row>
          ) : explanation.data ? (
            <Row>
              <span className="font-bold">
                {Math.round(explanation.data.decision.score * 100)}%
              </span>
              {` · ${explanation.data.decision.tier}`}
            </Row>
          ) : (
            <Row muted>Not scored by the ML pipeline</Row>
          )}
        </Stage>
      </div>
    </div>
  );
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
        "min-w-[152px] flex-1 border border-line px-3 py-2.5",
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
  muted = false,
  children,
}: {
  ok?: boolean;
  muted?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className={clsx(
        "text-[11px]",
        ok && "font-bold text-status-auto",
        muted && "italic text-gray",
        !ok && !muted && "text-gray-2",
      )}
    >
      {children}
    </div>
  );
}
