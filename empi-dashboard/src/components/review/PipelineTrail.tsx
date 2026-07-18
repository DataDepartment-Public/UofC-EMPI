"use client";

import clsx from "clsx";
import { useRawRecord } from "@/lib/hooks";
import type { ReviewQueueItem } from "@/lib/schemas";
import { fullName } from "@/lib/format";

/** Value lineage for a candidate pair: Raw -> Cleaned -> Deterministic rule
 * -> FS signal -> ML model. The first four stages show real data (raw via
 * the same `/records/:patid/raw` endpoint the raw-data drawer uses; cleaned
 * via the already-fetched display fields; rule + FS signal from the
 * candidate itself). Stage 5 is a labeled placeholder — no GBT/ML model
 * exists in production yet (empi-service/docs/FS-Matcher-Production-Guide.md)
 * — so it renders no data rather than fabricating a score. */
export function PipelineTrail({ item }: { item: ReviewQueueItem }) {
  const rawA = useRawRecord(item.patid_a);
  const rawB = useRawRecord(item.patid_b);

  const rawField = (
    data: { fields: Record<string, unknown> } | undefined,
    key: string,
  ) => {
    const v = data?.fields?.[key];
    return v == null || v === "" ? "—" : String(v);
  };

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
                {rawField(rawA.data, "BirthDT_raw")} vs {rawField(rawB.data, "BirthDT_raw")}
              </Row>
            </>
          )}
        </Stage>

        <Stage num={2} label="Normalized" title="Cleaned">
          <Row>
            {fullName(item.patient_a.first_name, item.patient_a.last_name)},{" "}
            {item.patient_a.birth_date ?? "—"}
          </Row>
          <Row>
            {fullName(item.patient_b.first_name, item.patient_b.last_name)},{" "}
            {item.patient_b.birth_date ?? "—"}
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

        <Stage num={4} label="FS matcher" title="FS signal">
          {item.fs_match_probability != null ? (
            <>
              <Row>
                <span className="font-bold">
                  {Math.round(item.fs_match_probability * 100)}%
                </span>
                {item.fs_classification_tier ? ` · ${item.fs_classification_tier}` : ""}
              </Row>
              <Row muted>Audit-only signal</Row>
            </>
          ) : (
            <Row muted>Not scored for this run</Row>
          )}
        </Stage>

        <Stage num={5} label="Future" title="ML model" placeholder>
          <Row muted>Not yet in production</Row>
        </Stage>
      </div>
    </div>
  );
}

function Stage({
  num,
  label,
  title,
  placeholder = false,
  children,
}: {
  num: number;
  label: string;
  title: string;
  placeholder?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className={clsx(
        "min-w-[152px] flex-1 border border-line px-3 py-2.5",
        "first:rounded-l-md last:rounded-r-md [&:not(:last-child)]:border-r-0",
        placeholder && "bg-bg/70 opacity-80",
      )}
      style={
        placeholder
          ? {
              backgroundImage:
                "repeating-linear-gradient(135deg, transparent, transparent 8px, var(--line) 8px, var(--line) 9px)",
            }
          : undefined
      }
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
