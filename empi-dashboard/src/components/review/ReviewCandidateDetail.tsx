"use client";

import { useState } from "react";
import { compareRecords } from "@/lib/compare";
import { fullName } from "@/lib/format";
import type { ReviewQueueItem } from "@/lib/schemas";
import { FeatureComparisonTable } from "@/components/shared/FeatureComparisonTable";
import { RawComparisonPanel } from "@/components/shared/RawComparisonPanel";
import { ShapWaterfall } from "@/components/shared/ShapWaterfall";
import { usePairExplanation } from "@/lib/hooks";
import { ManualMatchModal } from "./ManualMatchModal";
import { PipelineTrail } from "./PipelineTrail";

/** The right-hand panel of the Review Queue's two-panel layout — everything
 * needed to decide on the candidate selected in the left-hand list, with no
 * page navigation (selection swaps this panel's content in place). */
export function ReviewCandidateDetail({
  item,
  onMerge,
  onDismiss,
  dismissPending,
}: {
  item: ReviewQueueItem;
  onMerge: (mid: string, patids: string[]) => void;
  onDismiss: (patidA: string, patidB: string) => void;
  dismissPending: boolean;
}) {
  const [manualMatchOpen, setManualMatchOpen] = useState(false);
  const rows = compareRecords(item.patient_a, item.patient_b);
  const { data: explanation } = usePairExplanation(item.patid_a, item.patid_b);
  const predictedClass = item.match_rule
    ? "Confirmed duplicate (rule-matched)"
    : "Uncertain — pending review";

  return (
    <div className="card p-5">
      <div className="mb-4 flex items-start justify-between gap-4 border-b border-line pb-4">
        <div>
          <h3 className="text-[17px] font-extrabold text-ink-2">
            {fullName(item.patient_a.first_name, item.patient_a.last_name)}{" "}
            <span className="font-medium text-gray">vs</span>{" "}
            {fullName(item.patient_b.first_name, item.patient_b.last_name)}
          </h3>
          <p className="mt-0.5 font-mono text-[11px] text-gray">
            {item.patid_a} vs {item.patid_b}
            {(item.member_count_a > 1 || item.member_count_b > 1) && (
              <span className="ml-2 rounded-full bg-bg px-2 py-0.5 text-[10px] font-bold text-gray-2">
                part of a larger cluster
              </span>
            )}
          </p>
        </div>
        <div className="flex flex-shrink-0 gap-2">
          <button
            onClick={() => onMerge(item.mid_a, [item.patid_b])}
            className="rounded-md bg-status-auto px-3.5 py-1.5 text-xs font-bold text-white hover:opacity-90"
          >
            Merge
          </button>
          <button
            disabled={dismissPending}
            onClick={() => onDismiss(item.patid_a, item.patid_b)}
            className="rounded-md border border-status-nomatch px-3.5 py-1.5 text-xs font-bold text-status-nomatch hover:bg-status-nomatch/10 disabled:opacity-50"
          >
            Not a match
          </button>
        </div>
      </div>

      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetaCard
          label="Confidence"
          value={item.confidence != null ? `${Math.round(item.confidence * 100)}%` : "No rule signal"}
        />
        <MetaCard label="Rule fired" value={item.match_rule ?? "None (unconfirmed)"} />
        <MetaCard label="Predicted class" value={predictedClass} small />
        <MetaCard
          label="Cluster context"
          value={
            item.member_count_a > 1 || item.member_count_b > 1
              ? `${Math.max(item.member_count_a, item.member_count_b)} members`
              : "Standalone pair"
          }
        />
      </div>

      <div className="mb-5">
        <h4 className="mb-1 text-[13px] font-bold text-ink-2">Pipeline trail</h4>
        <PipelineTrail item={item} />
      </div>

      <div className="mb-5">
        <h4 className="mb-1 text-[13px] font-bold text-ink-2">Feature comparison</h4>
        <FeatureComparisonTable rows={rows} patidA={item.patid_a} patidB={item.patid_b} />
        <RawComparisonPanel patidA={item.patid_a} patidB={item.patid_b} />
      </div>

      {explanation && (
        <div className="mb-5">
          <h4 className="mb-1 text-[13px] font-bold text-ink-2">
            Feature contributions (SHAP)
          </h4>
          <ShapWaterfall explanation={explanation} />
        </div>
      )}

      <button
        onClick={() => setManualMatchOpen(true)}
        className="text-xs font-semibold text-brand-blue hover:underline"
      >
        Not the right match? Search manually for a different record →
      </button>

      {manualMatchOpen && (
        <ManualMatchModal
          sourceMid={item.mid_a}
          anchor={item.patient_a}
          onMerge={onMerge}
          onClose={() => setManualMatchOpen(false)}
        />
      )}
    </div>
  );
}

function MetaCard({
  label,
  value,
  small = false,
}: {
  label: string;
  value: string;
  small?: boolean;
}) {
  return (
    <div className="rounded-md border border-line px-3 py-2.5">
      <div className="text-[10px] font-bold tracking-wide text-gray uppercase">{label}</div>
      <div className={`mt-1 font-bold text-ink-2 ${small ? "text-xs" : "text-sm"}`}>
        {value}
      </div>
    </div>
  );
}
