"use client";

import { useState } from "react";
import { compareRecords } from "@/lib/compare";
import { fullName } from "@/lib/format";
import type { ReviewQueueItem } from "@/lib/schemas";
import { FeatureComparisonTable } from "@/components/shared/FeatureComparisonTable";
import { RawComparisonPanel } from "@/components/shared/RawComparisonPanel";
import { ClusterLinks } from "./ClusterLinks";
import { ExplanationPanel } from "./ExplanationPanel";
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
  const merged = item.mid_a === item.mid_b;

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
        {/* Both actions stay available in every section — that's what makes
            Auto-merged and Auto-rejected overridable rather than read-only.
            Only the labels change, so the button says what it will actually
            do: "Merge" on a pair the pipeline already merged would be a
            no-op, and "Not a match" on one is really a split. */}
        <div className="flex flex-shrink-0 gap-2">
          {!merged && (
            <button
              onClick={() => onMerge(item.mid_a, [item.patid_b])}
              className="rounded-md bg-status-auto px-3.5 py-1.5 text-xs font-bold text-white hover:opacity-90"
            >
              {item.bucket === "auto_rejected" ? "Merge anyway" : "Merge"}
            </button>
          )}
          <button
            disabled={dismissPending}
            onClick={() => onDismiss(item.patid_a, item.patid_b)}
            className="rounded-md border border-status-nomatch px-3.5 py-1.5 text-xs font-bold text-status-nomatch hover:bg-status-nomatch/10 disabled:opacity-50"
          >
            {merged ? "Not a match — split them" : "Not a match"}
          </button>
        </div>
      </div>

      <VerdictBanner item={item} />

      <div className="mb-5">
        <h4 className="mb-1 text-[13px] font-bold text-ink-2">Pipeline trail</h4>
        <PipelineTrail item={item} />
      </div>

      <div className="mb-5">
        <h4 className="mb-1.5 text-[13px] font-bold text-ink-2">Clusters</h4>
        <ClusterLinks item={item} />
      </div>

      <div className="mb-5">
        <h4 className="mb-1 text-[13px] font-bold text-ink-2">Feature comparison</h4>
        <FeatureComparisonTable rows={rows} patidA={item.patid_a} patidB={item.patid_b} />
        <RawComparisonPanel patidA={item.patid_a} patidB={item.patid_b} />
      </div>

      <ExplanationPanel item={item} />

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

/** The reviewed banner's palette. The other buckets used to need one too, for
 * the "Outcome" meta card; the left-hand queue list already badges the bucket
 * on every row, so the card was a second copy of it. */
const REVIEWED_TONE = "border-brand-blue/30 bg-brand-blue/5 text-brand-blue";

/** What the pipeline concluded, in the reviewer's words rather than the
 * stage's. Used only to say what a *human* decision overrode — the pipeline's
 * own story is told stage by stage, with real scores, by the pipeline trail
 * below, so restating it in prose above the trail would just be a vaguer
 * second copy. */
const VERDICT_TEXT: Record<string, string> = {
  auto_merge_rule: "a deterministic rule had merged it",
  ml_auto_merge: "the ML matcher had merged it",
  ml_human_review: "the ML matcher had left it for a human",
  reject: "the reject rules had discarded it",
  gate_dropped: "the non-match gate had discarded it",
  undecided: "no model stage scored it",
};

/** Only rendered for a pair a human ruled on — that's the one thing the
 * pipeline trail can't show, since a reviewer decision isn't a pipeline
 * stage. Every other bucket is fully explained by the trail. */
function VerdictBanner({ item }: { item: ReviewQueueItem }) {
  if (item.bucket !== "reviewed") return null;
  const overruled = item.verdict ? VERDICT_TEXT[item.verdict] : null;
  return (
    <div className={`mb-4 rounded-md border px-3 py-2 text-xs ${REVIEWED_TONE}`}>
      <span className="font-bold">Reviewed.</span>{" "}
      <span className="opacity-90">
        {item.reviewer_decision === "merged"
          ? "A reviewer merged these records."
          : "A reviewer marked this pair not a match."}
      </span>
      {overruled && (
        <span className="opacity-70"> The pipeline had said {overruled}.</span>
      )}
    </div>
  );
}
