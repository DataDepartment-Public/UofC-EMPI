"use client";

import { useState } from "react";
import { ApiError, ReviewQueueFilters } from "@/lib/api-client";
import {
  useDismissMutation,
  useMergeMutation,
  useReviewQueue,
} from "@/lib/hooks";
import { MergeModal } from "@/components/review/MergeModal";
import { RawDataDrawer } from "@/components/shared/RawDataDrawer";
import { ReviewCandidateDetail } from "@/components/review/ReviewCandidateDetail";
import { ReviewQueueList } from "@/components/review/ReviewQueueList";
import { Toast } from "@/components/shared/Toast";

const PAGE_SIZE = 30;

function keyOf(item: { patid_a: string; patid_b: string }) {
  return `${item.patid_a}-${item.patid_b}`;
}

export default function ReviewQueuePage() {
  const [filters, setFilters] = useState<ReviewQueueFilters>({
    reviewed: false,
    page: 1,
    page_size: PAGE_SIZE,
  });
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [rawPatid, setRawPatid] = useState<string | null>(null);
  const [pendingMerge, setPendingMerge] = useState<{
    mid: string;
    patids: string[];
  } | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const { data, isLoading, isError } = useReviewQueue(filters);
  const mergeMutation = useMergeMutation();
  const dismissMutation = useDismissMutation();

  const flash = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3200);
  };

  const items = data?.items ?? [];
  const selected =
    items.find((i) => keyOf(i) === selectedKey) ?? items[0] ?? null;

  const confirmMerge = () => {
    if (!pendingMerge) return;
    mergeMutation.mutate(pendingMerge, {
      onSuccess: () => {
        flash(`Merged into ${pendingMerge.mid}.`);
        setPendingMerge(null);
      },
      onError: (err) => {
        flash(err instanceof ApiError ? err.message : "Merge failed.");
      },
    });
  };

  const handleDismiss = (patidA: string, patidB: string) => {
    dismissMutation.mutate(
      { patidA, patidB },
      {
        onSuccess: () => flash("Marked not a match."),
        onError: (err) =>
          flash(err instanceof ApiError ? err.message : "Dismiss failed."),
      },
    );
  };

  return (
    <div>
      <div className="mb-5">
        <h2 className="text-[22px] font-extrabold text-ink-2">Review Queue</h2>
        <p className="mt-1 text-[13px] text-gray">
          Candidate-level triage — every pending pair, independent of which
          cluster it belongs to. Select a candidate on the left to see its
          full explanation on the right.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[400px_1fr]">
        <ReviewQueueList
          filters={filters}
          onFiltersChange={setFilters}
          items={items}
          total={data?.total ?? 0}
          isLoading={isLoading}
          isError={isError}
          selectedKey={selected ? keyOf(selected) : null}
          onSelect={setSelectedKey}
        />

        {selected ? (
          <ReviewCandidateDetail
            key={keyOf(selected)}
            item={selected}
            onMerge={(mid, patids) => setPendingMerge({ mid, patids })}
            onDismiss={handleDismiss}
            onViewRaw={setRawPatid}
            dismissPending={dismissMutation.isPending}
          />
        ) : (
          <div className="card flex h-[calc(100vh-220px)] min-h-[520px] items-center justify-center p-6 text-center text-sm text-gray">
            {isLoading
              ? "Loading…"
              : "Select a candidate from the list to see its full details."}
          </div>
        )}
      </div>

      <RawDataDrawer patid={rawPatid} onClose={() => setRawPatid(null)} />

      {pendingMerge && (
        <MergeModal
          targetMid={pendingMerge.mid}
          patids={pendingMerge.patids}
          pending={mergeMutation.isPending}
          onConfirm={confirmMerge}
          onCancel={() => setPendingMerge(null)}
        />
      )}

      <Toast message={toast} />
    </div>
  );
}
