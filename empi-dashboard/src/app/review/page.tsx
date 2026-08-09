"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ApiError, ReviewQueueFilters } from "@/lib/api-client";
import {
  useDismissMutation,
  useMergeMutation,
  useReviewCandidate,
  useReviewQueue,
} from "@/lib/hooks";
import type { ReviewBucket, ReviewQueueItem } from "@/lib/schemas";
import { MergeModal } from "@/components/review/MergeModal";
import { ReviewCandidateDetail } from "@/components/review/ReviewCandidateDetail";
import { ReviewQueueList } from "@/components/review/ReviewQueueList";
import { Toast } from "@/components/shared/Toast";

const PAGE_SIZE = 30;

function keyOf(item: { patid_a: string; patid_b: string }) {
  return `${item.patid_a}-${item.patid_b}`;
}

export default function ReviewQueuePage() {
  return (
    <Suspense fallback={<p className="text-sm text-gray">Loading…</p>}>
      <ReviewQueuePageContent />
    </Suspense>
  );
}

function ReviewQueuePageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  // A deep link from elsewhere in the UI (a cluster's comparison history
  // links here for its "awaiting review" pairs) — see `useReviewCandidate`.
  // Read once on mount; `deepLinkActive` (not the presence of these params)
  // is what actually gates the override below, so picking something else
  // from the list doesn't keep snapping back to this pair.
  const deepLinkPatidA = searchParams.get("patid_a");
  const deepLinkPatidB = searchParams.get("patid_b");
  const hasDeepLink = deepLinkPatidA !== null && deepLinkPatidB !== null;

  const [filters, setFilters] = useState<ReviewQueueFilters>({
    bucket: "needs_review",
    page: 1,
    page_size: PAGE_SIZE,
  });
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [deepLinkActive, setDeepLinkActive] = useState(hasDeepLink);
  const [pendingMerge, setPendingMerge] = useState<{
    mid: string;
    patids: string[];
  } | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const { data, isLoading, isError } = useReviewQueue(filters);
  const deepLinkCandidate = useReviewCandidate(
    hasDeepLink ? deepLinkPatidA : null,
    hasDeepLink ? deepLinkPatidB : null,
  );
  const mergeMutation = useMergeMutation();
  const dismissMutation = useDismissMutation();

  const flash = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3200);
  };

  // A deep link can land on any of the four sections, not just "Needs review"
  // — the registry links out from auto-merged, rejected and gate-dropped
  // comparisons too. Line the list's tabs up with where the pair actually
  // lives as soon as it resolves, so the reviewer sees it in its section
  // (and highlighted) rather than a "Needs review" list the linked pair isn't
  // in. Runs while the deep link is active; leaving it then keeps the section
  // rather than jumping back.
  //
  // Adjusted during render rather than in an effect (the React-sanctioned
  // "state derived from a change" pattern): `syncedBucket` records the
  // section we've already snapped to, so this fires once per resolved deep
  // link and a reviewer who then picks a different tab isn't dragged back.
  const deepLinkBucket = deepLinkCandidate.data?.bucket;
  const [syncedBucket, setSyncedBucket] = useState<ReviewBucket | null>(null);
  if (deepLinkActive && deepLinkBucket && deepLinkBucket !== syncedBucket) {
    setSyncedBucket(deepLinkBucket);
    setFilters((f) =>
      f.bucket === deepLinkBucket ? f : { ...f, bucket: deepLinkBucket, page: 1 },
    );
  }

  // Leaving deep-link mode — either the reviewer clicked "back to the full
  // queue" or picked a different candidate from the list.
  const exitDeepLink = (nextSelectedKey: string | null) => {
    setDeepLinkActive(false);
    setSelectedKey(nextSelectedKey);
    router.replace("/review");
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
        // Dismissing a pair the pipeline had merged splits the entity — say
        // so, since that's a bigger change than the reviewer just pressed a
        // button for, and it's undoable from the audit log.
        onSuccess: (res) =>
          flash(
            res.unmerged_to_mid
              ? `Marked not a match — ${patidB} split out into ${res.unmerged_to_mid}.`
              : "Marked not a match.",
          ),
        onError: (err) =>
          flash(err instanceof ApiError ? err.message : "Dismiss failed."),
      },
    );
  };

  return (
    <div>
      <div className="mb-5">
        <h2 className="text-[22px] font-extrabold text-ink-2">Review Queue</h2>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[400px_minmax(0,1fr)]">
        <ReviewQueueList
          filters={filters}
          onFiltersChange={setFilters}
          items={items}
          total={data?.total ?? 0}
          bucketCounts={data?.bucket_counts ?? {}}
          isLoading={isLoading}
          isError={isError}
          selectedKey={
            deepLinkActive
              ? deepLinkCandidate.data
                ? keyOf(deepLinkCandidate.data)
                : null
              : selected
                ? keyOf(selected)
                : null
          }
          onSelect={(key) => exitDeepLink(key)}
        />

        {deepLinkActive ? (
          <DeepLinkPanel
            candidate={deepLinkCandidate}
            onMerge={(mid, patids) => setPendingMerge({ mid, patids })}
            onDismiss={handleDismiss}
            dismissPending={dismissMutation.isPending}
            onExit={() => exitDeepLink(null)}
          />
        ) : selected ? (
          <ReviewCandidateDetail
            key={keyOf(selected)}
            item={selected}
            onMerge={(mid, patids) => setPendingMerge({ mid, patids })}
            onDismiss={handleDismiss}
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

/** The right-hand panel while a deep link (`?patid_a=&patid_b=`) is active —
 * renders straight from `useReviewCandidate`, independent of whatever page
 * or filter the list beside it happens to be on. Three states: still
 * resolving, found (the normal detail view plus a way back), or resolved to
 * nothing — the pair was dismissed, already merged elsewhere, or came from
 * a run that's since aged out of the index. */
function DeepLinkPanel({
  candidate,
  onMerge,
  onDismiss,
  dismissPending,
  onExit,
}: {
  candidate: { data?: ReviewQueueItem | null; isLoading: boolean };
  onMerge: (mid: string, patids: string[]) => void;
  onDismiss: (patidA: string, patidB: string) => void;
  dismissPending: boolean;
  onExit: () => void;
}) {
  if (candidate.isLoading) {
    return (
      <div className="card flex h-[calc(100vh-220px)] min-h-[520px] items-center justify-center p-6 text-center text-sm text-gray">
        Loading the linked candidate…
      </div>
    );
  }

  if (!candidate.data) {
    return (
      <div className="card flex h-[calc(100vh-220px)] min-h-[520px] flex-col items-center justify-center gap-3 p-6 text-center text-sm text-gray">
        <p>
          This pair isn&apos;t in the review queue anymore — it may have been
          dismissed, already merged elsewhere, or come from a run that&apos;s
          since aged out.
        </p>
        <button
          onClick={onExit}
          className="rounded-md border border-line px-3 py-1.5 text-xs font-semibold text-gray-2 hover:bg-bg"
        >
          Back to the full queue
        </button>
      </div>
    );
  }

  return (
    <div>
      <button
        onClick={onExit}
        className="mb-2.5 text-[11px] font-semibold text-brand-blue hover:underline"
      >
        ← Back to the full queue
      </button>
      <ReviewCandidateDetail
        key={keyOf(candidate.data)}
        item={candidate.data}
        onMerge={onMerge}
        onDismiss={onDismiss}
        dismissPending={dismissPending}
      />
    </div>
  );
}
