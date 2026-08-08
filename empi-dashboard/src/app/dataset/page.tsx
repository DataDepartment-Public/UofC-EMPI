"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ApiError, RecordsFilters } from "@/lib/api-client";
import { useCluster, useRecords, useUnmergeMutation } from "@/lib/hooks";
import { ClusterDetail } from "@/components/dataset/ClusterDetail";
import { ClusterList } from "@/components/dataset/ClusterList";
import { UnmergeModal } from "@/components/dataset/UnmergeModal";
import { AuditLog } from "@/components/shared/AuditLog";
import { Toast } from "@/components/shared/Toast";

const PAGE_SIZE = 25;

// The registry shows only resolved, final clusters — auto-matched,
// manually merged, or standalone with nothing pending. Anything still
// awaiting a decision (`origin: "review"`) belongs on the Review Queue tab,
// not here.
const FINAL_ORIGINS = "deterministic,merge,none";

/** Filters *and* the selected cluster live in the URL so a reviewer can link
 * a colleague straight to the cluster they're questioning, and so back/refresh
 * lands on the same page of the same list rather than a reset registry. (The
 * Review Queue keeps its selection in local state only; this page can do
 * better because a cluster has a stable id, where a pending pair does not.) */
interface UrlState {
  filters: RecordsFilters;
  mid: string | null;
  multiOnly: boolean;
}

function stateFromSearchParams(searchParams: URLSearchParams): UrlState {
  const sort = searchParams.get("sort");
  return {
    filters: {
      page: Number(searchParams.get("page")) || 1,
      page_size: PAGE_SIZE,
      search: searchParams.get("search") ?? undefined,
      birth_date: searchParams.get("birth_date") ?? undefined,
      ssn_last4: searchParams.get("ssn_last4") ?? undefined,
      sort: sort === "confidence" || sort === "name" ? sort : undefined,
    },
    mid: searchParams.get("mid"),
    // Multi-record-only is the default: a singleton cluster has nothing to
    // compare, and singletons are ~90% of the index.
    multiOnly: searchParams.get("all") !== "1",
  };
}

function stateToSearch({ filters, mid, multiOnly }: UrlState): string {
  const params = new URLSearchParams();
  if (filters.search) params.set("search", filters.search);
  if (filters.birth_date) params.set("birth_date", filters.birth_date);
  if (filters.ssn_last4) params.set("ssn_last4", filters.ssn_last4);
  if (filters.sort) params.set("sort", filters.sort);
  if (filters.page && filters.page !== 1) params.set("page", String(filters.page));
  if (!multiOnly) params.set("all", "1");
  if (mid) params.set("mid", mid);
  return params.toString();
}

export default function DatasetPage() {
  return (
    <Suspense fallback={<p className="text-sm text-gray">Loading…</p>}>
      <DatasetPageContent />
    </Suspense>
  );
}

function DatasetPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [state, setState] = useState<UrlState>(() =>
    stateFromSearchParams(searchParams),
  );

  // Syncing the URL is a side effect of `state` changing, not part of the
  // state update itself — a `router.replace` inside a "pure" state-updater
  // trips React's "Cannot update a component while rendering a different
  // component" warning.
  useEffect(() => {
    const qs = stateToSearch(state);
    router.replace(qs ? `/dataset?${qs}` : "/dataset", { scroll: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  const [pendingUnmerge, setPendingUnmerge] = useState<{
    mid: string;
    patid: string;
    patientName: string;
  } | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const { data, isLoading, isError } = useRecords({
    ...state.filters,
    origin: FINAL_ORIGINS,
    // `min_members`, not `is_merged`: the latter counts *unlocked* members at
    // publish time, so it hides any multi-record cluster a reviewer has
    // already touched — 168 of 762 in the current index.
    min_members: state.multiOnly ? 2 : undefined,
  });
  const unmergeMutation = useUnmergeMutation();

  const flash = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3200);
  };

  const items = data?.items ?? [];
  const onThisPage = items.find((e) => e.mid === state.mid) ?? null;

  // A requested `mid` usually sits in the current page of results, but not
  // always: a link shared with a colleague, or a jump out of the comparison
  // history into the cluster that a record was declined against, names a mid
  // that can be on page 40 of 237 — or excluded by the active filters
  // entirely. Falling back to `items[0]` there silently shows the wrong
  // patient, which is worse than slow. So fetch it directly when it isn't in
  // hand; `useCluster` hits `GET /clusters/{mid}` and is skipped whenever the
  // list already has the answer.
  const needsFetch = Boolean(state.mid) && !onThisPage;
  const detached = useCluster(needsFetch ? state.mid : null);

  // Only fall back to the first row when nothing specific was asked for. If a
  // mid *was* named and simply can't be resolved, the empty state below says
  // so rather than quietly substituting a different cluster.
  const selected =
    onThisPage ?? (needsFetch ? detached.data ?? null : items[0] ?? null);
  const selectionPending = needsFetch && detached.isLoading;
  const selectionMissing = needsFetch && detached.isError;

  const confirmUnmerge = () => {
    if (!pendingUnmerge) return;
    const { mid, patid } = pendingUnmerge;
    unmergeMutation.mutate(
      { mid, patid },
      {
        onSuccess: (res) => {
          flash(`${patid} split into ${res.new_mid}.`);
          setPendingUnmerge(null);
        },
        onError: (err) => {
          flash(err instanceof ApiError ? err.message : "Unmerge failed.");
        },
      },
    );
  };

  return (
    <div>
      <div className="mb-5">
        <h2 className="text-[22px] font-extrabold text-ink-2">Patient Registry</h2>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[380px_minmax(0,1fr)]">
        <ClusterList
          filters={state.filters}
          onFiltersChange={(filters) => setState((s) => ({ ...s, filters }))}
          items={items}
          total={data?.total ?? 0}
          isLoading={isLoading}
          isError={isError}
          selectedMid={selected?.mid ?? null}
          onSelect={(mid) => setState((s) => ({ ...s, mid }))}
          multiOnly={state.multiOnly}
          onMultiOnlyChange={(multiOnly) =>
            setState((s) => ({
              ...s,
              multiOnly,
              mid: null,
              filters: { ...s.filters, page: 1 },
            }))
          }
        />

        {selected ? (
          // Remount per cluster so the detail's internal state — which raw
          // panel is open, which pair cards are expanded — resets instead of
          // carrying over onto a different patient's record.
          <ClusterDetail
            key={selected.mid}
            entity={selected}
            onUnmerge={(mid, patid, patientName) =>
              setPendingUnmerge({ mid, patid, patientName })
            }
          />
        ) : (
          <div className="card flex h-[calc(100vh-220px)] min-h-[520px] items-center justify-center p-6 text-center text-sm text-gray">
            {selectionMissing ? (
              <span>
                No cluster <span className="font-mono">{state.mid}</span>. It may
                have been split or merged away since this link was made.
              </span>
            ) : isLoading || selectionPending ? (
              "Loading…"
            ) : (
              "Select a cluster from the list to compare its records."
            )}
          </div>
        )}
      </div>

      <div className="mt-8">
        <AuditLog onFlash={flash} />
      </div>

      {pendingUnmerge && (
        <UnmergeModal
          mid={pendingUnmerge.mid}
          patid={pendingUnmerge.patid}
          patientName={pendingUnmerge.patientName}
          pending={unmergeMutation.isPending}
          onConfirm={confirmUnmerge}
          onCancel={() => setPendingUnmerge(null)}
        />
      )}

      <Toast message={toast} />
    </div>
  );
}
