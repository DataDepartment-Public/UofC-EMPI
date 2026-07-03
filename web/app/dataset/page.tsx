"use client";

import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { ApiError, RecordsFilters } from "@/lib/api-client";
import { useMergeMutation, useRecords, useUnmergeMutation } from "@/lib/hooks";
import { AuditLog } from "@/components/AuditLog";
import { DatasetFilters } from "@/components/DatasetFilters";
import { DatasetRow } from "@/components/DatasetRow";
import { MergeModal } from "@/components/MergeModal";
import { RawDataDrawer } from "@/components/RawDataDrawer";
import { Toast } from "@/components/Toast";

const PAGE_SIZE = 25;

export default function DatasetPage() {
  return (
    <Suspense fallback={<p className="text-sm text-gray">Loading…</p>}>
      <DatasetPageContent />
    </Suspense>
  );
}

function DatasetPageContent() {
  const searchParams = useSearchParams();
  const [filters, setFilters] = useState<RecordsFilters>({
    page: 1,
    page_size: PAGE_SIZE,
    search: searchParams.get("search") ?? undefined,
  });
  const [rawPatid, setRawPatid] = useState<string | null>(null);
  const [pendingMerge, setPendingMerge] = useState<{
    mid: string;
    patids: string[];
  } | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const { data, isLoading, isError } = useRecords(filters);
  const mergeMutation = useMergeMutation();
  const unmergeMutation = useUnmergeMutation();

  const flash = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3200);
  };

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

  const handleUnmerge = (mid: string, patid: string) => {
    unmergeMutation.mutate(
      { mid, patid },
      {
        onSuccess: (res) => flash(`${patid} split into ${res.new_mid}.`),
        onError: (err) =>
          flash(err instanceof ApiError ? err.message : "Unmerge failed."),
      },
    );
  };

  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;
  const page = filters.page ?? 1;

  return (
    <div>
      <div className="mb-5">
        <h2 className="text-[22px] font-extrabold text-ink-2">Dataset</h2>
        <p className="mt-1 text-[13px] text-gray">
          Every master patient record — automatically matched, manually
          reviewed, manually merged, or unmatched. Expand a row to review
          candidates and take action.
        </p>
      </div>

      <DatasetFilters value={filters} onChange={setFilters} />

      {isLoading && <p className="text-sm text-gray">Loading records…</p>}
      {isError && (
        <p className="text-sm text-status-nomatch">
          Couldn&apos;t reach the eMPI API. Is the backend running?
        </p>
      )}

      {data && (
        <>
          <div className="mb-2 flex items-center justify-between text-xs text-gray">
            <span>
              {data.total.toLocaleString()} record{data.total === 1 ? "" : "s"}
            </span>
            <span>
              Page {page} of {totalPages}
            </span>
          </div>

          {data.items.length === 0 ? (
            <p className="card p-6 text-center text-sm text-gray">
              No records match these filters.
            </p>
          ) : (
            <>
              <div className="mb-1.5 grid grid-cols-[20px_1.4fr_1fr_0.9fr_0.9fr_1fr_1.3fr_1fr] gap-3 px-4 text-[10px] font-bold tracking-wide text-gray uppercase">
                <span />
                <span>Patient name</span>
                <span>Master patient ID</span>
                <span>Masked SSN</span>
                <span>Birthdate</span>
                <span>Match status</span>
                <span>Key matching features</span>
                <span>Last updated</span>
              </div>
              {data.items.map((entity) => (
                <DatasetRow
                  key={entity.mid}
                  entity={entity}
                  onMerge={(mid, patids) => setPendingMerge({ mid, patids })}
                  onUnmerge={handleUnmerge}
                  onViewRaw={setRawPatid}
                />
              ))}
            </>
          )}

          <div className="mt-4 flex justify-center gap-2">
            <button
              disabled={page <= 1}
              onClick={() => setFilters((f) => ({ ...f, page: page - 1 }))}
              className="rounded-md border border-line px-3 py-1.5 text-sm font-semibold text-gray-2 disabled:opacity-40 hover:bg-bg"
            >
              ← Prev
            </button>
            <button
              disabled={page >= totalPages}
              onClick={() => setFilters((f) => ({ ...f, page: page + 1 }))}
              className="rounded-md border border-line px-3 py-1.5 text-sm font-semibold text-gray-2 disabled:opacity-40 hover:bg-bg"
            >
              Next →
            </button>
          </div>
        </>
      )}

      <div className="mt-8">
        <AuditLog />
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
