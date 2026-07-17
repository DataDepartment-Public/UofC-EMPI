"use client";

import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { ApiError, RecordsFilters } from "@/lib/api-client";
import { useRecords, useUnmergeMutation } from "@/lib/hooks";
import { DatasetFilters } from "@/components/DatasetFilters";
import { DatasetRow } from "@/components/DatasetRow";
import { UnmergeModal } from "@/components/UnmergeModal";
import { RawDataDrawer } from "@/components/RawDataDrawer";
import { Toast } from "@/components/Toast";

const PAGE_SIZE = 25;

// The registry shows only resolved, final clusters — auto-matched,
// manually merged, or standalone with nothing pending. Anything still
// awaiting a decision (`origin: "review"`) belongs on the Review Queue tab,
// not here.
const FINAL_ORIGINS = "deterministic,merge,none";

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
  const [pendingUnmerge, setPendingUnmerge] = useState<{
    mid: string;
    patid: string;
    patientName: string;
  } | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const { data, isLoading, isError } = useRecords({
    ...filters,
    origin: FINAL_ORIGINS,
  });
  const unmergeMutation = useUnmergeMutation();

  const flash = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3200);
  };

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

  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;
  const page = filters.page ?? 1;

  return (
    <div>
      <div className="mb-5">
        <h2 className="text-[22px] font-extrabold text-ink-2">Patient Registry</h2>
        <p className="mt-1 text-[13px] text-gray">
          The final, resolved patient list — one row per distinct patient.
          Records still awaiting a match decision live on the Review Queue tab.
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
              {data.total.toLocaleString()} patient{data.total === 1 ? "" : "s"}
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
              <div className="mb-1.5 grid grid-cols-[20px_1.7fr_1fr_1fr_0.8fr_1fr] gap-3 px-4 text-[10px] font-bold tracking-wide text-gray uppercase">
                <span />
                <span>Patient name</span>
                <span>Masked SSN</span>
                <span>Birthdate</span>
                <span># of entries</span>
                <span>Last updated</span>
              </div>
              {data.items.map((entity) => (
                <DatasetRow
                  key={entity.mid}
                  entity={entity}
                  onUnmerge={(mid, patid, patientName) =>
                    setPendingUnmerge({ mid, patid, patientName })
                  }
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

      <RawDataDrawer patid={rawPatid} onClose={() => setRawPatid(null)} />

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
