"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ApiError, RecordsFilters } from "@/lib/api-client";
import { useRecords, useUnmergeMutation } from "@/lib/hooks";
import { DatasetFilters } from "@/components/dataset/DatasetFilters";
import { DatasetRow } from "@/components/dataset/DatasetRow";
import { UnmergeModal } from "@/components/dataset/UnmergeModal";
import { AuditLog } from "@/components/shared/AuditLog";
import { RawDataDrawer } from "@/components/shared/RawDataDrawer";
import { Toast } from "@/components/shared/Toast";

const PAGE_SIZE = 25;

// The registry shows only resolved, final clusters — auto-matched,
// manually merged, or standalone with nothing pending. Anything still
// awaiting a decision (`origin: "review"`) belongs on the Review Queue tab,
// not here.
const FINAL_ORIGINS = "deterministic,merge,none";

/** Filters live in the URL (not just component state) so that navigating
 * away — e.g. to a patient's Model Explanation page — and back with the
 * browser's own back button, or a plain page refresh, lands on the exact
 * same search/filter/page a reviewer had open, instead of a freshly reset
 * Patient Registry. */
function filtersFromSearchParams(searchParams: URLSearchParams): RecordsFilters {
  return {
    page: Number(searchParams.get("page")) || 1,
    page_size: PAGE_SIZE,
    search: searchParams.get("search") ?? undefined,
    birth_date: searchParams.get("birth_date") ?? undefined,
    ssn_last4: searchParams.get("ssn_last4") ?? undefined,
  };
}

function filtersToSearch(filters: RecordsFilters): string {
  const params = new URLSearchParams();
  if (filters.search) params.set("search", filters.search);
  if (filters.birth_date) params.set("birth_date", filters.birth_date);
  if (filters.ssn_last4) params.set("ssn_last4", filters.ssn_last4);
  if (filters.page && filters.page !== 1) params.set("page", String(filters.page));
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
  const [filters, setFiltersState] = useState<RecordsFilters>(() =>
    filtersFromSearchParams(searchParams),
  );

  const setFilters = (
    next: RecordsFilters | ((prev: RecordsFilters) => RecordsFilters),
  ) => {
    setFiltersState((prev) => {
      const resolved = typeof next === "function" ? next(prev) : next;
      const qs = filtersToSearch(resolved);
      router.replace(qs ? `/dataset?${qs}` : "/dataset", { scroll: false });
      return resolved;
    });
  };

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
              <div className="mb-1.5 grid grid-cols-[20px_1.5fr_0.9fr_1fr_1fr_0.8fr_1fr] gap-3 px-4 text-[10px] font-bold tracking-wide text-gray uppercase">
                <span />
                <span>Patient name</span>
                <span>Match status</span>
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

      <div className="mt-8">
        <AuditLog onFlash={flash} />
      </div>

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
