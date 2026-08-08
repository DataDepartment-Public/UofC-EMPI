"use client";

import { useRawRecord } from "@/lib/hooks";
import { formatRawField } from "@/lib/format";

/** FR-24: "View Raw Data" side drawer — the un-scrubbed source-system
 * fields (src/api/store.py `record_raw`), for data-steward audits.
 *
 * Single-record only, which is why the Review Queue uses
 * `RawComparisonPanel` instead: deciding a *pair* means reading both sides'
 * raw fields against each other. This drawer still backs the Patient
 * Registry, where one record at a time is the actual question. */
export function RawDataDrawer({
  patid,
  onClose,
}: {
  patid: string | null;
  onClose: () => void;
}) {
  const { data, isLoading, isError } = useRawRecord(patid);

  if (!patid) return null;

  return (
    <div className="fixed inset-0 z-[100] flex justify-end bg-black/30" onClick={onClose}>
      <div
        className="h-full w-full max-w-md overflow-y-auto bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h3 className="text-base font-bold text-ink-2">Raw source data</h3>
            <p className="mt-0.5 font-mono text-xs text-gray">{patid}</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-full px-2 py-1 text-lg text-gray hover:bg-bg"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <p className="mb-4 rounded-md bg-bg px-3 py-2 text-xs text-gray-2">
          Un-scrubbed, source-system fields — for data-steward review only.
          Not the reviewer-facing display values.
        </p>

        {isLoading && <p className="text-sm text-gray">Loading…</p>}
        {isError && (
          <p className="text-sm text-status-nomatch">
            No raw payload published for this record.
          </p>
        )}
        {data && (
          <dl className="divide-y divide-line">
            {Object.entries(data.fields).map(([key, value]) => (
              <div key={key} className="flex justify-between gap-3 py-2 text-[13px]">
                <dt className="font-semibold text-gray">{key}</dt>
                <dd className="text-right font-mono text-ink-2">
                  {formatRawField(key, value)}
                </dd>
              </div>
            ))}
          </dl>
        )}
      </div>
    </div>
  );
}
