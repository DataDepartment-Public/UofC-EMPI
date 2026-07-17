"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";
import { compareRecords } from "@/lib/compare";
import type { ExplainPatient } from "@/lib/explain";
import { fullName, maskSsn, formatDate } from "@/lib/format";
import { FeatureComparisonTable } from "./FeatureComparisonTable";

/** "Find a match manually" — search for and propose a match between two
 * records blocking never paired, not just merge auto-suggested candidates.
 * Reuses `compareRecords`/`FeatureComparisonTable` for the preview and hands
 * off to the caller's existing `onMerge` (the same callback `DatasetRow`
 * already wires to `MergeModal`'s confirmation) rather than building a
 * second confirm step. */
export function ManualMatchModal({
  sourceMid,
  anchor,
  onMerge,
  onClose,
}: {
  sourceMid: string;
  anchor: ExplainPatient;
  onMerge: (mid: string, patids: string[]) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<ExplainPatient | null>(null);

  const { data, isFetching } = useQuery({
    queryKey: ["manual-match-search", query],
    queryFn: () => api.listRecords({ search: query, page: 1, page_size: 8 }),
    enabled: query.trim().length >= 2,
  });

  const results = (data?.items ?? []).filter((e) => e.mid !== sourceMid);
  const rows = selected ? compareRecords(anchor, selected) : [];

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/30 p-4">
      <div className="card flex max-h-[85vh] w-full max-w-2xl flex-col p-0">
        <div className="flex items-center justify-between border-b border-line px-5 py-4">
          <h3 className="text-base font-bold text-ink-2">
            Find a match manually
          </h3>
          <button
            onClick={onClose}
            className="rounded-full px-2 py-1 text-lg text-gray hover:bg-bg"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="overflow-y-auto px-5 py-4">
          <p className="mb-3 text-[13px] text-gray-2">
            Comparing against{" "}
            <span className="font-semibold text-ink-2">
              {fullName(anchor.first_name, anchor.last_name)}
            </span>
            . Search for a record blocking didn&apos;t surface as a candidate.
          </p>

          <input
            autoFocus
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelected(null);
            }}
            placeholder="Name, birthdate, or master patient ID…"
            className="w-full rounded-md border border-line px-3 py-2 text-sm outline-none focus:border-brand-blue"
          />

          {!selected && (
            <div className="mt-3 space-y-1.5">
              {isFetching && (
                <p className="text-xs text-gray">Searching…</p>
              )}
              {!isFetching && query.trim().length >= 2 && results.length === 0 && (
                <p className="text-xs text-gray">No other records match.</p>
              )}
              {results.map((e) => {
                const primary = e.members.find((m) => m.is_primary) ?? e.members[0];
                if (!primary) return null;
                return (
                  <button
                    key={e.mid}
                    onClick={() =>
                      setSelected({
                        patid: primary.patid,
                        first_name: primary.first_name ?? null,
                        last_name: primary.last_name ?? null,
                        birth_date: primary.birth_date ?? null,
                        ssn_last4: primary.ssn_last4 ?? null,
                        email: primary.email ?? null,
                        zip_code: primary.zip_code ?? null,
                        address1: primary.address1 ?? null,
                        sex: primary.sex ?? null,
                        phone: primary.phone ?? null,
                      })
                    }
                    className="flex w-full items-center justify-between rounded-md border border-line px-3 py-2 text-left text-[13px] hover:border-brand-blue hover:bg-bg"
                  >
                    <span className="font-semibold text-ink-2">
                      {fullName(primary.first_name, primary.last_name)}
                    </span>
                    <span className="flex gap-3 text-xs text-gray-2">
                      <span>{primary.birth_date ?? "—"}</span>
                      <span>{maskSsn(primary.ssn_last4)}</span>
                      <span className="font-mono text-gray">{e.mid}</span>
                      <span>{formatDate(e.updated_utc)}</span>
                    </span>
                  </button>
                );
              })}
            </div>
          )}

          {selected && (
            <div className="mt-4">
              <div className="mb-2 flex items-center justify-between">
                <h4 className="text-[13px] font-bold text-ink-2">
                  Comparing with {fullName(selected.first_name, selected.last_name)}
                </h4>
                <button
                  onClick={() => setSelected(null)}
                  className="text-xs font-semibold text-brand-blue hover:underline"
                >
                  ‹ Back to search
                </button>
              </div>
              <FeatureComparisonTable
                rows={rows}
                patidA={anchor.patid}
                patidB={selected.patid}
              />
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2.5 border-t border-line px-5 py-3.5">
          <button
            onClick={onClose}
            className="rounded-md border border-line px-4 py-2 text-sm font-semibold text-gray-2 hover:bg-bg"
          >
            Cancel
          </button>
          <button
            disabled={!selected}
            onClick={() => {
              if (!selected) return;
              onMerge(sourceMid, [selected.patid]);
              onClose();
            }}
            className="rounded-md bg-brand-blue px-4 py-2 text-sm font-semibold text-white hover:bg-brand-blue-bright disabled:opacity-40"
          >
            Propose merge
          </button>
        </div>
      </div>
    </div>
  );
}
