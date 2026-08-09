"use client";

import { useState } from "react";
import { useQueries } from "@tanstack/react-query";
import clsx from "clsx";
import { api } from "@/lib/api-client";
import { compareRawFieldsMulti } from "@/lib/compare";
import type { RawRecord } from "@/lib/schemas";

/** The un-scrubbed source fields of every record in a cluster, in the same
 * columns as the cleaned table above it — the N-record counterpart of the
 * Review Queue's `RawComparisonPanel`.
 *
 * Collapsed by default and fetching nothing until opened, because every
 * successful `GET /records/{patid}/raw` writes a `view_raw` row to
 * `audit_log`. A reviewer scanning the registry shouldn't leave a trail of
 * PHI accesses they never asked for.
 *
 * `useQueries` rather than a `useRawRecord` per member: cluster size is
 * data-dependent, so a hook per record would be a hook loop. The query key
 * is the same `["raw", patid]` the Review Queue uses, so a record fetched
 * on either page is already cached for the other.
 */
export function ClusterRawPanel({ patids }: { patids: string[] }) {
  const [open, setOpen] = useState(false);

  const results = useQueries({
    queries: patids.map((patid) => ({
      queryKey: ["raw", patid],
      queryFn: () => api.getRaw(patid),
      enabled: open,
    })),
  });

  // "Still working on it" has to cover the gaps between retry attempts too,
  // where a query is pending but momentarily `fetchStatus: "idle"`. Keying
  // off `isPending` alone catches both, and it matters: without it a record
  // whose fetch is mid-backoff renders as a column of "(missing)", which
  // reads as "the source system had nothing" rather than "we haven't asked
  // successfully yet" — the two mean opposite things to a data steward.
  const loading = open && results.some((r) => r.isPending);
  const failed = patids.filter((_, i) => results[i]?.isError);
  const rows = compareRawFieldsMulti(
    results.map((r) => (r.data as RawRecord | undefined)?.fields),
  );

  return (
    <div className="mt-3 overflow-hidden rounded-md border border-line">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-3.5 py-2.5 text-left hover:bg-bg"
      >
        <span className="text-[13px] font-bold text-ink-2">
          <span className="mr-1.5 text-gray">{open ? "▾" : "▸"}</span>
          Compare raw source data
        </span>
        <span className="text-[11px] font-semibold text-gray">
          {open
            ? "Hide"
            : `Show all ${patids.length} records' un-scrubbed source fields`}
        </span>
      </button>

      {open && (
        <div className="border-t border-line px-3.5 py-3">
          <p className="mb-3 rounded-md bg-bg px-3 py-2 text-xs text-gray-2">
            Un-scrubbed, source-system fields for every record in this cluster
            — for data-steward review only. Not the reviewer-facing display
            values, so a difference here can still be an agreement after
            cleaning.
          </p>

          {loading && <p className="text-sm text-gray">Loading…</p>}

          {!loading && failed.length > 0 && (
            <p className="mb-3 text-xs font-semibold text-status-nomatch">
              Couldn&apos;t load raw source data for {failed.join(", ")} — those
              columns are blank below, not empty at the source.
            </p>
          )}

          {!loading && rows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-[13px]">
                <thead>
                  <tr className="border-b border-line text-left">
                    <th className="sticky left-0 z-10 min-w-[160px] bg-card px-3 py-2 text-[11px] font-bold tracking-wide text-gray uppercase">
                      Field
                    </th>
                    {patids.map((patid, i) => (
                      <th key={patid} className="min-w-[180px] px-3 py-2 align-bottom">
                        <div className="text-[11px] font-bold tracking-wide text-gray uppercase">
                          Record {i + 1}
                        </div>
                        <div className="font-mono text-[10px] font-normal normal-case text-gray">
                          {patid}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.label} className="border-b border-line last:border-none">
                      <td className="sticky left-0 z-10 bg-card px-3 py-2 align-top font-mono text-[12px] font-bold break-all text-ink-2">
                        {row.label}
                      </td>
                      {row.values.map((value, i) => (
                        <td
                          key={patids[i]}
                          className={clsx(
                            "px-3 py-2 align-top font-mono text-[12px] break-all",
                            value === "(missing)" ? "text-gray" : "text-gray-2",
                          )}
                        >
                          {value}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
