"use client";

import { useState } from "react";
import { compareRawFields } from "@/lib/compare";
import { useRawRecord } from "@/lib/hooks";
import { RESULT_STYLE } from "./FeatureComparisonTable";

/** FR-24, pair form: the un-scrubbed source-system fields of *both* records
 * in a candidate pair, side by side under the cleaned-value feature
 * comparison, with the same Result column.
 *
 * This replaces the single-record `RawDataDrawer` on the review panel. A
 * reviewer deciding a pair is always asking "do these two agree?", and a
 * drawer showing one side's raw fields alone can't answer that — they had to
 * hold one record's values in their head. `RawDataDrawer` still serves the
 * Patient Registry, where the question really is about one record.
 *
 * SSN is shown in full, exactly as the drawer does: `GET /records/{patid}/raw`
 * returns the un-scrubbed source payload and audit-logs the view, which is
 * the whole purpose of the raw view for a data steward. Masking belongs to
 * the cleaned-value table above it. */
export function RawComparisonPanel({
  patidA,
  patidB,
}: {
  patidA: string;
  patidB: string;
}) {
  const [open, setOpen] = useState(false);

  // Gated on `open` so the panel fetches nothing on its own until asked —
  // every successful `GET /records/{patid}/raw` writes a `view_raw` row to
  // `audit_log`. On the review panel this costs nothing either way: sibling
  // `PipelineTrail` already fetches both records eagerly under the same
  // `["raw", patid]` key, so expanding reads straight from the react-query
  // cache with no second request and no loading flash.
  const a = useRawRecord(open ? patidA : null);
  const b = useRawRecord(open ? patidB : null);

  const loading = a.isLoading || b.isLoading;
  const rows = compareRawFields(a.data?.fields, b.data?.fields);
  const unpublished = [
    a.isError ? patidA : null,
    b.isError ? patidB : null,
  ].filter(Boolean) as string[];

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
          {open ? "Hide" : "Show both records' un-scrubbed source fields"}
        </span>
      </button>

      {open && (
        <div className="border-t border-line px-3.5 py-3">
          <p className="mb-3 rounded-md bg-bg px-3 py-2 text-xs text-gray-2">
            Un-scrubbed, source-system fields for both records — for
            data-steward review only. Not the reviewer-facing display values,
            so a difference here can still be an agreement after cleaning.
          </p>

          {loading && <p className="text-sm text-gray">Loading…</p>}

          {!loading && unpublished.length > 0 && (
            <p className="mb-3 text-xs font-semibold text-status-nomatch">
              No raw payload published for {unpublished.join(" or ")}.
            </p>
          )}

          {!loading && rows.length > 0 && (
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b border-line text-left text-[11px] font-bold tracking-wide text-gray uppercase">
                  <th className="py-2">Field</th>
                  <th className="py-2">
                    Patient A
                    <div className="font-mono text-[10px] normal-case">
                      {patidA}
                    </div>
                  </th>
                  <th className="py-2">
                    Patient B
                    <div className="font-mono text-[10px] normal-case">
                      {patidB}
                    </div>
                  </th>
                  <th className="py-2">Result</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const style = RESULT_STYLE[row.result];
                  return (
                    <tr
                      key={row.label}
                      className="border-b border-line last:border-none"
                    >
                      <td className="py-2 pr-3 align-top font-mono text-[12px] font-bold break-all text-ink-2">
                        {row.label}
                      </td>
                      <td className="py-2 pr-3 align-top font-mono text-[12px] break-all text-gray-2">
                        {row.valueA}
                      </td>
                      <td className="py-2 pr-3 align-top font-mono text-[12px] break-all text-gray-2">
                        {row.valueB}
                      </td>
                      <td className={`py-2 align-top ${style.className}`}>
                        {style.label}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
