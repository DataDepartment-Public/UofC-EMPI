"use client";

import { useState } from "react";
import { useRawRecord } from "@/lib/hooks";

/** Reveal-in-place toggle for a masked SSN. Sourced from the same un-scrubbed
 * raw-record data the "View raw data" drawer already fetches
 * (`GET /records/:patid/raw`, `RawDataDrawer.tsx`) — no new PII exposure
 * surface, just a faster affordance than opening the full drawer. Fetch is
 * lazy (`enabled` only once revealed) and cached by react-query, so toggling
 * back and forth doesn't re-fetch. */
export function SsnReveal({ patid, masked }: { patid: string; masked: string }) {
  const [revealed, setRevealed] = useState(false);
  const { data, isLoading, isError } = useRawRecord(revealed ? patid : null);

  const rawSsn = data ? data.fields["SSN_raw"] : undefined;
  const fullSsn =
    rawSsn != null && rawSsn !== "" ? String(rawSsn) : null;

  let display = masked;
  if (revealed) {
    display = isLoading
      ? "Loading…"
      : fullSsn ?? (isError ? masked : "(not on file)");
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={revealed && fullSsn ? "font-mono" : undefined}>
        {display}
      </span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
          setRevealed((r) => !r);
        }}
        className="text-[10px] font-semibold text-brand-blue hover:underline"
        title={revealed ? "Hide SSN" : "Reveal full SSN"}
      >
        {revealed ? "hide" : "reveal"}
      </button>
    </span>
  );
}
