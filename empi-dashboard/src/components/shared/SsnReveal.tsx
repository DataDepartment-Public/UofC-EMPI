"use client";

import { useState } from "react";
import { useCleanSsn } from "@/lib/hooks";

/** Reveal-in-place toggle for a masked SSN. Sourced from `cleaned_attrs.ssn`
 * (`GET /records/:patid/ssn-clean`) — the pipeline-normalized value blocking
 * and the deterministic rules actually matched on — rather than `SSN_raw`,
 * which is the un-scrubbed source value and may be junk `clean_ssn` rejected
 * outright. Fetch is lazy (`enabled` only once revealed) and cached by
 * react-query, so toggling back and forth doesn't re-fetch. */
export function SsnReveal({ patid, masked }: { patid: string; masked: string }) {
  const [revealed, setRevealed] = useState(false);
  const { data, isLoading, isError } = useCleanSsn(revealed ? patid : null);

  const cleanSsn = data?.ssn;
  const fullSsn =
    cleanSsn != null && cleanSsn !== "" ? String(cleanSsn) : null;

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
