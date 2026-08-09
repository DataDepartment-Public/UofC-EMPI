"use client";

import { useState } from "react";

/** "Page [n] of N", where the n is typeable so a reviewer can jump straight to
 * page 13 instead of clicking Next twelve times. The draft lives in local state
 * while typing (so a half-typed "1" of "13" doesn't fire a fetch) and commits on
 * Enter or blur, clamped into range.
 *
 * Shared by the Review Queue and the Patient Registry — both sit next to their
 * own Prev/Next buttons, so the indicator is where the paging controls are. */
export function PageJumper({
  page,
  totalPages,
  onJump,
  className,
  inputClassName,
}: {
  page: number;
  totalPages: number;
  onJump: (page: number) => void;
  className?: string;
  inputClassName?: string;
}) {
  const [draft, setDraft] = useState(String(page));

  // Prev/Next, a filter change resetting to page 1 — anything that moves the
  // page from outside has to pull the box along with it. Adjusted during render
  // rather than in an effect so the box never paints a stale number.
  const [lastPage, setLastPage] = useState(page);
  if (lastPage !== page) {
    setLastPage(page);
    setDraft(String(page));
  }

  const commit = () => {
    const parsed = Number.parseInt(draft, 10);
    if (Number.isNaN(parsed)) {
      setDraft(String(page));
      return;
    }
    const clamped = Math.min(Math.max(parsed, 1), totalPages);
    setDraft(String(clamped));
    if (clamped !== page) onJump(clamped);
  };

  return (
    <span className={className ?? "flex items-center gap-1 text-[11px] text-gray"}>
      Page
      <input
        type="text"
        inputMode="numeric"
        aria-label={`Page number, ${totalPages} pages total`}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onFocus={(e) => e.target.select()}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.currentTarget.blur();
          } else if (e.key === "Escape") {
            setDraft(String(page));
            e.currentTarget.blur();
          }
        }}
        className={
          inputClassName ??
          "w-10 rounded-md border border-line px-1 py-0.5 text-center font-mono text-[11px] tabular-nums text-ink-2 outline-none focus:border-brand-blue"
        }
      />
      of {totalPages}
    </span>
  );
}
