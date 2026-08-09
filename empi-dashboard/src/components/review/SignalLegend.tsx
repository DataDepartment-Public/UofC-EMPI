"use client";

import { useState } from "react";
import clsx from "clsx";
import { BAND_DEFS, bandRangeLabel, type LiveThresholds } from "@/lib/pair-signal";

/** The ⓘ beside the Confidence filter: the whole band scale in one panel,
 * one short line each.
 *
 * One panel rather than a tooltip per chip — a reviewer opening this wants
 * to compare the bands against each other, not interrogate one of them.
 *
 * The two threshold-derived edges come from `useThresholds()` (live,
 * `GET /admin/thresholds`) and are never hardcoded: both are
 * operator-tunable, and a frontend constant for the merge bar was wrong the
 * moment `.env` set 0.90 while the code said 0.70.
 *
 * Opens on hover *and* on click/focus. Hover alone would leave this
 * unreachable by keyboard and dead on touch.
 *
 * POSITIONING: the panel anchors to the **filter row**, not to the ⓘ — the
 * caller's row must carry `relative`, and this component's own wrapper
 * deliberately does not. The icon sits partway into a 400px column, so a
 * panel anchored to it has no good side: `left-0` pushes it off the card's
 * right edge and `right-0` off its left. Pinned `left-0 right-0` to the row
 * instead, it is exactly as wide as the controls above it and cannot
 * overflow either way, at any column width.
 *
 * TYPOGRAPHY: the panel resets `normal-case` / `whitespace-normal` /
 * `font-normal` / `tracking-normal` explicitly. The filter label beside it
 * is an uppercase bold `whitespace-nowrap` chip, and this panel used to be
 * nested inside it — inheriting all four, which shouted every line in caps
 * and, because of `nowrap`, ran the text clean out of the box. The caller
 * now renders the ⓘ as a sibling of that label rather than a child, so
 * these resets are belt-and-braces against it drifting back in. */
export function SignalLegend({ thresholds }: { thresholds: LiveThresholds }) {
  const [open, setOpen] = useState(false);

  return (
    <span
      className="inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        aria-label="What the confidence bands mean"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
        className="flex h-4 w-4 items-center justify-center rounded-full border border-line text-[9px] font-bold text-gray-2 hover:border-brand-blue hover:text-brand-blue"
      >
        i
      </button>

      {open && (
        <div
          role="tooltip"
          className="absolute top-full right-0 left-0 z-20 mt-1 rounded-md border border-line bg-card p-2.5 font-normal tracking-normal normal-case whitespace-normal shadow-lg"
        >
          <ul className="flex flex-col gap-1.5">
            {BAND_DEFS.map((d) => (
              <li key={d.band} className="flex items-start gap-1.5">
                <span
                  className={clsx(
                    "mt-px shrink-0 rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase",
                    d.tone,
                  )}
                >
                  {d.label}
                </span>
                <span className="min-w-0 flex-1 break-words text-[10px] leading-snug text-gray-2">
                  {bandRangeLabel(d.band, thresholds)}
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-2 border-t border-line pt-1.5 text-[10px] leading-snug text-gray">
            Low ≠ non-match — the matcher only scores confidence <em>in</em> a
            match.
          </p>
        </div>
      )}
    </span>
  );
}
