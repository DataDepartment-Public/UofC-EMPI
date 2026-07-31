import type { Origin } from "@/lib/schemas";

const STYLES: Record<Origin, { label: string; bg: string; fg: string }> = {
  deterministic: { label: "Auto-match", bg: "#e9f6e9", fg: "#3f8a3f" },
  merge: { label: "Manually merged", bg: "#e3f1fb", fg: "var(--brand-blue)" },
  review: { label: "Needs review", bg: "#e0f5f4", fg: "#0a7a78" },
  none: { label: "No match", bg: "#e0f4fa", fg: "#0a6a8a" },
};

/** FR-21/22: "Match Status / Row Origin" pill — one fixed style per origin,
 * reusing the FR-13 color family (green/gold/red) plus blue for a
 * reviewer-confirmed manual merge. */
export function StatusBadge({ origin }: { origin: Origin }) {
  const s = STYLES[origin];
  return (
    <span
      className="inline-block rounded-full px-2.5 py-0.5 text-[11px] font-bold"
      style={{ background: s.bg, color: s.fg }}
    >
      {s.label}
    </span>
  );
}
