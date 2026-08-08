export function maskSsn(last4: string | null | undefined): string {
  return last4 ? `***-**-${last4}` : "—";
}

export function fullName(
  first?: string | null,
  last?: string | null,
): string {
  const parts = [first, last].filter(Boolean);
  return parts.length ? parts.join(" ") : "(no name on file)";
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/** Strip the time component off a raw source-system date for display.
 *
 * Source exports carry a meaningless midnight time on date-only fields
 * (`1974-02-12 00:00:00`, `2/12/1974 12:00:00 AM`), which is noise on a
 * birthdate. Deliberately string-level: re-parsing into a `Date` and
 * reformatting would shift a `00:00:00` UTC timestamp back a day in any
 * negative-offset timezone, silently displaying the wrong birthdate. Any
 * value that doesn't end in a time is returned untouched, so unparseable
 * raw values stay verbatim. */
export function formatRawDate(value: unknown): string {
  if (value == null || value === "") return "—";
  const s = String(value).trim();
  return s.replace(/[\sT]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?\s*([AP]\.?M\.?)?\s*(Z|[+-]\d{2}:?\d{2})?$/i, "");
}

export function formatPct(v: number): string {
  return `${v.toFixed(1)}%`;
}
