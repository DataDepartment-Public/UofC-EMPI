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

export function formatPct(v: number): string {
  return `${v.toFixed(1)}%`;
}
