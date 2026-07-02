export function KpiCard({
  label,
  value,
  accent = "var(--brand-blue)",
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div
      className="card relative overflow-hidden p-4"
      style={{ borderLeftWidth: 4, borderLeftColor: accent }}
    >
      <div className="text-[25px] font-extrabold tracking-tight text-ink-2">
        {value}
      </div>
      <div className="mt-1 text-[11px] font-semibold leading-tight text-gray">
        {label}
      </div>
    </div>
  );
}

export function KpiGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-2 gap-3.5 md:grid-cols-4">{children}</div>
  );
}
