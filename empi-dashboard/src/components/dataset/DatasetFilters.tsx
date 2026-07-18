"use client";

import type { RecordsFilters } from "@/lib/api-client";

/** Patient Registry search/filter — the registry is read-mostly (final,
 * resolved clusters only), so this stays to just what a data steward needs
 * to locate one patient: name/ID, birthdate, SSN. Match/merge status and
 * confidence filters live on the Review Queue tab instead, where they're
 * actually actionable. */
export function DatasetFilters({
  value,
  onChange,
}: {
  value: RecordsFilters;
  onChange: (next: RecordsFilters) => void;
}) {
  const set = (patch: Partial<RecordsFilters>) =>
    onChange({ ...value, ...patch, page: 1 });

  return (
    <div className="card mb-4 flex flex-wrap items-end gap-3 p-4">
      <Field label="Search (ID, name)">
        <input
          type="text"
          placeholder="Master Patient ID or name…"
          value={value.search ?? ""}
          onChange={(e) => set({ search: e.target.value || undefined })}
          className="w-56 rounded-md border border-line px-2.5 py-1.5 text-sm outline-none focus:border-brand-blue"
        />
      </Field>
      <Field label="Birthdate">
        <input
          type="date"
          value={value.birth_date ?? ""}
          onChange={(e) => set({ birth_date: e.target.value || undefined })}
          className="rounded-md border border-line px-2.5 py-1.5 text-sm outline-none focus:border-brand-blue"
        />
      </Field>
      <Field label="SSN last 4">
        <input
          type="text"
          maxLength={4}
          placeholder="1234"
          value={value.ssn_last4 ?? ""}
          onChange={(e) => set({ ssn_last4: e.target.value || undefined })}
          className="w-20 rounded-md border border-line px-2.5 py-1.5 text-sm outline-none focus:border-brand-blue"
        />
      </Field>
      <button
        onClick={() => onChange({ page: 1, page_size: value.page_size })}
        className="rounded-md border border-line px-3 py-1.5 text-sm font-semibold text-gray-2 hover:bg-bg"
      >
        Clear filters
      </button>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] font-bold tracking-wide text-gray uppercase">
        {label}
      </span>
      {children}
    </label>
  );
}
