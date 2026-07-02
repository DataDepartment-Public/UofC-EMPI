"use client";

import type { RecordsFilters } from "@/lib/api-client";

/** FR-22: search/filter by Master Patient ID, name, masked SSN, birthdate,
 * match status, merge status, last-updated date. */
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
      <Field label="Match status">
        <select
          value={value.origin ?? ""}
          onChange={(e) => set({ origin: e.target.value || undefined })}
          className="rounded-md border border-line px-2.5 py-1.5 text-sm outline-none focus:border-brand-blue"
        >
          <option value="">All</option>
          <option value="deterministic">Auto-match</option>
          <option value="review">Needs review</option>
          <option value="merge">Manually merged</option>
          <option value="none">No match</option>
        </select>
      </Field>
      <Field label="Merge status">
        <select
          value={
            value.is_merged === undefined ? "" : value.is_merged ? "merged" : "unmerged"
          }
          onChange={(e) =>
            set({
              is_merged:
                e.target.value === "" ? undefined : e.target.value === "merged",
            })
          }
          className="rounded-md border border-line px-2.5 py-1.5 text-sm outline-none focus:border-brand-blue"
        >
          <option value="">All</option>
          <option value="merged">Merged</option>
          <option value="unmerged">Unmerged</option>
        </select>
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
