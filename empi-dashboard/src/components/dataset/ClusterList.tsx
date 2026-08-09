"use client";

import clsx from "clsx";
import type { RecordsFilters } from "@/lib/api-client";
import type { Entity } from "@/lib/schemas";
import { fullName, maskSsn, formatRawDate } from "@/lib/format";
import { PageJumper } from "@/components/shared/PageJumper";
import { StatusBadge } from "@/components/dataset/StatusBadge";

interface Props {
  filters: RecordsFilters;
  onFiltersChange: (next: RecordsFilters) => void;
  items: Entity[];
  total: number;
  isLoading: boolean;
  isError: boolean;
  selectedMid: string | null;
  onSelect: (mid: string) => void;
  /** True when the list is restricted to clusters holding 2+ records. */
  multiOnly: boolean;
  onMultiOnlyChange: (next: boolean) => void;
}

/** Left pane of the Patient Registry: the cluster list a reviewer picks from.
 *
 * Same shell as the Review Queue's `ReviewQueueList` — fixed header, scrolling
 * body, pinned pagination footer, all inside a viewport-height card — because
 * the two pages now do the same thing with different subjects (a pending pair
 * there, a resolved cluster here) and shouldn't feel like different apps. */
export function ClusterList({
  filters,
  onFiltersChange,
  items,
  total,
  isLoading,
  isError,
  selectedMid,
  onSelect,
  multiOnly,
  onMultiOnlyChange,
}: Props) {
  // Every filter edit resets to page 1; paging does not (it *is* the page
  // edit). Same split the Review Queue's list makes.
  const set = (patch: Partial<RecordsFilters>) =>
    onFiltersChange({ ...filters, ...patch, page: 1 });

  const page = filters.page ?? 1;
  const pageSize = filters.page_size ?? 25;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <div className="card flex h-[calc(100vh-220px)] min-h-[520px] flex-col">
      <div className="border-b border-line px-4 pt-3.5 pb-3">
        <div className="mb-2.5 flex items-center justify-between">
          <span className="text-[13px] font-bold text-ink-2">
            {multiOnly ? "Multi-record clusters" : "All clusters"}
          </span>
          <span className="rounded-full bg-bg px-2 py-0.5 text-[11px] font-bold text-gray">
            {total.toLocaleString()}
          </span>
        </div>

        <div className="mb-2.5 flex rounded-md bg-bg p-0.5">
          <SegmentButton
            active={multiOnly}
            onClick={() => onMultiOnlyChange(true)}
          >
            2+ records
          </SegmentButton>
          <SegmentButton
            active={!multiOnly}
            onClick={() => onMultiOnlyChange(false)}
          >
            All
          </SegmentButton>
        </div>

        {!multiOnly && (
          <div className="mb-2 text-[10.5px] text-gray">
            Includes standalone records still pending review (badged{" "}
            <span className="font-semibold text-[#0a7a78]">Needs review</span>
            ) — decide those from the Review Queue tab.
          </div>
        )}

        <input
          type="text"
          placeholder="Master Patient ID or name…"
          value={filters.search ?? ""}
          onChange={(e) => set({ search: e.target.value || undefined })}
          className="mb-2 w-full rounded-md border border-line px-2.5 py-1.5 text-sm outline-none focus:border-brand-blue"
        />

        <div className="flex gap-2">
          <input
            type="date"
            aria-label="Birthdate"
            value={filters.birth_date ?? ""}
            onChange={(e) => set({ birth_date: e.target.value || undefined })}
            className="min-w-0 flex-1 rounded-md border border-line px-2 py-1 text-xs outline-none focus:border-brand-blue"
          />
          <input
            type="text"
            maxLength={4}
            aria-label="SSN last 4"
            placeholder="SSN4"
            value={filters.ssn_last4 ?? ""}
            onChange={(e) => set({ ssn_last4: e.target.value || undefined })}
            className="w-16 rounded-md border border-line px-2 py-1 text-xs outline-none focus:border-brand-blue"
          />
          <select
            aria-label="Sort by"
            value={filters.sort ?? "updated"}
            onChange={(e) => {
              const next = e.target.value;
              set({
                sort:
                  next === "updated"
                    ? undefined
                    : (next as RecordsFilters["sort"]),
              });
            }}
            className="rounded-md border border-line px-1.5 py-1 text-xs outline-none focus:border-brand-blue"
          >
            <option value="updated">Updated</option>
            <option value="confidence">Confidence</option>
            <option value="name">Name</option>
          </select>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {isLoading && <p className="p-3 text-sm text-gray">Loading…</p>}
        {isError && (
          <p className="p-3 text-sm text-status-nomatch">
            Couldn&apos;t reach the eMPI API. Is the backend running?
          </p>
        )}
        {!isLoading && !isError && items.length === 0 && (
          <p className="p-3 text-sm text-gray">No clusters match these filters.</p>
        )}
        {items.map((entity) => (
          <ClusterRow
            key={entity.mid}
            entity={entity}
            selected={entity.mid === selectedMid}
            onSelect={() => onSelect(entity.mid)}
          />
        ))}
      </div>

      <div className="flex items-center justify-between gap-2 border-t border-line px-3 py-2.5">
        <button
          disabled={page <= 1}
          onClick={() => onFiltersChange({ ...filters, page: page - 1 })}
          className="rounded-md border border-line px-2.5 py-1 text-xs font-semibold text-gray-2 hover:bg-bg disabled:opacity-40"
        >
          ← Prev
        </button>
        <PageJumper
          page={page}
          totalPages={totalPages}
          onJump={(next) => onFiltersChange({ ...filters, page: next })}
          className="flex items-center gap-1.5 text-xs text-gray"
          inputClassName="w-11 rounded-md border border-line px-1 py-0.5 text-center font-mono text-xs tabular-nums text-ink-2 outline-none focus:border-brand-blue"
        />
        <button
          disabled={page >= totalPages}
          onClick={() => onFiltersChange({ ...filters, page: page + 1 })}
          className="rounded-md border border-line px-2.5 py-1 text-xs font-semibold text-gray-2 hover:bg-bg disabled:opacity-40"
        >
          Next →
        </button>
      </div>
    </div>
  );
}

function ClusterRow({
  entity,
  selected,
  onSelect,
}: {
  entity: Entity;
  selected: boolean;
  onSelect: () => void;
}) {
  // The cluster has no single name, so the primary member stands in for it —
  // the same record the old registry table showed on the collapsed row.
  const primary = entity.members.find((m) => m.is_primary) ?? entity.members[0];
  const count = entity.members.length;

  return (
    <button
      type="button"
      onClick={onSelect}
      className={clsx(
        "mb-1 w-full rounded-md border px-3 py-2.5 text-left",
        selected
          ? "border-brand-blue bg-brand-blue/5"
          : "border-transparent hover:bg-bg",
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-[13px] font-bold text-ink-2">
          {fullName(primary?.first_name, primary?.last_name)}
        </span>
        <span
          className={clsx(
            "shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold",
            count > 1 ? "bg-brand-blue/10 text-brand-blue" : "bg-bg text-gray",
          )}
        >
          {count} record{count === 1 ? "" : "s"}
        </span>
      </div>
      <div className="mt-0.5 flex items-center gap-2 text-[11px] text-gray">
        <span className="font-mono" title="Master Patient ID">
          {entity.mid}
        </span>
        <span>·</span>
        <span>{maskSsn(primary?.ssn_last4)}</span>
        <span>·</span>
        <span>{formatRawDate(primary?.birth_date)}</span>
      </div>
      <div className="mt-1.5">
        <StatusBadge origin={entity.origin} />
      </div>
    </button>
  );
}

function SegmentButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        "flex-1 rounded px-2 py-1 text-xs font-bold",
        active ? "bg-white text-brand-blue shadow-sm" : "text-gray",
      )}
    >
      {children}
    </button>
  );
}
