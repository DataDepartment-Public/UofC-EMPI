"use client";

import clsx from "clsx";
import type { ReviewQueueFilters } from "@/lib/api-client";
import type { ReviewQueueItem } from "@/lib/schemas";
import { fullName, maskSsn } from "@/lib/format";

/** Left panel of the Review Queue's two-panel layout: the candidate-grain
 * list itself (one row per pending pair, not per cluster), plus its own
 * filters/sort/segmented reviewed-state control. Selecting a row never
 * navigates — it just updates the caller's `selectedKey` state, which
 * `ReviewCandidateDetail` re-renders from. */
export function ReviewQueueList({
  filters,
  onFiltersChange,
  items,
  total,
  isLoading,
  isError,
  selectedKey,
  onSelect,
}: {
  filters: ReviewQueueFilters;
  onFiltersChange: (next: ReviewQueueFilters) => void;
  items: ReviewQueueItem[];
  total: number;
  isLoading: boolean;
  isError: boolean;
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  const set = (patch: Partial<ReviewQueueFilters>) =>
    onFiltersChange({ ...filters, ...patch, page: 1 });

  const pageSize = filters.page_size ?? 30;
  const page = filters.page ?? 1;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <div className="card flex h-[calc(100vh-220px)] min-h-[520px] flex-col">
      <div className="border-b border-line px-4 pt-3.5 pb-3">
        <div className="mb-2.5 flex items-center justify-between">
          <h3 className="text-[15px] font-bold text-ink-2">
            {filters.reviewed ? "Already reviewed" : "Needs review"}
          </h3>
          <span className="rounded-full bg-brand-blue/10 px-2 py-0.5 text-[11px] font-bold text-brand-blue">
            {total.toLocaleString()} pair{total === 1 ? "" : "s"}
          </span>
        </div>

        <div className="mb-2.5 flex rounded-md border border-line bg-bg p-0.5">
          <SegmentButton active={!filters.reviewed} onClick={() => set({ reviewed: false })}>
            Needs review
          </SegmentButton>
          <SegmentButton active={!!filters.reviewed} onClick={() => set({ reviewed: true })}>
            Already reviewed
          </SegmentButton>
        </div>

        <input
          type="text"
          placeholder="Search name…"
          value={filters.search ?? ""}
          onChange={(e) => set({ search: e.target.value || undefined })}
          className="mb-2.5 w-full rounded-md border border-line px-2.5 py-1.5 text-sm outline-none focus:border-brand-blue"
        />

        <div className="flex items-center gap-2 text-[11px] text-gray-2">
          <span className="whitespace-nowrap font-bold uppercase tracking-wide text-gray">
            Min. conf.
          </span>
          <input
            type="range"
            min={0}
            max={100}
            value={filters.confidence_min != null ? Math.round(filters.confidence_min * 100) : 0}
            onChange={(e) => {
              const pct = Number(e.target.value);
              set({ confidence_min: pct > 0 ? pct / 100 : undefined });
            }}
            className="flex-1 accent-brand-blue"
          />
          <span className="w-8 font-mono">
            {filters.confidence_min != null ? `${Math.round(filters.confidence_min * 100)}%` : "Any"}
          </span>
        </div>
        <div className="mt-1.5 text-[10.5px] text-gray">
          ↓ Sorted by confidence, highest first
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {isLoading && <p className="p-3 text-xs text-gray">Loading…</p>}
        {isError && (
          <p className="p-3 text-xs text-status-nomatch">
            Couldn&apos;t reach the eMPI API. Is the backend running?
          </p>
        )}
        {!isLoading && !isError && items.length === 0 && (
          <p className="p-3 text-xs text-gray">
            {filters.reviewed
              ? "No reviewed candidates yet."
              : "Nothing needs review right now."}
          </p>
        )}
        {items.map((item) => {
          const key = `${item.patid_a}-${item.patid_b}`;
          return (
            <CandidateRow
              key={key}
              item={item}
              selected={key === selectedKey}
              onClick={() => onSelect(key)}
            />
          );
        })}
      </div>

      <div className="flex items-center justify-center gap-2 border-t border-line py-2">
        <button
          disabled={page <= 1}
          onClick={() => onFiltersChange({ ...filters, page: page - 1 })}
          className="rounded-md border border-line px-2.5 py-1 text-xs font-semibold text-gray-2 disabled:opacity-40 hover:bg-bg"
        >
          ← Prev
        </button>
        <span className="text-[11px] text-gray">
          Page {page} of {totalPages}
        </span>
        <button
          disabled={page >= totalPages}
          onClick={() => onFiltersChange({ ...filters, page: page + 1 })}
          className="rounded-md border border-line px-2.5 py-1 text-xs font-semibold text-gray-2 disabled:opacity-40 hover:bg-bg"
        >
          Next →
        </button>
      </div>
    </div>
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
      onClick={onClick}
      className={clsx(
        "flex-1 rounded px-2 py-1.5 text-[11.5px] font-bold transition-colors",
        active ? "bg-white text-brand-blue shadow-sm" : "text-gray-2 hover:text-brand-blue",
      )}
    >
      {children}
    </button>
  );
}

function CandidateRow({
  item,
  selected,
  onClick,
}: {
  item: ReviewQueueItem;
  selected: boolean;
  onClick: () => void;
}) {
  const confPct = item.confidence != null ? Math.round(item.confidence * 100) : null;
  const confColor =
    confPct == null
      ? "text-gray"
      : confPct >= 90
        ? "text-status-auto"
        : confPct >= 75
          ? "text-status-review"
          : "text-status-nomatch";
  const inCluster = item.member_count_a > 1 || item.member_count_b > 1;

  return (
    <button
      onClick={onClick}
      className={clsx(
        "mb-1 flex w-full items-start justify-between gap-3 rounded-md border px-3 py-2.5 text-left",
        selected
          ? "border-brand-blue bg-brand-blue/5"
          : "border-transparent hover:bg-bg",
      )}
    >
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13px] font-bold text-ink-2">
          {fullName(item.patient_a.first_name, item.patient_a.last_name)}
          <span className="mx-1 font-medium text-gray">vs</span>
          {fullName(item.patient_b.first_name, item.patient_b.last_name)}
        </div>
        <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[11px] text-gray-2">
          <span>{item.patient_a.birth_date ?? "—"}</span>
          <span>·</span>
          <span>{maskSsn(item.patient_a.ssn_last4)}</span>
          {inCluster && (
            <span className="rounded-full bg-bg px-1.5 py-0.5 text-[9.5px] font-bold text-gray-2">
              +{Math.max(item.member_count_a, item.member_count_b) - 1} in cluster
            </span>
          )}
        </div>
        <span
          className={clsx(
            "mt-1 inline-block rounded-full px-2 py-0.5 text-[9.5px] font-bold",
            item.match_rule
              ? "bg-status-review/15 text-status-review"
              : "bg-bg text-gray-2",
          )}
        >
          {item.match_rule ?? "Blocking only — no rule"}
        </span>
      </div>
      <div className="flex flex-shrink-0 flex-col items-end gap-1">
        <span className={clsx("text-[13px] font-extrabold tabular-nums", confColor)}>
          {confPct != null ? `${confPct}%` : "—"}
        </span>
        {item.reviewed && (
          <span className="rounded-full bg-status-auto/15 px-1.5 py-0.5 text-[9px] font-bold uppercase text-status-auto">
            {item.mid_a === item.mid_b ? "Merged" : "Dismissed"}
          </span>
        )}
      </div>
    </button>
  );
}
