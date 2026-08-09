"use client";

import clsx from "clsx";
import type { ReviewQueueFilters } from "@/lib/api-client";
import type { ReviewBucket, ReviewQueueItem } from "@/lib/schemas";
import { formatRawDate, fullName, maskSsn } from "@/lib/format";
import { PageJumper } from "@/components/shared/PageJumper";

/** The four sections, in the order they're shown. The two human sections
 * come first because that's where a reviewer's work is; the two pipeline
 * sections are there to be audited, not worked through.
 *
 * `blurb` is the one-line answer to "why is this pair here?", which matters
 * most for the pipeline sections — nothing else on the row explains that the
 * system decided it unattended. */
const SECTIONS: {
  bucket: ReviewBucket;
  label: string;
  empty: string;
  blurb: string;
}[] = [
  {
    bucket: "needs_review",
    label: "Needs review",
    empty: "Nothing needs review right now.",
    blurb: "Pairs the pipeline couldn't decide. Waiting on you.",
  },
  {
    bucket: "reviewed",
    label: "Reviewed",
    empty: "No pairs have been reviewed yet.",
    blurb: "Pairs you or another reviewer has ruled on.",
  },
  {
    bucket: "auto_merged",
    label: "Auto-merged",
    empty: "The pipeline merged nothing on its own.",
    blurb:
      "Merged by a deterministic rule or the ML matcher, with no reviewer involved.",
  },
  {
    bucket: "auto_rejected",
    label: "Auto-rejected",
    empty: "The pipeline rejected nothing on its own.",
    blurb:
      "Discarded by the reject rules or the non-match gate, with no reviewer involved.",
  },
];

/** Left panel of the Review Queue's two-panel layout: the candidate-grain
 * list itself (one row per candidate pair, not per cluster), plus its
 * section tabs and filters/sort. Selecting a row never navigates — it just
 * updates the caller's `selectedKey` state, which `ReviewCandidateDetail`
 * re-renders from. */
export function ReviewQueueList({
  filters,
  onFiltersChange,
  items,
  total,
  bucketCounts,
  isLoading,
  isError,
  selectedKey,
  onSelect,
}: {
  filters: ReviewQueueFilters;
  onFiltersChange: (next: ReviewQueueFilters) => void;
  items: ReviewQueueItem[];
  total: number;
  /** Whole-index pair count per section — deliberately not affected by the
   * search/confidence filters, so the tab counts stay a stable picture of
   * the queue while you narrow the list under them. */
  bucketCounts: Record<string, number>;
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
  const section =
    SECTIONS.find((s) => s.bucket === filters.bucket) ?? SECTIONS[0];

  return (
    <div className="card flex h-[calc(100vh-220px)] min-h-[520px] flex-col">
      <div className="border-b border-line px-4 pt-3.5 pb-3">
        <div className="mb-2.5 flex items-center justify-between">
          <h3 className="text-[15px] font-bold text-ink-2">{section.label}</h3>
          <span className="rounded-full bg-brand-blue/10 px-2 py-0.5 text-[11px] font-bold text-brand-blue">
            {total.toLocaleString()} pair{total === 1 ? "" : "s"}
          </span>
        </div>

        <div className="mb-1.5 grid grid-cols-2 gap-0.5 rounded-md border border-line bg-bg p-0.5">
          {SECTIONS.map((s) => (
            <SegmentButton
              key={s.bucket}
              active={section.bucket === s.bucket}
              onClick={() => set({ bucket: s.bucket })}
            >
              {s.label}
              <span className="ml-1 font-mono text-[10px] font-bold opacity-60">
                {(bucketCounts[s.bucket] ?? 0).toLocaleString()}
              </span>
            </SegmentButton>
          ))}
        </div>
        <p className="mb-2.5 text-[10.5px] leading-snug text-gray">
          {section.blurb}
        </p>

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
          ↓ Sorted by confidence (rule, or ML match score), highest first
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
          <p className="p-3 text-xs text-gray">{section.empty}</p>
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
        <PageJumper
          page={page}
          totalPages={totalPages}
          onJump={(next) => onFiltersChange({ ...filters, page: next })}
        />
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
  // No rule fired for most queue pairs (confidence null) — fall back to the
  // Stage 4.5 ML matcher's score so the list isn't just showing "—" for the
  // vast majority of rows. Matches the backend's own sort/filter fallback
  // (sql_backend.list_review_candidates' COALESCE).
  const displayScore = item.confidence ?? item.ml_match_probability;
  const confPct = displayScore != null ? Math.round(displayScore * 100) : null;
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
          <span>{formatRawDate(item.patient_a.birth_date)}</span>
          <span>·</span>
          <span>{maskSsn(item.patient_a.ssn_last4)}</span>
          {inCluster && (
            <span className="rounded-full bg-bg px-1.5 py-0.5 text-[9.5px] font-bold text-gray-2">
              +{Math.max(item.member_count_a, item.member_count_b) - 1} in cluster
            </span>
          )}
        </div>
      </div>
      <div className="flex flex-shrink-0 flex-col items-end gap-1">
        <span className="text-[13px] font-extrabold tabular-nums text-status-auto">
          {confPct != null ? `${confPct}%` : "—"}
        </span>
        <OutcomeBadge item={item} />
      </div>
    </button>
  );
}

/** What happened to this pair, in one word. Reads `reviewer_decision` rather
 * than inferring from `mid_a === mid_b`: a pair can share a mid because the
 * pipeline merged it, because clustering merged it transitively, or because
 * a reviewer merged it, and only the third is a review outcome. Inferring
 * was the bug that put auto-merges under "Already reviewed". */
function OutcomeBadge({ item }: { item: ReviewQueueItem }) {
  const outcome =
    item.bucket === "reviewed"
      ? {
          text: item.reviewer_decision === "merged" ? "Merged" : "Not a match",
          tone: "text-status-auto bg-status-auto/15",
        }
      : item.bucket === "auto_merged"
        ? { text: "Auto-merged", tone: "text-status-auto bg-status-auto/15" }
        : item.bucket === "auto_rejected"
          ? {
              text: "Auto-rejected",
              tone: "text-status-nomatch bg-status-nomatch/15",
            }
          : null;

  if (!outcome) return null;
  return (
    <span
      className={clsx(
        "rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase",
        outcome.tone,
      )}
    >
      {outcome.text}
    </span>
  );
}
