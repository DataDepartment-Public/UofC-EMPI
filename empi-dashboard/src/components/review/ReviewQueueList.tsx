"use client";

import clsx from "clsx";
import type { ReviewQueueFilters } from "@/lib/api-client";
import type { ReviewBucket, ReviewQueueItem } from "@/lib/schemas";
import { formatRawDate, fullName, maskSsn } from "@/lib/format";
import {
  BAND_DEFS,
  bandDef,
  bandFilter,
  bandFromFilters,
  bandRangeLabel,
  bucketBadge,
  signalFor,
  signalTooltip,
  verdictLabel,
  type LiveThresholds,
  type SignalBand,
} from "@/lib/pair-signal";
import { useThresholds } from "@/lib/hooks";
import { PageJumper } from "@/components/shared/PageJumper";
import { SignalLegend } from "@/components/review/SignalLegend";

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
  // Both bars are operator-tunable at runtime, so they're fetched rather
  // than compiled in — see `SignalLegend`.
  const t = useThresholds().data;
  const thresholds: LiveThresholds = {
    ml: t?.ml_auto_merge_threshold ?? null,
    gate: t?.gate_threshold ?? null,
  };
  const selectedBand = bandFromFilters(filters, thresholds);

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
              onClick={() =>
                set({
                  bucket: s.bucket,
                  // The mirror of the band select's tab switch: a gate band
                  // left applied while moving off Auto-rejected would empty
                  // the list for a reason nothing on screen explains.
                  ...(selectedBand &&
                  bandDef(selectedBand).axis === "gate" &&
                  s.bucket !== "auto_rejected"
                    ? bandFilter(null, thresholds)
                    : {}),
                })
              }
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

        {/* `relative` is the positioning context for SignalLegend's panel —
            see its docstring; it anchors to this row, not to the ⓘ. */}
        <div className="relative flex items-center gap-2 text-[11px] text-gray-2">
          {/* The ⓘ sits *beside* the label, not inside it: this span's
              `uppercase` / `font-bold` / `whitespace-nowrap` would otherwise
              inherit into the legend panel and stop its text wrapping. */}
          <span className="whitespace-nowrap font-bold uppercase tracking-wide text-gray">
            Confidence
          </span>
          <SignalLegend thresholds={thresholds} />
          <select
            value={selectedBand ?? ""}
            onChange={(e) => {
              const band = (e.target.value || null) as SignalBand | null;
              // Every filter ANDs, so a gate band under the "Needs review"
              // tab can never match anything: a pair the gate dropped is in
              // Auto-rejected by construction (`pair_verdicts.bucket_for`).
              // Follow the reviewer's intent and move the tab with them,
              // rather than returning a silently empty list.
              const axis = band ? bandDef(band).axis : null;
              set({
                ...bandFilter(band, thresholds),
                ...(axis === "gate" ? { bucket: "auto_rejected" as const } : {}),
              });
            }}
            className="flex-1 rounded-md border border-line bg-card px-2 py-1 text-[11px] outline-none focus:border-brand-blue"
          >
            <option value="">Any</option>
            {BAND_DEFS.map((d) => (
              <option key={d.band} value={d.band}>
                {d.label}
              </option>
            ))}
          </select>
        </div>
        <div className="mt-1.5 text-[10.5px] leading-snug text-gray">
          {selectedBand
            ? bandRangeLabel(selectedBand, thresholds)
            : "↓ Strongest first — a rule's precision, or the ML matcher's score."}
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
  // A band, not a percentage — see `lib/pair-signal.ts` for why a bare
  // `confidence ?? ml_match_probability` in the merge-green misread as
  // "1% likely the same person" on pairs the gate had passed as plausible.
  const { band } = signalFor(item);
  const def = bandDef(band);
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
        {/* Why this pair is in front of you — the stage that routed it, and
            the rule that fired if one did. The old row showed only a number,
            which is what let a 1% ML score read as a verdict of its own. */}
        <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[10.5px] text-gray">
          <span className="font-semibold">{verdictLabel(item.verdict)}</span>
          {item.match_rule && (
            <>
              <span>·</span>
              <span className="font-mono">{item.match_rule}</span>
            </>
          )}
        </div>
      </div>
      <div className="flex flex-shrink-0 flex-col items-end gap-1">
        <span
          title={signalTooltip(item)}
          className={clsx(
            "rounded-full px-1.5 py-0.5 text-[9.5px] font-bold uppercase",
            def.tone,
          )}
        >
          {def.label}
        </span>
        <OutcomeBadge item={item} />
      </div>
    </button>
  );
}

/** What happened to this pair, in one word — the same `bucketBadge` the
 * detail header shows, so a row and the panel beside it can't disagree.
 *
 * Suppressed for `needs_review`: every row under that tab would carry the
 * same badge, which is noise. The detail header does show it, because there
 * the reviewer is about to act and "this is still open work" is the point. */
function OutcomeBadge({ item }: { item: ReviewQueueItem }) {
  if (item.bucket === "needs_review") return null;
  const { label, tone } = bucketBadge(item);
  return (
    <span
      className={clsx(
        "rounded-full px-1.5 py-0.5 text-[9px] font-bold uppercase",
        tone,
      )}
    >
      {label}
    </span>
  );
}
