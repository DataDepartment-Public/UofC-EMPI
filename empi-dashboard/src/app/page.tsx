"use client";

import { useDashboardSummary } from "@/lib/hooks";
import { KpiCard, KpiGrid } from "@/components/dashboard/KpiCard";
import { MatchStatusChart } from "@/components/dashboard/MatchStatusChart";
import { TrendChart } from "@/components/dashboard/TrendChart";
import { ModelInfoPanel } from "@/components/dashboard/ModelInfoPanel";

export default function DashboardPage() {
  const { data: summary, isLoading, isError } = useDashboardSummary();

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-end justify-between gap-2.5">
        <div>
          <h2 className="text-[22px] font-extrabold text-ink-2">Dashboard</h2>
          <p className="mt-1 text-[13px] text-gray">
            High-level KPIs and match-quality visualizations. All review and
            merge workflows live on the Dataset tab.
          </p>
        </div>
      </div>

      {isLoading && <p className="text-sm text-gray">Loading KPIs…</p>}
      {isError && (
        <p className="text-sm text-status-nomatch">
          Couldn&apos;t reach the eMPI API. Is the backend running?
        </p>
      )}

      {summary && (
        <>
          <section>
            <SectionLabel>Summary metrics</SectionLabel>
            <KpiGrid>
              <KpiCard
                label="Total patient records"
                value={summary.total_records.toLocaleString()}
                accent="var(--brand-blue)"
              />
              <KpiCard
                label="Duplicate clusters found"
                value={summary.duplicate_clusters.toLocaleString()}
                accent="var(--brand-cyan)"
              />
              <KpiCard
                label="Matched vs. unmerged"
                value={`${summary.matched_pct}% / ${summary.unmerged_pct}%`}
                accent="var(--status-auto)"
              />
              <KpiCard
                label="Auto-match rate"
                value={`${summary.auto_match_rate}%`}
                accent="var(--brand-gold)"
              />
              <KpiCard
                label="Needs manual review"
                value={summary.needs_review_records.toLocaleString()}
                accent="var(--status-review)"
              />
              <KpiCard
                label="Invalid records"
                value={summary.invalid_records.toLocaleString()}
                accent="var(--status-nomatch)"
              />
              <KpiCard
                label="Manually merged vs. unmerged"
                value={`${summary.manual_merge_pct}% / ${summary.manual_unmerge_pct}%`}
                accent="var(--brand-teal)"
              />
              <KpiCard
                label="Raw rows ingested"
                value={summary.total_raw_rows.toLocaleString()}
                accent="var(--brand-lime)"
              />
            </KpiGrid>
          </section>

          <section className="mt-6 grid grid-cols-1 gap-5 lg:grid-cols-[1.4fr_1fr]">
            <MatchStatusChart
              autoMatch={summary.status_counts.auto_match}
              needsReview={summary.status_counts.needs_review}
              noMatch={summary.status_counts.no_match}
            />
            <ModelInfoPanel summary={summary} />
          </section>

          <section className="mt-6">
            <TrendChart history={summary.history} />
          </section>
        </>
      )}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-3 text-[13px] font-bold tracking-wide text-gray uppercase">
      {children}
    </h3>
  );
}
