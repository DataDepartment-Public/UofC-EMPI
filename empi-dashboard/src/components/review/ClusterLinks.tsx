"use client";

import Link from "next/link";
import { fullName } from "@/lib/format";
import type { ReviewQueueItem } from "@/lib/schemas";

/** The cluster each side of the pair currently belongs to, with a way out to
 * the Patient Registry.
 *
 * A candidate pair is two *records*, but the decision is about two *clusters*:
 * merging pulls in every other record already grouped with each side. The
 * pipeline trail above says what happened to the pair; this says what the pair
 * is attached to, so a reviewer can see a 6-member cluster on one side before
 * merging rather than after. The link is the inverse of `ClusterComparisons`'
 * "Open in Review Queue" — registry ↔ queue in both directions. */
export function ClusterLinks({ item }: { item: ReviewQueueItem }) {
  const merged = item.mid_a === item.mid_b;

  if (merged) {
    return (
      <ClusterCard
        mid={item.mid_a}
        count={item.member_count_a}
        label="Both records are in this cluster"
        names={[
          fullName(item.patient_a.first_name, item.patient_a.last_name),
          fullName(item.patient_b.first_name, item.patient_b.last_name),
        ]}
      />
    );
  }

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <ClusterCard
        mid={item.mid_a}
        count={item.member_count_a}
        label={`Cluster of ${item.patid_a}`}
        names={[fullName(item.patient_a.first_name, item.patient_a.last_name)]}
      />
      <ClusterCard
        mid={item.mid_b}
        count={item.member_count_b}
        label={`Cluster of ${item.patid_b}`}
        names={[fullName(item.patient_b.first_name, item.patient_b.last_name)]}
      />
    </div>
  );
}

function ClusterCard({
  mid,
  count,
  label,
  names,
}: {
  mid: string;
  count: number;
  label: string;
  names: string[];
}) {
  return (
    <div className="rounded-md border border-line px-3 py-2.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-[10px] font-bold tracking-wide text-gray uppercase">
          {label}
        </span>
        <span
          className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold ${
            count > 1 ? "bg-brand-blue/10 text-brand-blue" : "bg-bg text-gray"
          }`}
        >
          {count} record{count === 1 ? "" : "s"}
        </span>
      </div>
      <div className="mt-1 truncate text-[13px] font-bold text-ink-2">
        {names.join(" · ")}
      </div>
      <div className="mt-0.5 flex items-center justify-between gap-2">
        <span className="truncate font-mono text-[11px] text-gray" title="Master Patient ID">
          {mid}
        </span>
        <Link
          href={`/dataset?mid=${encodeURIComponent(mid)}`}
          className="shrink-0 whitespace-nowrap text-[11px] font-bold text-brand-blue hover:underline"
        >
          Open in Patient Registry →
        </Link>
      </div>
    </div>
  );
}
