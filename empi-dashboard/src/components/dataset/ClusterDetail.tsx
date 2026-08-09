"use client";

import type { Entity } from "@/lib/schemas";
import { formatDate, fullName } from "@/lib/format";
import { StatusBadge } from "@/components/dataset/StatusBadge";
import {
  ClusterComparisonHistory,
  ClusterComparisons,
} from "@/components/dataset/ClusterComparisons";
import { ClusterRawPanel } from "@/components/dataset/ClusterRawPanel";
import { ClusterRecordTable } from "@/components/dataset/ClusterRecordTable";

interface Props {
  entity: Entity;
  onUnmerge: (mid: string, patid: string, patientName: string) => void;
}

/** Right pane of the Patient Registry: one cluster, top to bottom.
 *
 * The order is the reviewer's question order — *what* was grouped (the
 * side-by-side records, with Unmerge on each column), then *what the source
 * really said* (the raw toggle), then *why* it was grouped (the pairwise
 * comparison trace). Evidence last, because a reviewer usually decides from
 * the values alone and only drills into the model's reasoning when they
 * disagree with the grouping.
 */
export function ClusterDetail({ entity, onUnmerge }: Props) {
  const primary = entity.members.find((m) => m.is_primary) ?? entity.members[0];
  const count = entity.members.length;

  return (
    <div className="card p-5">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3 border-b border-line pb-4">
        <div>
          <h3 className="text-[17px] font-extrabold text-ink-2">
            {fullName(primary?.first_name, primary?.last_name)}
          </h3>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-gray">
            <span className="font-mono" title="Master Patient ID">
              {entity.mid}
            </span>
            <span>·</span>
            <span>
              {count} record{count === 1 ? "" : "s"}
            </span>
            <span>·</span>
            <span>run <span className="font-mono">{entity.run_id}</span></span>
            <span>·</span>
            <span>updated {formatDate(entity.updated_utc)}</span>
          </div>
        </div>
        <StatusBadge origin={entity.origin} />
      </div>

      <Section title={`Records in this cluster (${count})`}>
        <ClusterRecordTable entity={entity} onUnmerge={onUnmerge} />
        <ClusterRawPanel patids={entity.members.map((m) => m.patid)} />
      </Section>

      <Section
        title="Comparisons of cluster records"
        hint="How these records were matched to each other"
      >
        <ClusterComparisons entity={entity} />
      </Section>

      <Section
        title="Comparison history"
        hint="Records outside this cluster that were checked and not merged"
      >
        <ClusterComparisonHistory entity={entity} />
      </Section>
    </div>
  );
}

function Section({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mb-6 last:mb-0">
      <h4 className="text-[11px] font-bold tracking-wide text-gray uppercase">
        {title}
      </h4>
      {/* The two comparison sections are easy to confuse — one is about the
          records inside the cluster, the other about records outside it — so
          each carries a one-line subtitle rather than relying on the reader
          inferring the distinction from the heading alone. */}
      {hint && <p className="mb-2.5 text-[11px] text-gray">{hint}</p>}
      {!hint && <div className="mb-2.5" />}
      {children}
    </div>
  );
}
