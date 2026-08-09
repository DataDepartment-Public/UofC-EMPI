"use client";

import clsx from "clsx";
import type { Entity, EntityMember } from "@/lib/schemas";
import { compareRecordsMulti, MultiComparisonRow } from "@/lib/compare";
import { memberToExplainPatient } from "@/lib/explain";
import { fullName } from "@/lib/format";
import { SsnReveal } from "@/components/shared/SsnReveal";

interface Props {
  entity: Entity;
  onUnmerge: (mid: string, patid: string, patientName: string) => void;
}

/** Every record in a cluster side by side — fields down, records across.
 *
 * The Review Queue's `FeatureComparisonTable` answers "do these two agree?"
 * and prints a verdict per row. This one answers a different question: "what
 * did we fold together, and does it look right?" With three or more columns a
 * single per-row verdict stops being definable (A and B agree, C differs — is
 * that an agreement?), so the table shows the values and nothing else, and
 * lets the reader scan across the row.
 *
 * Wide clusters scroll horizontally with the field column pinned — without
 * the pin, scrolling right leaves the reader looking at a grid of values with
 * no labels.
 */
export function ClusterRecordTable({ entity, onUnmerge }: Props) {
  const members = entity.members;
  const rows = compareRecordsMulti(members.map(memberToExplainPatient));

  if (members.length === 0) {
    return (
      <p className="rounded-md border border-line bg-bg px-3 py-2 text-sm text-gray">
        This cluster has no published records.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-line text-left align-bottom">
            <Th sticky>Field</Th>
            {members.map((m, i) => (
              <th
                key={m.patid}
                className="min-w-[200px] px-3 py-2 align-bottom font-normal"
              >
                <RecordHeader
                  member={m}
                  index={i}
                  canUnmerge={members.length > 1}
                  onUnmerge={() =>
                    onUnmerge(
                      entity.mid,
                      m.patid,
                      fullName(m.first_name, m.last_name),
                    )
                  }
                />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <Row key={row.label} row={row} members={members} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Row({
  row,
  members,
}: {
  row: MultiComparisonRow;
  members: EntityMember[];
}) {
  return (
    <tr className="border-b border-line last:border-none">
      <Td sticky>
        <span className="font-bold text-ink-2">{row.label}</span>
      </Td>
      {row.values.map((value, i) => (
        <Td key={members[i].patid}>
          <CellValue
            value={value}
            values={row.valueLists?.[i]}
            patid={members[i].patid}
            isSsn={row.label.startsWith("SSN")}
          />
        </Td>
      ))}
    </tr>
  );
}

function CellValue({
  value,
  values,
  patid,
  isSsn,
}: {
  value: string;
  values?: string[];
  patid: string;
  isSsn: boolean;
}) {
  if (isSsn && value !== "(missing)") {
    return <SsnReveal patid={patid} masked={value} />;
  }
  if (values && values.length > 0) {
    // A record can carry four phone numbers; comma-joining them onto one line
    // makes the columns impossible to scan against each other.
    return (
      <div className="flex flex-col gap-0.5">
        {values.map((v) => (
          <span key={v}>{v}</span>
        ))}
      </div>
    );
  }
  return (
    <span className={value === "(missing)" ? "text-gray" : undefined}>
      {value}
    </span>
  );
}

function RecordHeader({
  member,
  index,
  canUnmerge,
  onUnmerge,
}: {
  member: EntityMember;
  index: number;
  canUnmerge: boolean;
  onUnmerge: () => void;
}) {
  const tag = member.is_primary
    ? "Primary"
    : member.added_by === "pipeline"
      ? null
      : `by ${member.added_by}`;

  return (
    <div className="flex flex-col items-start gap-1 py-1">
      <span className="text-[10px] font-bold tracking-wide text-gray uppercase">
        Record {index + 1}
      </span>
      <span className="font-bold text-ink-2">
        {fullName(member.first_name, member.last_name)}
      </span>
      <span className="font-mono text-[10px] text-gray">{member.patid}</span>
      {tag && (
        <span className="rounded-full bg-bg px-2 py-0.5 text-[10px] font-bold text-gray">
          {tag}
        </span>
      )}
      {canUnmerge && (
        <button
          type="button"
          onClick={onUnmerge}
          className="mt-0.5 rounded-md border border-status-nomatch px-2.5 py-1 text-[11px] font-bold text-status-nomatch hover:bg-status-nomatch hover:text-white"
        >
          Unmerge
        </button>
      )}
    </div>
  );
}

/** `sticky left-0` needs an opaque background of its own, or the value cells
 * scroll visibly underneath it. */
function Th({ children, sticky }: { children: React.ReactNode; sticky?: boolean }) {
  return (
    <th
      className={clsx(
        "px-3 py-2 text-[11px] font-bold tracking-wide text-gray uppercase",
        sticky && "sticky left-0 z-10 min-w-[130px] bg-card",
      )}
    >
      {children}
    </th>
  );
}

function Td({ children, sticky }: { children: React.ReactNode; sticky?: boolean }) {
  return (
    <td
      className={clsx(
        "px-3 py-2.5 align-top text-gray-2",
        sticky && "sticky left-0 z-10 bg-card",
      )}
    >
      {children}
    </td>
  );
}
