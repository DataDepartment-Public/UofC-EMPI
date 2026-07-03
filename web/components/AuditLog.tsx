"use client";

import { useAuditLog } from "@/lib/hooks";
import { formatDate } from "@/lib/format";

const ACTION_LABEL: Record<string, string> = {
  merge: "Merged",
  unmerge: "Unmerged",
  split: "Split out",
};

/** FR-30/31: immutable merge/unmerge/split audit trail — user, timestamp,
 * patient IDs affected, previous & new status, final master patient ID. */
export function AuditLog() {
  const { data, isLoading, isError } = useAuditLog();

  return (
    <div>
      <h3 className="mb-3 text-[13px] font-bold tracking-wide text-gray uppercase">
        Merge audit log
      </h3>
      <div className="card overflow-hidden">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="bg-bg">
              <Th>User</Th>
              <Th>Timestamp</Th>
              <Th>Patient IDs</Th>
              <Th>Prev → New status</Th>
              <Th>Master ID</Th>
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <EmptyRow>Loading audit log…</EmptyRow>
            )}
            {isError && (
              <EmptyRow className="text-status-nomatch">
                Couldn&apos;t reach the eMPI API.
              </EmptyRow>
            )}
            {data && data.length === 0 && (
              <EmptyRow>
                No merge actions yet — merge a cluster to record an entry.
              </EmptyRow>
            )}
            {data?.map((row) => (
              <tr key={row.id} className="border-t border-line">
                <Td className="font-semibold text-ink-2">{row.user}</Td>
                <Td className="text-gray">{formatDate(row.ts_utc)}</Td>
                <Td className="font-mono text-xs">{row.patids}</Td>
                <Td>
                  {row.prev_state}{" "}
                  <span className="text-gray">→</span>{" "}
                  <b>{ACTION_LABEL[row.action] ?? row.next_state}</b>
                </Td>
                <Td className="font-mono text-xs font-bold">{row.mid}</Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="border-b border-line px-4 py-2.5 text-left text-[10px] font-bold tracking-wide text-gray uppercase">
      {children}
    </th>
  );
}

function Td({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <td className={`px-4 py-2.5 text-ink-2 ${className}`}>{children}</td>;
}

function EmptyRow({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <tr>
      <td
        colSpan={5}
        className={`p-6 text-center text-sm text-gray italic ${className}`}
      >
        {children}
      </td>
    </tr>
  );
}
