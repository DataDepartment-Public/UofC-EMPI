"use client";

import clsx from "clsx";
import { ApiError } from "@/lib/api-client";
import { useAuditLog, useUndoMutation } from "@/lib/hooks";
import { formatDate } from "@/lib/format";
import type { AuditLogRow } from "@/lib/schemas";

const ACTION_LABEL: Record<string, string> = {
  merge: "Merged",
  unmerge: "Unmerged",
  split: "Split out",
  dismiss: "Dismissed",
};

const UNDOABLE_ACTIONS = new Set(["merge", "unmerge"]);

interface Props {
  /** Reuses the host page's own toast/flash mechanism rather than rendering
   * a second `<Toast/>` — this and e.g. the Dataset page's unmerge flow would
   * otherwise stack two fixed-position toasts on top of each other. */
  onFlash: (message: string) => void;
}

/** FR-30/31: immutable merge/unmerge/split audit trail — user, timestamp,
 * patient IDs affected, previous & new status, final master patient ID. */
export function AuditLog({ onFlash }: Props) {
  const { data, isLoading, isError } = useAuditLog();
  const undoMutation = useUndoMutation();

  const handleUndo = (row: AuditLogRow) => {
    undoMutation.mutate(row.id, {
      onSuccess: () =>
        onFlash(
          `Undid ${ACTION_LABEL[row.action]?.toLowerCase() ?? row.action} entry #${row.id}.`,
        ),
      onError: (err) =>
        onFlash(err instanceof ApiError ? err.message : "Undo failed."),
    });
  };

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
              <Th>Actions</Th>
            </tr>
          </thead>
          <tbody>
            {isLoading && <EmptyRow>Loading audit log…</EmptyRow>}
            {isError && (
              <EmptyRow className="text-status-nomatch">
                Couldn&apos;t reach the eMPI API.
              </EmptyRow>
            )}
            {data && data.length === 0 && (
              <EmptyRow>
                No merge actions yet.
              </EmptyRow>
            )}
            {data?.map((row) => {
              const pending =
                undoMutation.isPending && undoMutation.variables === row.id;
              const canUndo = UNDOABLE_ACTIONS.has(row.action) && !row.undone;
              return (
                <tr key={row.id} className="border-t border-line">
                  <Td className="font-semibold text-ink-2">{row.user}</Td>
                  <Td className="text-gray">{formatDate(row.ts_utc)}</Td>
                  <Td className="font-mono text-xs">{row.patids}</Td>
                  <Td>
                    {row.prev_state} <span className="text-gray">→</span>{" "}
                    <b>{ACTION_LABEL[row.action] ?? row.next_state}</b>
                  </Td>
                  <Td className="font-mono text-xs font-bold">{row.mid}</Td>
                  <Td>
                    {row.undone ? (
                      <span className="text-xs font-semibold text-gray italic">
                        Undone
                      </span>
                    ) : canUndo ? (
                      <button
                        onClick={() => handleUndo(row)}
                        disabled={pending}
                        className="rounded-md bg-status-nomatch px-3 py-1 text-xs font-bold text-white hover:opacity-90 disabled:opacity-40"
                      >
                        {pending ? "Undoing…" : "Undo"}
                      </button>
                    ) : (
                      <span className="text-xs text-gray">—</span>
                    )}
                  </Td>
                </tr>
              );
            })}
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
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <td className={clsx("px-4 py-2.5 text-ink-2", className)}>{children}</td>;
}

function EmptyRow({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <tr>
      <td
        colSpan={6}
        className={clsx("p-6 text-center text-sm text-gray italic", className)}
      >
        {children}
      </td>
    </tr>
  );
}
