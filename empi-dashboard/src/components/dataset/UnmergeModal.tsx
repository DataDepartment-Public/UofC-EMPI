"use client";

/** Confirmation before splitting a record out of a merged cluster — mirrors
 * MergeModal's confirm-before-permanent-action pattern (docs/Dashboard-Guide.md
 * FR-28), applied to the unmerge/split direction, which previously fired
 * immediately with no confirmation. */
export function UnmergeModal({
  mid,
  patid,
  patientName,
  pending,
  onConfirm,
  onCancel,
}: {
  mid: string;
  patid: string;
  patientName: string;
  pending: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/30 p-4">
      <div className="card w-full max-w-md p-0">
        <div className="border-b border-line px-5 py-4">
          <h3 className="text-base font-bold text-ink-2">Confirm unmerge</h3>
        </div>
        <div className="px-5 py-4 text-sm text-ink-2">
          <span className="font-semibold">{patientName}</span> will be split
          out of master record <span className="font-mono font-bold">{mid}</span>{" "}
          into its own new master record. This action is recorded in the audit
          log.
          <div className="mt-3 rounded-md border border-line bg-bg px-3 py-2 font-mono text-xs text-gray-2">
            {patid}
          </div>
        </div>
        <div className="flex justify-end gap-2.5 border-t border-line px-5 py-3.5">
          <button
            onClick={onCancel}
            disabled={pending}
            className="rounded-md border border-line px-4 py-2 text-sm font-semibold text-gray-2 hover:bg-bg disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={pending}
            className="rounded-md bg-status-nomatch px-4 py-2 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-50"
          >
            {pending ? "Unmerging…" : "Confirm unmerge"}
          </button>
        </div>
      </div>
    </div>
  );
}
