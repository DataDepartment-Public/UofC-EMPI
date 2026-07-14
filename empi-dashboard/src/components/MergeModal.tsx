"use client";

/** Confirmation before a permanent merge — names exactly which records will
 * be combined (docs/Dashboard-Guide.md FR-28). */
export function MergeModal({
  targetMid,
  patids,
  pending,
  onConfirm,
  onCancel,
}: {
  targetMid: string;
  patids: string[];
  pending: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/30 p-4">
      <div className="card w-full max-w-md p-0">
        <div className="border-b border-line px-5 py-4">
          <h3 className="text-base font-bold text-ink-2">Confirm merge</h3>
        </div>
        <div className="px-5 py-4 text-sm text-ink-2">
          The following record{patids.length > 1 ? "s" : ""} will be combined
          into master record <span className="font-mono font-bold">{targetMid}</span>.
          This action is recorded in the audit log.
          <ul className="mt-3 list-disc space-y-1 pl-5 font-mono text-xs text-gray-2">
            {patids.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
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
            className="rounded-md bg-brand-blue px-4 py-2 text-sm font-semibold text-white hover:bg-brand-blue-bright disabled:opacity-50"
          >
            {pending ? "Merging…" : "Confirm merge"}
          </button>
        </div>
      </div>
    </div>
  );
}
