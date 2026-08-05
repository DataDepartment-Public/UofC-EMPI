"use client";

import { useState } from "react";
import { ApiError } from "@/lib/api-client";
import { useThresholds, useUpdateThresholds } from "@/lib/hooks";
import { Toast } from "@/components/shared/Toast";
import { SystemStatusPanel } from "@/components/admin/SystemStatusPanel";
import type { ThresholdSettings } from "@/lib/schemas";

const FIELDS: {
  key: keyof ThresholdSettings;
  label: string;
  description: string;
}[] = [
  {
    key: "gate_threshold",
    label: "Gate threshold",
    description:
      "P(plausible) at/above which a pair passes the non-match gate and " +
      "reaches the ML matcher. Below it, the pair is a confident non-match " +
      "and is discarded.",
  },
  {
    key: "ml_auto_merge_threshold",
    label: "ML auto-merge threshold",
    description:
      'Match-probability at/above which the ML matcher tiers a pair "auto_merge".',
  },
  {
    key: "ml_review_floor",
    label: "ML review floor",
    description:
      'Match-probability at/above which the ML matcher tiers a pair "human_review" ' +
      "(also the candidate-inclusion floor for the ML matcher's feature parquet).",
  },
];

/** Admin console for the live ML decision thresholds — no auth (handled
 * elsewhere) and no audit trail: this is operator configuration, not a
 * reviewer action on a specific patient pair. Saving takes effect
 * immediately on the running backend and persists across a restart, but
 * only ever affects future scoring — it never rewrites tiers a prior run
 * already computed and published (see src/api/threshold_store.py). */
export default function AdminPage() {
  const { data, isLoading, isError } = useThresholds();

  return (
    <div>
      <div className="mb-5">
        <h2 className="text-[22px] font-extrabold text-ink-2">Admin</h2>
        <p className="mt-1 max-w-2xl text-[13px] text-gray">
          Tune the ML pipeline&apos;s live decision thresholds. Changes apply
          immediately to the running service and persist across a restart —
          but only affect future scoring, never tiers already computed and
          published by a prior run.
        </p>
      </div>

      <div className="flex flex-wrap justify-center gap-5">
        <div className="w-full max-w-md">
          {isLoading && (
            <p className="text-sm text-gray">Loading thresholds…</p>
          )}
          {isError && (
            <p className="text-sm text-status-nomatch">
              Couldn&apos;t reach the eMPI API. Is the backend running?
            </p>
          )}
          {data && <ThresholdForm initial={data} />}
        </div>

        <SystemStatusPanel />
      </div>
    </div>
  );
}

/** Mounted only once `initial` is loaded, so `useState(initial)` captures it
 * directly as the form's starting draft — no effect needed to sync fetched
 * data into local state. */
function ThresholdForm({ initial }: { initial: ThresholdSettings }) {
  const updateMutation = useUpdateThresholds();
  const [values, setValues] = useState<ThresholdSettings>(initial);
  const [toast, setToast] = useState<string | null>(null);

  const flash = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3200);
  };

  const handleSave = () => {
    updateMutation.mutate(values, {
      onSuccess: () => flash("Thresholds saved."),
      onError: (err) =>
        flash(err instanceof ApiError ? err.message : "Save failed."),
    });
  };

  return (
    <div className="card w-full max-w-md p-5">
      <div className="space-y-5">
        {FIELDS.map((f) => (
          <div key={f.key}>
            <label className="mb-1 block text-[13px] font-bold text-ink-2">
              {f.label}
            </label>
            <p className="mb-2 text-[11.5px] text-gray">{f.description}</p>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={values[f.key]}
              onChange={(e) => {
                const next = Number(e.target.value);
                setValues((v) => ({ ...v, [f.key]: next }));
              }}
              className="w-32 rounded-md border border-line px-3 py-1.5 text-sm text-ink-2"
            />
          </div>
        ))}
      </div>

      <button
        onClick={handleSave}
        disabled={updateMutation.isPending}
        className="mt-5 rounded-md bg-brand-blue px-4 py-2 text-sm font-bold text-white hover:opacity-90 disabled:opacity-50"
      >
        {updateMutation.isPending ? "Saving…" : "Save"}
      </button>

      <Toast message={toast} />
    </div>
  );
}
