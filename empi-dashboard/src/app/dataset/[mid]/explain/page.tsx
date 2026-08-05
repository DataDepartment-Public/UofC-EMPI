"use client";

import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { compareRecords } from "@/lib/compare";
import { decodeExplainPayload } from "@/lib/explain";
import { fullName } from "@/lib/format";
import { FeatureComparisonTable } from "@/components/shared/FeatureComparisonTable";
import { ShapWaterfall } from "@/components/shared/ShapWaterfall";
import { useDashboardSummary, usePairExplanation } from "@/lib/hooks";

/** FR-31..FR-38: per-pair "why was this a match?" detail, reached from a
 * Dataset dropdown. The deterministic side (rule fired, fixed confidence,
 * field-by-field comparison) is always real. The probabilistic side — a
 * calibrated match probability and SHAP waterfall — comes from
 * `GET /explanations/{model}/{a}/{b}` (the gate + ML matcher's persisted,
 * exact-TreeSHAP contributions); it's `null` for a pair neither model ever
 * scored (a purely deterministic-rule match), which this page shows
 * honestly rather than fabricating a probability. */
export default function ExplainPage() {
  return (
    <Suspense fallback={<p className="text-sm text-gray">Loading…</p>}>
      <ExplainPageContent />
    </Suspense>
  );
}

function ExplainPageContent() {
  const searchParams = useSearchParams();
  const payload = decodeExplainPayload(searchParams.get("d"));
  const { data: summary } = useDashboardSummary();
  const { data: explanation, isLoading: explanationLoading } = usePairExplanation(
    payload?.patientA.patid ?? null,
    payload?.patientB.patid ?? null,
  );

  if (!payload) {
    return (
      <div>
        <BackLink />
        <p className="card mt-4 p-6 text-sm text-status-nomatch">
          This explanation link is missing its comparison data. Go back to the
          Dataset tab and click a match again.
        </p>
      </div>
    );
  }

  const { patientA, patientB, rule, confidence, evidence, updated } = payload;
  const rows = compareRecords(patientA, patientB);
  const predictedClass = rule ? "Confirmed duplicate (rule-matched)" : "Uncertain — pending review";

  return (
    <div>
      <BackLink />

      <div className="mb-5 mt-3">
        <h2 className="text-[22px] font-extrabold text-ink-2">
          Match explanation — {fullName(patientA.first_name, patientA.last_name)} vs{" "}
          {fullName(patientB.first_name, patientB.last_name)}
        </h2>
        <p
          className="mt-0.5 font-mono text-[11px] text-gray"
          title="Patient IDs compared"
        >
          {patientA.patid} vs {patientB.patid}
        </p>
        <p className="mt-1 text-[13px] text-gray">
          Why this pair was — or wasn&apos;t yet — classified as a likely duplicate.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        <MetaCard label="Rule fired" value={rule ?? "None (unconfirmed)"} />
        <MetaCard label="Predicted class" value={predictedClass} />
        <MetaCard label="Rule confidence" value={confidence != null ? confidence.toFixed(3) : "—"} mono />
        <MetaCard label="Model / git version" value={summary?.model_version?.slice(0, 12) ?? "—"} mono />
        <MetaCard
          label="Generated"
          value={updated ? new Date(updated).toLocaleString() : "at review time"}
        />
      </div>

      <div className="card mt-5 p-5">
        <h4 className="mb-1 text-[15px] font-semibold text-ink-2">Feature comparison</h4>
        <p className="mb-4 text-xs text-gray">
          Every field the deterministic rules consider, compared exactly as the
          pipeline sees them (cleaned/normalized values, not raw source text).
        </p>
        <FeatureComparisonTable
          rows={rows}
          patidA={patientA.patid}
          patidB={patientB.patid}
        />
      </div>

      {evidence && evidence !== rule && (
        <div className="card mt-5 p-5">
          <h4 className="mb-1 text-[15px] font-semibold text-ink-2">
            All rules that fired
          </h4>
          <p className="text-[13px] text-gray-2">{evidence}</p>
        </div>
      )}

      <div className="card mt-5 flex items-center justify-between gap-4 border-[#cfe6f7] bg-[#f3f9fe] p-5">
        <div>
          <h4 className="text-[15px] font-semibold text-brand-blue">
            {explanation
              ? explanation.model === "ml_matcher"
                ? "ML matcher signal"
                : "Non-match gate signal"
              : "ML pipeline signal"}
          </h4>
          <p className="mt-1 max-w-md text-xs text-gray-2">
            {explanationLoading
              ? "Loading model explanation…"
              : explanation
                ? `Scored by ${explanation.model}${explanation.run_id ? ` (run ${explanation.run_id})` : ""}.`
                : "Not scored by the ML pipeline — this pair was resolved by deterministic rules alone."}
          </p>
        </div>
        {explanation && (
          <div className="text-right">
            <div className="text-2xl font-extrabold text-brand-blue tabular-nums">
              {Math.round(explanation.decision.score * 100)}%
            </div>
            <div className="text-xs font-semibold text-gray-2">
              {explanation.decision.tier}
            </div>
          </div>
        )}
      </div>

      {explanation && (
        <div className="card mt-5 p-5">
          <h4 className="mb-1 text-[15px] font-semibold text-ink-2">
            Feature contributions (SHAP)
          </h4>
          <p className="mb-4 text-xs text-gray">
            Exact TreeSHAP contributions from the{" "}
            {explanation.model === "ml_matcher" ? "ML matcher" : "non-match gate"}{" "}
            model, in log-odds. The dashed line marks the model&apos;s base
            rate; each bar shows how one feature moved the decision from
            there.
          </p>
          <ShapWaterfall explanation={explanation} />
        </div>
      )}
    </div>
  );
}

function MetaCard({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="card p-3.5">
      <div className="text-[10px] font-bold tracking-wide text-gray uppercase">
        {label}
      </div>
      <div className={`mt-1 text-sm font-bold text-ink-2 ${mono ? "font-mono text-xs" : ""}`}>
        {value}
      </div>
    </div>
  );
}

/** Uses browser history rather than a constructed href so the reviewer
 * lands back on whatever Patient Registry search/filter/page they were on
 * — not a fresh view refiltered to this one pair. Safe because this page
 * is only ever reached via a same-tab push from that list. */
function BackLink() {
  const router = useRouter();
  return (
    <button
      type="button"
      onClick={() => router.back()}
      className="text-sm font-semibold text-brand-blue hover:underline"
    >
      ‹ Back to Patient Registry
    </button>
  );
}
