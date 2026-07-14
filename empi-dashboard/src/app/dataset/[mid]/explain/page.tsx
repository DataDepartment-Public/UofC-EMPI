"use client";

import { Suspense, use } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { compareRecords, plainLanguageSummary } from "@/lib/compare";
import { decodeExplainPayload } from "@/lib/explain";
import { fullName } from "@/lib/format";
import { FeatureComparisonTable } from "@/components/FeatureComparisonTable";
import { useDashboardSummary } from "@/lib/hooks";

/** FR-31..FR-38: per-pair "why was this a match?" detail, reached from a
 * Dataset dropdown. Scope note: FR-34's Fellegi-Sunter waterfall and a
 * calibrated match probability need the probabilistic model stage, which
 * docs/Application-Architecture.md explicitly defers ("Model Explanation
 * sub-page once the probabilistic stage lands") — only deterministic rules
 * run in production today. This page shows what's real: the rule that
 * fired (or didn't), its fixed confidence, and a genuine field-by-field
 * comparison — no fabricated probability or waterfall. */
export default function ExplainPage({
  params,
}: {
  params: Promise<{ mid: string }>;
}) {
  const { mid } = use(params);
  return (
    <Suspense fallback={<p className="text-sm text-gray">Loading…</p>}>
      <ExplainPageContent mid={mid} />
    </Suspense>
  );
}

function ExplainPageContent({ mid }: { mid: string }) {
  const searchParams = useSearchParams();
  const payload = decodeExplainPayload(searchParams.get("d"));
  const { data: summary } = useDashboardSummary();

  if (!payload) {
    return (
      <div>
        <BackLink mid={mid} />
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
      <BackLink mid={mid} />

      <div className="mb-5 mt-3">
        <h2 className="text-[22px] font-extrabold text-ink-2">
          Match explanation — {fullName(patientA.first_name, patientA.last_name)} vs{" "}
          {fullName(patientB.first_name, patientB.last_name)}
        </h2>
        <p className="mt-1 text-[13px] text-gray">
          Why this pair was — or wasn&apos;t yet — classified as a likely duplicate.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        <MetaCard label="Patients compared" value={`${patientA.patid.slice(0, 10)}… vs ${patientB.patid.slice(0, 10)}…`} mono />
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
        <FeatureComparisonTable rows={rows} />
      </div>

      {evidence && evidence !== rule && (
        <div className="card mt-5 p-5">
          <h4 className="mb-1 text-[15px] font-semibold text-ink-2">
            All rules that fired
          </h4>
          <p className="text-[13px] text-gray-2">{evidence}</p>
        </div>
      )}

      <div className="card mt-5 border-[#cfe6f7] bg-[#f3f9fe] p-5">
        <h4 className="mb-2 text-[15px] font-semibold text-brand-blue">
          Plain-language summary
        </h4>
        <p className="text-[14px] text-ink-2">{plainLanguageSummary(rule, rows)}</p>
      </div>
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

function BackLink({ mid }: { mid: string }) {
  return (
    <Link
      href={`/dataset?search=${encodeURIComponent(mid)}`}
      className="text-sm font-semibold text-brand-blue hover:underline"
    >
      ‹ Back to {mid} in Dataset
    </Link>
  );
}
