"use client";

import { useState } from "react";
import clsx from "clsx";
import { usePairExplanations } from "@/lib/hooks";
import type { PairExplanation, ReviewQueueItem } from "@/lib/schemas";
import { ShapWaterfall } from "@/components/shared/ShapWaterfall";
import { skipReason } from "@/components/shared/PipelineStages";

type ExplanationModel = "nonmatch_gate" | "ml_matcher";

/** What each model's waterfall is explaining. Both are `P(same patient)` —
 * see `PipelineStages`' note on why neither is ever shown as its complement —
 * but they answer different questions at different stages, and a waterfall
 * carries no label of its own. Reading the gate's chart as the matcher's is
 * the whole failure this switcher exists to prevent: the two look identical,
 * share most of their features (the gate reuses `FeatureBuilderV5`), and
 * routinely disagree — a pair can be 99.9% plausible and 1.1% a confident
 * match at the same time, both correct. */
const MODELS: Record<
  ExplanationModel,
  { label: string; quantity: string; decides: string; stage: "gate" | "matcher" }
> = {
  nonmatch_gate: {
    label: "Non-match gate",
    quantity: "P(plausible)",
    decides: "whether the matcher sees this pair at all",
    stage: "gate",
  },
  ml_matcher: {
    label: "ML matcher",
    quantity: "P(confident match)",
    decides: "whether this pair auto-merges",
    stage: "matcher",
  },
};

/** The SHAP waterfall plus the one thing it can't say for itself: which model
 * drew it.
 *
 * Defaults to the pair's *decisive* model — the matcher when it ran, since
 * that's the stage that decided whether an edge formed; otherwise the gate,
 * which is then where the pair's journey ended. A model that didn't run has
 * no persisted explanation, so its tab is disabled and says why rather than
 * being hidden: "the gate dropped this pair" is information a reviewer wants,
 * and a tab that silently vanishes reads as a bug. */
export function ExplanationPanel({ item }: { item: ReviewQueueItem }) {
  const { gate, ml, isLoading } = usePairExplanations(item.patid_a, item.patid_b);
  // Null means "follow the default"; the panel is keyed by pair upstream, so
  // an explicit pick never leaks across candidates.
  const [picked, setPicked] = useState<ExplanationModel | null>(null);

  if (isLoading) {
    return (
      <Section>
        <p className="text-sm text-gray">Loading explanation…</p>
      </Section>
    );
  }

  const available: Record<ExplanationModel, PairExplanation | null> = {
    nonmatch_gate: gate.data ?? null,
    ml_matcher: ml.data ?? null,
  };
  if (!available.nonmatch_gate && !available.ml_matcher) return null;

  const active: ExplanationModel =
    picked && available[picked]
      ? picked
      : available.ml_matcher
        ? "ml_matcher"
        : "nonmatch_gate";
  const explanation = available[active]!;
  const meta = MODELS[active];

  /** Why a model produced no explanation for this pair. The matcher's reason
   * is usually the gate's decision, which the pair's verdict only names when
   * the gate is what ended its journey — so check the gate's own tier first. */
  const reasonFor = (model: ExplanationModel) =>
    available[model]
      ? null
      : skipReason(
          gate.data?.decision.tier === "no_match" ? "gate_dropped" : item.verdict,
          MODELS[model].stage,
        );
  const missing = (Object.keys(MODELS) as ExplanationModel[]).find(
    (m) => !available[m],
  );

  return (
    <Section
      tabs={(Object.keys(MODELS) as ExplanationModel[]).map((model) => (
        <ModelTab
          key={model}
          label={MODELS[model].label}
          active={model === active}
          reason={reasonFor(model)}
          onClick={() => setPicked(model)}
        />
      ))}
    >
      <p className="text-[11px] text-gray-2">
        <span className="font-bold text-ink-2">
          {meta.label} · {meta.quantity} {pct(explanation.decision.score)}
        </span>{" "}
        — the stage that decides {meta.decides}. Bars show what moved this
        model&apos;s confidence, not the other&apos;s.
      </p>
      {/* Spelled out rather than left to the disabled tab's tooltip: a
          disabled button fires no hover events, so its `title` may never
          appear — and "the gate dropped this pair" is the reason the other
          chart is missing, which a reviewer should not have to discover. */}
      {missing && (
        <p className="text-[11px] text-gray italic">
          {MODELS[missing].label}: {lowerFirst(reasonFor(missing)!)}
        </p>
      )}
      <div className="mt-2">
        <ShapWaterfall explanation={explanation} />
      </div>
    </Section>
  );
}

/** The skip reasons read as sentences ("Didn't run — …"); after a model name
 * and colon they should read as clauses. */
function lowerFirst(s: string): string {
  return s.charAt(0).toLowerCase() + s.slice(1);
}

function Section({
  tabs,
  children,
}: {
  tabs?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="mb-5">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-[13px] font-bold text-ink-2">
          Feature contributions (SHAP)
        </h4>
        {tabs && (
          <div className="flex overflow-hidden rounded-md border border-line">
            {tabs}
          </div>
        )}
      </div>
      {children}
    </div>
  );
}

/** `reason` non-null means this model produced no explanation for the pair:
 * the tab is inert and carries the reason as its tooltip. */
function ModelTab({
  label,
  active,
  reason,
  onClick,
}: {
  label: string;
  active: boolean;
  reason: string | null;
  onClick: () => void;
}) {
  const disabled = reason !== null;
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      title={reason ?? undefined}
      aria-pressed={active}
      className={clsx(
        "px-3 py-1 text-[11px] font-bold",
        "[&:not(:last-child)]:border-r [&:not(:last-child)]:border-line",
        disabled
          ? "cursor-not-allowed bg-bg text-gray italic"
          : active
            ? "bg-brand-blue text-white"
            : "text-gray-2 hover:bg-bg",
      )}
    >
      {label}
      {disabled && " · didn't run"}
    </button>
  );
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}
