"use client";

import { useState } from "react";
import Link from "next/link";
import clsx from "clsx";
import { useClusterPairs, usePairExplanation } from "@/lib/hooks";
import type {
  ClusterExternalPair,
  ClusterPair,
  ClusterPairVerdict,
  Entity,
} from "@/lib/schemas";
import { formatRawDate, fullName, maskSsn } from "@/lib/format";
import { ShapWaterfall } from "@/components/shared/ShapWaterfall";

/** How each verdict reads to a reviewer, and which of the three status
 * colors it wears. Order matters twice over: it is the backend's
 * most-decisive-stage-first ladder (`CLUSTER_PAIR_VERDICTS`), and the sort
 * order of the list below — a reviewer should see the edges that actually
 * built the cluster before the pairs nothing decided. */
const VERDICT: Record<
  ClusterPairVerdict,
  { label: string; tone: "auto" | "review" | "nomatch" | "muted"; blurb: string }
> = {
  auto_merge_rule: {
    label: "Deterministic rule",
    tone: "auto",
    blurb: "A deterministic auto-merge rule confirmed this pair outright.",
  },
  reject: {
    label: "Rejected by rules",
    tone: "nomatch",
    blurb:
      "The deterministic rules found enough strong-identifier conflicts to " +
      "call this a confident non-match.",
  },
  ml_auto_merge: {
    label: "ML matcher · merged",
    tone: "auto",
    blurb: "The ML matcher scored this pair a confident match.",
  },
  ml_human_review: {
    label: "ML matcher · ambiguous",
    tone: "review",
    blurb:
      "The ML matcher scored this pair below its auto-merge threshold, so it " +
      "formed no edge of its own.",
  },
  gate_dropped: {
    label: "Dropped by non-match gate",
    tone: "nomatch",
    blurb:
      "The non-match gate scored this pair a confident non-match and dropped " +
      "it before the ML matcher ever saw it.",
  },
  blocked_undecided: {
    label: "Blocked, not decided",
    tone: "review",
    blurb:
      "Blocking put these two in the same candidate set, but no stage in this " +
      "run recorded a decision about them.",
  },
  not_compared: {
    label: "Not compared",
    tone: "muted",
    blurb:
      "These two were never directly compared. They share a cluster because " +
      "each matched a third record — clustering joined them transitively.",
  },
};

const VERDICT_ORDER: ClusterPairVerdict[] = [
  "auto_merge_rule",
  "ml_auto_merge",
  "ml_human_review",
  "blocked_undecided",
  "gate_dropped",
  "reject",
  "not_compared",
];

const TONE: Record<string, string> = {
  auto: "bg-status-auto/10 text-status-auto",
  review: "bg-status-review/10 text-status-review",
  nomatch: "bg-status-nomatch/10 text-status-nomatch",
  muted: "bg-bg text-gray",
};

/** The evidence section: every pair of the cluster's members, what compared
 * them and what it concluded.
 *
 * This is the question the old registry couldn't answer at all. `Entity`
 * carries a single founding rule for the whole cluster, so a three-record
 * cluster built from two different rules — plus a third pair nothing ever
 * looked at — presented as one confidence number. Here each pair states its
 * own case, including the ones that were rejected, dropped, or joined only
 * by transitivity.
 */
export function ClusterComparisons({ entity }: { entity: Entity }) {
  const { data, isLoading, isError } = useClusterPairs(entity.mid);
  const nameOf = (patid: string) => {
    const m = entity.members.find((x) => x.patid === patid);
    return m ? fullName(m.first_name, m.last_name) : patid;
  };

  if (isLoading) {
    return <p className="text-sm text-gray">Loading comparisons…</p>;
  }
  if (isError || !data) {
    return (
      <p className="text-sm text-status-nomatch">
        Couldn&apos;t load the comparison trace for this cluster.
      </p>
    );
  }
  if (data.pairs.length === 0) {
    return (
      <p className="rounded-md border border-line bg-bg px-3 py-2 text-sm text-gray">
        A single-record cluster has no internal pairs — see the comparison
        history below for what it was checked against.
      </p>
    );
  }

  return (
    <div>
      {!data.artifacts_available && (
        <p className="mb-3 rounded-md border border-line bg-bg px-3 py-2 text-xs text-gray-2">
          The pipeline artifacts for run{" "}
          <span className="font-mono">{data.run_id ?? entity.run_id}</span> are no
          longer on disk, so no stage detail is available for these pairs.
        </p>
      )}
      <div className="space-y-2">
        {byVerdict(data.pairs).map((pair) => (
          <PairCard
            key={`${pair.patid_a}-${pair.patid_b}`}
            pair={pair}
            nameA={nameOf(pair.patid_a)}
            nameB={nameOf(pair.patid_b)}
            nameOf={nameOf}
            runId={data.run_id ?? undefined}
            thresholds={data.thresholds}
          />
        ))}
      </div>
    </div>
  );
}

/** Everything this cluster's records were checked against and *not* merged
 * with — the near-misses.
 *
 * The section above explains how the cluster was assembled; this one explains
 * where it stopped. For a singleton that is the entire story, and the
 * difference between "nothing was ever compared to this record" and "six
 * records were compared and every one was rejected" is the difference between
 * a gap in blocking and a confident answer.
 *
 * Rendered as its own section rather than folded in above because the two
 * lists answer different questions and a reviewer scans them at different
 * times — and because the counterpart lives in another cluster, so each row
 * needs to name it and link the reader out.
 */
export function ClusterComparisonHistory({ entity }: { entity: Entity }) {
  const { data, isLoading, isError } = useClusterPairs(entity.mid);

  if (isLoading) {
    return <p className="text-sm text-gray">Loading comparison history…</p>;
  }
  if (isError || !data) {
    return (
      <p className="text-sm text-status-nomatch">
        Couldn&apos;t load the comparison history for this cluster.
      </p>
    );
  }
  if (data.external_pairs.length === 0) {
    return (
      <p className="rounded-md border border-line bg-bg px-3 py-2 text-sm text-gray">
        Blocking never paired {data.members.length === 1 ? "this record" : "these records"}{" "}
        with anything outside this cluster, so no other candidate was ever
        scored.
      </p>
    );
  }

  const pairs = byVerdict(data.external_pairs);
  const pending = pairs.filter((p) => isAwaitingReview(p.verdict)).length;
  const declined = pairs.filter(
    (p) => p.verdict === "reject" || p.verdict === "gate_dropped",
  ).length;

  return (
    <div>
      <p className="mb-2.5 text-[11px] text-gray">
        {pairs.length} record{pairs.length === 1 ? "" : "s"} outside this cluster{" "}
        {pairs.length === 1 ? "was" : "were"} compared against{" "}
        {data.members.length === 1 ? "it" : "its members"} and not merged
        {pending > 0 || declined > 0 ? " — " : "."}
        {declined > 0 && (
          <span className="font-bold text-status-nomatch">
            {declined} ruled out
          </span>
        )}
        {declined > 0 && pending > 0 && ", "}
        {pending > 0 && (
          <span className="font-bold text-status-review">
            {pending} awaiting review
          </span>
        )}
        {(pending > 0 || declined > 0) && "."}
      </p>
      <div className="space-y-2">
        {pairs.map((pair) => (
          <ExternalPairCard
            key={`${pair.patid_a}-${pair.patid_b}`}
            pair={pair}
            memberName={nameOfMember(entity, pair.member_patid)}
            runId={data.run_id ?? undefined}
            thresholds={data.thresholds}
          />
        ))}
      </div>
    </div>
  );
}

function nameOfMember(entity: Entity, patid: string): string {
  const m = entity.members.find((x) => x.patid === patid);
  return m ? fullName(m.first_name, m.last_name) : patid;
}

/** Most-decisive stage first, so the edges that actually built the cluster
 * (or the rejections a reviewer is most likely to want to overturn) come
 * before the pairs nothing decided. */
function byVerdict<T extends { verdict: ClusterPairVerdict }>(pairs: T[]): T[] {
  return [...pairs].sort(
    (a, b) => VERDICT_ORDER.indexOf(a.verdict) - VERDICT_ORDER.indexOf(b.verdict),
  );
}

function ExternalPairCard({
  pair,
  memberName,
  runId,
  thresholds,
}: {
  pair: ClusterExternalPair;
  memberName: string;
  runId?: string;
  thresholds: Record<string, number>;
}) {
  const [open, setOpen] = useState(false);
  const verdict = VERDICT[pair.verdict];
  const otherName = fullName(pair.other_first_name, pair.other_last_name);

  return (
    <div className="overflow-hidden rounded-md border border-line">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-3.5 py-2.5 text-left hover:bg-bg"
      >
        <span className="flex min-w-0 items-center gap-2.5">
          <span className="text-gray">{open ? "▾" : "▸"}</span>
          <span className="truncate text-[13px]">
            <span className="text-gray-2">{memberName}</span>
            <span className="mx-1.5 text-gray">↔</span>
            <span className="font-bold text-ink-2">{otherName}</span>
          </span>
          <span className="shrink-0 font-mono text-[10px] text-gray">
            {pair.other_patid}
            {pair.other_mid && ` · ${pair.other_mid}`}
          </span>
        </span>
        <Badge tone={verdict.tone}>{verdict.label}</Badge>
      </button>

      {open && (
        <div className="border-t border-line px-3.5 py-3">
          <div className="mb-3 flex items-start justify-between gap-3">
            <p className="text-xs text-gray-2">{verdict.blurb}</p>
            {isAwaitingReview(pair.verdict) && (
              <Link
                href={`/review?patid_a=${encodeURIComponent(pair.patid_a)}&patid_b=${encodeURIComponent(pair.patid_b)}`}
                className="shrink-0 whitespace-nowrap text-[11px] font-bold text-brand-blue hover:underline"
              >
                Open in Review Queue →
              </Link>
            )}
          </div>
          <OtherRecordSummary pair={pair} />
          <StageStrip pair={pair} thresholds={thresholds} internal={false} />
          <Waterfall pair={pair} runId={runId} />
        </div>
      )}
    </div>
  );
}

/** Whether a pair's verdict is exactly the set the Review Queue's "Needs
 * review" tab would surface for it — see `ClusterComparisonHistory`'s
 * `pending` count, which uses this same pair. Not `reject`/`gate_dropped`
 * (a confident automated non-match, never queued) and not the merge
 * verdicts (already resolved). */
function isAwaitingReview(verdict: ClusterPairVerdict): boolean {
  return verdict === "ml_human_review" || verdict === "blocked_undecided";
}

/** Enough of the declined record to judge the decision without leaving the
 * page. Deliberately not a full comparison table — that record belongs to
 * another cluster, and the link is there for a reviewer who wants the whole
 * picture. */
function OtherRecordSummary({ pair }: { pair: ClusterExternalPair }) {
  return (
    <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md bg-bg px-3 py-2 text-[11px] text-gray-2">
      <span>
        <span className="text-gray">Compared against </span>
        <span className="font-bold text-ink-2">
          {fullName(pair.other_first_name, pair.other_last_name)}
        </span>
      </span>
      <span>
        <span className="text-gray">DOB </span>
        {formatRawDate(pair.other_birth_date) || "—"}
      </span>
      <span>
        <span className="text-gray">SSN </span>
        {maskSsn(pair.other_ssn_last4)}
      </span>
      {pair.other_mid && (
        <Link
          href={`/dataset?mid=${encodeURIComponent(pair.other_mid)}`}
          className="font-bold text-brand-blue hover:underline"
        >
          Open {pair.other_mid} →
        </Link>
      )}
    </div>
  );
}

function PairCard({
  pair,
  nameA,
  nameB,
  nameOf,
  runId,
  thresholds,
}: {
  pair: ClusterPair;
  nameA: string;
  nameB: string;
  nameOf: (patid: string) => string;
  runId?: string;
  thresholds: Record<string, number>;
}) {
  const [open, setOpen] = useState(false);
  const verdict = VERDICT[pair.verdict];

  return (
    <div className="overflow-hidden rounded-md border border-line">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-3.5 py-2.5 text-left hover:bg-bg"
      >
        <span className="flex min-w-0 items-center gap-2.5">
          <span className="text-gray">{open ? "▾" : "▸"}</span>
          <span className="truncate text-[13px] font-bold text-ink-2">
            {nameA} ↔ {nameB}
          </span>
          <span className="shrink-0 font-mono text-[10px] text-gray">
            {pair.patid_a} · {pair.patid_b}
          </span>
        </span>
        <span className="flex shrink-0 items-center gap-2">
          {pair.joined_by === "reviewer" && (
            <Badge tone="review">Manually merged by {pair.reviewer_id}</Badge>
          )}
          <Badge tone={verdict.tone}>{verdict.label}</Badge>
        </span>
      </button>

      {open && (
        <div className="border-t border-line px-3.5 py-3">
          <p className="mb-3 text-xs text-gray-2">{verdict.blurb}</p>
          <StageStrip pair={pair} thresholds={thresholds} nameOf={nameOf} internal />
          <Waterfall pair={pair} runId={runId} />
        </div>
      )}
    </div>
  );
}

/** Blocking -> rules -> gate -> matcher -> clustering for one pair, in the
 * same joined-cards language as the Review Queue's `PipelineTrail`. Reads
 * entirely from the pair payload — no fetch — so the strip renders
 * instantly on expand and the (slower) waterfall below fills in after. */
function StageStrip({
  pair,
  thresholds,
  nameOf = (patid) => patid,
  internal = true,
}: {
  pair: ClusterPair;
  thresholds: Record<string, number>;
  /** Maps a chain member's patid to a display name, for the clustering
   * stage's path. Defaults to the identity (bare patid) when the caller has
   * no name lookup handy — `cluster_path` is never populated in that case
   * anyway (external comparisons), so it only ever affects rendering. */
  nameOf?: (patid: string) => string;
  /** Whether `pair` is an internal (within-cluster) pair, for which the
   * clustering stage is meaningful, vs. an external comparison, for which
   * it never is — the two records live in different clusters by
   * definition. */
  internal?: boolean;
}) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const isDirectEdge =
    pair.verdict === "auto_merge_rule" ||
    pair.verdict === "ml_auto_merge" ||
    pair.joined_by === "reviewer";

  return (
    <div className="overflow-x-auto pb-1">
      <div className="flex min-w-[700px]">
        <Stage num={1} label="Blocking">
          {pair.blocked ? (
            <>
              <Row ok>Same candidate set</Row>
              <Row>{pair.source_blocks ?? "—"}</Row>
            </>
          ) : (
            <Row muted>Never blocked together</Row>
          )}
        </Stage>

        <Stage num={2} label="Rules">
          {pair.match_rule ? (
            <>
              <Row ok>{pair.match_rule} fired</Row>
              <Row>
                {pair.rules_fired && pair.rules_fired !== pair.match_rule
                  ? `All: ${pair.rules_fired}`
                  : `Confidence ${
                      pair.confidence != null ? pct(pair.confidence) : "—"
                    }`}
              </Row>
            </>
          ) : pair.reject_rule ? (
            <>
              <Row bad>{pair.reject_rule}</Row>
              <Row>{pair.n_contradictions} strong contradictions</Row>
            </>
          ) : (
            <Row muted>No rule confirmed or rejected</Row>
          )}
        </Stage>

        <Stage num={3} label="Non-match gate">
          {pair.gate_score != null ? (
            <>
              <Row bad={pair.gate_tier === "no_match"} ok={pair.gate_tier !== "no_match"}>
                {pct(pair.gate_score)} plausible
              </Row>
              <Row>
                {pair.gate_tier === "no_match" ? "Dropped" : "Passed"} · threshold{" "}
                {thresholds.gate_threshold != null
                  ? pct(thresholds.gate_threshold)
                  : "—"}
              </Row>
            </>
          ) : (
            <Row muted>Not scored by the gate</Row>
          )}
        </Stage>

        <Stage num={4} label="ML matcher">
          {pair.ml_score != null ? (
            <>
              <Row ok={pair.ml_tier === "auto_merge"}>{pct(pair.ml_score)} match</Row>
              <Row>
                {pair.ml_tier} · threshold{" "}
                {thresholds.ml_auto_merge_threshold != null
                  ? pct(thresholds.ml_auto_merge_threshold)
                  : "—"}
              </Row>
            </>
          ) : (
            <Row muted>Not scored by the matcher</Row>
          )}
        </Stage>

        <Stage num={5} label="Clustering">
          {!internal ? (
            <Row muted>Different clusters — not applicable</Row>
          ) : isDirectEdge ? (
            <Row ok>This pair&apos;s own match formed the merge edge</Row>
          ) : pair.cluster_path && pair.cluster_path.length > 1 ? (
            <>
              <Row ok>
                Joined transitively ({pair.cluster_path.length - 1} hop
                {pair.cluster_path.length - 1 === 1 ? "" : "s"})
              </Row>
              <PathChain path={pair.cluster_path} nameOf={nameOf} />
            </>
          ) : (
            <Row muted>No merge path found for this run</Row>
          )}
        </Stage>
      </div>
    </div>
  );
}

/** The chain of members whose own confirmed merges connected two records
 * that formed no edge of their own — the "record ids that determined the
 * merge" a reviewer needs to trace a surprising grouping back to its
 * source. Patids lead (they're what a reviewer looks records up by); the
 * name rides along as a tooltip. */
function PathChain({
  path,
  nameOf,
}: {
  path: string[];
  nameOf: (patid: string) => string;
}) {
  return (
    <div className="mt-0.5 flex flex-wrap items-center gap-x-1 gap-y-0.5">
      {path.map((patid, i) => (
        <span key={`${patid}-${i}`} className="flex items-center gap-1">
          {i > 0 && <span className="text-[10px] text-gray">→</span>}
          <span
            className="font-mono text-[10px] text-gray-2"
            title={nameOf(patid)}
          >
            {patid}
          </span>
        </span>
      ))}
    </div>
  );
}

/** The SHAP waterfall, fetched only when a pair is expanded and only for the
 * verdicts a model actually produced. A deterministic auto-merge or a
 * transitive pair has no model decision to explain — its provenance is the
 * rule name in the strip above — so asking would just 404. */
function Waterfall({ pair, runId }: { pair: ClusterPair; runId?: string }) {
  const scored =
    pair.verdict === "ml_auto_merge" ||
    pair.verdict === "ml_human_review" ||
    pair.verdict === "gate_dropped";

  const { data, isLoading } = usePairExplanation(
    scored ? pair.patid_a : null,
    scored ? pair.patid_b : null,
    runId,
  );

  if (!scored) return null;

  return (
    <div className="mt-4">
      <h4 className="mb-2 text-[11px] font-bold tracking-wide text-gray uppercase">
        Feature contributions (SHAP)
      </h4>
      {isLoading && <p className="text-sm text-gray">Loading explanation…</p>}
      {!isLoading && !data && (
        <p className="text-xs text-gray">
          No stored explanation for this pair in this run.
        </p>
      )}
      {data && <ShapWaterfall explanation={data} />}
    </div>
  );
}

function Badge({
  tone,
  children,
}: {
  tone: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className={clsx(
        "rounded-full px-2.5 py-0.5 text-[10px] font-bold whitespace-nowrap",
        TONE[tone],
      )}
    >
      {children}
    </span>
  );
}

function Stage({
  num,
  label,
  children,
}: {
  num: number;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className={clsx(
        "min-w-0 flex-1 basis-0 border border-line px-3 py-2.5",
        "first:rounded-l-md last:rounded-r-md [&:not(:last-child)]:border-r-0",
      )}
    >
      <div className="mb-1.5 text-[10px] font-bold tracking-wide text-gray uppercase">
        {num} · {label}
      </div>
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}

function Row({
  ok = false,
  bad = false,
  muted = false,
  children,
}: {
  ok?: boolean;
  bad?: boolean;
  muted?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      className={clsx(
        "text-[11px]",
        bad && "font-bold text-status-nomatch",
        ok && !bad && "font-bold text-status-auto",
        muted && "text-gray italic",
        !ok && !bad && !muted && "text-gray-2",
      )}
    >
      {children}
    </div>
  );
}
