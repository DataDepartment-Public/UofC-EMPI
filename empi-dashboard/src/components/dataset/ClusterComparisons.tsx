"use client";

import { useState } from "react";
import Link from "next/link";
import clsx from "clsx";
import { useClusterPairs } from "@/lib/hooks";
import type {
  ClusterExternalPair,
  ClusterPair,
  ClusterPairVerdict,
  Entity,
} from "@/lib/schemas";
import { fullName } from "@/lib/format";
import {
  BlockingStage,
  ClusteringStage,
  GateStage,
  MatcherStage,
  RulesStage,
  StageStrip,
  skipReason,
} from "@/components/shared/PipelineStages";

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
  if (data.pairs_truncated) {
    // The API omits the pairwise enumeration above its cluster-size cap —
    // it grows quadratically, so a large hub cluster would otherwise cost a
    // multi-minute request. Say so rather than showing an empty list.
    return (
      <p className="rounded-md border border-line bg-bg px-3 py-2 text-sm text-gray">
        This cluster has {data.members.length} records — too many to trace
        pair by pair. The comparison history below still lists what its
        records were checked against outside the cluster.
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
      {data.unresolved_run_id ? (
        // Not "the artifacts aged out": this entity was last touched by
        // incremental scoring, which records no run artifacts at all. The
        // trace is blank because there is nothing to read — not because the
        // pipeline decided nothing about these records.
        <p className="mb-3 rounded-md border border-line bg-bg px-3 py-2 text-xs text-gray-2">
          These records were last scored outside a full pipeline run (
          <span className="font-mono">{data.unresolved_run_id}</span>), which
          keeps no per-stage artifacts — so no stage detail is available for
          these pairs.
        </p>
      ) : (
        !data.artifacts_available && (
          <p className="mb-3 rounded-md border border-line bg-bg px-3 py-2 text-xs text-gray-2">
            The pipeline artifacts for run{" "}
            <span className="font-mono">{data.run_id ?? entity.run_id}</span> are
            no longer on disk, so no stage detail is available for these pairs.
          </p>
        )
      )}
      <div className="space-y-2">
        {byVerdict(data.pairs).map((pair) => (
          <PairCard
            key={`${pair.patid_a}-${pair.patid_b}`}
            pair={pair}
            nameA={nameOf(pair.patid_a)}
            nameB={nameOf(pair.patid_b)}
            nameOf={nameOf}
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
  thresholds,
}: {
  pair: ClusterExternalPair;
  memberName: string;
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
          {/* Both PATIDs, same as the internal cards. Naming only the
              counterpart made several rows of this list identical on screen
              whenever two members of this cluster were each compared against
              the same outside record — and each of those rows links to a
              different pair. */}
          <span className="shrink-0 font-mono text-[10px] text-gray">
            {pair.member_patid} · {pair.other_patid}
            {pair.other_mid && ` · ${pair.other_mid}`}
          </span>
        </span>
        <Badge tone={verdict.tone}>{verdict.label}</Badge>
      </button>

      {open && (
        <div className="border-t border-line px-3.5 py-3">
          <div className="mb-3 flex items-start justify-between gap-3">
            <p className="text-xs text-gray-2">{verdict.blurb}</p>
            <ReviewQueueLink pair={pair} resolvable={pair.other_mid != null} />
          </div>
          <PairStages pair={pair} thresholds={thresholds} internal={false} />
        </div>
      )}
    </div>
  );
}

/** Whether a pair's verdict is the set the Review Queue's "Needs review"
 * section would surface for it — see `ClusterComparisonHistory`'s `pending`
 * count. Not `reject`/`gate_dropped` (those land in "Auto-rejected") and not
 * the merge verdicts ("Auto-merged"). Used for the wording of the counts
 * only: every one of those pairs is still linkable — see `hasQueueRow`. */
function isAwaitingReview(verdict: ClusterPairVerdict): boolean {
  return verdict === "ml_human_review" || verdict === "blocked_undecided";
}

/** Whether the Review Queue holds a row for this pair, and so whether a deep
 * link to it resolves.
 *
 * Publish writes a `review_candidate` row for **every** candidate pair the
 * run decided, not just the ones still awaiting a reviewer — the queue's four
 * sections then split them into needs-review / reviewed / auto-merged /
 * auto-rejected. So a rejected, gate-dropped or auto-merged pair is just as
 * addressable as an ambiguous one, and every card here should link out.
 *
 * The single exception is `not_compared`: those two records were never
 * blocked together, no stage ever looked at them, and no queue row exists.
 * That pair shares a cluster only by transitivity — the clustering stage of
 * its strip is the explanation, and a link would resolve to nothing. */
function hasQueueRow(verdict: ClusterPairVerdict): boolean {
  return verdict !== "not_compared";
}

/** The deep link into the Review Queue for one pair. The queue canonicalizes
 * the two PATIDs itself, so the order sent here doesn't matter; it resolves
 * the pair independently of whatever section or page the queue happens to be
 * showing (`useReviewCandidate`).
 *
 * `resolvable` is the caller's extra precondition — the external list uses it
 * to drop the link for a counterpart that isn't in the index, since the
 * queue joins both sides to `entity_member` and would return nothing. */
function ReviewQueueLink({
  pair,
  resolvable = true,
}: {
  pair: ClusterPair;
  resolvable?: boolean;
}) {
  if (!resolvable || !hasQueueRow(pair.verdict)) return null;
  return (
    <Link
      href={`/review?patid_a=${encodeURIComponent(pair.patid_a)}&patid_b=${encodeURIComponent(pair.patid_b)}`}
      className="shrink-0 whitespace-nowrap text-[11px] font-bold text-brand-blue hover:underline"
    >
      Open in Review Queue →
    </Link>
  );
}

function PairCard({
  pair,
  nameA,
  nameB,
  nameOf,
  thresholds,
}: {
  pair: ClusterPair;
  nameA: string;
  nameB: string;
  nameOf: (patid: string) => string;
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
          <div className="mb-3 flex items-start justify-between gap-3">
            <p className="text-xs text-gray-2">{verdict.blurb}</p>
            <ReviewQueueLink pair={pair} />
          </div>
          <PairStages pair={pair} thresholds={thresholds} nameOf={nameOf} internal />
        </div>
      )}
    </div>
  );
}

/** One pair's journey through the five shared stages
 * (`components/shared/PipelineStages`) — the same strip, in the same order
 * and wording, that the Review Queue's `PipelineTrail` renders.
 *
 * The difference is the source, not the shape: here everything comes from the
 * run's immutable artifacts, already in the `/clusters/{mid}/pairs` payload,
 * so the strip renders on expand with no further fetch. The Review Queue
 * reads the two model stages from the persisted per-pair explanations, which
 * is also where its thresholds come from; this side takes them from the
 * run's settings, returned alongside the pairs. */
function PairStages({
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
  return (
    <StageStrip>
      <BlockingStage blocked={pair.blocked} sourceBlocks={pair.source_blocks} />

      <RulesStage
        matchRule={pair.match_rule}
        rulesFired={pair.rules_fired}
        confidence={pair.confidence}
        rejectRule={pair.reject_rule}
        nContradictions={pair.n_contradictions}
      />

      <GateStage
        score={pair.gate_score}
        tier={pair.gate_tier}
        threshold={thresholds.gate_threshold}
        skipReason={skipReason(pair.verdict, "gate")}
      />

      <MatcherStage
        score={pair.ml_score}
        tier={pair.ml_tier}
        threshold={thresholds.ml_auto_merge_threshold}
        skipReason={skipReason(pair.verdict, "matcher")}
      />

      <ClusteringStage
        sameCluster={internal}
        directEdge={
          pair.verdict === "auto_merge_rule" || pair.verdict === "ml_auto_merge"
        }
        byReviewer={pair.joined_by === "reviewer"}
        path={pair.cluster_path}
        tracksPath
        nameOf={nameOf}
      />
    </StageStrip>
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
