"use client";

import clsx from "clsx";

/** The pipeline trail for one candidate pair, as five stages — the single
 * definition both the Review Queue's detail panel and the Patient Registry's
 * comparison cards render.
 *
 * Blocking → Deterministic rules → Non-match gate → ML matcher → Clustering,
 * numbered and labelled here rather than by the callers, so the two screens
 * cannot drift into showing the same pair's journey as different numbers of
 * differently-named steps. What each caller still owns is where the *data*
 * comes from: the registry reads a run's immutable artifacts through
 * `/clusters/{mid}/pairs`, the Review Queue reads the published candidate row
 * plus the two models' persisted explanations. Same stages, same wording,
 * different sources.
 *
 * The FS matcher is deliberately not a stage: it's audit-only in the backend,
 * kept for lineage, not a reviewer-facing decision.
 */
export function StageStrip({ children }: { children: React.ReactNode }) {
  // The five stages stretch to fill whatever width their container has; the
  // scroll floor is for a genuinely narrow window, below which the stage text
  // stops being readable at all.
  return (
    <div className="overflow-x-auto pb-1">
      <div className="flex min-w-[800px]">{children}</div>
    </div>
  );
}

// ── 1 · Blocking ─────────────────────────────────────────────────────────────

export function BlockingStage({
  blocked,
  sourceBlocks,
}: {
  blocked: boolean;
  sourceBlocks?: string | null;
}) {
  return (
    <Stage num={1} label="Blocking" title="Candidate pair">
      {blocked ? (
        <>
          <Row ok>Same candidate set</Row>
          <Row>{sourceBlocks || "—"}</Row>
        </>
      ) : (
        <Row muted>Never blocked together</Row>
      )}
    </Stage>
  );
}

// ── 2 · Deterministic rules ──────────────────────────────────────────────────

/** `rejected` covers the caller that knows the reject rules ended this pair
 * but not which one fired — the Review Queue's candidate row carries the
 * verdict without the rule name, where the registry reads it straight from
 * the rejects artifact. */
export function RulesStage({
  matchRule,
  rulesFired,
  confidence,
  rejectRule,
  nContradictions,
  rejected = false,
}: {
  matchRule?: string | null;
  rulesFired?: string | null;
  confidence?: number | null;
  rejectRule?: string | null;
  nContradictions?: number | null;
  rejected?: boolean;
}) {
  return (
    <Stage num={2} label="Deterministic rules" title="Rule">
      {matchRule ? (
        <>
          <Row ok>{matchRule} fired</Row>
          <Row>
            {rulesFired && rulesFired !== matchRule
              ? `All: ${rulesFired}`
              : `Confidence ${confidence != null ? pct(confidence) : "—"}`}
          </Row>
        </>
      ) : rejectRule ? (
        <>
          <Row bad>{rejectRule}</Row>
          <Row>{nContradictions} strong contradictions</Row>
        </>
      ) : rejected ? (
        <Row bad>Rejected — enough strong contradictions</Row>
      ) : (
        <Row muted>No rule confirmed or rejected</Row>
      )}
    </Stage>
  );
}

// ── 3 · Non-match gate ───────────────────────────────────────────────────────

/** **Both model stages report the probability of the pair being the same
 * patient**, never its complement — the gate's `P(plausible)`, the matcher's
 * `P(confident match)`. The threshold beside it is a `>=` bound on that same
 * quantity, so score and threshold compare directly, and the SHAP waterfall
 * in the Review Queue explains the score in this same orientation
 * (`Explanations-Guide.md` §2). A headline number running opposite to the
 * waterfall explaining it would read plausibly and be exactly backwards.
 *
 * Showing the threshold is what makes a low score legible: bare "0%" reads as
 * "no signal"; "0% · needs 30%" reads as the deliberate drop it was. */
export function GateStage({
  score,
  tier,
  threshold,
  skipReason,
  loading = false,
}: {
  score?: number | null;
  tier?: string | null;
  threshold?: number | null;
  skipReason: string;
  loading?: boolean;
}) {
  const dropped = tier === "no_match";
  return (
    <Stage num={3} label="Non-match gate" title="P(plausible)">
      {loading ? (
        <Row muted>Loading…</Row>
      ) : score != null ? (
        <>
          <Score score={score} threshold={threshold} />
          <Row ok={!dropped} bad={dropped}>
            {dropped ? "Dropped — confident non-match" : "Passed to the matcher"}
          </Row>
        </>
      ) : (
        <Row muted>{skipReason}</Row>
      )}
    </Stage>
  );
}

// ── 4 · ML matcher ───────────────────────────────────────────────────────────

export function MatcherStage({
  score,
  tier,
  threshold,
  skipReason,
  loading = false,
}: {
  score?: number | null;
  tier?: string | null;
  threshold?: number | null;
  skipReason: string;
  loading?: boolean;
}) {
  const merged = tier === "auto_merge";
  return (
    <Stage num={4} label="ML matcher" title="P(confident match)">
      {loading ? (
        <Row muted>Loading…</Row>
      ) : score != null ? (
        <>
          <Score score={score} threshold={threshold} />
          <Row ok={merged}>
            {merged
              ? "Confident match — auto-merged"
              : "Ambiguous — sent to review"}
          </Row>
        </>
      ) : (
        <Row muted>{skipReason}</Row>
      )}
    </Stage>
  );
}

// ── 5 · Clustering ───────────────────────────────────────────────────────────

/** Whether the two records ended up together, and what put them there.
 *
 * `path` is the chain of members whose own confirmed merges connected a pair
 * that formed no edge of its own — available from the registry's per-run
 * trace, not from the Review Queue's candidate row, which knows only that the
 * two share a `mid` today.
 *
 * `tracksPath` is what separates the two callers when there is no path to
 * show, and the distinction is worth the prop: for the registry that means
 * it walked this run's merge graph and found nothing connecting a pair that
 * nonetheless shares a cluster — a real gap worth flagging — while for the
 * Review Queue it means only that a candidate row doesn't carry paths. Same
 * empty `path`, opposite conclusions. */
export function ClusteringStage({
  sameCluster,
  mid,
  directEdge = false,
  byReviewer = false,
  path,
  tracksPath = false,
  nameOf = (patid) => patid,
}: {
  sameCluster: boolean;
  mid?: string | null;
  directEdge?: boolean;
  byReviewer?: boolean;
  path?: string[] | null;
  tracksPath?: boolean;
  nameOf?: (patid: string) => string;
}) {
  return (
    <Stage num={5} label="Clustering" title="Cluster outcome">
      {!sameCluster ? (
        <Row muted>Different clusters — not merged</Row>
      ) : byReviewer ? (
        <Row ok>A reviewer merged these records</Row>
      ) : directEdge ? (
        <Row ok>This pair&apos;s own match formed the merge edge</Row>
      ) : path && path.length > 1 ? (
        <>
          <Row ok>
            Joined transitively ({path.length - 1} hop
            {path.length - 1 === 1 ? "" : "s"})
          </Row>
          <PathChain path={path} nameOf={nameOf} />
        </>
      ) : tracksPath ? (
        <Row muted>Same cluster — no merge path found for this run</Row>
      ) : (
        <>
          <Row ok>Joined through other records</Row>
          <Row>{mid ? `Same cluster · ${mid}` : "Same cluster"}</Row>
        </>
      )}
    </Stage>
  );
}

/** The chain of members whose own confirmed merges connected two records that
 * formed no edge of their own — the record ids a reviewer needs to trace a
 * surprising grouping back to its source. PATIDs lead (they're what records
 * are looked up by); the name rides along as a tooltip. */
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
          <span className="font-mono text-[10px] text-gray-2" title={nameOf(patid)}>
            {patid}
          </span>
        </span>
      ))}
    </div>
  );
}

// ── Shared primitives ────────────────────────────────────────────────────────

/** Why a model stage has no score, given what the pipeline concluded about
 * the pair. The verdict is the authority rather than the missing score
 * itself: an absent score tells you the stage didn't run, but only the
 * verdict says which earlier stage ended the pair's journey — and "no rule
 * fired, no model scored it either" (an ungated run) is a genuinely
 * different state from "a rule decided it at stage 2". */
export function skipReason(
  verdict: string | null | undefined,
  stage: "gate" | "matcher",
): string {
  if (verdict === "auto_merge_rule") {
    return "Not scored — an auto-merge rule decided this at stage 2";
  }
  if (verdict === "reject") {
    return "Not scored — the reject rules decided this at stage 2";
  }
  if (stage === "matcher" && verdict === "gate_dropped") {
    return "Didn't run — the gate dropped this pair";
  }
  return stage === "gate"
    ? "Not scored — no gate model ran for this run"
    : "Didn't run — no ML model scored this run";
}

/** One decimal on the score, none on the threshold. The extremes are where
 * these numbers earn their keep — a gate score of 99.9% rounded to 100%, or
 * 0.04% rounded to 0%, reads as a certainty the model never expressed —
 * while thresholds are round settings (30%, 70%) that gain nothing from it. */
function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function Score({
  score,
  threshold,
}: {
  score: number;
  threshold?: number | null;
}) {
  return (
    <Row>
      <span className="font-bold">{pct(score)}</span>
      {threshold != null && (
        <span className="text-gray">
          {" · needs "}
          {Math.round(threshold * 100)}%
        </span>
      )}
    </Row>
  );
}

function Stage({
  num,
  label,
  title,
  children,
}: {
  num: number;
  label: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className={clsx(
        "min-w-0 flex-1 basis-0 border border-line px-3 py-2.5",
        "first:rounded-l-md last:rounded-r-md [&:not(:last-child)]:border-r-0",
      )}
    >
      <div className="text-[10px] font-bold tracking-wide text-gray uppercase">
        {num} · {label}
      </div>
      <div className="mb-1.5 text-[12px] font-bold text-ink-2">{title}</div>
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
