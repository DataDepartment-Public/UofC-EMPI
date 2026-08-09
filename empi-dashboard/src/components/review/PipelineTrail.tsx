"use client";

import { usePairExplanations } from "@/lib/hooks";
import type { ReviewQueueItem } from "@/lib/schemas";
import {
  BlockingStage,
  ClusteringStage,
  GateStage,
  MatcherStage,
  RulesStage,
  StageStrip,
  skipReason,
} from "@/components/shared/PipelineStages";

/** What the pipeline did with this candidate pair — the five shared stages of
 * `components/shared/PipelineStages`, so a pair reads identically here and in
 * the Patient Registry's comparison cards.
 *
 * Source and cleaned values used to lead this trail as two more columns; the
 * feature comparison directly below shows both, raw drawer included, so they
 * were a smaller duplicate of the panel's main table.
 *
 * The two model stages read `GET /explanations/{model}/{a}/{b}` rather than
 * the candidate row: those are the models' *persisted* scores, computed at
 * score time against the model and feature vector that produced the recorded
 * decision, and they carry the threshold each pair was actually judged by.
 * The gate's score is only available there at all. */
export function PipelineTrail({ item }: { item: ReviewQueueItem }) {
  const { gate, ml } = usePairExplanations(item.patid_a, item.patid_b);
  const merged = item.mid_a === item.mid_b;

  return (
    <StageStrip>
      {/* Every row in the queue is a candidate pair, so blocking always put
          these two together; only which blocks fired varies. */}
      <BlockingStage blocked sourceBlocks={item.source_blocks} />

      <RulesStage
        matchRule={item.match_rule}
        confidence={item.confidence}
        rejected={item.verdict === "reject"}
      />

      <GateStage
        loading={gate.isLoading}
        score={gate.data?.decision.score}
        tier={gate.data?.decision.tier}
        threshold={gate.data?.decision.threshold}
        skipReason={skipReason(item.verdict, "gate")}
      />

      <MatcherStage
        loading={ml.isLoading}
        score={ml.data?.decision.score}
        tier={ml.data?.decision.tier}
        threshold={ml.data?.decision.threshold}
        skipReason={skipReason(
          gate.data?.decision.tier === "no_match" ? "gate_dropped" : item.verdict,
          "matcher",
        )}
      />

      <ClusteringStage
        sameCluster={merged}
        mid={item.mid_a}
        directEdge={
          item.verdict === "auto_merge_rule" || item.verdict === "ml_auto_merge"
        }
        byReviewer={item.reviewer_decision === "merged"}
      />
    </StageStrip>
  );
}
