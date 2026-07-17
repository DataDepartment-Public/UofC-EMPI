import type { RecordAttrs } from "./schemas";

/**
 * The Model Explanation page (FR-31..FR-38) needs both compared records'
 * full display fields plus the deterministic rule's evidence. The Dataset
 * page already has all of this in hand (from `Entity`/`ReviewCandidate`) —
 * rather than adding a new backend "compare two arbitrary PATIDs" endpoint,
 * we pass it through the URL as one compact payload. That also means the
 * explanation page works correctly on a hard refresh or direct link, with
 * no extra fetch/loading state.
 */
export interface ExplainPatient extends RecordAttrs {
  patid: string;
}

export interface ExplainPayload {
  mid: string;
  patientA: ExplainPatient;
  patientB: ExplainPatient;
  rule: string | null;
  confidence: number | null;
  evidence: string | null;
  updated: string | null;
  // Audit-only FS matcher signal (docs/FS-Matcher-Production-Guide.md) — only
  // present for candidates scored via the incremental path.
  fsMatchProbability?: number | null;
  fsClassificationTier?: string | null;
}

export function encodeExplainPayload(payload: ExplainPayload): string {
  return encodeURIComponent(JSON.stringify(payload));
}

export function decodeExplainPayload(raw: string | null): ExplainPayload | null {
  if (!raw) return null;
  try {
    return JSON.parse(decodeURIComponent(raw)) as ExplainPayload;
  } catch {
    return null;
  }
}
