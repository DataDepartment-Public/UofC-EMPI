/**
 * zod models mirroring the FastAPI backend's response shapes
 * (src/api/schemas.py). Kept as the single source of truth for the
 * frontend's view of the contract — a backend field rename surfaces here as
 * a runtime parse failure instead of a silent `undefined` in the UI.
 */
import { z } from "zod";

export const RunStatusSchema = z.enum([
  "queued",
  "running",
  "succeeded",
  "failed",
]);
export type RunStatus = z.infer<typeof RunStatusSchema>;

export const RunCreateResponseSchema = z.object({
  run_id: z.string(),
  status: RunStatusSchema,
});
export type RunCreateResponse = z.infer<typeof RunCreateResponseSchema>;

export const RunSummarySchema = z.object({
  run_id: z.string(),
  status: RunStatusSchema,
  counts: z.record(z.string(), z.number()).default({}),
  created_utc: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
});
export type RunSummary = z.infer<typeof RunSummarySchema>;

export const RunDetailSchema = z.object({
  run_id: z.string(),
  status: RunStatusSchema,
  created_utc: z.string().optional(),
  git_sha: z.string().nullable().optional(),
  counts: z.record(z.string(), z.number()).default({}),
  error: z.string().nullable().optional(),
});
export type RunDetail = z.infer<typeof RunDetailSchema>;

const recordAttrsFields = {
  first_name: z.string().nullable().optional(),
  last_name: z.string().nullable().optional(),
  birth_date: z.string().nullable().optional(),
  ssn_last4: z.string().nullable().optional(),
  email: z.string().nullable().optional(),
  zip_code: z.string().nullable().optional(),
  address1: z.string().nullable().optional(),
  sex: z.string().nullable().optional(),
  phone: z.string().nullable().optional(),
};

export const RecordAttrsSchema = z.object(recordAttrsFields);
export type RecordAttrs = z.infer<typeof RecordAttrsSchema>;

export const EntityMemberSchema = z.object({
  patid: z.string(),
  is_primary: z.boolean(),
  added_by: z.string(),
  updated_utc: z.string(),
  ...recordAttrsFields,
});
export type EntityMember = z.infer<typeof EntityMemberSchema>;

export const CandidatePatientSchema = z.object({
  patid: z.string(),
  ...recordAttrsFields,
});
export type CandidatePatient = z.infer<typeof CandidatePatientSchema>;

export const ReviewCandidateSchema = z.object({
  patid_a: z.string(),
  patid_b: z.string(),
  match_rule: z.string().nullable(),
  confidence: z.number().nullable(),
  evidence: z.string().nullable(),
  source_blocks: z.string().nullable(),
  ml_match_probability: z.number().nullable().optional(),
  ml_classification_tier: z.string().nullable().optional(),
  patient_a: CandidatePatientSchema,
  patient_b: CandidatePatientSchema,
});
export type ReviewCandidate = z.infer<typeof ReviewCandidateSchema>;

export const OriginSchema = z.enum([
  "deterministic",
  "review",
  "merge",
  "none",
]);
export type Origin = z.infer<typeof OriginSchema>;

export const EntitySchema = z.object({
  mid: z.string(),
  run_id: z.string(),
  origin: OriginSchema,
  is_merged: z.boolean(),
  confidence: z.number().nullable(),
  match_rule: z.string().nullable().optional(),
  evidence: z.string().nullable().optional(),
  updated_utc: z.string(),
  members: z.array(EntityMemberSchema).default([]),
  review_candidates: z.array(ReviewCandidateSchema).default([]),
});
export type Entity = z.infer<typeof EntitySchema>;

export const RecordsPageSchema = z.object({
  total: z.number(),
  page: z.number(),
  page_size: z.number(),
  items: z.array(EntitySchema),
});
export type RecordsPage = z.infer<typeof RecordsPageSchema>;

export const ReviewQueueItemSchema = z.object({
  patid_a: z.string(),
  patid_b: z.string(),
  mid_a: z.string(),
  mid_b: z.string(),
  member_count_a: z.number(),
  member_count_b: z.number(),
  match_rule: z.string().nullable(),
  confidence: z.number().nullable(),
  evidence: z.string().nullable(),
  source_blocks: z.string().nullable(),
  ml_match_probability: z.number().nullable().optional(),
  ml_classification_tier: z.string().nullable().optional(),
  reviewed: z.boolean(),
  patient_a: CandidatePatientSchema,
  patient_b: CandidatePatientSchema,
});
export type ReviewQueueItem = z.infer<typeof ReviewQueueItemSchema>;

export const ReviewQueuePageSchema = z.object({
  total: z.number(),
  page: z.number(),
  page_size: z.number(),
  items: z.array(ReviewQueueItemSchema),
});
export type ReviewQueuePage = z.infer<typeof ReviewQueuePageSchema>;

export const DismissResponseSchema = z.object({
  audit_id: z.number(),
});

export const RawRecordSchema = z.object({
  patid: z.string(),
  fields: z.record(z.string(), z.unknown()),
});
export type RawRecord = z.infer<typeof RawRecordSchema>;

export const CleanSsnSchema = z.object({
  patid: z.string(),
  ssn: z.string().nullable(),
});
export type CleanSsn = z.infer<typeof CleanSsnSchema>;

export const MatchStatusCountsSchema = z.object({
  auto_match: z.number(),
  needs_review: z.number(),
  no_match: z.number(),
});

export const DashboardHistoryPointSchema = z.object({
  run_id: z.string(),
  created_utc: z.string(),
  auto_match_rate: z.number(),
  review_rate: z.number(),
});

export const DashboardSummarySchema = z.object({
  last_run_id: z.string().nullable(),
  last_run_created_utc: z.string().nullable(),
  model_version: z.string().nullable(),
  total_raw_rows: z.number(),
  total_records: z.number(),
  invalid_records: z.number(),
  duplicate_clusters: z.number(),
  matched_records: z.number(),
  matched_pct: z.number(),
  unmerged_pct: z.number(),
  needs_review_records: z.number(),
  manual_merge_actions: z.number(),
  manual_unmerge_actions: z.number(),
  manual_merge_pct: z.number(),
  manual_unmerge_pct: z.number(),
  auto_match_rate: z.number(),
  status_counts: MatchStatusCountsSchema,
  confidence_thresholds: z.record(z.string(), z.number()),
  history: z.array(DashboardHistoryPointSchema),
});
export type DashboardSummary = z.infer<typeof DashboardSummarySchema>;

export const AuditLogRowSchema = z.object({
  id: z.number(),
  ts_utc: z.string(),
  user: z.string(),
  action: z.enum(["merge", "unmerge", "split", "dismiss"]),
  patids: z.string(),
  mid: z.string(),
  prev_state: z.string(),
  next_state: z.string(),
  run_id: z.string().nullable(),
  prev_mid: z.string().nullable().optional(),
  undo_of: z.number().nullable().optional(),
  undone: z.boolean().default(false),
});
export type AuditLogRow = z.infer<typeof AuditLogRowSchema>;

export const MergeResponseSchema = z.object({
  audit_id: z.number(),
  entity: EntitySchema,
});

export const UnmergeResponseSchema = z.object({
  audit_id: z.number(),
  new_mid: z.string(),
  entity: EntitySchema,
});

export const UndoResponseSchema = z.object({
  audit_id: z.number(),
  reversed_action: z.enum(["merge", "unmerge"]),
  entity: EntitySchema.nullable().optional(),
  new_mids: z.array(z.string()).default([]),
});
export type UndoResponse = z.infer<typeof UndoResponseSchema>;

// ── Explanations (GET /explanations/{model}/{a}/{b}) ────────────────────────
// Mirrors src/api/schemas.py::PairExplanation exactly — this API layer is a
// 1:1 pass-through of src/models/explanations.py's build_payload(), not
// renamed anywhere, so field names here match the backend verbatim.
export const ExplanationFeatureSchema = z.object({
  name: z.string(),
  label: z.string(),
  value: z.union([z.number(), z.string()]).nullable().optional(),
  display_value: z.string().nullable().optional(),
  shap: z.number(),
  start: z.number(),
  end: z.number(),
  direction: z.enum(["positive", "negative"]),
  cumulative_prob: z.number(),
});
export type ExplanationFeature = z.infer<typeof ExplanationFeatureSchema>;

export const ExplanationDecisionSchema = z.object({
  score: z.number(),
  tier: z.string(),
  threshold: z.number().nullable().optional(),
});
export type ExplanationDecision = z.infer<typeof ExplanationDecisionSchema>;

export const ExplanationAxisSchema = z.object({
  min: z.number(),
  max: z.number(),
});
export type ExplanationAxis = z.infer<typeof ExplanationAxisSchema>;

export const PairExplanationSchema = z.object({
  model: z.string(),
  run_id: z.string().nullable().optional(),
  model_file: z.string().nullable().optional(),
  patid_a: z.string(),
  patid_b: z.string(),
  decision: ExplanationDecisionSchema,
  base_value: z.number(),
  final_margin: z.number(),
  units: z.literal("log_odds"),
  top_n: z.number(),
  axis: ExplanationAxisSchema,
  features: z.array(ExplanationFeatureSchema),
});
export type PairExplanation = z.infer<typeof PairExplanationSchema>;

// ── Admin (GET/PUT /admin/thresholds) ────────────────────────────────────────
export const ThresholdSettingsSchema = z.object({
  gate_threshold: z.number().min(0).max(1),
  ml_auto_merge_threshold: z.number().min(0).max(1),
  fs_review_floor: z.number().min(0).max(1),
});
export type ThresholdSettings = z.infer<typeof ThresholdSettingsSchema>;

// ── Health (GET /health/ready) ───────────────────────────────────────────────
export const ReadyChecksSchema = z.object({
  db: z.boolean(),
  data_dirs: z.boolean(),
  last_run_id: z.string().nullable(),
});
export type ReadyChecks = z.infer<typeof ReadyChecksSchema>;

// `checks` is omitted by the BFF's health route when FastAPI is completely
// unreachable (a plain fetch failure, not an HTTP error response) — see
// app/api/health/route.ts's generic catch branch.
export const ReadyResponseSchema = z.object({
  status: z.enum(["ok", "not_ready"]),
  checks: ReadyChecksSchema.optional(),
});
export type ReadyResponse = z.infer<typeof ReadyResponseSchema>;
