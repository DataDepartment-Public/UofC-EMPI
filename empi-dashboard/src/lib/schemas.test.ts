import { describe, expect, it } from "vitest";
import {
  AuditLogRowSchema,
  DismissResponseSchema,
  EntitySchema,
  MergeResponseSchema,
  ThresholdSettingsSchema,
  UndoResponseSchema,
  UnmergeResponseSchema,
} from "./schemas";

/** These schemas are the frontend's only defense against a backend contract
 * drift reaching the UI silently (see schemas.ts's module docstring) — a
 * field the backend stops sending, or a shape it changes, should fail loudly
 * here rather than surface as a blank cell or a crash deep in a component.
 * The audit/threshold shapes in particular were the subject of a real
 * contract mismatch (a merge accidentally reverted backend router wiring,
 * and separately a dashboard field was named `ml_review_floor` when the
 * backend's actual field was `fs_review_floor` — same live-field-behind-a-
 * wrong-name bug class) — these tests pin the shapes that actually matter. */

const baseAuditRow = {
  id: 1,
  ts_utc: "2026-08-05T12:00:00Z",
  user: "reviewer.jclark",
  action: "merge" as const,
  patids: "P1,P2",
  mid: "M-000001",
  prev_state: "Needs review",
  next_state: "Merged",
  run_id: "20260805T000000Z-abc123",
};

describe("AuditLogRowSchema", () => {
  it("accepts a full row with undo provenance", () => {
    const parsed = AuditLogRowSchema.parse({
      ...baseAuditRow,
      prev_mid: "M-000005",
      undo_of: 3,
      undone: true,
    });
    expect(parsed.undone).toBe(true);
    expect(parsed.prev_mid).toBe("M-000005");
    expect(parsed.undo_of).toBe(3);
  });

  it("defaults undone to false when the backend omits it", () => {
    const parsed = AuditLogRowSchema.parse(baseAuditRow);
    expect(parsed.undone).toBe(false);
  });

  it("accepts prev_mid/undo_of as absent (a pair predating undo support)", () => {
    const parsed = AuditLogRowSchema.parse(baseAuditRow);
    expect(parsed.prev_mid).toBeUndefined();
    expect(parsed.undo_of).toBeUndefined();
  });

  it("rejects an unknown action", () => {
    expect(() =>
      AuditLogRowSchema.parse({ ...baseAuditRow, action: "delete" }),
    ).toThrow();
  });
});

describe("ThresholdSettingsSchema", () => {
  it("accepts gate_threshold, ml_auto_merge_threshold, and fs_review_floor", () => {
    const parsed = ThresholdSettingsSchema.parse({
      gate_threshold: 0.3,
      ml_auto_merge_threshold: 0.9,
      fs_review_floor: 0.4,
    });
    expect(parsed).toEqual({
      gate_threshold: 0.3,
      ml_auto_merge_threshold: 0.9,
      fs_review_floor: 0.4,
    });
  });

  it("rejects a value out of [0, 1] range", () => {
    expect(() =>
      ThresholdSettingsSchema.parse({
        gate_threshold: 1.5,
        ml_auto_merge_threshold: 0.9,
        fs_review_floor: 0.4,
      }),
    ).toThrow();
  });

  it("rejects a missing required field", () => {
    expect(() =>
      ThresholdSettingsSchema.parse({ gate_threshold: 0.3, ml_auto_merge_threshold: 0.9 }),
    ).toThrow();
  });
});

const baseEntity = {
  mid: "M-000001",
  run_id: "20260805T000000Z-abc123",
  origin: "merge",
  is_merged: true,
  confidence: 0.98,
  updated_utc: "2026-08-05T12:00:00Z",
};

describe("MergeResponseSchema / UnmergeResponseSchema / UndoResponseSchema", () => {
  it("parses a merge response", () => {
    const parsed = MergeResponseSchema.parse({
      audit_id: 5,
      entity: EntitySchema.parse(baseEntity),
    });
    expect(parsed.audit_id).toBe(5);
    expect(parsed.entity.mid).toBe("M-000001");
  });

  it("parses an unmerge response", () => {
    const parsed = UnmergeResponseSchema.parse({
      audit_id: 6,
      new_mid: "M-000099",
      entity: EntitySchema.parse(baseEntity),
    });
    expect(parsed.new_mid).toBe("M-000099");
  });

  it("parses an undo response for a reversed merge (entity present)", () => {
    const parsed = UndoResponseSchema.parse({
      audit_id: 5,
      reversed_action: "merge",
      entity: null,
      new_mids: ["M-000010", "M-000011"],
    });
    expect(parsed.reversed_action).toBe("merge");
    expect(parsed.new_mids).toHaveLength(2);
  });

  it("defaults new_mids to an empty array when the backend omits it", () => {
    const parsed = UndoResponseSchema.parse({
      audit_id: 5,
      reversed_action: "unmerge",
    });
    expect(parsed.new_mids).toEqual([]);
  });

  it("rejects an unknown reversed_action", () => {
    expect(() =>
      UndoResponseSchema.parse({ audit_id: 5, reversed_action: "dismiss" }),
    ).toThrow();
  });
});

describe("DismissResponseSchema", () => {
  it("parses a dismiss response", () => {
    expect(DismissResponseSchema.parse({ audit_id: 7 })).toEqual({ audit_id: 7 });
  });
});
