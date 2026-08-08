import { describe, expect, it } from "vitest";
import { compareRecords } from "./compare";
import type { ExplainPatient } from "./explain";

const patient = (over: Partial<ExplainPatient>): ExplainPatient => ({
  patid: "PAT000001",
  first_name: "Luis",
  last_name: "Hernandez",
  birth_date: null,
  ssn_last4: null,
  email: null,
  zip_code: null,
  address1: null,
  sex: null,
  phone: null,
  ...over,
});

const birthRow = (a: ExplainPatient, b: ExplainPatient) =>
  compareRecords(a, b).find((r) => r.label === "Birthdate")!;

describe("compareRecords birthdate display", () => {
  it("never shows an hour on a birthdate", () => {
    const row = birthRow(
      patient({ birth_date: "1974-02-12 00:00:00" }),
      patient({ birth_date: "1974-02-12T00:00:00.000Z" }),
    );
    expect(row.valueA).toBe("1974-02-12");
    expect(row.valueB).toBe("1974-02-12");
    expect(row.result).toBe("exact");
  });

  it("leaves a plain date and a missing value alone", () => {
    const row = birthRow(
      patient({ birth_date: "1974-02-12" }),
      patient({ birth_date: null }),
    );
    expect(row.valueA).toBe("1974-02-12");
    expect(row.valueB).toBe("(missing)");
    expect(row.result).toBe("missing");
  });

  it("masks SSN as before", () => {
    const rows = compareRecords(
      patient({ ssn_last4: "0042" }),
      patient({ ssn_last4: "0042" }),
    );
    expect(rows.find((r) => r.label === "SSN")!.valueA).toBe("***-**-0042");
  });
});

const phoneRow = (a: ExplainPatient, b: ExplainPatient) =>
  compareRecords(a, b).find((r) => r.label === "Phones")!;

describe("compareRecords phones", () => {
  it("lists every phone, not just the primary one", () => {
    const row = phoneRow(
      patient({ phone: "3125551234", phones: ["3125551234", "7735559999"] }),
      patient({ phone: "3125551234", phones: ["3125551234", "7735559999"] }),
    );
    expect(row.valueA).toBe("3125551234, 7735559999");
    expect(row.result).toBe("exact");
  });

  it("reports overlapping-but-unequal sets as partial, not different", () => {
    const row = phoneRow(
      patient({ phones: ["3125551234", "7735559999"] }),
      patient({ phones: ["7735559999"] }),
    );
    expect(row.result).toBe("partial");
  });

  it("reports disjoint sets as different", () => {
    const row = phoneRow(
      patient({ phones: ["3125551234"] }),
      patient({ phones: ["7735559999"] }),
    );
    expect(row.result).toBe("different");
  });

  it("falls back to the primary phone when the set is absent", () => {
    const row = phoneRow(
      patient({ phone: "3125551234" }),
      patient({ phone: "3125551234" }),
    );
    expect(row.valueA).toBe("3125551234");
    expect(row.result).toBe("exact");
  });

  it("treats an empty set on either side as missing", () => {
    const row = phoneRow(
      patient({ phones: ["3125551234"] }),
      patient({ phones: [] }),
    );
    expect(row.valueB).toBe("(missing)");
    expect(row.result).toBe("missing");
  });
});

describe("compareRecords added identity fields", () => {
  it("compares middle name, suffix and city", () => {
    const rows = compareRecords(
      patient({ middle_name: "Ann", suffix: "JR", city: "Chicago" }),
      patient({ middle_name: "Anne", suffix: "JR", city: null }),
    );
    const byLabel = (label: string) => rows.find((r) => r.label === label)!;
    expect(byLabel("Middle name").result).toBe("different");
    expect(byLabel("Suffix").result).toBe("exact");
    expect(byLabel("City").result).toBe("missing");
  });
});
