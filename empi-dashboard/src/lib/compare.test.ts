import { describe, expect, it } from "vitest";
import { compareRawFields, compareRecords } from "./compare";
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

describe("compareRawFields", () => {
  const rowsBy = (
    a: Record<string, unknown> | undefined,
    b: Record<string, unknown> | undefined,
  ) => {
    const rows = compareRawFields(a, b);
    return (label: string) => rows.find((r) => r.label === label)!;
  };

  it("verdicts each field the same way the cleaned-value table does", () => {
    const row = rowsBy(
      { FirstNM_raw: "jane", LastNM_raw: "Doe", CityNM_raw: "Chicago" },
      { FirstNM_raw: "JANE", LastNM_raw: "Doh", CityNM_raw: null },
    );
    expect(row("FirstNM_raw").result).toBe("exact");
    expect(row("LastNM_raw").result).toBe("different");
    expect(row("CityNM_raw").result).toBe("missing");
    expect(row("CityNM_raw").valueB).toBe("(missing)");
  });

  it("strips a source export's midnight time off a raw birthdate", () => {
    const row = rowsBy(
      { BirthDT_raw: "1974-02-12 00:00:00" },
      { BirthDT_raw: "1974-02-12" },
    )("BirthDT_raw");
    expect(row.valueA).toBe("1974-02-12");
    expect(row.result).toBe("exact");
  });

  it("shows raw values verbatim — no cleaning that would hide a difference", () => {
    const row = rowsBy(
      { PrimaryPhoneNBR_raw: "(312) 555-1234" },
      { PrimaryPhoneNBR_raw: "3125551234" },
    )("PrimaryPhoneNBR_raw");
    expect(row.valueA).toBe("(312) 555-1234");
    expect(row.result).toBe("different");
  });

  it("treats a whitespace-only source value as missing, not as text", () => {
    const row = rowsBy({ SuffixNM_raw: "   " }, { SuffixNM_raw: "JR" })(
      "SuffixNM_raw",
    );
    expect(row.valueA).toBe("(missing)");
    expect(row.result).toBe("missing");
  });

  it("unions both key sets, keeping A's order and appending B-only fields", () => {
    const rows = compareRawFields(
      { FirstNM_raw: "Jane", LastNM_raw: "Doe" },
      { LastNM_raw: "Doe", Email_raw: "jane@example.com" },
    );
    expect(rows.map((r) => r.label)).toEqual([
      "FirstNM_raw",
      "LastNM_raw",
      "Email_raw",
    ]);
    expect(rows[2].valueA).toBe("(missing)");
  });

  it("renders every field as missing when one side has no payload at all", () => {
    const rows = compareRawFields({ FirstNM_raw: "Jane" }, undefined);
    expect(rows).toHaveLength(1);
    expect(rows[0].valueA).toBe("Jane");
    expect(rows[0].result).toBe("missing");
  });
});
