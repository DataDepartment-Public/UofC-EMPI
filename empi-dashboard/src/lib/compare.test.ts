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
