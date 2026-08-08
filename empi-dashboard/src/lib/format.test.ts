import { describe, expect, it } from "vitest";
import { formatRawDate } from "./format";

describe("formatRawDate", () => {
  it("strips the midnight time real source exports carry", () => {
    expect(formatRawDate("1974-02-12 00:00:00")).toBe("1974-02-12");
    expect(formatRawDate("1974-02-12T00:00:00")).toBe("1974-02-12");
    expect(formatRawDate("1974-02-12T00:00:00.000Z")).toBe("1974-02-12");
    expect(formatRawDate("1974-02-12 00:00:00+00:00")).toBe("1974-02-12");
    expect(formatRawDate("2/12/1974 12:00:00 AM")).toBe("2/12/1974");
    expect(formatRawDate("2/12/1974 12:00 AM")).toBe("2/12/1974");
  });

  it("leaves a date-only value untouched", () => {
    expect(formatRawDate("1974-02-12")).toBe("1974-02-12");
    expect(formatRawDate("2/12/1974")).toBe("2/12/1974");
  });

  it("keeps the calendar date the source wrote, never shifting it a day", () => {
    // The reason this is string-level: `new Date("1974-02-12 00:00:00")`
    // reformatted through UTC lands on the 11th west of Greenwich.
    expect(formatRawDate("1974-02-12 00:00:00")).toBe("1974-02-12");
    expect(formatRawDate("2024-01-01T00:00:00Z")).toBe("2024-01-01");
  });

  it("renders an em dash for missing values", () => {
    expect(formatRawDate(null)).toBe("—");
    expect(formatRawDate(undefined)).toBe("—");
    expect(formatRawDate("")).toBe("—");
  });

  it("leaves an unparseable raw value verbatim rather than blanking it", () => {
    expect(formatRawDate("UNKNOWN")).toBe("UNKNOWN");
    expect(formatRawDate("19740212")).toBe("19740212");
  });
});
