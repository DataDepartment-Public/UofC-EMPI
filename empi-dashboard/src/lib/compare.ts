import type { ExplainPatient } from "./explain";
import { maskSsn } from "./format";

export interface ComparisonRow {
  label: string;
  valueA: string;
  valueB: string;
  result: "exact" | "different" | "missing";
}

const FIELDS: { key: keyof ExplainPatient; label: string }[] = [
  { key: "ssn_last4", label: "SSN" },
  { key: "birth_date", label: "Birthdate" },
  { key: "first_name", label: "First name" },
  { key: "last_name", label: "Last name" },
  { key: "phone", label: "Phone" },
  { key: "zip_code", label: "ZIP code" },
  { key: "address1", label: "Address" },
  { key: "email", label: "Email" },
  { key: "sex", label: "Sex" },
];

/** FR-36/FR-37: a real, field-by-field comparison — no fabricated similarity
 * scores. "Exact match" is a case-insensitive string comparison of the two
 * *display* values (already cleaned/normalized upstream by the pipeline);
 * there's no fuzzy-similarity model in production to compute a percentage
 * from, so unequal-but-present values are reported as "Different", not
 * scored. */
export function compareRecords(
  a: ExplainPatient,
  b: ExplainPatient,
): ComparisonRow[] {
  return FIELDS.map(({ key, label }) => {
    const va = a[key];
    const vb = b[key];
    const va_ = va == null || va === "" ? null : String(va);
    const vb_ = vb == null || vb === "" ? null : String(vb);
    let result: ComparisonRow["result"];
    if (va_ === null || vb_ === null) {
      result = "missing";
    } else if (va_.trim().toLowerCase() === vb_.trim().toLowerCase()) {
      result = "exact";
    } else {
      result = "different";
    }
    const displayA = key === "ssn_last4" && va_ !== null ? maskSsn(va_) : va_ ?? "(missing)";
    const displayB = key === "ssn_last4" && vb_ !== null ? maskSsn(vb_) : vb_ ?? "(missing)";
    return { label, valueA: displayA, valueB: displayB, result };
  });
}

