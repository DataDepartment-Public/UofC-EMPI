import type { ExplainPatient } from "./explain";

export interface ComparisonRow {
  label: string;
  valueA: string;
  valueB: string;
  result: "exact" | "different" | "missing";
}

const FIELDS: { key: keyof ExplainPatient; label: string }[] = [
  { key: "ssn_last4", label: "SSN (last 4)" },
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
    return { label, valueA: va_ ?? "(missing)", valueB: vb_ ?? "(missing)", result };
  });
}

/** The rule's required fields (label -> the ComparisonRow.label it maps to),
 * so the plain-language summary cites only what that specific rule actually
 * checked (src/models/deterministic_rules.py RULES). */
const RULE_FIELDS: Record<string, string[]> = {
  SSN_DOB: ["SSN (last 4)", "Birthdate"],
  NAME_DOB_EMAIL: ["First name", "Last name", "Birthdate", "Email"],
  NAME_DOB_PHONE: ["First name", "Last name", "Birthdate", "Phone"],
  NAME_DOB_SEX: ["First name", "Last name", "Birthdate", "Sex"],
  NAME_DOB_ADDRESS: ["First name", "Last name", "Birthdate", "Address"],
};

export function plainLanguageSummary(
  rule: string | null,
  rows: ComparisonRow[],
): string {
  if (!rule) {
    return (
      "This pair was surfaced by blocking as a potential duplicate, but no " +
      "deterministic rule confirmed it — it needs clerical review before any " +
      "merge decision is made."
    );
  }
  const fields = RULE_FIELDS[rule];
  if (!fields) {
    return `This pair was confirmed a match by the ${rule} rule.`;
  }
  const byLabel = new Map(rows.map((r) => [r.label, r]));
  const matched = fields.filter((f) => byLabel.get(f)?.result === "exact");
  return (
    `This pair was classified as a likely duplicate by the ${rule} rule, which ` +
    `requires ${fields.join(", ").toLowerCase()} to agree. In this pair, ` +
    `${matched.join(", ").toLowerCase() || "the required fields"} matched exactly.`
  );
}
