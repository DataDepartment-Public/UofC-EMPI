import type { ExplainPatient } from "./explain";
import { formatRawDate, maskSsn } from "./format";

export interface ComparisonRow {
  label: string;
  valueA: string;
  valueB: string;
  /**
   * Set on multi-valued fields (only `Phones` today) so the table can stack
   * the values one per line instead of running them together on one. When
   * present these are the display source; `valueA`/`valueB` stay populated as
   * the flat equivalent for callers that just want text.
   */
  valuesA?: string[];
  valuesB?: string[];
  result: "exact" | "partial" | "different" | "missing";
}

const FIELDS: { key: keyof ExplainPatient; label: string }[] = [
  { key: "ssn_last4", label: "SSN" },
  { key: "birth_date", label: "Birthdate" },
  { key: "first_name", label: "First name" },
  { key: "middle_name", label: "Middle name" },
  { key: "last_name", label: "Last name" },
  { key: "suffix", label: "Suffix" },
  { key: "zip_code", label: "ZIP code" },
  { key: "address1", label: "Address" },
  { key: "city", label: "City" },
  { key: "email", label: "Email" },
  { key: "sex", label: "Sex" },
];

/** Every phone on the record, not just the primary one. A record's phone
 * *set* is what B5 blocking and the NAME_DOB_PHONE rule intersect on, so a
 * pair can genuinely agree on a phone that isn't either side's primary —
 * showing only `phone` would render that agreement as a disagreement.
 * `phones` is absent on a payload published before the field existed; the
 * primary `phone` is the best available stand-in there. */
function phoneList(p: ExplainPatient): string[] {
  const all = p.phones ?? (p.phone ? [p.phone] : []);
  return [...new Set(all.map((v) => String(v).trim()).filter(Boolean))].sort();
}

function comparePhones(a: ExplainPatient, b: ExplainPatient): ComparisonRow {
  const listA = phoneList(a);
  const listB = phoneList(b);
  const shared = listA.filter((v) => listB.includes(v));

  let result: ComparisonRow["result"];
  if (listA.length === 0 || listB.length === 0) {
    result = "missing";
  } else if (shared.length === 0) {
    result = "different";
  } else if (listA.length === listB.length && shared.length === listA.length) {
    result = "exact";
  } else {
    // Overlapping but not identical — the signal the matcher acts on is
    // present, so this must not read as "Different", but calling it an exact
    // match would overstate what the reviewer is looking at.
    result = "partial";
  }

  return {
    label: "Phones",
    valueA: listA.length ? listA.join(", ") : "(missing)",
    valueB: listB.length ? listB.join(", ") : "(missing)",
    valuesA: listA,
    valuesB: listB,
    result,
  };
}

/** FR-36/FR-37: a real, field-by-field comparison — no fabricated similarity
 * scores. "Exact match" is a case-insensitive string comparison of the two
 * *display* values (already cleaned/normalized upstream by the pipeline);
 * there's no fuzzy-similarity model in production to compute a percentage
 * from, so unequal-but-present values are reported as "Different", not
 * scored. The one multi-valued field, `phones`, is compared as a set — see
 * `comparePhones`. */
export function compareRecords(
  a: ExplainPatient,
  b: ExplainPatient,
): ComparisonRow[] {
  const rows: ComparisonRow[] = FIELDS.map(({ key, label }) => {
    const va = a[key];
    const vb = b[key];
    // A birthdate carries a meaningless midnight time in some source exports;
    // never show an hour on it. `formatRawDate` is string-level, so it can't
    // shift the date across a timezone offset. Stripped *before* the equality
    // check so two spellings of the same midnight don't read as "Different".
    const norm = (v: unknown) =>
      v == null || v === "" ? null : key === "birth_date" ? formatRawDate(v) : String(v);
    const va_ = norm(va);
    const vb_ = norm(vb);
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

  // Kept next to Address/City rather than appended, so the table still reads
  // identity fields → contact fields top to bottom.
  const emailAt = rows.findIndex((r) => r.label === "Email");
  rows.splice(emailAt, 0, comparePhones(a, b));
  return rows;
}
