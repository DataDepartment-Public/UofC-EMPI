import type { ExplainPatient } from "./explain";
import { formatRawDate, maskSsn, RAW_DATE_FIELD } from "./format";

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

/** One raw source value, normalized only as far as display requires: an
 * empty-ish value becomes `null` ("missing"), a date-only field loses its
 * meaningless midnight time, and everything else is left verbatim. No
 * cleaning beyond that — the whole point of a raw view is to show the
 * source text the pipeline started from. */
function normalizeRaw(key: string, value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const s = RAW_DATE_FIELD.test(key) ? formatRawDate(value) : String(value);
  return s.trim() === "" || s === "—" ? null : s;
}

/** The un-scrubbed source fields of two records, aligned key-by-key into the
 * same agree/disagree rows `compareRecords` produces for cleaned values, so
 * both tables read identically.
 *
 * Agreement is a trimmed, case-insensitive string comparison, the same test
 * `compareRecords` uses — but applied to *source* text, so two spellings the
 * cleaner would have reconciled (`"312-555-1234"` vs `"3125551234"`) honestly
 * show up as Different here. That divergence between the two tables is the
 * information a data steward opens the raw view for.
 *
 * Never returns `partial`: raw fields are single-valued (the multi-phone set
 * is a cleaned-side artifact, arriving here as separate `Phone0nNBR_raw`
 * columns). A side whose payload is absent — never published, or 404 — is
 * passed as `undefined` and renders as missing on every row. */
export function compareRawFields(
  fieldsA: Record<string, unknown> | undefined,
  fieldsB: Record<string, unknown> | undefined,
): ComparisonRow[] {
  // A's key order first (the publisher writes them in a deliberate
  // name → address → contact order, which JSON round-trips preserve), then
  // any key only B carries — an older payload published before a column
  // existed shouldn't silently drop its extra fields off the table.
  const keys = [
    ...new Set([...Object.keys(fieldsA ?? {}), ...Object.keys(fieldsB ?? {})]),
  ];

  return keys.map((key) => {
    const va = normalizeRaw(key, fieldsA?.[key]);
    const vb = normalizeRaw(key, fieldsB?.[key]);
    let result: ComparisonRow["result"];
    if (va === null || vb === null) {
      result = "missing";
    } else if (va.trim().toLowerCase() === vb.trim().toLowerCase()) {
      result = "exact";
    } else {
      result = "different";
    }
    return {
      label: key,
      valueA: va ?? "(missing)",
      valueB: vb ?? "(missing)",
      result,
    };
  });
}
