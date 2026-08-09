import type { ComparisonRow } from "@/lib/compare";
import { SsnReveal } from "./SsnReveal";

/** Shared with `RawComparisonPanel` so the cleaned-value table and the raw
 * one render an identical Result column — a reviewer reads them stacked, and
 * two vocabularies for the same verdict would imply a distinction that
 * doesn't exist. */
export const RESULT_STYLE: Record<ComparisonRow["result"], { label: string; className: string }> = {
  exact: { label: "Exact match", className: "text-status-auto font-bold" },
  // Only a multi-valued field (Phones) can land here: the two sets overlap
  // without being equal, which supports a duplicate without confirming one.
  partial: { label: "Some shared", className: "text-status-review font-bold" },
  different: { label: "Different", className: "text-status-nomatch font-bold" },
  missing: { label: "One or both missing", className: "text-gray font-semibold" },
};

/** FR-33/36/37: structured per-feature comparison table. FR-36's "which
 * traits increased vs decreased match probability" is adapted (no
 * probabilistic model in production) into a plain agree/disagree signal per
 * field — the row highlight color communicates the same "supports duplicate
 * vs doesn't" intuition without fabricating a numeric contribution.
 *
 * `patidA`/`patidB`, when provided, let the SSN row render a reveal-in-place
 * toggle (`SsnReveal`) instead of a static masked value. */
export function FeatureComparisonTable({
  rows,
  patidA,
  patidB,
}: {
  rows: ComparisonRow[];
  patidA?: string;
  patidB?: string;
}) {
  return (
    <table className="w-full text-[13px]">
      <thead>
        <tr className="border-b border-line text-left text-[11px] font-bold tracking-wide text-gray uppercase">
          <th className="py-2">Feature</th>
          <th className="py-2">Patient A</th>
          <th className="py-2">Patient B</th>
          <th className="py-2">Result</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const style = RESULT_STYLE[row.result];
          const isSsnRow = row.label.startsWith("SSN") && patidA && patidB;
          return (
            <tr key={row.label} className="border-b border-line last:border-none">
              <td className="py-2.5 align-top font-bold text-ink-2">{row.label}</td>
              <td className="py-2.5 align-top text-gray-2">
                {isSsnRow ? (
                  <SsnReveal patid={patidA} masked={row.valueA} />
                ) : (
                  <Value text={row.valueA} values={row.valuesA} />
                )}
              </td>
              <td className="py-2.5 align-top text-gray-2">
                {isSsnRow ? (
                  <SsnReveal patid={patidB} masked={row.valueB} />
                ) : (
                  <Value text={row.valueB} values={row.valuesB} />
                )}
              </td>
              <td className={`py-2.5 align-top ${style.className}`}>{style.label}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** A multi-valued field (`values`) is stacked one entry per line — a record
 * can carry four phone numbers, and comma-joining them onto one line makes
 * the two sides impossible to scan against each other. Anything else renders
 * as the plain string it already was. */
function Value({ text, values }: { text: string; values?: string[] }) {
  if (!values || values.length === 0) return <>{text}</>;
  return (
    <div className="flex flex-col gap-0.5">
      {values.map((v) => (
        <span key={v}>{v}</span>
      ))}
    </div>
  );
}
