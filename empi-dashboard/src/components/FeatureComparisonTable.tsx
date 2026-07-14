import type { ComparisonRow } from "@/lib/compare";

const RESULT_STYLE: Record<ComparisonRow["result"], { label: string; className: string }> = {
  exact: { label: "Exact match", className: "text-status-auto font-bold" },
  different: { label: "Different", className: "text-status-nomatch font-bold" },
  missing: { label: "One or both missing", className: "text-gray font-semibold" },
};

/** FR-33/36/37: structured per-feature comparison table. FR-36's "which
 * traits increased vs decreased match probability" is adapted (no
 * probabilistic model in production) into a plain agree/disagree signal per
 * field — the row highlight color communicates the same "supports duplicate
 * vs doesn't" intuition without fabricating a numeric contribution. */
export function FeatureComparisonTable({ rows }: { rows: ComparisonRow[] }) {
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
          return (
            <tr key={row.label} className="border-b border-line last:border-none">
              <td className="py-2.5 font-bold text-ink-2">{row.label}</td>
              <td className="py-2.5 text-gray-2">{row.valueA}</td>
              <td className="py-2.5 text-gray-2">{row.valueB}</td>
              <td className={`py-2.5 ${style.className}`}>{style.label}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
