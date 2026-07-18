"use client";

import { useState } from "react";
import Link from "next/link";
import clsx from "clsx";
import type { Entity, EntityMember } from "@/lib/schemas";
import { encodeExplainPayload, ExplainPatient } from "@/lib/explain";
import { formatDate, fullName, maskSsn } from "@/lib/format";
import { SsnReveal } from "@/components/shared/SsnReveal";

interface Props {
  entity: Entity;
  onUnmerge: (mid: string, patid: string, patientName: string) => void;
  onViewRaw: (patid: string) => void;
  defaultExpanded?: boolean;
}

/** Patient Registry row — a resolved, final cluster. Expanding reveals its
 * constituent entries (the raw records folded into this patient) so a data
 * steward can trace provenance or correct a wrong merge; it does not surface
 * pending review candidates — those live entirely on the Review Queue tab. */
export function DatasetRow({
  entity,
  onUnmerge,
  onViewRaw,
  defaultExpanded = false,
}: Props) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const primary = entity.members.find((m) => m.is_primary) ?? entity.members[0];
  const expandable = entity.members.length > 0;

  return (
    <div className="card mb-2.5 overflow-hidden">
      <button
        onClick={() => expandable && setExpanded((e) => !e)}
        className={clsx(
          "grid w-full grid-cols-[20px_1.7fr_1fr_1fr_0.8fr_1fr] items-center gap-3 px-4 py-3 text-left text-[13px]",
          expandable && "cursor-pointer hover:bg-bg",
        )}
      >
        <span className="text-gray">
          {expandable ? (expanded ? "▾" : "▸") : ""}
        </span>
        <span>
          <span className="font-bold text-ink-2">
            {fullName(primary?.first_name, primary?.last_name)}
          </span>
          <div
            className="mt-0.5 font-mono text-[10px] text-gray"
            title="Master Patient ID"
          >
            {entity.mid}
          </div>
        </span>
        <span className="text-gray-2">{maskSsn(primary?.ssn_last4)}</span>
        <span className="text-gray-2">{primary?.birth_date ?? "—"}</span>
        <span className="text-gray-2">{entity.members.length}</span>
        <span className="text-xs text-gray">{formatDate(entity.updated_utc)}</span>
      </button>

      {expanded && (
        <div className="border-t border-line bg-bg/50 px-4 py-3">
          <Section title={`Entries (${entity.members.length})`}>
            {entity.members.map((m) => {
              const other = entity.members.find((x) => x.patid !== m.patid);
              return (
                <RecordLine
                  key={m.patid}
                  patid={m.patid}
                  name={fullName(m.first_name, m.last_name)}
                  ssn={maskSsn(m.ssn_last4)}
                  dob={m.birth_date}
                  tag={m.is_primary ? "Primary" : m.added_by === "pipeline" ? undefined : `by ${m.added_by}`}
                  explainHref={
                    other ? buildExplainHref(entity, m, other) : undefined
                  }
                  onViewRaw={() => onViewRaw(m.patid)}
                  action={
                    entity.is_merged
                      ? {
                          label: "Unmerge",
                          variant: "danger" as const,
                          onClick: () =>
                            onUnmerge(
                              entity.mid,
                              m.patid,
                              fullName(m.first_name, m.last_name),
                            ),
                        }
                      : undefined
                  }
                />
              );
            })}
          </Section>
        </div>
      )}
    </div>
  );
}

function toExplainPatient(m: EntityMember): ExplainPatient {
  return {
    patid: m.patid,
    first_name: m.first_name ?? null,
    last_name: m.last_name ?? null,
    birth_date: m.birth_date ?? null,
    ssn_last4: m.ssn_last4 ?? null,
    email: m.email ?? null,
    zip_code: m.zip_code ?? null,
    address1: m.address1 ?? null,
    sex: m.sex ?? null,
    phone: m.phone ?? null,
  };
}

function buildExplainHref(
  entity: Entity,
  a: EntityMember,
  b: EntityMember,
): string {
  const payload = encodeExplainPayload({
    mid: entity.mid,
    patientA: toExplainPatient(a),
    patientB: toExplainPatient(b),
    rule: entity.match_rule ?? null,
    confidence: entity.confidence,
    evidence: entity.evidence ?? null,
    updated: entity.updated_utc,
  });
  return `/dataset/${encodeURIComponent(entity.mid)}/explain?d=${payload}`;
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mb-3 last:mb-0">
      <div className="mb-1.5 text-[11px] font-bold tracking-wide text-gray uppercase">
        {title}
      </div>
      <div className="space-y-1.5">{children}</div>
    </div>
  );
}

function RecordLine({
  patid,
  name,
  ssn,
  dob,
  tag,
  action,
  explainHref,
  onViewRaw,
}: {
  patid: string;
  name: string;
  ssn: string;
  dob?: string | null;
  tag?: string;
  action?: { label: string; variant: "primary" | "danger"; onClick: () => void };
  explainHref?: string;
  onViewRaw: () => void;
}) {
  return (
    <div className="flex items-center justify-between rounded-md border border-line bg-white px-3 py-2 text-[13px]">
      <div className="flex min-w-0 items-center gap-3">
        {explainHref ? (
          <Link href={explainHref} className="font-semibold text-brand-blue hover:underline">
            {name}
          </Link>
        ) : (
          <span className="font-semibold text-ink-2">{name}</span>
        )}
        <span className="font-mono text-xs text-gray">{patid}</span>
        <span className="text-xs text-gray-2">
          <SsnReveal patid={patid} masked={ssn} />
        </span>
        <span className="text-xs text-gray-2">{dob ?? "—"}</span>
        {tag && (
          <span className="rounded-full bg-bg px-2 py-0.5 text-[10px] font-bold text-gray">
            {tag}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={onViewRaw}
          className="text-xs font-semibold text-gray-2 hover:text-brand-blue"
        >
          View raw data
        </button>
        {action && (
          <button
            onClick={action.onClick}
            className={clsx(
              "rounded-md px-3 py-1 text-xs font-bold text-white",
              action.variant === "danger"
                ? "bg-status-nomatch hover:opacity-90"
                : "bg-status-auto hover:opacity-90",
            )}
          >
            {action.label}
          </button>
        )}
      </div>
    </div>
  );
}
