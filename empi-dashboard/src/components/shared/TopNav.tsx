"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

/** FR-1/2/3: top nav (logo + title) + the top-level tabs (Dashboard /
 * Review Queue / Patient Registry). Tab switches are client-side Next.js
 * navigation — no full page reload. "Patient Registry" is the clinical-facing
 * label for the `/dataset` route — the final, resolved patient list, as
 * opposed to Review Queue's in-progress candidate work. */
export function TopNav() {
  const pathname = usePathname();
  const onDataset = pathname.startsWith("/dataset");
  const onReview = pathname.startsWith("/review");

  return (
    <div className="sticky top-0 z-50">
      <header className="flex h-16 items-center gap-4 border-b-[3px] border-brand-blue bg-white px-6 shadow-sm">
        <div className="flex items-center">
          <Image
            src="/alliance-logo.png"
            alt="AllianceChicago"
            width={800}
            height={148}
            priority
            className="h-9 w-auto"
          />
        </div>
        <div className="h-[30px] w-px bg-line" />
        <div className="text-[15px] font-semibold text-ink-2">
          Entity Matching Dashboard
          <span className="mt-0.5 block text-[11px] font-medium tracking-wide text-gray uppercase">
            eMPI reviewer console
          </span>
        </div>
      </header>
      <nav className="flex gap-1 border-b border-line bg-white px-6">
        <TabLink href="/" active={!onDataset && !onReview}>
          Dashboard
        </TabLink>
        <TabLink href="/review" active={onReview}>
          Review Queue
        </TabLink>
        <TabLink href="/dataset" active={onDataset}>
          Patient Registry
        </TabLink>
      </nav>
    </div>
  );
}

function TabLink({
  href,
  active,
  children,
}: {
  href: string;
  active: boolean;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className={clsx(
        "border-b-[3px] px-[18px] py-3.5 text-sm font-semibold transition-colors",
        active
          ? "border-brand-blue text-brand-blue"
          : "border-transparent text-gray-2 hover:text-brand-blue",
      )}
    >
      {children}
    </Link>
  );
}
