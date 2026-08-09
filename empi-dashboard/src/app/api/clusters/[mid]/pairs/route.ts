import { NextRequest, NextResponse } from "next/server";
import { apiGet, UpstreamError } from "@/lib/server-api";

/** Read-only, and deliberately not reviewer-tagged: unlike
 * `/records/{patid}/raw`, this returns no unmasked PHI — only pair-level
 * pipeline decisions — so it writes no `audit_log` row and needs no
 * `X-Reviewer-Id`. */
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ mid: string }> },
) {
  const { mid } = await params;
  try {
    const data = await apiGet(
      `/clusters/${encodeURIComponent(mid)}/pairs${req.nextUrl.search}`,
    );
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}
