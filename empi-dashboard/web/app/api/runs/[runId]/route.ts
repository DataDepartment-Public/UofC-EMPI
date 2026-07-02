import { NextResponse } from "next/server";
import { apiGet, UpstreamError } from "@/lib/server-api";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ runId: string }> },
) {
  const { runId } = await params;
  try {
    const data = await apiGet(`/runs/${encodeURIComponent(runId)}`);
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}
