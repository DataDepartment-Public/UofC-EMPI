import { NextResponse } from "next/server";
import { apiGet, UpstreamError } from "@/lib/server-api";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ mid: string }> },
) {
  const { mid } = await params;
  try {
    const data = await apiGet(`/clusters/${encodeURIComponent(mid)}`);
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}
