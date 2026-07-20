import { NextResponse } from "next/server";
import { apiGet, UpstreamError } from "@/lib/server-api";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ patid: string }> },
) {
  const { patid } = await params;
  try {
    const data = await apiGet(`/records/${encodeURIComponent(patid)}/raw`);
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}
