import { NextRequest, NextResponse } from "next/server";
import { apiGet, UpstreamError } from "@/lib/server-api";

export async function GET(req: NextRequest) {
  const qs = req.nextUrl.search;
  try {
    const data = await apiGet(`/audit${qs}`);
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}
