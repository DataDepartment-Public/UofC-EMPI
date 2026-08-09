import { NextRequest, NextResponse } from "next/server";
import { apiPostJson, UpstreamError } from "@/lib/server-api";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const data = await apiPostJson("/audit/unmerge", body);
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}
