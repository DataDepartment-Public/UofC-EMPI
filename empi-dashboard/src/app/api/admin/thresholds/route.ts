import { NextRequest, NextResponse } from "next/server";
import { apiGet, apiPutJson, UpstreamError } from "@/lib/server-api";

export async function GET() {
  try {
    const data = await apiGet("/admin/thresholds");
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}

export async function PUT(req: NextRequest) {
  try {
    const body = await req.json();
    const data = await apiPutJson("/admin/thresholds", body);
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}
