import { NextResponse } from "next/server";
import { apiGet, UpstreamError } from "@/lib/server-api";

export async function GET() {
  try {
    const data = await apiGet("/health/ready");
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    return NextResponse.json({ status: "not_ready" }, { status: 503 });
  }
}
