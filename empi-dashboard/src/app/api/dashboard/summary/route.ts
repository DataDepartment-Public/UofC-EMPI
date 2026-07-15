import { NextResponse } from "next/server";
import { apiGet, UpstreamError } from "@/lib/server-api";

export async function GET() {
  try {
    const data = await apiGet("/dashboard/summary");
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}
