import { NextRequest, NextResponse } from "next/server";
import { apiGet, apiPostForm, UpstreamError } from "@/lib/server-api";

export async function GET() {
  try {
    const data = await apiGet("/runs");
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}

export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const data = await apiPostForm("/runs", form);
    return NextResponse.json(data, { status: 202 });
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}
