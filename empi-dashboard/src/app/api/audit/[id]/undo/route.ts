import { NextResponse } from "next/server";
import { apiPostJson, UpstreamError } from "@/lib/server-api";

export async function POST(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  try {
    const data = await apiPostJson(`/audit/${encodeURIComponent(id)}/undo`, {});
    return NextResponse.json(data);
  } catch (err) {
    if (err instanceof UpstreamError) {
      return NextResponse.json(err.body, { status: err.status });
    }
    throw err;
  }
}
