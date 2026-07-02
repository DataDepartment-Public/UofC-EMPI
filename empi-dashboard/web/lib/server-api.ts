/**
 * Server-only FastAPI client — used exclusively inside `app/api/*` Route
 * Handlers (the BFF layer). The browser never imports this file or talks to
 * FastAPI directly (docs/Application-Architecture.md §3): the BFF proxies
 * every call and injects `X-Reviewer-Id` server-side so the audit trail
 * can't be spoofed from the client.
 */
import "server-only";

const API_BASE = process.env.EMPI_API_URL ?? "http://localhost:8000";

/**
 * Local-build reviewer identity (docs/Application-Architecture.md §3
 * "Identity / auth" — no login UI was requested for this build; a real
 * deploy would source this from a server-side session instead).
 */
export const REVIEWER_ID = "reviewer.jclark";

export class UpstreamError extends Error {
  constructor(
    public status: number,
    public body: unknown,
  ) {
    super(`FastAPI responded ${status}`);
  }
}

async function request(
  path: string,
  init?: RequestInit & { reviewer?: boolean },
): Promise<Response> {
  const headers = new Headers(init?.headers);
  if (init?.reviewer) headers.set("X-Reviewer-Id", REVIEWER_ID);
  return fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await request(path);
  if (!res.ok) throw new UpstreamError(res.status, await safeJson(res));
  return res.json() as Promise<T>;
}

export async function apiPostJson<T>(path: string, body: unknown): Promise<T> {
  const res = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    reviewer: true,
  });
  if (!res.ok) throw new UpstreamError(res.status, await safeJson(res));
  return res.json() as Promise<T>;
}

export async function apiPostForm<T>(path: string, form: FormData): Promise<T> {
  const res = await request(path, { method: "POST", body: form });
  if (!res.ok) throw new UpstreamError(res.status, await safeJson(res));
  return res.json() as Promise<T>;
}

async function safeJson(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch {
    return { detail: res.statusText };
  }
}

export { API_BASE };
