/**
 * Server-only FastAPI client — used exclusively inside `app/api/*` Route
 * Handlers (the BFF layer). The browser never imports this file or talks to
 * FastAPI directly (docs/Application-Architecture.md §3): the BFF proxies
 * every call and injects `X-Reviewer-Id` server-side so the audit trail
 * can't be spoofed from the client.
 */
import "server-only";
import { headers as nextHeaders } from "next/headers";

const API_BASE = process.env.EMPI_API_URL ?? "http://localhost:8000";

/**
 * Local-dev fallback identity — used only when no platform-authenticated
 * principal is present, i.e. running outside Azure App Service's Easy Auth
 * (docker-compose, `npm run dev`). There's no Azure platform in front of
 * the container in those cases, so this keeps local dev exactly as easy as
 * it was before Entra ID SSO existed (terraform/auth.tf, to-do.md A5).
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

/**
 * The signed-in reviewer's identity for this request. In Azure, App
 * Service's Easy Auth authenticates the request before it ever reaches this
 * container and injects `X-MS-CLIENT-PRINCIPAL-NAME` — for the Entra ID
 * provider, populated from the user's UPN — so there is nothing here to
 * spoof from the client (the browser can't set platform-injected headers).
 * Falls back to `REVIEWER_ID` when that header is absent, which is always
 * the case locally.
 */
async function reviewerId(): Promise<string> {
  const incoming = await nextHeaders();
  const principalName = incoming.get("x-ms-client-principal-name");
  if (!principalName) return REVIEWER_ID;
  try {
    return decodeURIComponent(principalName);
  } catch {
    return principalName;
  }
}

async function request(
  path: string,
  init?: RequestInit & { reviewer?: boolean },
): Promise<Response> {
  const headers = new Headers(init?.headers);
  if (init?.reviewer) headers.set("X-Reviewer-Id", await reviewerId());
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
