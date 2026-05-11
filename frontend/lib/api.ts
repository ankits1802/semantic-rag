/**
 * Typed API client functions.
 * All functions call the FastAPI backend via the NEXT_PUBLIC_API_URL env var.
 */

import type {
  BenchmarkResponse,
  ExpansionMode,
  HealthResponse,
  SearchRequest,
  SearchResponse,
  Strategy,
} from "./types";

const BASE_URL =
  typeof window === "undefined"
    ? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
    : (window as Window & { __NEXT_PUBLIC_API_URL__?: string }).__NEXT_PUBLIC_API_URL__ ??
      process.env.NEXT_PUBLIC_API_URL ??
      "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(`API ${res.status}: ${error}`);
  }
  return res.json() as Promise<T>;
}

// ── Search ────────────────────────────────────────────────────────────────────

export async function search(req: SearchRequest): Promise<SearchResponse> {
  return apiFetch<SearchResponse>("/api/search", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

export async function runBenchmark(topK = 5): Promise<BenchmarkResponse> {
  return apiFetch<BenchmarkResponse>("/api/benchmark", {
    method: "POST",
    body: JSON.stringify({ top_k: topK, save_report: true }),
  });
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/health");
}

// ── Documents ─────────────────────────────────────────────────────────────────

export async function getDocuments(): Promise<{ count: number; sources: string[] }> {
  return apiFetch<{ count: number; sources: string[] }>("/api/documents");
}
