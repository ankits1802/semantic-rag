/**
 * TypeScript type definitions matching the FastAPI response schemas.
 */

export interface SearchResultItem {
  chunk_id: string;
  rank: number;
  score: number;
  source: string;
  section: string | null;
  text: string;
  metadata: Record<string, unknown>;
}

export interface StrategyResult {
  strategy: string;
  latency_ms: number;
  results: SearchResultItem[];
  expanded_query?: string;
  keywords_added?: string[];
}

export interface SearchResponse {
  query: string;
  strategy_a?: StrategyResult;
  strategy_b?: StrategyResult;
  hybrid?: StrategyResult;
}

export type Strategy = "both" | "a" | "b" | "hybrid";
export type ExpansionMode = "full" | "synonyms" | "technical" | "hyde";

export interface SearchRequest {
  query: string;
  top_k: number;
  strategy: Strategy;
  expansion_mode: ExpansionMode;
}

// ── Benchmark types ──────────────────────────────────────────────────────────

export interface MetricsMap {
  [key: string]: number;
}

export interface MetricComparison {
  strategy_a: number;
  strategy_b: number;
  delta: number;
  "relative_improvement_%": number | null;
  /** @deprecated use `relative_improvement_%` */
  relative_improvement_percent?: number | null;
}

export interface QueryBenchmarkResult {
  query: string;
  category: string;
  relevant_keywords: string[];
  strategy_a: {
    retrieved_chunks: SearchResultItem[];
    latency_ms: number;
    metrics: MetricsMap;
  };
  strategy_b: {
    expanded_query: string;
    keywords_added: string[];
    retrieved_chunks: SearchResultItem[];
    latency_ms: number;
    metrics: MetricsMap;
  };
  comparison: Record<string, MetricComparison>;
  analysis: string;
}

export interface BenchmarkResponse {
  timestamp: string;
  num_queries: number;
  aggregate_metrics_a: MetricsMap;
  aggregate_metrics_b: MetricsMap;
  aggregate_comparison: Record<string, MetricComparison>;
  overall_analysis: string;
  query_results: QueryBenchmarkResult[];
}

// ── Health ───────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  num_chunks: number;
  model: string;
  uptime_s: number;
}
