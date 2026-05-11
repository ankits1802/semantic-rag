"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { search, runBenchmark } from "@/lib/api";
import type {
  BenchmarkResponse,
  ExpansionMode,
  SearchResponse,
  Strategy,
} from "@/lib/types";
import { QueryInput } from "@/components/QueryInput";
import { ResultsComparison } from "@/components/ResultsComparison";
import { MetricsDashboard } from "@/components/MetricsDashboard";

export default function HomePage() {
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"search" | "benchmark">("search");

  async function handleSearch(
    query: string,
    strategy: Strategy,
    topK: number,
    expansionMode: ExpansionMode
  ) {
    setIsSearching(true);
    setError(null);
    try {
      const result = await search({ query, strategy, top_k: topK, expansion_mode: expansionMode });
      setSearchResult(result);
      setActiveTab("search");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsSearching(false);
    }
  }

  async function handleBenchmark() {
    setIsBenchmarking(true);
    setError(null);
    try {
      const result = await runBenchmark(5);
      setBenchmarkResult(result);
      setActiveTab("benchmark");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIsBenchmarking(false);
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="max-w-screen-2xl mx-auto px-4 py-8 space-y-6"
    >
      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="text-center space-y-3 mb-8"
      >
        <motion.div
          initial={{ scale: 0.9 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.5, type: "spring", stiffness: 120 }}
        >
          <h1 className="text-3xl font-bold text-white tracking-tight">
            Context-Aware Retrieval Engine
          </h1>
        </motion.div>
        <p className="text-gray-400 text-sm max-w-2xl mx-auto leading-relaxed">
          Dual-strategy RAG system — compare direct vector search (Strategy A) with
          AI-enhanced query expansion + Reciprocal Rank Fusion (Strategy B) powered by{" "}
          <span className="text-indigo-400 font-medium">gemini-3.1-pro-preview</span>.
        </p>
        <div className="flex justify-center gap-3 pt-1">
          {["FAISS Vector Store", "BM25 Hybrid Search", "Cross-Encoder Reranking"].map((tag, i) => (
            <motion.span
              key={tag}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 + i * 0.08 }}
              className="text-xs px-2.5 py-0.5 rounded-full bg-gray-800 text-gray-400 border border-gray-700"
            >
              {tag}
            </motion.span>
          ))}
        </div>
      </motion.div>

      {/* Query Input */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
      >
        <QueryInput
          onSearch={handleSearch}
          onBenchmark={handleBenchmark}
          isSearching={isSearching}
          isBenchmarking={isBenchmarking}
        />
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8, height: 0 }}
            animate={{ opacity: 1, y: 0, height: "auto" }}
            exit={{ opacity: 0, y: -8, height: 0 }}
            transition={{ duration: 0.25 }}
            className="rounded-lg border border-red-700 bg-red-950/40 px-4 py-3 text-sm text-red-300"
          >
            <strong>Error:</strong> {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Tabs */}
      <AnimatePresence>
        {(searchResult || benchmarkResult) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-2 border-b border-gray-800"
          >
            {searchResult && (
              <button
                onClick={() => setActiveTab("search")}
                className={`relative px-4 py-2 text-sm font-medium transition-colors ${
                  activeTab === "search"
                    ? "text-white"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                Search Results
                {activeTab === "search" && (
                  <motion.div
                    layoutId="tab-indicator"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-500"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
              </button>
            )}
            {benchmarkResult && (
              <button
                onClick={() => setActiveTab("benchmark")}
                className={`relative px-4 py-2 text-sm font-medium transition-colors ${
                  activeTab === "benchmark"
                    ? "text-white"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                Benchmark ({benchmarkResult.num_queries} queries)
                {activeTab === "benchmark" && (
                  <motion.div
                    layoutId="tab-indicator"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-500"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Content */}
      <AnimatePresence mode="wait">
        {activeTab === "search" && searchResult && (
          <motion.div
            key="search"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.3 }}
          >
            <ResultsComparison response={searchResult} />
          </motion.div>
        )}
        {activeTab === "benchmark" && benchmarkResult && (
          <motion.div
            key="benchmark"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.3 }}
          >
            <MetricsDashboard report={benchmarkResult} />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
