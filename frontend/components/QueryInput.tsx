"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { ExpansionMode, Strategy } from "@/lib/types";

interface Props {
  onSearch: (query: string, strategy: Strategy, topK: number, mode: ExpansionMode) => void;
  onBenchmark: () => void;
  isSearching: boolean;
  isBenchmarking: boolean;
}

const EXAMPLE_QUERIES = [
  "How does the system handle peak load?",
  "What happens when a node fails?",
  "How is traffic distributed across servers?",
  "How do we prevent API abuse?",
  "How does Kubernetes scale applications automatically?",
];

function Spinner() {
  return (
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }}
      className="inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full"
    />
  );
}

export function QueryInput({ onSearch, onBenchmark, isSearching, isBenchmarking }: Props) {
  const [query, setQuery] = useState("");
  const [strategy, setStrategy] = useState<Strategy>("both");
  const [topK, setTopK] = useState(5);
  const [expansionMode, setExpansionMode] = useState<ExpansionMode>("full");
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    onSearch(query.trim(), strategy, topK, expansionMode);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="rounded-xl border border-gray-800 bg-gray-900 p-6 space-y-4 shadow-xl"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Main input with focus glow */}
        <div className="flex gap-3">
          <motion.div
            animate={{
              boxShadow: isFocused
                ? "0 0 0 2px rgba(99, 102, 241, 0.4)"
                : "0 0 0 0px rgba(99, 102, 241, 0)",
            }}
            transition={{ duration: 0.2 }}
            className="flex-1 rounded-lg overflow-hidden"
          >
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Enter a technical question about system design…"
              className="w-full rounded-lg border border-gray-700 bg-gray-800 px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500 transition-colors"
            />
          </motion.div>
          <motion.button
            type="submit"
            disabled={isSearching || !query.trim()}
            whileHover={{ scale: isSearching || !query.trim() ? 1 : 1.03 }}
            whileTap={{ scale: isSearching || !query.trim() ? 1 : 0.97 }}
            className="px-6 py-3 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {isSearching ? (
              <>
                <Spinner />
                <span>Searching…</span>
              </>
            ) : (
              "Search"
            )}
          </motion.button>
        </div>

        {/* Options row */}
        <div className="flex flex-wrap gap-4 items-center">
          {/* Strategy */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400 whitespace-nowrap">Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as Strategy)}
              className="rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-colors"
            >
              <option value="both">Both (A + B)</option>
              <option value="a">A — Direct Vector</option>
              <option value="b">B — AI-Enhanced</option>
              <option value="hybrid">Hybrid (BM25 + Dense)</option>
            </select>
          </div>

          {/* Top-K */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400">Top-K</label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-colors"
            >
              {[1, 3, 5, 10].map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </div>

          {/* Expansion mode */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400 whitespace-nowrap">Expansion mode</label>
            <select
              value={expansionMode}
              onChange={(e) => setExpansionMode(e.target.value as ExpansionMode)}
              className="rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-colors"
            >
              <option value="full">Full</option>
              <option value="synonyms">Synonyms</option>
              <option value="technical">Technical</option>
              <option value="hyde">HyDE</option>
            </select>
          </div>

          {/* Benchmark button */}
          <motion.button
            type="button"
            onClick={onBenchmark}
            disabled={isBenchmarking}
            whileHover={{ scale: isBenchmarking ? 1 : 1.03 }}
            whileTap={{ scale: isBenchmarking ? 1 : 0.97 }}
            className="ml-auto px-4 py-1.5 rounded border border-gray-600 text-xs text-gray-300 hover:bg-gray-800 hover:border-gray-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-1.5"
          >
            {isBenchmarking ? (
              <>
                <Spinner />
                <span>Benchmarking…</span>
              </>
            ) : (
              "Run Benchmark Suite"
            )}
          </motion.button>
        </div>
      </form>

      {/* Example queries */}
      <div className="flex flex-wrap gap-2 pt-1 border-t border-gray-800">
        <span className="text-xs text-gray-500 mt-0.5">Examples:</span>
        {EXAMPLE_QUERIES.map((q, i) => (
          <motion.button
            key={q}
            type="button"
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.05 * i }}
            onClick={() => setQuery(q)}
            className="text-xs text-indigo-400 hover:text-indigo-300 underline underline-offset-2 transition-colors"
          >
            {q}
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}
