"use client";

import { motion, AnimatePresence } from "framer-motion";
import type { SearchResponse } from "@/lib/types";
import { ChunkCard } from "./ChunkCard";
import { ExpandedQueryDisplay } from "./ExpandedQueryDisplay";
import { ScoreVisualization } from "./ScoreVisualization";

interface Props {
  response: SearchResponse;
}

const columnVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, delay: i * 0.12, ease: [0.25, 0.46, 0.45, 0.94] },
  }),
};

function StrategyColumn({
  title,
  color,
  badge,
  result,
  showExpansion,
  columnIndex,
}: {
  title: string;
  color: string;
  badge: string;
  result: NonNullable<SearchResponse["strategy_a"]>;
  showExpansion?: boolean;
  columnIndex?: number;
}) {
  return (
    <motion.div
      custom={columnIndex ?? 0}
      variants={columnVariants}
      initial="hidden"
      animate="visible"
      className="space-y-3 min-w-0"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${badge}`}>{title.split("—")[0].trim()}</span>
          <h3 className={`text-sm font-semibold ${color}`}>{title.split("—").slice(1).join("—").trim()}</h3>
        </div>
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="text-xs text-gray-500 font-mono bg-gray-800 px-2 py-0.5 rounded"
        >
          {result.latency_ms.toFixed(1)} ms
        </motion.span>
      </div>

      {showExpansion && <ExpandedQueryDisplay result={result} />}

      <div className="space-y-2">
        {result.results.map((item, i) => (
          <ChunkCard key={item.chunk_id} result={item} index={i} />
        ))}
      </div>
    </motion.div>
  );
}

export function ResultsComparison({ response }: Props) {
  const { query, strategy_a, strategy_b, hybrid } = response;
  const showBoth = !!(strategy_a && strategy_b);

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={query}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.3 }}
        className="space-y-6"
      >
        {/* Query header */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="rounded-lg border border-gray-700 bg-gray-900/60 px-4 py-3 backdrop-blur-sm"
        >
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Query</p>
          <p className="text-sm text-white font-medium">{query}</p>
        </motion.div>

        {/* Score chart (when both strategies present) */}
        {showBoth && (
          <motion.div
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="rounded-xl border border-gray-800 bg-gray-900 p-5"
          >
            <ScoreVisualization
              strategyAResults={strategy_a?.results}
              strategyBResults={strategy_b?.results}
              label="Similarity score by rank position"
            />
          </motion.div>
        )}

        {/* Results columns */}
        {showBoth ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <StrategyColumn
              title="Strategy A — Direct Vector Search"
              color="text-indigo-300"
              badge="border-indigo-700/50 text-indigo-400 bg-indigo-900/20"
              result={strategy_a!}
              columnIndex={0}
            />
            <StrategyColumn
              title="Strategy B — AI-Enhanced Retrieval"
              color="text-emerald-300"
              badge="border-emerald-700/50 text-emerald-400 bg-emerald-900/20"
              result={strategy_b!}
              showExpansion
              columnIndex={1}
            />
          </div>
        ) : (
          <div className="space-y-6">
            {strategy_a && (
              <StrategyColumn
                title="Strategy A — Direct Vector Search"
                color="text-indigo-300"
                badge="border-indigo-700/50 text-indigo-400 bg-indigo-900/20"
                result={strategy_a}
                columnIndex={0}
              />
            )}
            {strategy_b && (
              <StrategyColumn
                title="Strategy B — AI-Enhanced Retrieval"
                color="text-emerald-300"
                badge="border-emerald-700/50 text-emerald-400 bg-emerald-900/20"
                result={strategy_b}
                showExpansion
                columnIndex={0}
              />
            )}
            {hybrid && (
              <StrategyColumn
                title="Hybrid — BM25 + Dense Fusion"
                color="text-amber-300"
                badge="border-amber-700/50 text-amber-400 bg-amber-900/20"
                result={hybrid}
                columnIndex={0}
              />
            )}
          </div>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
