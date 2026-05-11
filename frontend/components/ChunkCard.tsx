"use client";

import { motion } from "framer-motion";
import type { SearchResultItem } from "@/lib/types";

interface Props {
  result: SearchResultItem;
  index?: number;
}

const SCORE_COLOR = (score: number) => {
  if (score >= 0.75) return "text-green-400";
  if (score >= 0.5) return "text-yellow-400";
  return "text-red-400";
};

const SCORE_BG = (score: number) => {
  if (score >= 0.75) return "bg-green-900/30 border-green-700/40";
  if (score >= 0.5) return "bg-yellow-900/20 border-yellow-700/30";
  return "bg-red-900/20 border-red-700/30";
};

export function ChunkCard({ result, index = 0 }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{
        duration: 0.35,
        delay: index * 0.07,
        ease: [0.25, 0.46, 0.45, 0.94],
      }}
      whileHover={{
        scale: 1.012,
        transition: { duration: 0.15 },
      }}
      className="rounded-lg border border-gray-800 bg-gray-900/60 p-4 space-y-2 hover:border-gray-600 hover:bg-gray-900/90 hover:shadow-lg hover:shadow-black/30 transition-colors cursor-default"
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <motion.span
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: index * 0.07 + 0.15, type: "spring", stiffness: 300 }}
            className="shrink-0 inline-flex items-center justify-center h-5 w-5 rounded-full bg-brand-600 text-xs font-bold text-white"
          >
            {result.rank}
          </motion.span>
          <span className="text-xs text-gray-400 truncate" title={result.source}>
            {result.source}
          </span>
          {result.section && (
            <span className="text-xs text-gray-600">› {result.section}</span>
          )}
        </div>
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: index * 0.07 + 0.2 }}
          className={`shrink-0 font-mono text-xs font-semibold px-1.5 py-0.5 rounded border ${SCORE_COLOR(result.score)} ${SCORE_BG(result.score)}`}
        >
          {result.score.toFixed(4)}
        </motion.span>
      </div>

      {/* Score bar */}
      <div className="h-0.5 w-full bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(result.score * 100, 100)}%` }}
          transition={{ delay: index * 0.07 + 0.25, duration: 0.5, ease: "easeOut" }}
          className={`h-full rounded-full ${
            result.score >= 0.75
              ? "bg-green-500"
              : result.score >= 0.5
              ? "bg-yellow-500"
              : "bg-red-500"
          }`}
        />
      </div>

      {/* Text */}
      <p className="text-sm text-gray-300 leading-relaxed line-clamp-4">{result.text}</p>

      {/* Chunk ID */}
      <p className="text-xs text-gray-600 font-mono truncate">{result.chunk_id}</p>
    </motion.div>
  );
}
