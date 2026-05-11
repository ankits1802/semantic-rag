"use client";

import { motion } from "framer-motion";
import type { StrategyResult } from "@/lib/types";

interface Props {
  result: StrategyResult;
}

export function ExpandedQueryDisplay({ result }: Props) {
  if (!result.expanded_query && (!result.keywords_added || result.keywords_added.length === 0)) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="rounded-lg border border-indigo-800/50 bg-indigo-950/30 px-4 py-3 space-y-2"
    >
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
        className="text-xs font-semibold text-indigo-300 uppercase tracking-wider"
      >
        Query Expansion (Strategy B)
      </motion.p>
      {result.expanded_query && (
        <motion.p
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.15, duration: 0.25 }}
          className="text-sm text-gray-300 italic leading-relaxed"
        >
          &ldquo;{result.expanded_query}&rdquo;
        </motion.p>
      )}
      {result.keywords_added && result.keywords_added.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {result.keywords_added.map((kw, i) => (
            <motion.span
              key={kw}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 + i * 0.04, type: "spring", stiffness: 260, damping: 18 }}
              className="inline-block rounded-full bg-indigo-900/60 border border-indigo-700 px-2 py-0.5 text-xs text-indigo-200"
            >
              +{kw}
            </motion.span>
          ))}
        </div>
      )}
    </motion.div>
  );
}
