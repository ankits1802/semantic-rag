"use client";

import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
} from "recharts";
import type { BenchmarkResponse, MetricsMap } from "@/lib/types";

interface Props {
  report: BenchmarkResponse;
}

const KEY_METRICS = ["precision@5", "recall@5", "mrr", "hit_rate@5", "ndcg@5"];

const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.4, delay: i * 0.08, ease: [0.25, 0.46, 0.45, 0.94] },
  }),
};

function MetricBar({ label, a, b, index = 0 }: { label: string; a: number; b: number; index?: number }) {
  const delta = b - a;
  return (
    <motion.div
      custom={index}
      variants={cardVariants}
      initial="hidden"
      animate="visible"
      className="space-y-1"
    >
      <div className="flex justify-between text-xs text-gray-400">
        <span className="font-medium">{label}</span>
        <span className={delta > 0 ? "text-green-400" : delta < 0 ? "text-red-400" : "text-gray-500"}>
          {delta > 0 ? "+" : ""}{delta.toFixed(4)}
        </span>
      </div>
      <div className="flex gap-1 items-center">
        <div className="w-24 text-right text-xs text-indigo-300 font-mono">{a.toFixed(4)}</div>
        <div className="flex-1 h-2 rounded-full bg-gray-800 overflow-hidden relative">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${a * 100}%` }}
            transition={{ delay: 0.3 + index * 0.04, duration: 0.5, ease: "easeOut" }}
            className="absolute left-0 top-0 h-full bg-indigo-500 rounded-full"
          />
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${b * 100}%` }}
            transition={{ delay: 0.35 + index * 0.04, duration: 0.5, ease: "easeOut" }}
            className="absolute left-0 top-0 h-full bg-emerald-500 rounded-full opacity-70"
          />
        </div>
        <div className="w-24 text-left text-xs text-emerald-300 font-mono">{b.toFixed(4)}</div>
      </div>
    </motion.div>
  );
}

export function MetricsDashboard({ report }: Props) {
  const { aggregate_metrics_a: agg_a, aggregate_metrics_b: agg_b, overall_analysis, query_results } = report;

  // Bar chart data: key metrics comparison
  const chartData = KEY_METRICS.map((k) => ({
    metric: k.replace(/@/, "@"),
    "Strategy A": agg_a[k] ?? 0,
    "Strategy B": agg_b[k] ?? 0,
  }));

  // Radar chart data
  const radarData = KEY_METRICS.map((k) => ({
    metric: k,
    A: +(agg_a[k] ?? 0).toFixed(4),
    B: +(agg_b[k] ?? 0).toFixed(4),
  }));

  // Per-query MRR data
  const mrrData = query_results.map((r, i) => ({
    name: `Q${i + 1}`,
    title: r.query.slice(0, 40),
    "Strategy A": r.strategy_a.metrics["mrr"] ?? 0,
    "Strategy B": r.strategy_b.metrics["mrr"] ?? 0,
  }));

  return (
    <div className="space-y-8">
      {/* Summary */}
      <motion.div
        custom={0}
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        className="rounded-xl border border-gray-800 bg-gray-900 p-5 space-y-4"
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white">Benchmark Summary</h2>
          <span className="text-xs text-gray-500">{report.timestamp}</span>
        </div>
        <p className="text-sm text-gray-300 leading-relaxed">{overall_analysis}</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {["mrr", "precision@5", "recall@5", "ndcg@5"].map((k, ki) => {
            const a = agg_a[k] ?? 0;
            const b = agg_b[k] ?? 0;
            const improved = b > a;
            return (
              <motion.div
                key={k}
                custom={ki}
                variants={cardVariants}
                initial="hidden"
                animate="visible"
                whileHover={{ scale: 1.03, transition: { duration: 0.15 } }}
                className="rounded-lg bg-gray-800/60 border border-gray-700 p-3 space-y-1"
              >
                <p className="text-xs text-gray-500 uppercase tracking-wider">{k}</p>
                <div className="flex gap-2 items-end">
                  <span className="text-lg font-bold text-white">{b.toFixed(3)}</span>
                  <span className={`text-xs mb-0.5 ${improved ? "text-green-400" : "text-red-400"}`}>
                    {improved ? "▲" : "▼"}{Math.abs(b - a).toFixed(3)}
                  </span>
                </div>
                <p className="text-xs text-gray-600">A: {a.toFixed(3)}</p>
              </motion.div>
            );
          })}
        </div>
      </motion.div>

      {/* Grouped bar chart */}
      <motion.div
        custom={1}
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        className="rounded-xl border border-gray-800 bg-gray-900 p-5"
      >
        <h3 className="text-sm font-semibold text-gray-300 mb-4">Aggregate Metrics Comparison</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="metric" tick={{ fontSize: 11, fill: "#9ca3af" }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#9ca3af" }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 12 }}
              formatter={(v: number) => v.toFixed(4)}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="Strategy A" fill="#6366f1" radius={[3, 3, 0, 0]} />
            <Bar dataKey="Strategy B" fill="#10b981" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Per-query MRR chart */}
      <motion.div
        custom={2}
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        className="rounded-xl border border-gray-800 bg-gray-900 p-5"
      >
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          MRR per Query — Strategy A vs B
        </h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={mrrData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#9ca3af" }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#9ca3af" }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 12 }}
              formatter={(v: number) => [v.toFixed(4)]}
              labelFormatter={(label: string) => {
                const item = mrrData.find((d) => d.name === label);
                return item ? item.title : label;
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="Strategy A" fill="#6366f1" radius={[3, 3, 0, 0]} />
            <Bar dataKey="Strategy B" fill="#10b981" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Detailed metric comparison (all metrics) */}
      <motion.div
        custom={3}
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        className="rounded-xl border border-gray-800 bg-gray-900 p-5 space-y-3"
      >
        <h3 className="text-sm font-semibold text-gray-300">All Metrics Detail</h3>
        <div className="flex gap-6 text-xs text-gray-500 mb-1">
          <span className="w-24 text-right">Strategy A</span>
          <span className="flex-1" />
          <span className="w-24">Strategy B</span>
        </div>
        {Object.keys(agg_a)
          .sort()
          .map((k, ki) => (
            <MetricBar key={k} label={k} a={agg_a[k] ?? 0} b={agg_b[k] ?? 0} index={ki} />
          ))}
      </motion.div>

      {/* Per-query detail table */}
      <motion.div
        custom={4}
        variants={cardVariants}
        initial="hidden"
        animate="visible"
        className="rounded-xl border border-gray-800 bg-gray-900 p-5 space-y-3"
      >
        <h3 className="text-sm font-semibold text-gray-300">Per-Query Analysis</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-gray-700 text-gray-500">
                <th className="py-2 pr-3 font-medium">Query</th>
                <th className="py-2 px-2 font-medium">A MRR</th>
                <th className="py-2 px-2 font-medium">B MRR</th>
                <th className="py-2 px-2 font-medium">A P@5</th>
                <th className="py-2 px-2 font-medium">B P@5</th>
                <th className="py-2 px-2 font-medium">Analysis</th>
              </tr>
            </thead>
            <tbody>
              {query_results.map((r, ri) => (
                <motion.tr
                  key={r.query}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.6 + ri * 0.05 }}
                  className="border-b border-gray-800 hover:bg-gray-800/30 transition-colors"
                >
                  <td className="py-2 pr-3 text-gray-300 max-w-xs truncate" title={r.query}>
                    {r.query}
                  </td>
                  <td className="py-2 px-2 font-mono text-indigo-300">
                    {(r.strategy_a.metrics["mrr"] ?? 0).toFixed(3)}
                  </td>
                  <td className="py-2 px-2 font-mono text-emerald-300">
                    {(r.strategy_b.metrics["mrr"] ?? 0).toFixed(3)}
                  </td>
                  <td className="py-2 px-2 font-mono text-indigo-300">
                    {(r.strategy_a.metrics["precision@5"] ?? 0).toFixed(3)}
                  </td>
                  <td className="py-2 px-2 font-mono text-emerald-300">
                    {(r.strategy_b.metrics["precision@5"] ?? 0).toFixed(3)}
                  </td>
                  <td className="py-2 px-2 text-gray-500 max-w-xs truncate" title={r.analysis}>
                    {r.analysis}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  );
}
