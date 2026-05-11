"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { SearchResultItem } from "@/lib/types";

interface Props {
  strategyAResults?: SearchResultItem[];
  strategyBResults?: SearchResultItem[];
  label?: string;
}

interface ChartDatum {
  rank: string;
  scoreA?: number;
  scoreB?: number;
}

export function ScoreVisualization({ strategyAResults, strategyBResults, label }: Props) {
  const maxLen = Math.max(
    strategyAResults?.length ?? 0,
    strategyBResults?.length ?? 0
  );

  if (maxLen === 0) return null;

  const data: ChartDatum[] = Array.from({ length: maxLen }, (_, i) => ({
    rank: `#${i + 1}`,
    scoreA: strategyAResults?.[i]?.score,
    scoreB: strategyBResults?.[i]?.score,
  }));

  return (
    <div className="space-y-2">
      {label && <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{label}</p>}
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="rank" tick={{ fontSize: 11, fill: "#9ca3af" }} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#9ca3af" }} />
          <Tooltip
            contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", fontSize: 12 }}
            formatter={(value: number) => value.toFixed(4)}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {strategyAResults && (
            <Bar dataKey="scoreA" name="Strategy A" fill="#6366f1" radius={[3, 3, 0, 0]} />
          )}
          {strategyBResults && (
            <Bar dataKey="scoreB" name="Strategy B" fill="#10b981" radius={[3, 3, 0, 0]} />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
