import { useState } from "react";
import { globalResults, kaggleStrokeResults, categoryColors, categoryLabels } from "../data/benchmarkData";

const SORT_ICONS = {
  asc: "↑",
  desc: "↓",
  none: "↕",
};

function getBestValue(data, key, lowerBetter = false) {
  const vals = data.map((d) => d[key]).filter((v) => v !== undefined);
  return lowerBetter ? Math.min(...vals) : Math.max(...vals);
}

export default function BenchmarkTable() {
  const [dataset, setDataset] = useState("global");
  const [sortKey, setSortKey] = useState(null);
  const [sortDir, setSortDir] = useState("desc");

  const isGlobal = dataset === "global";
  const data = isGlobal ? globalResults : kaggleStrokeResults;

  const globalCols = [
    { key: "name", label: "Method", lowerBetter: false, format: (v) => v },
    { key: "prAuc", label: "PR-AUC ↑", lowerBetter: false, format: (v) => v.toFixed(4) },
    { key: "f1", label: "F1 ↑", lowerBetter: false, format: (v) => v.toFixed(4) },
    { key: "balancedAcc", label: "Bal. Acc ↑", lowerBetter: false, format: (v) => v.toFixed(4) },
    { key: "ece", label: "ECE ↓", lowerBetter: true, format: (v) => v.toFixed(4) },
    { key: "kendallTau", label: "Kendall τ ↑", lowerBetter: false, format: (v) => v.toFixed(4) },
    { key: "wasserstein", label: "Wasserstein ↓", lowerBetter: true, format: (v) => v.toFixed(4) },
    { key: "ksPass", label: "KS Pass% ↑", lowerBetter: false, format: (v) => `${v.toFixed(1)}%` },
    { key: "latency", label: "Latency ↓", lowerBetter: true, format: (v) => `${v.toFixed(2)}s` },
  ];

  const strokeCols = [
    { key: "name", label: "Method", lowerBetter: false, format: (v) => v },
    { key: "prAuc", label: "PR-AUC ↑", lowerBetter: false, format: (v) => v.toFixed(4) },
    { key: "minorityRecall", label: "Recall ↑", lowerBetter: false, format: (v) => v.toFixed(4) },
    { key: "minorityF1", label: "F1 ↑", lowerBetter: false, format: (v) => v.toFixed(4) },
    { key: "ece", label: "ECE ↓", lowerBetter: true, format: (v) => v.toFixed(4) },
    { key: "kendallTau", label: "Kendall τ ↑", lowerBetter: false, format: (v) => v.toFixed(4) },
  ];

  const cols = isGlobal ? globalCols : strokeCols;

  const handleSort = (key) => {
    if (key === "name") return;
    if (sortKey === key) {
      setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const sortedData = [...data].sort((a, b) => {
    if (!sortKey) return 0;
    const col = cols.find((c) => c.key === sortKey);
    if (!col) return 0;
    const aVal = a[sortKey] ?? 0;
    const bVal = b[sortKey] ?? 0;
    const dir = col.lowerBetter ? (sortDir === "asc" ? -1 : 1) : (sortDir === "desc" ? -1 : 1);
    return (aVal - bVal) * dir;
  });

  return (
    <section id="results" className="py-24 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-10">
          <span className="badge bg-secondary/20 text-secondary border border-secondary/30 mb-4">
            Objective 4 · Interactive Results
          </span>
          <h2 className="section-heading">Benchmark Results</h2>
          <p className="section-subheading">
            Click column headers to sort. Toggle between global results (all datasets) and the
            primary Kaggle Stroke benchmark.
          </p>
        </div>

        {/* Dataset toggle */}
        <div className="flex items-center gap-2 mb-6">
          {[
            { id: "global", label: "🌐 Global Results (All Datasets)" },
            { id: "stroke", label: "🏥 Kaggle Stroke (19.5:1 imbalance)" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => { setDataset(tab.id); setSortKey(null); }}
              className={`px-4 py-2 rounded-xl text-sm font-semibold transition-colors ${
                dataset === tab.id
                  ? "bg-primary text-white"
                  : "bg-slate-800 text-slate-400 hover:text-white hover:bg-slate-700"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Category legend */}
        <div className="flex flex-wrap gap-3 mb-6">
          {Object.entries(categoryLabels).map(([key, label]) => (
            <span
              key={key}
              className="flex items-center gap-2 text-xs text-slate-400"
            >
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: categoryColors[key] }}
              />
              {label}
            </span>
          ))}
        </div>

        {/* Table */}
        <div className="card p-0 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  {cols.map((col) => (
                    <th
                      key={col.key}
                      onClick={() => handleSort(col.key)}
                      className={`px-4 py-3 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider whitespace-nowrap ${
                        col.key !== "name" ? "cursor-pointer hover:text-white select-none" : ""
                      }`}
                    >
                      <span className="flex items-center gap-1">
                        {col.label}
                        {col.key !== "name" && (
                          <span className="text-slate-600">
                            {sortKey === col.key ? SORT_ICONS[sortDir] : SORT_ICONS["none"]}
                          </span>
                        )}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedData.map((row, i) => {
                  const isShap = row.category === "shap";
                  return (
                    <tr
                      key={row.name}
                      className={`border-b border-slate-800 transition-colors ${
                        isShap ? "bg-highlight/5 hover:bg-highlight/10" : "hover:bg-slate-800/50"
                      }`}
                    >
                      {cols.map((col) => {
                        const val = row[col.key];
                        const best = col.key !== "name" ? getBestValue(data, col.key, col.lowerBetter) : null;
                        const isBest = best !== null && val === best;
                        return (
                          <td key={col.key} className="px-4 py-3.5 whitespace-nowrap">
                            {col.key === "name" ? (
                              <div className="flex items-center gap-2">
                                <span
                                  className="w-2 h-2 rounded-full flex-shrink-0"
                                  style={{ backgroundColor: categoryColors[row.category] }}
                                />
                                <span className={`font-medium ${isShap ? "text-highlight" : "text-slate-200"}`}>
                                  {val}
                                </span>
                                {isShap && (
                                  <span className="badge bg-highlight/10 text-highlight border-highlight/30 text-xs">Ours</span>
                                )}
                              </div>
                            ) : (
                              <span className={isBest ? "metric-best font-bold" : "text-slate-300"}>
                                {col.format(val)}
                                {isBest && <span className="text-highlight ml-1">★</span>}
                              </span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <p className="text-xs text-slate-600 mt-3">
          ★ = Best value in column &nbsp;·&nbsp; ↑ = higher is better &nbsp;·&nbsp; ↓ = lower is better &nbsp;·&nbsp; Results from 5-fold Stratified Nested Cross-Validation
        </p>
      </div>
    </section>
  );
}
