import React, { useEffect, useState } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { api } from "../services/api";
import { ExperimentStatusResponse, ExperimentResultDetail } from "../types/api";
import { Badge } from "../components/common/Badge";
import { LoadingSpinner } from "../components/common/LoadingSpinner";
import { EmptyState } from "../components/common/EmptyState";
import {
  History,
  Filter,
  Sparkles,
  Download,
  AlertCircle,
  Clock,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileText,
  RotateCcw,
  BarChart2,
  ExternalLink,
} from "lucide-react";

export const ExperimentHistory: React.FC = () => {
  const [searchParams] = useSearchParams();
  const highlightId = searchParams.get("highlight");

  const [experiments, setExperiments] = useState<ExperimentStatusResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterType, setFilterType] = useState<string>("all"); // "all", "live", "benchmark"
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [expandedId, setExpandedId] = useState<string | null>(highlightId);
  const [detailCache, setDetailCache] = useState<Record<string, ExperimentResultDetail>>({});
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);

  const fetchList = async () => {
    try {
      const isBenchParam = filterType === "benchmark" ? true : filterType === "live" ? false : undefined;
      const statusParam = statusFilter !== "all" ? statusFilter : undefined;
      const list = await api.getExperiments({
        is_benchmark: isBenchParam,
        status: statusParam,
        limit: 300,
      });
      setExperiments(list);
    } catch (err: any) {
      console.error("Failed to fetch experiments:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchList();
    // Poll active experiments every 3 seconds if any are running
    const interval = setInterval(() => {
      api.getExperiments({ status: "running" }).then((running) => {
        if (running.length > 0) {
          fetchList();
        }
      });
    }, 3000);
    return () => clearInterval(interval);
  }, [filterType, statusFilter]);

  const toggleExpand = async (id: string) => {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(id);
    if (!detailCache[id]) {
      try {
        setLoadingDetail(true);
        const detail = await api.getExperimentResults(id);
        setDetailCache((prev) => ({ ...prev, [id]: detail }));
      } catch (err) {
        console.error("Failed to fetch result detail:", err);
      } finally {
        setLoadingDetail(false);
      }
    }
  };

  if (loading) {
    return <LoadingSpinner message="Loading experiment runs from persistent database..." />;
  }

  return (
    <div className="space-y-6">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
            <History className="h-6 w-6 text-blue-600" />
            Experiment Runs & History
          </h1>
          <p className="mt-1 text-xs text-slate-500">
            Real-time execution monitor, persistent runs archive, and detailed evaluation inspector.
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Benchmark vs Live */}
          <div className="flex rounded-lg border border-slate-200 bg-white p-1 text-xs">
            <button
              onClick={() => setFilterType("all")}
              className={`rounded-md px-2.5 py-1 font-medium transition ${
                filterType === "all" ? "bg-blue-600 text-white font-semibold" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              All ({experiments.length})
            </button>
            <button
              onClick={() => setFilterType("live")}
              className={`rounded-md px-2.5 py-1 font-medium transition ${
                filterType === "live" ? "bg-blue-600 text-white font-semibold" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Live Runs
            </button>
            <button
              onClick={() => setFilterType("benchmark")}
              className={`rounded-md px-2.5 py-1 font-medium transition ${
                filterType === "benchmark" ? "bg-blue-600 text-white font-semibold" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Benchmarks (108)
            </button>
          </div>

          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none"
          >
            <option value="all">All Statuses</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="failed">Failed</option>
            <option value="pending">Pending</option>
          </select>
        </div>
      </div>

      {/* Experiments List Table */}
      {experiments.length === 0 ? (
        <EmptyState
          title="No experiments found"
          description="Try adjusting your filters or launch a new run in the Experiment Lab."
        />
      ) : (
        <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50/80 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  <th className="py-3 px-4">Experiment ID & Date</th>
                  <th className="py-3 px-4">Dataset</th>
                  <th className="py-3 px-4">Oversampling Method</th>
                  <th className="py-3 px-4">Classifier</th>
                  <th className="py-3 px-4">Status & Progress</th>
                  <th className="py-3 px-4">Wall-Clock</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {experiments.map((exp) => {
                  const isHighlighted = exp.id === highlightId;
                  const isExpanded = exp.id === expandedId;
                  const detail = detailCache[exp.id];

                  return (
                    <React.Fragment key={exp.id}>
                      <tr
                        className={`transition hover:bg-slate-50/60 ${
                          isHighlighted ? "bg-blue-50/50" : ""
                        }`}
                      >
                        <td className="py-3 px-4">
                          <div className="font-mono font-medium text-slate-900 flex items-center gap-1.5">
                            {exp.id}
                            {exp.is_benchmark && (
                              <Badge variant="neutral" size="sm">Benchmark</Badge>
                            )}
                          </div>
                          <div className="text-[10px] text-slate-400 mt-0.5">
                            {new Date(exp.created_at).toLocaleString()}
                          </div>
                        </td>

                        <td className="py-3 px-4 font-medium text-slate-800">
                          {exp.dataset_name}
                        </td>

                        <td className="py-3 px-4 font-semibold text-slate-800">
                          {exp.oversampler}
                        </td>

                        <td className="py-3 px-4 font-mono text-slate-600">
                          {exp.classifier}
                        </td>

                        <td className="py-3 px-4">
                          {exp.status === "completed" && (
                            <Badge variant="success">Completed</Badge>
                          )}
                          {exp.status === "running" && (
                            <div className="space-y-1 w-36">
                              <div className="flex items-center justify-between text-[10px] font-medium text-amber-700">
                                <span className="flex items-center gap-1">
                                  <span className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-ping" />
                                  Running
                                </span>
                                <span>{exp.progress_percent}%</span>
                              </div>
                              <div className="h-1.5 w-full bg-amber-100 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-amber-500 transition-all duration-300"
                                  style={{ width: `${exp.progress_percent}%` }}
                                />
                              </div>
                              <div className="text-[9px] text-slate-400 truncate">
                                {exp.current_step}
                              </div>
                            </div>
                          )}
                          {exp.status === "failed" && (
                            <Badge variant="danger">Failed</Badge>
                          )}
                          {exp.status === "pending" && (
                            <Badge variant="neutral">Pending</Badge>
                          )}
                        </td>

                        <td className="py-3 px-4 font-mono text-slate-500 text-[11px]">
                          {exp.wall_clock_seconds ? `${exp.wall_clock_seconds.toFixed(2)}s` : "—"}
                        </td>

                        <td className="py-3 px-4 text-right space-x-2 whitespace-nowrap">
                          {exp.status === "completed" && (
                            <>
                              <Link
                                to={`/shap?experiment_id=${exp.id}`}
                                className="text-blue-600 hover:text-blue-800 font-medium inline-flex items-center gap-1"
                              >
                                <Sparkles className="h-3 w-3" /> SHAP
                              </Link>
                              <a
                                href={api.getExportUrl(exp.id, "json")}
                                download
                                className="text-slate-500 hover:text-slate-800 inline-flex items-center gap-0.5"
                                title="Download JSON"
                              >
                                <Download className="h-3 w-3" />
                              </a>
                            </>
                          )}
                          <button
                            onClick={() => toggleExpand(exp.id)}
                            className="rounded p-1 text-slate-400 hover:text-slate-700"
                          >
                            {isExpanded ? (
                              <ChevronUp className="h-4 w-4" />
                            ) : (
                              <ChevronDown className="h-4 w-4" />
                            )}
                          </button>
                        </td>
                      </tr>

                      {/* Expanded Details Row */}
                      {isExpanded && (
                        <tr className="bg-slate-50/80">
                          <td colSpan={7} className="p-4 border-y border-slate-200">
                            {exp.status === "failed" ? (
                              <div className="rounded-lg bg-rose-50 border border-rose-200 p-4 text-xs text-rose-800">
                                <span className="font-semibold block mb-1">Execution Failure Traceback:</span>
                                <pre className="font-mono text-[11px] whitespace-pre-wrap bg-white/70 p-3 rounded border border-rose-100">
                                  {exp.error_message || "Unknown computation error occurred."}
                                </pre>
                              </div>
                            ) : loadingDetail && !detail ? (
                              <div className="text-center py-4 text-xs text-slate-500">
                                Loading metric breakdown...
                              </div>
                            ) : detail ? (
                              <div className="space-y-3 text-xs">
                                <div className="flex items-center justify-between">
                                  <span className="font-bold uppercase tracking-wider text-slate-500 text-[10px]">
                                    Recorded Multi-Dimensional Metrics (5-Fold Stratified Nested CV)
                                  </span>
                                  <div className="flex gap-2">
                                    <a
                                      href={api.getExportUrl(exp.id, "csv")}
                                      download
                                      className="rounded bg-white border border-slate-200 px-2 py-1 text-[11px] text-slate-700 hover:bg-slate-50"
                                    >
                                      Export CSV
                                    </a>
                                    <a
                                      href={api.getExportUrl(exp.id, "markdown")}
                                      download
                                      className="rounded bg-white border border-slate-200 px-2 py-1 text-[11px] text-slate-700 hover:bg-slate-50"
                                    >
                                      Export Report (.md)
                                    </a>
                                  </div>
                                </div>

                                <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
                                  <div className="rounded bg-white p-2.5 border border-slate-200">
                                    <span className="text-[10px] text-slate-400 uppercase">PR-AUC (Avg Prec)</span>
                                    <div className="font-mono font-bold text-sm text-blue-700 mt-0.5">
                                      {detail.metrics.pr_auc_mean.toFixed(4)}
                                    </div>
                                    <span className="text-[9px] text-slate-400">±{detail.metrics.pr_auc_std.toFixed(4)}</span>
                                  </div>

                                  <div className="rounded bg-white p-2.5 border border-slate-200">
                                    <span className="text-[10px] text-slate-400 uppercase">ROC-AUC</span>
                                    <div className="font-mono font-bold text-sm text-slate-800 mt-0.5">
                                      {detail.metrics.roc_auc_mean.toFixed(4)}
                                    </div>
                                    <span className="text-[9px] text-slate-400">±{detail.metrics.roc_auc_std.toFixed(4)}</span>
                                  </div>

                                  <div className="rounded bg-white p-2.5 border border-slate-200">
                                    <span className="text-[10px] text-slate-400 uppercase">Minority F1</span>
                                    <div className="font-mono font-bold text-sm text-slate-800 mt-0.5">
                                      {detail.metrics.minority_f1_mean.toFixed(4)}
                                    </div>
                                    <span className="text-[9px] text-slate-400">±{detail.metrics.minority_f1_std.toFixed(4)}</span>
                                  </div>

                                  <div className="rounded bg-white p-2.5 border border-slate-200">
                                    <span className="text-[10px] text-slate-400 uppercase">Minority Recall</span>
                                    <div className="font-mono font-bold text-sm text-slate-800 mt-0.5">
                                      {detail.metrics.minority_recall_mean.toFixed(4)}
                                    </div>
                                    <span className="text-[9px] text-slate-400">±{detail.metrics.minority_recall_std.toFixed(4)}</span>
                                  </div>

                                  <div className="rounded bg-white p-2.5 border border-slate-200">
                                    <span className="text-[10px] text-slate-400 uppercase">Calibration ECE (↓)</span>
                                    <div className="font-mono font-bold text-sm text-emerald-700 mt-0.5">
                                      {detail.metrics.ece_mean.toFixed(4)}
                                    </div>
                                    <span className="text-[9px] text-slate-400">Lower is better</span>
                                  </div>

                                  <div className="rounded bg-white p-2.5 border border-slate-200">
                                    <span className="text-[10px] text-slate-400 uppercase">Kendall's Tau</span>
                                    <div className="font-mono font-bold text-sm text-purple-700 mt-0.5">
                                      {detail.metrics.attribution_kendall_tau_mean.toFixed(4)}
                                    </div>
                                    <span className="text-[9px] text-slate-400">Attribution Stability</span>
                                  </div>
                                </div>
                              </div>
                            ) : null}
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
