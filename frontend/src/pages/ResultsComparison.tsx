import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { ExperimentStatusResponse, ExperimentResultDetail } from "../types/api";
import { Badge } from "../components/common/Badge";
import { LoadingSpinner } from "../components/common/LoadingSpinner";
import { EmptyState } from "../components/common/EmptyState";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  BarChart3,
  Filter,
  ArrowUpDown,
  Download,
  HelpCircle,
  TrendingUp,
  Sparkles,
  ShieldCheck,
  AlertTriangle,
} from "lucide-react";

export const ResultsComparison: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentStatusResponse[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("synthetic");
  const [selectedClassifier, setSelectedClassifier] = useState<string>("random_forest");
  const [loading, setLoading] = useState<boolean>(true);
  const [metricDetails, setMetricDetails] = useState<Record<string, ExperimentResultDetail>>({});
  const [loadingMetrics, setLoadingMetrics] = useState<boolean>(false);
  const [sortField, setSortField] = useState<string>("pr_auc");
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  // Fetch all experiments list
  useEffect(() => {
    const fetchAll = async () => {
      try {
        setLoading(true);
        const list = await api.getExperiments({ status: "completed", limit: 300 });
        setExperiments(list);
      } catch (err) {
        console.error("Failed to load experiments:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchAll();
  }, []);

  // Filter experiments by dataset and classifier
  const filteredExps = experiments.filter(
    (e) =>
      e.dataset_id === selectedDataset &&
      e.classifier.toLowerCase().includes(selectedClassifier.toLowerCase())
  );

  // Fetch full metrics for filtered experiments
  useEffect(() => {
    if (filteredExps.length === 0) return;

    const fetchDetails = async () => {
      setLoadingMetrics(true);
      const newDetails: Record<string, ExperimentResultDetail> = { ...metricDetails };

      await Promise.all(
        filteredExps.map(async (exp) => {
          if (!newDetails[exp.id]) {
            try {
              const res = await api.getExperimentResults(exp.id);
              newDetails[exp.id] = res;
            } catch (err) {
              console.error(`Failed to fetch results for ${exp.id}:`, err);
            }
          }
        })
      );

      setMetricDetails(newDetails);
      setLoadingMetrics(false);
    };

    fetchDetails();
  }, [selectedDataset, selectedClassifier, experiments]);

  if (loading) {
    return <LoadingSpinner message="Loading benchmark results database..." />;
  }

  // Datasets and classifiers available in the experiments
  const availableDatasets = Array.from(new Set(experiments.map((e) => e.dataset_id)));
  const availableClassifiers = Array.from(new Set(experiments.map((e) => e.classifier)));

  // Combine rows with their fetched metrics
  const rows = filteredExps.map((exp) => {
    const detail = metricDetails[exp.id];
    const m = detail?.metrics;
    return {
      id: exp.id,
      oversampler: exp.oversampler,
      classifier: exp.classifier,
      dataset_name: exp.dataset_name,
      is_benchmark: exp.is_benchmark,
      pr_auc: m ? m.pr_auc_mean : 0,
      roc_auc: m ? m.roc_auc_mean : 0,
      recall: m ? m.minority_recall_mean : 0,
      precision: m ? m.minority_precision_mean : 0,
      f1: m ? m.minority_f1_mean : 0,
      bal_acc: m ? m.balanced_accuracy_mean : 0,
      g_mean: m ? m.g_mean_mean : 0,
      ece: m ? m.ece_mean : 1,
      brier: m ? m.brier_score_mean : 1,
      tau: m ? m.attribution_kendall_tau_mean : 0,
      wasserstein: m ? m.mean_wasserstein_distance : 0,
    };
  });

  // Sort rows
  rows.sort((a: any, b: any) => {
    const valA = a[sortField];
    const valB = b[sortField];
    return sortAsc ? valA - valB : valB - valA;
  });

  // Chart data for key metrics comparison
  const chartData = rows.map((r) => ({
    name: r.oversampler.replace(" (No Balancing)", "").replace(" (ROS)", ""),
    "PR-AUC": Number(r.pr_auc.toFixed(4)),
    "Minority F1": Number(r.f1.toFixed(4)),
    "Minority Recall": Number(r.recall.toFixed(4)),
    "ECE (↓)": Number(r.ece.toFixed(4)),
  }));

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
            <BarChart3 className="h-6 w-6 text-blue-600" />
            Empirical Results & Benchmark Comparison
          </h1>
          <p className="mt-1 text-xs text-slate-500">
            Head-to-head empirical comparison of SHAP-guided methods vs classical oversamplers under zero data leakage.
          </p>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-3">
          <div>
            <label className="block text-[10px] font-semibold uppercase text-slate-400 mb-0.5">
              Dataset
            </label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-800 focus:outline-none"
            >
              {availableDatasets.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-[10px] font-semibold uppercase text-slate-400 mb-0.5">
              Classifier
            </label>
            <select
              value={selectedClassifier}
              onChange={(e) => setSelectedClassifier(e.target.value)}
              className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-800 focus:outline-none"
            >
              {availableClassifiers.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Scientific Integrity Advisory */}
      <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-4 text-xs text-amber-900 flex items-start gap-3">
        <AlertTriangle className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-semibold">Research Evaluation Standards:</span>
          <p className="text-[11px] text-amber-800 leading-relaxed">
            All models displayed below are evaluated on identical, stratified cross-validation folds on the <strong>{selectedDataset}</strong> dataset.
            Do not compare results across disparate datasets or fold configurations.
            PR-AUC (Average Precision) represents the primary utility metric for severely imbalanced tabular outcomes.
          </p>
        </div>
      </div>

      {/* Interactive Comparison Bar Chart */}
      {rows.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between pb-4 border-b border-slate-100">
            <div>
              <h2 className="text-sm font-bold text-slate-800">
                Multi-Metric Method Comparison ({selectedDataset} | {selectedClassifier})
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Side-by-side performance across minority discrimination and probability calibration.
              </p>
            </div>
          </div>

          <div className="mt-6 h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#64748b" }} interval={0} angle={-15} textAnchor="end" />
                <YAxis tick={{ fontSize: 11, fill: "#64748b" }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#334155",
                    borderRadius: "8px",
                    color: "#fff",
                    fontSize: "12px",
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: "15px", fontSize: "11px" }} />
                <Bar dataKey="PR-AUC" fill="#2563eb" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Minority F1" fill="#10b981" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Minority Recall" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
                <Bar dataKey="ECE (↓)" fill="#f59e0b" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Detailed Benchmark Comparison Table */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Complete Multi-Dimensional Assessment Matrix
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Click any column header to sort. Colors highlight optimization direction.
            </p>
          </div>
        </div>

        {loadingMetrics ? (
          <LoadingSpinner message="Calculating and aggregating fold evaluation scores..." />
        ) : rows.length === 0 ? (
          <EmptyState
            title="No Results Matching Filter"
            description="Select a different dataset or classifier combination above."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse font-mono">
              <thead>
                <tr className="border-b border-slate-200 bg-slate-50 text-[11px] font-semibold text-slate-600">
                  <th className="py-2.5 px-3">Oversampling Method</th>
                  <th
                    className="py-2.5 px-3 cursor-pointer hover:bg-slate-100 transition"
                    onClick={() => handleSort("pr_auc")}
                  >
                    <div className="flex items-center gap-1 text-blue-700">
                      <span>PR-AUC (↑)</span>
                      <ArrowUpDown className="h-3 w-3" />
                    </div>
                  </th>
                  <th
                    className="py-2.5 px-3 cursor-pointer hover:bg-slate-100 transition"
                    onClick={() => handleSort("roc_auc")}
                  >
                    <div className="flex items-center gap-1">
                      <span>ROC-AUC (↑)</span>
                      <ArrowUpDown className="h-3 w-3" />
                    </div>
                  </th>
                  <th
                    className="py-2.5 px-3 cursor-pointer hover:bg-slate-100 transition"
                    onClick={() => handleSort("f1")}
                  >
                    <div className="flex items-center gap-1 text-emerald-700">
                      <span>Minority F1 (↑)</span>
                      <ArrowUpDown className="h-3 w-3" />
                    </div>
                  </th>
                  <th
                    className="py-2.5 px-3 cursor-pointer hover:bg-slate-100 transition"
                    onClick={() => handleSort("recall")}
                  >
                    <div className="flex items-center gap-1">
                      <span>Recall (↑)</span>
                      <ArrowUpDown className="h-3 w-3" />
                    </div>
                  </th>
                  <th
                    className="py-2.5 px-3 cursor-pointer hover:bg-slate-100 transition"
                    onClick={() => handleSort("ece")}
                  >
                    <div className="flex items-center gap-1 text-amber-700">
                      <span>ECE (↓)</span>
                      <ArrowUpDown className="h-3 w-3" />
                    </div>
                  </th>
                  <th
                    className="py-2.5 px-3 cursor-pointer hover:bg-slate-100 transition"
                    onClick={() => handleSort("tau")}
                  >
                    <div className="flex items-center gap-1 text-purple-700">
                      <span>Tau (↑)</span>
                      <ArrowUpDown className="h-3 w-3" />
                    </div>
                  </th>
                  <th
                    className="py-2.5 px-3 cursor-pointer hover:bg-slate-100 transition"
                    onClick={() => handleSort("wasserstein")}
                  >
                    <div className="flex items-center gap-1">
                      <span>Wasserstein (↓)</span>
                      <ArrowUpDown className="h-3 w-3" />
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {rows.map((r) => {
                  const isShap = r.oversampler.includes("SHAP");
                  return (
                    <tr
                      key={r.id}
                      className={`hover:bg-slate-50 transition ${
                        isShap ? "bg-blue-50/30 font-medium" : ""
                      }`}
                    >
                      <td className="py-2.5 px-3 font-sans font-semibold text-slate-900 flex items-center gap-1.5">
                        {r.oversampler}
                        {isShap && <Badge variant="default" size="sm">Proposed</Badge>}
                      </td>
                      <td className="py-2.5 px-3 font-bold text-blue-700">
                        {r.pr_auc.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3">{r.roc_auc.toFixed(4)}</td>
                      <td className="py-2.5 px-3 font-semibold text-emerald-700">
                        {r.f1.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3">{r.recall.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-amber-700">
                        {r.ece.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 text-purple-700">
                        {r.tau.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3">{r.wasserstein.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
