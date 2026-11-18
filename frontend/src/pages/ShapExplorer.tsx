import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../services/api";
import {
  ExperimentStatusResponse,
  SHAPExplanationResponse,
  SHAPFeatureAttribution,
} from "../types/api";
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
  Cell,
} from "recharts";
import {
  Sparkles,
  Info,
  CheckCircle2,
  XCircle,
  HelpCircle,
  TrendingUp,
  Sliders,
  Layers,
} from "lucide-react";

export const ShapExplorer: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedParamId = searchParams.get("experiment_id");

  const [experiments, setExperiments] = useState<ExperimentStatusResponse[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string>("");
  const [explanations, setExplanations] = useState<SHAPExplanationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [loadingExplanations, setLoadingExplanations] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [chartView, setChartView] = useState<"masv" | "weight">("masv");

  // Load completed experiments
  useEffect(() => {
    const fetchExps = async () => {
      try {
        setLoading(true);
        const list = await api.getExperiments({ status: "completed", limit: 100 });
        setExperiments(list);

        if (selectedParamId && list.some((e) => e.id === selectedParamId)) {
          setSelectedExpId(selectedParamId);
        } else if (list.length > 0) {
          // Prefer a shap experiment if available
          const preferred = list.find((e) => e.oversampler.includes("SHAP")) || list[0];
          setSelectedExpId(preferred.id);
        }
      } catch (err: any) {
        setError(err.message || "Failed to load experiments.");
      } finally {
        setLoading(false);
      }
    };
    fetchExps();
  }, [selectedParamId]);

  // Load SHAP explanations when selectedExpId changes
  useEffect(() => {
    if (!selectedExpId) return;

    const fetchShap = async () => {
      try {
        setLoadingExplanations(true);
        setError(null);
        const res = await api.getExperimentExplanations(selectedExpId);
        setExplanations(res);
      } catch (err: any) {
        setError(err.message || "Could not retrieve SHAP explanations.");
        setExplanations(null);
      } finally {
        setLoadingExplanations(false);
      }
    };
    fetchShap();
  }, [selectedExpId]);

  const handleSelectExperiment = (id: string) => {
    setSelectedExpId(id);
    setSearchParams({ experiment_id: id });
  };

  if (loading) {
    return <LoadingSpinner message="Loading explainability models..." />;
  }

  // Prepare chart data (top 15 features for clarity)
  const chartData = (explanations?.features || []).slice(0, 15).map((f) => ({
    name: f.feature_name,
    raw_masv: f.masv,
    masv: Number(f.masv.toFixed(4)),
    weight: Number((f.normalized_weight * 100).toFixed(2)),
    is_selected: f.is_selected_in_subspace,
  }));

  return (
    <div className="space-y-8 max-w-6xl">
      {/* Title & Research Context */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-blue-600" />
            SHAP Explainability Explorer
          </h1>
          <p className="mt-1 text-xs text-slate-500 max-w-2xl">
            Inspect upstream Mean Absolute SHAP Values (MASV) and normalized distance weights calculated strictly on inner training partitions.
          </p>
        </div>

        {/* Experiment Selector Dropdown */}
        <div className="w-full sm:w-80">
          <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
            Active Experiment
          </label>
          <select
            value={selectedExpId}
            onChange={(e) => handleSelectExperiment(e.target.value)}
            className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-800 shadow-sm focus:border-blue-500 focus:outline-none"
          >
            {experiments.map((e) => (
              <option key={e.id} value={e.id}>
                {e.oversampler} on {e.dataset_name} ({e.classifier})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Educational Notice: Global MASV vs Local Predictions */}
      <div className="rounded-xl border border-blue-200 bg-blue-50/60 p-4 text-xs text-blue-900 flex items-start gap-3">
        <Info className="h-4 w-4 text-blue-600 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-semibold">Game-Theoretic Upstream Attribution Formulation:</span>
          <p className="text-[11px] text-blue-800 leading-relaxed">
            This module displays <strong>Global Mean Absolute SHAP Values</strong>:
            <span className="font-mono ml-1 font-bold">MASV_j = (1/N) ∑_i |φ_j(x_i)|</span>.
            Unlike heuristic Gini impurity, TreeSHAP accounts for feature interactions and non-linear boundaries.
            Normalized weights <span className="font-mono">w_j = MASV_j / ∑ MASV</span> modulate the Riemannian metric tensor for nearest-neighbour search.
          </p>
        </div>
      </div>

      {loadingExplanations ? (
        <LoadingSpinner message="Computing SHAP feature attributions on training fold..." />
      ) : error ? (
        <EmptyState
          title="Attribution Data Not Available"
          description={error}
        />
      ) : explanations && explanations.features.length > 0 ? (
        <div className="space-y-6">
          {/* Summary Banner */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Oversampling Method
              </span>
              <div className="mt-1 text-sm font-bold text-slate-800 truncate">
                {explanations.method}
              </div>
              <span className="text-[10px] text-slate-400">{explanations.dataset_name}</span>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Evaluated Dimensions
              </span>
              <div className="mt-1 text-xl font-bold font-mono text-slate-800">
                {explanations.total_features_count} Features
              </div>
              <span className="text-[10px] text-slate-400">Total feature space</span>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Subspace Retained
              </span>
              <div className="mt-1 text-xl font-bold font-mono text-blue-600">
                {explanations.retained_features_count || explanations.total_features_count} Features
              </div>
              <span className="text-[10px] text-slate-400">
                Cutoff threshold: {explanations.cumulative_threshold ? `${(explanations.cumulative_threshold * 100).toFixed(0)}%` : "N/A"}
              </span>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Top Attribution
              </span>
              <div className="mt-1 text-sm font-bold text-emerald-600 truncate font-mono">
                {explanations.features[0]?.feature_name}
              </div>
              <span className="text-[10px] text-slate-400 font-mono">
                Weight: {(explanations.features[0]?.normalized_weight * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Horizontal Bar Chart */}
          <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-slate-100">
              <div>
                <h3 className="text-sm font-bold text-slate-800">
                  Ranked Feature Attribution Profile
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Top 15 predictive dimensions ranked by Mean Absolute SHAP Values (MASV).
                </p>
              </div>

              <div className="flex rounded-lg border border-slate-200 bg-slate-50 p-1 text-xs">
                <button
                  onClick={() => setChartView("masv")}
                  className={`rounded-md px-2.5 py-1 font-medium transition ${
                    chartView === "masv" ? "bg-white text-blue-700 shadow-sm font-semibold" : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  Raw MASV Value
                </button>
                <button
                  onClick={() => setChartView("weight")}
                  className={`rounded-md px-2.5 py-1 font-medium transition ${
                    chartView === "weight" ? "bg-white text-blue-700 shadow-sm font-semibold" : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  Normalized Weight (%)
                </button>
              </div>
            </div>

            <div className="mt-6 h-96 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} />
                  <YAxis
                    dataKey="name"
                    type="category"
                    tick={{ fontSize: 11, fill: "#334155", fontWeight: 500 }}
                  />
                  <Tooltip
                    formatter={(val: any) => [
                      chartView === "masv" ? val : `${val}%`,
                      chartView === "masv" ? "Mean Absolute SHAP Value" : "Normalized Weight",
                    ]}
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      borderRadius: "8px",
                      color: "#fff",
                      fontSize: "12px",
                    }}
                  />
                  <Bar
                    dataKey={chartView === "masv" ? "masv" : "weight"}
                    radius={[0, 4, 4, 0]}
                  >
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.is_selected ? "#2563eb" : "#94a3b8"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-3 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-100 pt-3">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5 text-slate-700">
                  <span className="h-2.5 w-2.5 rounded bg-blue-600" /> Retained in Subspace
                </span>
                <span className="flex items-center gap-1.5 text-slate-700">
                  <span className="h-2.5 w-2.5 rounded bg-slate-400" /> Pruned / Downweighted
                </span>
              </div>
              <span>Extracted using TreeExplainer on inner training fold.</span>
            </div>
          </div>

          {/* Detailed Feature Ranking Table */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-3">
              Comprehensive Feature Ranking & Weighting Matrix
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse font-mono">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50 text-[11px] text-slate-500 font-semibold">
                    <th className="py-2.5 px-3 w-16">Rank</th>
                    <th className="py-2.5 px-3">Feature Name</th>
                    <th className="py-2.5 px-3">Raw MASV</th>
                    <th className="py-2.5 px-3">Normalized Weight (w_j)</th>
                    <th className="py-2.5 px-3">Weight Share</th>
                    <th className="py-2.5 px-3 text-right">Subspace Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  {explanations.features.map((f) => (
                    <tr key={f.rank} className="hover:bg-slate-50 transition">
                      <td className="py-2 px-3 font-bold text-slate-500">#{f.rank}</td>
                      <td className="py-2 px-3 font-semibold text-slate-900">{f.feature_name}</td>
                      <td className="py-2 px-3 font-mono">{f.masv.toFixed(6)}</td>
                      <td className="py-2 px-3 font-mono font-bold text-blue-700">
                        {f.normalized_weight.toFixed(6)}
                      </td>
                      <td className="py-2 px-3">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-16 bg-slate-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-blue-600 rounded-full"
                              style={{ width: `${Math.min(100, f.normalized_weight * 100 * 2.5)}%` }}
                            />
                          </div>
                          <span className="text-[11px]">{(f.normalized_weight * 100).toFixed(2)}%</span>
                        </div>
                      </td>
                      <td className="py-2 px-3 text-right">
                        {f.is_selected_in_subspace ? (
                          <Badge variant="success" size="sm">
                            <CheckCircle2 className="h-3 w-3 mr-1" /> Retained
                          </Badge>
                        ) : (
                          <Badge variant="neutral" size="sm">
                            Pruned
                          </Badge>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : (
        <EmptyState
          title="No Explanations Found"
          description="Select an experiment run to inspect its SHAP attributions."
        />
      )}
    </div>
  );
};
