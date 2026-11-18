import React, { useEffect, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from "recharts";
import { Badge } from "../components/common/Badge";
import { LoadingSpinner } from "../components/common/LoadingSpinner";
import {
  Sparkles,
  Sliders,
  Play,
  Layers,
  Info,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Eye,
} from "lucide-react";

interface ManifoldData {
  dataset_id: string;
  dataset_name: string;
  majority_points: { x: number; y: number }[];
  minority_points: { x: number; y: number }[];
  smote_synthetic: { x: number; y: number }[];
  shap_synthetic: { x: number; y: number }[];
  key_features: string[];
  noise_features: string[];
  explained_variance_ratio: number[];
  weights: { feature: string; weight: number }[];
}

export const SimulationPage: React.FC = () => {
  const [datasetId, setDatasetId] = useState<string>("kaggle_stroke");
  const [kNeighbors, setKNeighbors] = useState<number>(5);
  const [samplingRatio, setSamplingRatio] = useState<number>(1.0);
  const [viewMode, setViewMode] = useState<"side-by-side" | "overlay">("side-by-side");
  const [activeMethod, setActiveMethod] = useState<"all" | "shap" | "smote">("all");

  const [manifold, setManifold] = useState<ManifoldData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [simulating, setSimulating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSimulation = async () => {
    try {
      setSimulating(true);
      setError(null);
      const res = await fetch(
        `/api/simulation/manifold?dataset_id=${datasetId}&k_neighbors=${kNeighbors}&sampling_ratio=${samplingRatio}`
      );
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned ${res.status}: ${res.statusText}`);
      }
      const data = await res.json();
      setManifold(data);
    } catch (err: any) {
      console.error("Simulation failed:", err);
      setError(err?.message || "Failed to load geometric manifold simulation");
    } finally {
      setSimulating(false);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSimulation();
  }, [datasetId]);

  return (
    <div className="space-y-8 max-w-7xl">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-blue-600" />
            Interactive 2D Manifold Oversampling Simulator
          </h1>
          <p className="mt-1 text-xs text-slate-500 max-w-3xl">
            Live geometric simulation of synthetic sample interpolation. Observe how standard Euclidean SMOTE bleeds across Bayes decision boundaries due to unweighted noise features, while SHAP-Weighted SMOTE constrains interpolation strictly along the true predictive manifold.
          </p>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-1 rounded-lg border border-slate-200 bg-white p-1 text-xs">
          <button
            onClick={() => setViewMode("side-by-side")}
            className={`rounded-md px-3 py-1 font-medium transition ${
              viewMode === "side-by-side" ? "bg-blue-600 text-white font-semibold" : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Side-by-Side View
          </button>
          <button
            onClick={() => setViewMode("overlay")}
            className={`rounded-md px-3 py-1 font-medium transition ${
              viewMode === "overlay" ? "bg-blue-600 text-white font-semibold" : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Overlay Comparison
          </button>
        </div>
      </div>

      {/* Simulator Control Bar */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm grid grid-cols-1 sm:grid-cols-4 gap-4 items-end">
        <div>
          <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
            Dataset Under Simulation
          </label>
          <select
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
            className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-800 focus:outline-none"
          >
            <option value="kaggle_stroke">Kaggle Stroke Prediction (Clinical 19.5:1)</option>
            <option value="synthetic">Synthetic Imbalanced (Known Signal/Noise)</option>
            <option value="breast_cancer">Breast Cancer Wisconsin (Diagnostic)</option>
            <option value="credit_fraud">Credit Card Fraud (PCA Simulation)</option>
            <option value="nsl_kdd">NSL-KDD Network Intrusion</option>
          </select>
        </div>

        <div>
          <div className="flex justify-between text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
            <span>k-Nearest Neighbours</span>
            <span className="font-mono text-blue-600 font-bold">{kNeighbors}</span>
          </div>
          <input
            type="range"
            min="2"
            max="12"
            value={kNeighbors}
            onChange={(e) => setKNeighbors(Number(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>

        <div>
          <div className="flex justify-between text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-1">
            <span>Sampling Ratio</span>
            <span className="font-mono text-blue-600 font-bold">{(samplingRatio * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range"
            min="0.2"
            max="1.0"
            step="0.1"
            value={samplingRatio}
            onChange={(e) => setSamplingRatio(Number(e.target.value))}
            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
        </div>

        <div>
          <button
            onClick={fetchSimulation}
            disabled={simulating}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-navy-900 py-2 px-4 text-xs font-semibold text-white shadow-sm hover:bg-navy-800 disabled:opacity-50 transition"
          >
            {simulating ? (
              <>
                <span className="h-3.5 w-3.5 rounded-full border-2 border-white border-t-transparent animate-spin" />
                <span>Simulating...</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-current text-blue-400" />
                <span>Re-Simulate Geometry</span>
              </>
            )}
          </button>
        </div>
      </div>

      {loading ? (
        <LoadingSpinner message="Calculating 2D PCA projection of real manifold & oversampling interpolations..." />
      ) : error ? (
        <div className="rounded-xl border border-rose-200 bg-rose-50/60 p-6 text-center space-y-3">
          <AlertTriangle className="h-8 w-8 text-rose-500 mx-auto" />
          <h3 className="text-sm font-semibold text-rose-900">Simulation Error</h3>
          <p className="text-xs text-rose-700 max-w-md mx-auto">{error}</p>
          <button
            onClick={fetchSimulation}
            className="inline-flex items-center gap-1.5 rounded-lg bg-rose-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-rose-700 transition"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Retry Simulation
          </button>
        </div>
      ) : manifold ? (
        <div className="space-y-6">
          {/* Key Attribution & Geometric Insights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Top Predictive Dimensions
              </span>
              <div className="mt-1 flex flex-wrap gap-1">
                {manifold.key_features.map((f) => (
                  <Badge key={f} variant="default" size="sm">
                    {f}
                  </Badge>
                ))}
              </div>
              <span className="text-[10px] text-blue-700 mt-1 block">
                Dictate neighborhood in SHAP-Weighted SMOTE
              </span>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Uninformative / Noise Dimensions
              </span>
              <div className="mt-1 flex flex-wrap gap-1">
                {manifold.noise_features.map((f) => (
                  <Badge key={f} variant="neutral" size="sm">
                    {f}
                  </Badge>
                ))}
              </div>
              <span className="text-[10px] text-rose-600 mt-1 block">
                Distort Standard Euclidean SMOTE nearest neighbors
              </span>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                2D PCA Variance Explained
              </span>
              <div className="mt-1 text-sm font-bold font-mono text-slate-800">
                PC1: {(manifold.explained_variance_ratio[0] * 100).toFixed(1)}% | PC2: {(manifold.explained_variance_ratio[1] * 100).toFixed(1)}%
              </div>
              <span className="text-[10px] text-slate-400">
                Total 2D projection coverage: {((manifold.explained_variance_ratio[0] + manifold.explained_variance_ratio[1]) * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Manifold Visualizer */}
          {viewMode === "side-by-side" ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Standard Euclidean SMOTE Plot */}
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
                <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                  <div>
                    <h3 className="font-bold text-xs uppercase tracking-wider text-purple-700 flex items-center gap-1.5">
                      <AlertTriangle className="h-3.5 w-3.5 text-purple-600" />
                      Standard Euclidean SMOTE
                    </h3>
                    <p className="text-[11px] text-slate-500">
                      Isotropic distance d(a,b) ignores feature relevance; synthetic points bleed into majority territory.
                    </p>
                  </div>
                  <Badge variant="purple" size="sm">Baseline</Badge>
                </div>

                <div className="h-80 w-full" style={{ minHeight: 320 }}>
                  <ResponsiveContainer width="100%" height="100%" minHeight={320}>
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis type="number" dataKey="x" name="PC1" tick={{ fontSize: 10 }} />
                      <YAxis type="number" dataKey="y" name="PC2" tick={{ fontSize: 10 }} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                      <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "10px" }} />
                      <Scatter name="Real Majority" data={manifold.majority_points || []} fill="#94a3b8" shape="circle" />
                      <Scatter name="Real Minority" data={manifold.minority_points || []} fill="#dc2626" shape="circle" />
                      <Scatter name="Synthetic SMOTE" data={manifold.smote_synthetic || []} fill="#a855f7" shape="cross" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* SHAP-Weighted SMOTE Plot */}
              <div className="rounded-xl border border-blue-200 bg-white p-5 shadow-sm space-y-3">
                <div className="flex items-center justify-between border-b border-blue-100 pb-2">
                  <div>
                    <h3 className="font-bold text-xs uppercase tracking-wider text-blue-700 flex items-center gap-1.5">
                      <Sparkles className="h-3.5 w-3.5 text-blue-600" />
                      SHAP-Weighted MASV-SMOTE (Ours)
                    </h3>
                    <p className="text-[11px] text-slate-500">
                      Weighted Riemannian metric d_W(a,b) aligns interpolation along true predictive cluster boundaries.
                    </p>
                  </div>
                  <Badge variant="default" size="sm">Proposed</Badge>
                </div>

                <div className="h-80 w-full" style={{ minHeight: 320 }}>
                  <ResponsiveContainer width="100%" height="100%" minHeight={320}>
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis type="number" dataKey="x" name="PC1" tick={{ fontSize: 10 }} />
                      <YAxis type="number" dataKey="y" name="PC2" tick={{ fontSize: 10 }} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                      <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "10px" }} />
                      <Scatter name="Real Majority" data={manifold.majority_points || []} fill="#94a3b8" shape="circle" />
                      <Scatter name="Real Minority" data={manifold.minority_points || []} fill="#dc2626" shape="circle" />
                      <Scatter name="Synthetic SHAP-SMOTE" data={manifold.shap_synthetic || []} fill="#2563eb" shape="diamond" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          ) : (
            /* Overlay View */
            <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                <div>
                  <h3 className="font-bold text-xs uppercase tracking-wider text-slate-800">
                    Direct Geometric Overlay Comparison
                  </h3>
                  <p className="text-xs text-slate-500">
                    Comparing Standard SMOTE synthetic instances (purple cross) vs SHAP-Weighted synthetic instances (blue diamond).
                  </p>
                </div>
              </div>

              <div className="h-96 w-full" style={{ minHeight: 384 }}>
                <ResponsiveContainer width="100%" height="100%" minHeight={384}>
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis type="number" dataKey="x" name="PC1" tick={{ fontSize: 11 }} />
                    <YAxis type="number" dataKey="y" name="PC2" tick={{ fontSize: 11 }} />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                    <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "10px" }} />
                    <Scatter name="Real Majority" data={manifold.majority_points || []} fill="#cbd5e1" shape="circle" />
                    <Scatter name="Real Minority" data={manifold.minority_points || []} fill="#dc2626" shape="circle" />
                    <Scatter name="Standard SMOTE" data={manifold.smote_synthetic || []} fill="#a855f7" shape="cross" />
                    <Scatter name="SHAP-Weighted SMOTE" data={manifold.shap_synthetic || []} fill="#2563eb" shape="diamond" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Theoretical Proof Summary */}
          <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-5 text-xs text-slate-700 space-y-2">
            <span className="font-bold text-slate-900 block flex items-center gap-1.5">
              <Info className="h-4 w-4 text-blue-600" />
              Mathematical Derivation of Manifold Preservation:
            </span>
            <p className="text-[11px] text-slate-600 leading-relaxed">
              Standard Euclidean distance treats each of the feature dimensions uniformly.
              In high-dimensional tabular data, nearest neighbors identified by isotropic Euclidean distance
              are contaminated by irrelevant noise features, causing synthetic minority interpolations to bleed
              directly into majority class territory.
              By contrast, SHAP-Weighted SMOTE defines a feature-weighted Riemannian metric tensor
              W = diag(w₁, ..., wₚ) where each weight wⱼ is proportional to MASVⱼ (Mean Absolute SHAP Value).
              This confines the nearest-neighbour graph strictly along true predictive manifold dimensions,
              preventing synthetic samples from crossing Bayes decision boundaries.
            </p>
          </div>
        </div>
      ) : null}
    </div>
  );
};
