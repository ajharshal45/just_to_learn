import React, { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../services/api";
import { DatasetMetadata, ExperimentConfigRequest } from "../types/api";
import { Badge } from "../components/common/Badge";
import { LoadingSpinner } from "../components/common/LoadingSpinner";
import {
  FlaskConical,
  Play,
  ShieldCheck,
  HelpCircle,
  Sparkles,
  Layers,
  Settings2,
  Info,
  CheckCircle2,
  Sliders,
  AlertCircle,
} from "lucide-react";

export const ExperimentLab: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Form State
  const [datasetId, setDatasetId] = useState<string>("synthetic");
  const [oversampler, setOversampler] = useState<string>("shap_weighted");
  const [classifier, setClassifier] = useState<string>("random_forest");
  const [kNeighbors, setKNeighbors] = useState<number>(5);
  const [samplingStrategy, setSamplingStrategy] = useState<string>("auto");
  const [randomState, setRandomState] = useState<number>(42);
  const [nSplits, setNSplits] = useState<number>(5);
  const [cumulativeThreshold, setCumulativeThreshold] = useState<number>(0.85);
  const [scaleFeatures, setScaleFeatures] = useState<boolean>(true);

  useEffect(() => {
    const init = async () => {
      try {
        setLoading(true);
        const list = await api.getDatasets();
        setDatasets(list);

        const paramDs = searchParams.get("dataset_id");
        if (paramDs && list.some((d) => d.id === paramDs)) {
          setDatasetId(paramDs);
        } else if (list.length > 0) {
          setDatasetId(list[0].id);
        }
      } catch (err: any) {
        setError(err.message || "Failed to load datasets.");
      } finally {
        setLoading(false);
      }
    };
    init();
  }, [searchParams]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);

    const config: ExperimentConfigRequest = {
      dataset_id: datasetId,
      oversampler,
      classifier,
      k_neighbors: Number(kNeighbors),
      sampling_strategy: samplingStrategy === "auto" ? "auto" : parseFloat(samplingStrategy),
      random_state: Number(randomState),
      n_splits: Number(nSplits),
      n_repeats: 1,
      scale_features: scaleFeatures,
      cumulative_threshold: oversampler === "shap_filtered" ? Number(cumulativeThreshold) : null,
    };

    try {
      const exp = await api.createExperiment(config);
      navigate(`/history?highlight=${exp.id}`);
    } catch (err: any) {
      setError(err.message || "Failed to launch experiment.");
      setSubmitting(false);
    }
  };

  if (loading) {
    return <LoadingSpinner message="Preparing experiment lab..." />;
  }

  const selectedDataset = datasets.find((d) => d.id === datasetId);

  return (
    <div className="space-y-8 max-w-6xl">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
          <FlaskConical className="h-6 w-6 text-blue-600" />
          Experiment Configuration Lab
        </h1>
        <p className="mt-1 text-xs text-slate-500 max-w-3xl">
          Configure an empirical benchmark run under strict Zero-Data-Leakage nested cross-validation.
          Attribution weights and oversampling are computed strictly inside the inner training partition of each fold.
        </p>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-800 flex items-start gap-2">
          <AlertCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold">Execution Error:</span> {error}
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Core Parameters */}
          <div className="lg:col-span-2 space-y-6">
            {/* 1. Dataset Selection */}
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
              <label className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 text-[11px] font-bold text-blue-700">1</span>
                Dataset Selection
              </label>

              <select
                value={datasetId}
                onChange={(e) => setDatasetId(e.target.value)}
                className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-800 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.domain}) — {d.n_samples} samples | Imbalance: {d.imbalance_ratio}
                  </option>
                ))}
              </select>

              {selectedDataset && (
                <div className="rounded-lg bg-slate-50 border border-slate-200 p-3 text-xs text-slate-600 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-slate-700">{selectedDataset.name}</span>
                    <Badge variant={selectedDataset.is_custom ? "purple" : "neutral"} size="sm">
                      {selectedDataset.is_custom ? "Custom Upload" : "Standard Benchmark"}
                    </Badge>
                  </div>
                  <p className="text-[11px] text-slate-500">
                    {selectedDataset.description || "Tabular classification dataset."}
                  </p>
                  <div className="flex gap-4 pt-1 font-mono text-[11px] text-slate-500">
                    <span>N = {selectedDataset.n_samples}</span>
                    <span>P = {selectedDataset.n_features} features</span>
                    <span className="text-rose-600 font-semibold">IR: {selectedDataset.imbalance_ratio}</span>
                  </div>
                </div>
              )}
            </div>

            {/* 2. Oversampling Method Selection */}
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
              <label className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 text-[11px] font-bold text-blue-700">2</span>
                Oversampling Method
              </label>

              {/* Proposed Methods */}
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-blue-700 flex items-center gap-1 mb-2">
                  <Sparkles className="h-3.5 w-3.5" /> Proposed Explainability-Aware Methods
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <label
                    className={`relative flex flex-col rounded-lg border p-3.5 cursor-pointer transition-all ${
                      oversampler === "shap_weighted"
                        ? "border-blue-500 bg-blue-50/50 shadow-sm ring-1 ring-blue-500"
                        : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-bold text-slate-800">SHAP-Weighted SMOTE</span>
                      <input
                        type="radio"
                        name="oversampler"
                        value="shap_weighted"
                        checked={oversampler === "shap_weighted"}
                        onChange={() => setOversampler("shap_weighted")}
                        className="text-blue-600 focus:ring-blue-500"
                      />
                    </div>
                    <p className="text-[11px] text-slate-500 leading-snug">
                      Distance metric modification: scales feature differences by normalized Mean Absolute SHAP Values (MASV):
                      <span className="block mt-1 font-mono text-[10px] text-blue-700">
                        d_W(a,b) = √(∑ w_j · (a_j - b_j)²)
                      </span>
                    </p>
                  </label>

                  <label
                    className={`relative flex flex-col rounded-lg border p-3.5 cursor-pointer transition-all ${
                      oversampler === "shap_filtered"
                        ? "border-blue-500 bg-blue-50/50 shadow-sm ring-1 ring-blue-500"
                        : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-bold text-slate-800">SHAP-Filtered SMOTE</span>
                      <input
                        type="radio"
                        name="oversampler"
                        value="shap_filtered"
                        checked={oversampler === "shap_filtered"}
                        onChange={() => setOversampler("shap_filtered")}
                        className="text-blue-600 focus:ring-blue-500"
                      />
                    </div>
                    <p className="text-[11px] text-slate-500 leading-snug">
                      Subspace pre-filtering: discovers nearest neighbours strictly within top predictive feature subspace, then interpolates full feature space.
                    </p>
                  </label>
                </div>
              </div>

              {/* Classical Baselines */}
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2 block">
                  Classical & Hybrid Baselines
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {[
                    { id: "smote", name: "Standard SMOTE" },
                    { id: "borderline_smote", name: "Borderline-SMOTE" },
                    { id: "adasyn", name: "ADASYN" },
                    { id: "smote_enn", name: "SMOTE-ENN" },
                    { id: "smote_tomek", name: "SMOTE-Tomek" },
                    { id: "ros", name: "Random Oversampling" },
                    { id: "raw", name: "Raw (No Balancing)" },
                  ].map((m) => (
                    <label
                      key={m.id}
                      className={`flex items-center justify-between rounded-lg border px-3 py-2 text-xs cursor-pointer transition ${
                        oversampler === m.id
                          ? "border-blue-500 bg-blue-50 text-blue-900 font-semibold"
                          : "border-slate-200 hover:bg-slate-50 text-slate-700"
                      }`}
                    >
                      <span className="truncate">{m.name}</span>
                      <input
                        type="radio"
                        name="oversampler"
                        value={m.id}
                        checked={oversampler === m.id}
                        onChange={() => setOversampler(m.id)}
                        className="sr-only"
                      />
                    </label>
                  ))}
                </div>
              </div>
            </div>

            {/* 3. Classifier Selection */}
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
              <label className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-100 text-[11px] font-bold text-blue-700">3</span>
                Downstream Classifier
              </label>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {[
                  { id: "random_forest", name: "Random Forest", desc: "100 trees, max depth 6. Bagging ensemble." },
                  { id: "logistic_regression", name: "Logistic Regression", desc: "L2 regularized linear boundary (liblinear)." },
                  { id: "xgboost", name: "XGBoost", desc: "Gradient boosting on trees (100 estimators, lr 0.08)." },
                ].map((c) => (
                  <label
                    key={c.id}
                    className={`flex flex-col rounded-lg border p-3 cursor-pointer transition ${
                      classifier === c.id
                        ? "border-blue-500 bg-blue-50/50 shadow-sm ring-1 ring-blue-500"
                        : "border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-800">{c.name}</span>
                      <input
                        type="radio"
                        name="classifier"
                        value={c.id}
                        checked={classifier === c.id}
                        onChange={() => setClassifier(c.id)}
                        className="text-blue-600 focus:ring-blue-500"
                      />
                    </div>
                    <p className="mt-1 text-[11px] text-slate-500 leading-snug">{c.desc}</p>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Hyperparameters & Pre-Flight Review */}
          <div className="space-y-6">
            {/* Hyperparameters Card */}
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                <Sliders className="h-4 w-4 text-slate-400" />
                Hyperparameters
              </h3>

              {/* k_neighbors */}
              <div>
                <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                  <span>k-Nearest Neighbours</span>
                  <span className="font-mono font-bold text-blue-600">{kNeighbors}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="15"
                  value={kNeighbors}
                  onChange={(e) => setKNeighbors(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
              </div>

              {/* sampling_strategy */}
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Sampling Strategy
                </label>
                <select
                  value={samplingStrategy}
                  onChange={(e) => setSamplingStrategy(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-800"
                >
                  <option value="auto">auto (1:1 balanced ratio)</option>
                  <option value="0.5">0.5 (1:2 ratio)</option>
                  <option value="0.75">0.75 (3:4 ratio)</option>
                  <option value="1.0">1.0 (Full parity)</option>
                </select>
              </div>

              {/* SHAP Cumulative Threshold (conditional) */}
              {oversampler === "shap_filtered" && (
                <div className="rounded-lg bg-blue-50 p-3 border border-blue-200">
                  <div className="flex justify-between text-xs font-medium text-blue-900 mb-1">
                    <span>SHAP Cumulative Threshold</span>
                    <span className="font-mono font-bold">{cumulativeThreshold}</span>
                  </div>
                  <input
                    type="range"
                    min="0.50"
                    max="1.0"
                    step="0.05"
                    value={cumulativeThreshold}
                    onChange={(e) => setCumulativeThreshold(Number(e.target.value))}
                    className="w-full h-1.5 bg-blue-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <p className="mt-1 text-[10px] text-blue-700">
                    Retains top features whose cumulative MASV reaches {(cumulativeThreshold * 100).toFixed(0)}%.
                  </p>
                </div>
              )}

              {/* Cross-Validation Splits */}
              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-100">
                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    CV Splits (K)
                  </label>
                  <select
                    value={nSplits}
                    onChange={(e) => setNSplits(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-800"
                  >
                    <option value="3">3 Folds (Fast)</option>
                    <option value="5">5 Folds (Standard)</option>
                    <option value="10">10 Folds (Rigorous)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-xs font-medium text-slate-700 mb-1">
                    Random Seed
                  </label>
                  <input
                    type="number"
                    value={randomState}
                    onChange={(e) => setRandomState(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-800 font-mono"
                  />
                </div>
              </div>

              {/* In-fold Scaling */}
              <div className="pt-2 border-t border-slate-100">
                <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-700">
                  <input
                    type="checkbox"
                    checked={scaleFeatures}
                    onChange={(e) => setScaleFeatures(e.target.checked)}
                    className="rounded text-blue-600 focus:ring-blue-500"
                  />
                  <span>Standardize Features (StandardScaler)</span>
                </label>
                <p className="text-[10px] text-slate-400 ml-5 mt-0.5">
                  Fit strictly on inner training folds; test fold transformed independently.
                </p>
              </div>
            </div>

            {/* Pre-flight Review & Launch */}
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                Zero-Leakage Safeguard Check
              </h3>

              <div className="space-y-1.5 text-xs text-slate-600">
                <div className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                  <span>Outer Stratified Split applied first</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                  <span>Attribution tree fit on training fold only</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                  <span>Zero synthetic samples in test partitions</span>
                </div>
              </div>

              <button
                type="submit"
                disabled={submitting}
                className="w-full flex items-center justify-center gap-2 rounded-lg bg-blue-600 py-2.5 px-4 text-xs font-semibold text-white shadow-sm hover:bg-blue-700 disabled:opacity-50 transition"
              >
                {submitting ? (
                  <>
                    <span className="h-3.5 w-3.5 rounded-full border-2 border-white border-t-transparent animate-spin"></span>
                    <span>Submitting Experiment...</span>
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-current" />
                    <span>Run Experiment (Real ML Pipeline)</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
};
