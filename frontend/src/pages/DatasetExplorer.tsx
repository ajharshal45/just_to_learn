import React, { useEffect, useState, useRef } from "react";
import { api } from "../services/api";
import {
  DatasetMetadata,
  DatasetSummary,
  DatasetPreviewResponse,
} from "../types/api";
import { Badge } from "../components/common/Badge";
import { LoadingSpinner } from "../components/common/LoadingSpinner";
import { EmptyState } from "../components/common/EmptyState";
import {
  Upload,
  Database,
  FileSpreadsheet,
  AlertCircle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  Columns,
  Layers,
  ArrowRight,
} from "lucide-react";
import { Link } from "react-router-dom";

export const DatasetExplorer: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("synthetic");
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [preview, setPreview] = useState<DatasetPreviewResponse | null>(null);
  const [page, setPage] = useState<number>(1);
  const [loading, setLoading] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [targetColumnHint, setTargetColumnHint] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDatasetList = async () => {
    try {
      const list = await api.getDatasets();
      setDatasets(list);
      if (list.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(list[0].id);
      }
    } catch (err: any) {
      console.error("Failed to load datasets:", err);
    }
  };

  const loadDatasetDetails = async (id: string, pg = 1) => {
    try {
      setLoading(true);
      const [sum, prev] = await Promise.all([
        api.getDatasetSummary(id),
        api.getDatasetPreview(id, pg, 15),
      ]);
      setSummary(sum);
      setPreview(prev);
      setPage(pg);
    } catch (err: any) {
      console.error("Failed to load dataset details:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasetList();
  }, []);

  useEffect(() => {
    if (selectedDatasetId) {
      loadDatasetDetails(selectedDatasetId, 1);
    }
  }, [selectedDatasetId]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const meta = await api.uploadDataset(file, targetColumnHint || undefined);
      setUploadSuccess(`Successfully uploaded '${file.name}'. Found ${meta.n_samples} rows and ${meta.n_features} features.`);
      await loadDatasetList();
      setSelectedDatasetId(meta.id);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err: any) {
      setUploadError(err.message || "Failed to upload dataset.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            Dataset Explorer & Preprocessor
          </h1>
          <p className="mt-1 text-xs text-slate-500 max-w-2xl">
            Inspect benchmark distributions, upload proprietary CSVs, and examine safe preprocessing workflows before cross-validation.
          </p>
        </div>

        <Link
          to={`/lab?dataset_id=${selectedDatasetId}`}
          className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-3.5 py-2 text-xs font-medium text-white shadow-sm hover:bg-blue-700 transition"
        >
          <span>Use in Experiment Lab</span>
          <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>

      {/* Dataset Selection & Upload Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Dataset Selector List */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5 mb-3">
            <Database className="h-4 w-4 text-slate-400" />
            Select Dataset
          </h3>

          <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
            {datasets.map((d) => (
              <button
                key={d.id}
                onClick={() => setSelectedDatasetId(d.id)}
                className={`w-full text-left p-3 rounded-lg border text-xs transition-all ${
                  selectedDatasetId === d.id
                    ? "border-blue-500 bg-blue-50/50 shadow-sm"
                    : "border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-800">{d.name}</span>
                  {d.is_custom ? (
                    <Badge variant="purple" size="sm">Custom</Badge>
                  ) : (
                    <Badge variant="neutral" size="sm">Benchmark</Badge>
                  )}
                </div>
                <div className="mt-1 text-[11px] text-slate-500 flex items-center justify-between">
                  <span>{d.domain}</span>
                  <span className="font-mono text-slate-600">{d.n_samples} samples</span>
                </div>
                <div className="mt-1 text-[10px] text-rose-600 font-medium">
                  Imbalance Ratio: {d.imbalance_ratio}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Upload Custom CSV Form */}
        <div className="lg:col-span-2 rounded-xl border border-slate-200 bg-white p-5 shadow-sm flex flex-col justify-between">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5 mb-2">
              <Upload className="h-4 w-4 text-slate-400" />
              Upload Custom Tabular CSV
            </h3>
            <p className="text-xs text-slate-500 mb-4">
              Upload an imbalanced tabular dataset with a binary classification target. Safe parsing ensures no path exposure or arbitrary code execution.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Target Column Name (Optional Hint)
                </label>
                <input
                  type="text"
                  placeholder="e.g. stroke, target, label, fraud"
                  value={targetColumnHint}
                  onChange={(e) => setTargetColumnHint(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-800 placeholder-slate-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-700 mb-1">
                  Select CSV File
                </label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  disabled={uploading}
                  className="w-full text-xs text-slate-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
                />
              </div>
            </div>

            {uploading && (
              <div className="flex items-center gap-2 text-xs text-blue-600 mb-2">
                <span className="h-3 w-3 rounded-full border-2 border-blue-600 border-t-transparent animate-spin"></span>
                <span>Validating schema and target column classes...</span>
              </div>
            )}

            {uploadSuccess && (
              <div className="flex items-center gap-2 rounded-lg bg-emerald-50 border border-emerald-200 p-3 text-xs text-emerald-800 mb-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
                <span>{uploadSuccess}</span>
              </div>
            )}

            {uploadError && (
              <div className="flex items-start gap-2 rounded-lg bg-rose-50 border border-rose-200 p-3 text-xs text-rose-800 mb-2">
                <AlertCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold">Validation Error:</span> {uploadError}
                </div>
              </div>
            )}
          </div>

          <div className="rounded-lg bg-slate-50 border border-slate-200 p-3 text-[11px] text-slate-500 flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-emerald-600 shrink-0" />
            <span>Files are securely isolated in a controlled storage partition with UUID hashing.</span>
          </div>
        </div>
      </div>

      {/* Dataset Details & Preprocessing Blueprint */}
      {loading ? (
        <LoadingSpinner message="Inspecting dataset schema and class distributions..." />
      ) : summary ? (
        <div className="space-y-6">
          {/* Key Stat Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Total Samples
              </span>
              <div className="mt-1 text-2xl font-bold font-mono text-slate-900">
                {summary.n_samples.toLocaleString()}
              </div>
              <span className="text-[10px] text-slate-400">Rows in dataset</span>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Feature Space
              </span>
              <div className="mt-1 text-2xl font-bold font-mono text-slate-900">
                {summary.n_features}
              </div>
              <span className="text-[10px] text-slate-400">
                {summary.numeric_features.length} numeric, {summary.categorical_features.length} categorical
              </span>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Target Distribution
              </span>
              <div className="mt-1 text-lg font-bold font-mono text-slate-900">
                {Object.entries(summary.class_distribution).map(([cls, cnt]) => (
                  <span key={cls} className="mr-2">
                    {cls}: {cnt}
                  </span>
                ))}
              </div>
              <span className="text-[10px] text-slate-400">
                Minority Class: <span className="font-semibold text-rose-600">{String(summary.minority_class)}</span>
              </span>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                Imbalance Ratio
              </span>
              <div className="mt-1 text-2xl font-bold font-mono text-rose-600">
                {summary.imbalance_ratio}
              </div>
              <span className="text-[10px] text-slate-400">Majority-to-minority ratio</span>
            </div>
          </div>

          {/* Safe Preprocessing Blueprint Card */}
          <div className="rounded-xl border border-blue-200 bg-blue-50/40 p-5">
            <h3 className="text-xs font-bold uppercase tracking-wider text-blue-900 flex items-center gap-1.5 mb-3">
              <ShieldCheck className="h-4 w-4 text-blue-600" />
              Safe Preprocessing Blueprint (Zero Data Leakage)
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="rounded-lg bg-white p-3 border border-blue-100 shadow-sm">
                <span className="font-semibold text-slate-800">1. Missing Imputation</span>
                <p className="mt-1 text-[11px] text-slate-600">
                  {summary.preprocessing_strategy.numeric_imputation}
                </p>
              </div>
              <div className="rounded-lg bg-white p-3 border border-blue-100 shadow-sm">
                <span className="font-semibold text-slate-800">2. Categorical Encoding</span>
                <p className="mt-1 text-[11px] text-slate-600">
                  {summary.preprocessing_strategy.categorical_encoding}
                </p>
              </div>
              <div className="rounded-lg bg-white p-3 border border-blue-100 shadow-sm">
                <span className="font-semibold text-slate-800">3. Scaling Isolation</span>
                <p className="mt-1 text-[11px] text-slate-600">
                  {summary.preprocessing_strategy.scaling}
                </p>
              </div>
            </div>
          </div>

          {/* Paginated Data Preview Table */}
          {preview && (
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-slate-100">
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                    <FileSpreadsheet className="h-4 w-4 text-slate-400" />
                    Dataset Preview (Rows {Math.min((page - 1) * preview.page_size + 1, preview.total_rows)} - {Math.min(page * preview.page_size, preview.total_rows)} of {preview.total_rows})
                  </h3>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Target column is highlighted. Values shown prior to in-fold scaling.
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    disabled={page <= 1}
                    onClick={() => loadDatasetDetails(selectedDatasetId, page - 1)}
                    className="rounded-lg border border-slate-200 p-1.5 text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </button>
                  <span className="text-xs font-medium font-mono text-slate-600">
                    Page {page} / {Math.ceil(preview.total_rows / preview.page_size)}
                  </span>
                  <button
                    disabled={page >= Math.ceil(preview.total_rows / preview.page_size)}
                    onClick={() => loadDatasetDetails(selectedDatasetId, page + 1)}
                    className="rounded-lg border border-slate-200 p-1.5 text-slate-600 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse font-mono">
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50/80 text-[11px] text-slate-500">
                      <th className="py-2 px-3 w-12 text-slate-400">#</th>
                      {preview.columns.map((col) => {
                        const isTarget = col === summary.target_column || col === "target";
                        return (
                          <th
                            key={col}
                            className={`py-2 px-3 whitespace-nowrap ${
                              isTarget ? "bg-blue-100/70 text-blue-900 font-bold" : "text-slate-600"
                            }`}
                          >
                            {col}
                            {isTarget && " [Target]"}
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-700">
                    {preview.rows.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-50 transition">
                        <td className="py-1.5 px-3 text-slate-400 text-[10px]">
                          {(page - 1) * preview.page_size + idx + 1}
                        </td>
                        {preview.columns.map((col) => {
                          const isTarget = col === summary.target_column || col === "target";
                          const val = row[col];
                          const isMinority = isTarget && String(val) === String(summary.minority_class);
                          return (
                            <td
                              key={col}
                              className={`py-1.5 px-3 whitespace-nowrap ${
                                isTarget
                                  ? isMinority
                                    ? "bg-rose-50 font-bold text-rose-700"
                                    : "bg-blue-50/40 text-slate-800"
                                  : ""
                              }`}
                            >
                              {val !== null && val !== undefined ? String(val) : "—"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        <EmptyState
          title="Dataset Not Found"
          description="Could not load the selected dataset."
        />
      )}
    </div>
  );
};
