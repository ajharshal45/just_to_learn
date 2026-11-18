import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { ExperimentStatusResponse } from "../types/api";
import { Badge } from "../components/common/Badge";
import { LoadingSpinner } from "../components/common/LoadingSpinner";
import { EmptyState } from "../components/common/EmptyState";
import {
  FileText,
  Download,
  Copy,
  Check,
  FileSpreadsheet,
  FileCode,
  Sparkles,
} from "lucide-react";

export const ReportsAndExports: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentStatusResponse[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string>("");
  const [reportMarkdown, setReportMarkdown] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [loadingReport, setLoadingReport] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  useEffect(() => {
    const fetchList = async () => {
      try {
        setLoading(true);
        const list = await api.getExperiments({ status: "completed", limit: 100 });
        setExperiments(list);
        if (list.length > 0) {
          setSelectedExpId(list[0].id);
        }
      } catch (err) {
        console.error("Failed to fetch experiments:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchList();
  }, []);

  useEffect(() => {
    if (!selectedExpId) return;

    const fetchReport = async () => {
      try {
        setLoadingReport(true);
        const res = await fetch(api.getExportUrl(selectedExpId, "markdown"));
        const text = await res.text();
        setReportMarkdown(text);
      } catch (err) {
        console.error("Failed to fetch report:", err);
      } finally {
        setLoadingReport(false);
      }
    };
    fetchReport();
  }, [selectedExpId]);

  const handleCopy = () => {
    navigator.clipboard.writeText(reportMarkdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (loading) {
    return <LoadingSpinner message="Loading report generation engine..." />;
  }

  const selectedExp = experiments.find((e) => e.id === selectedExpId);

  return (
    <div className="space-y-8 max-w-5xl">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
          <FileText className="h-6 w-6 text-blue-600" />
          Experiment Reports & Data Exports
        </h1>
        <p className="mt-1 text-xs text-slate-500 max-w-2xl">
          Generate publication-ready academic reports, export fold-level metrics, and download raw configurations in standard research formats.
        </p>
      </div>

      {experiments.length === 0 ? (
        <EmptyState
          title="No Completed Experiments"
          description="Complete an experiment run in the Experiment Lab to generate reports."
        />
      ) : (
        <div className="space-y-6">
          {/* Controls Bar */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="flex-1 max-w-md">
              <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                Select Completed Experiment
              </label>
              <select
                value={selectedExpId}
                onChange={(e) => setSelectedExpId(e.target.value)}
                className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-800 focus:border-blue-500 focus:outline-none"
              >
                {experiments.map((e) => (
                  <option key={e.id} value={e.id}>
                    {e.oversampler} on {e.dataset_name} ({e.classifier}) — {e.id}
                  </option>
                ))}
              </select>
            </div>

            {/* Quick Export Actions */}
            <div className="flex flex-wrap items-center gap-2">
              <a
                href={api.getExportUrl(selectedExpId, "json")}
                download
                className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm hover:bg-slate-50 transition"
              >
                <FileCode className="h-3.5 w-3.5 text-blue-600" />
                <span>JSON</span>
              </a>

              <a
                href={api.getExportUrl(selectedExpId, "csv")}
                download
                className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm hover:bg-slate-50 transition"
              >
                <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-600" />
                <span>CSV</span>
              </a>

              <a
                href={api.getExportUrl(selectedExpId, "markdown")}
                download
                className="flex items-center gap-1.5 rounded-lg bg-navy-900 px-3.5 py-2 text-xs font-medium text-white shadow-sm hover:bg-navy-800 transition"
              >
                <Download className="h-3.5 w-3.5 text-blue-400" />
                <span>Download Report (.md)</span>
              </a>
            </div>
          </div>

          {/* Report Viewer */}
          <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50/80 px-5 py-3">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-semibold text-slate-700">
                  experiment_report_{selectedExpId}.md
                </span>
                {selectedExp?.is_benchmark && (
                  <Badge variant="neutral" size="sm">Pre-computed Benchmark</Badge>
                )}
              </div>

              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 shadow-sm transition"
              >
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-emerald-600" />
                    <span className="text-emerald-700">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5 text-slate-400" />
                    <span>Copy Markdown</span>
                  </>
                )}
              </button>
            </div>

            <div className="p-6">
              {loadingReport ? (
                <LoadingSpinner message="Rendering markdown research report..." />
              ) : (
                <pre className="font-mono text-xs text-slate-800 whitespace-pre-wrap leading-relaxed bg-slate-50/50 p-6 rounded-lg border border-slate-100 overflow-x-auto">
                  {reportMarkdown}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
