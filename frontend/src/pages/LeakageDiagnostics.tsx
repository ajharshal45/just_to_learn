import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { LeakageDiagnosticData } from "../types/api";
import { Badge } from "../components/common/Badge";
import { LoadingSpinner } from "../components/common/LoadingSpinner";
import {
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  TrendingDown,
  TrendingUp,
  FileCode2,
  Lock,
} from "lucide-react";

export const LeakageDiagnostics: React.FC = () => {
  const [data, setData] = useState<LeakageDiagnosticData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const res = await api.getLeakageDiagnostics();
        setData(res);
      } catch (err) {
        console.error("Failed to fetch leakage diagnostics:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <LoadingSpinner message="Loading zero-data-leakage verification suite..." />;
  }

  if (!data) {
    return (
      <div className="rounded-xl border border-rose-200 bg-rose-50 p-6 text-rose-800 text-xs">
        Failed to load leakage verification specifications.
      </div>
    );
  }

  const { protocol_steps, empirical_proof, safeguards } = data;

  return (
    <div className="space-y-8 max-w-6xl">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
          <ShieldAlert className="h-6 w-6 text-blue-600" />
          Zero-Data-Leakage Diagnostics & Empirical Proof
        </h1>
        <p className="mt-1 text-xs text-slate-500 max-w-3xl">
          Scientific proof and architectural validation that preprocessing, preliminary SHAP estimators,
          and synthetic sample generation occur strictly inside inner training folds without contaminating test partitions.
        </p>
      </div>

      {/* The Empirical Data Leakage Trap Demonstration */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-6">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-rose-600 flex items-center gap-1.5">
            <AlertTriangle className="h-4 w-4" />
            Empirical Benchmark Proof: The Data Leakage Trap
          </span>
          <h2 className="text-base font-bold text-slate-900 mt-1">
            Contaminated vs Strict Leak-Free Cross-Validation on Kaggle Healthcare Stroke
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Prior literature frequently oversamples the full dataset prior to cross-validation splits.
            Below is our direct head-to-head empirical measurement on the official Kaggle dataset ({empirical_proof.imbalance_ratio}).
          </p>
        </div>

        {/* Side-by-Side Comparison Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Flawed Pipeline */}
          <div className="rounded-xl border border-rose-200 bg-rose-50/50 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs uppercase tracking-wider text-rose-700 flex items-center gap-1">
                <AlertTriangle className="h-3.5 w-3.5" /> Flawed Leaked Pipeline
              </span>
              <Badge variant="danger" size="sm">Severe Data Leakage</Badge>
            </div>

            <p className="text-xs text-rose-800 leading-snug">
              {empirical_proof.leaked_pipeline.description}
            </p>

            <div className="grid grid-cols-3 gap-2 pt-2 border-t border-rose-200/60 font-mono">
              <div className="rounded bg-white p-2 border border-rose-200 text-center">
                <span className="text-[10px] text-slate-500 uppercase">PR-AUC</span>
                <div className="text-lg font-bold text-rose-600">
                  {empirical_proof.leaked_pipeline.pr_auc.toFixed(4)}
                </div>
              </div>
              <div className="rounded bg-white p-2 border border-rose-200 text-center">
                <span className="text-[10px] text-slate-500 uppercase">Recall</span>
                <div className="text-lg font-bold text-rose-600">
                  {empirical_proof.leaked_pipeline.minority_recall.toFixed(4)}
                </div>
              </div>
              <div className="rounded bg-white p-2 border border-rose-200 text-center">
                <span className="text-[10px] text-slate-500 uppercase">Minority F1</span>
                <div className="text-lg font-bold text-rose-600">
                  {empirical_proof.leaked_pipeline.minority_f1.toFixed(4)}
                </div>
              </div>
            </div>

            <p className="text-[11px] text-rose-700 bg-rose-100/70 p-2.5 rounded font-mono">
              *Synthetic sample interpolation creates synthetic twins across train and test partitions, inflating reported performance by +128.6%!
            </p>
          </div>

          {/* Strict Leak-Free Pipeline */}
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/50 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="font-bold text-xs uppercase tracking-wider text-emerald-800 flex items-center gap-1">
                <ShieldCheck className="h-4 w-4 text-emerald-600" /> Strict Leak-Free Pipeline (Ours)
              </span>
              <Badge variant="success" size="sm">Zero Data Leakage</Badge>
            </div>

            <p className="text-xs text-emerald-900 leading-snug">
              {empirical_proof.leak_free_pipeline.description}
            </p>

            <div className="grid grid-cols-3 gap-2 pt-2 border-t border-emerald-200/60 font-mono">
              <div className="rounded bg-white p-2 border border-emerald-200 text-center">
                <span className="text-[10px] text-slate-500 uppercase">PR-AUC</span>
                <div className="text-lg font-bold text-emerald-700">
                  {empirical_proof.leak_free_pipeline.pr_auc.toFixed(4)}
                </div>
              </div>
              <div className="rounded bg-white p-2 border border-emerald-200 text-center">
                <span className="text-[10px] text-slate-500 uppercase">Recall</span>
                <div className="text-lg font-bold text-emerald-700">
                  {empirical_proof.leak_free_pipeline.minority_recall.toFixed(4)}
                </div>
              </div>
              <div className="rounded bg-white p-2 border border-emerald-200 text-center">
                <span className="text-[10px] text-slate-500 uppercase">Minority F1</span>
                <div className="text-lg font-bold text-emerald-700">
                  {empirical_proof.leak_free_pipeline.minority_f1.toFixed(4)}
                </div>
              </div>
            </div>

            <p className="text-[11px] text-emerald-800 bg-emerald-100/70 p-2.5 rounded font-mono">
              *Evaluated on untouched held-out clinical data with zero synthetic leakage. Reflects true out-of-fold generalization.
            </p>
          </div>
        </div>

        {/* Delta Callout */}
        <div className="rounded-lg bg-slate-900 text-white p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 text-xs">
          <div>
            <span className="font-bold text-amber-400">Artificial Inflation Delta:</span>
            <span className="ml-2 font-mono text-sm font-semibold text-rose-400">
              {empirical_proof.delta.pr_auc_inflation}
            </span>
          </div>
          <div className="text-[11px] text-slate-300 max-w-xl">
            {empirical_proof.delta.finding}
          </div>
        </div>
      </div>

      {/* 7-Step Protocol Flowchart */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm space-y-4">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Pipeline Architecture
          </span>
          <h2 className="text-base font-bold text-slate-900 mt-0.5">
            The 7-Step Zero-Data-Leakage Sequence
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Every experiment in this platform follows this strict sequence implemented in <code className="font-mono text-blue-600">src/pipeline/leak_free_cv.py</code>.
          </p>
        </div>

        <div className="space-y-3 mt-4">
          {protocol_steps.map((st) => (
            <div
              key={st.step}
              className="flex items-start gap-4 rounded-xl border border-slate-200 bg-slate-50/50 p-4 transition hover:bg-slate-50"
            >
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-blue-600 font-mono text-xs font-bold text-white">
                {st.step}
              </div>

              <div className="flex-1 text-xs space-y-1">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-slate-900 text-xs">{st.title}</h3>
                  <Badge variant="success" size="sm">
                    <CheckCircle2 className="h-3 w-3 mr-1" /> Leak-Free
                  </Badge>
                </div>
                <p className="text-slate-600 leading-relaxed text-[11px]">
                  {st.description}
                </p>
                <div className="mt-1 text-[11px] text-rose-700 bg-rose-50/70 p-2 rounded border border-rose-100 flex items-start gap-1.5">
                  <span className="font-semibold shrink-0">Failure Mode in Naive Pipelines:</span>
                  <span>{st.danger_in_flawed}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Code Safeguards Reference */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
          <FileCode2 className="h-4 w-4 text-blue-600" />
          Backend Implementation Safeguards
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          {safeguards.map((s, idx) => (
            <div key={idx} className="rounded-lg bg-slate-50 p-3.5 border border-slate-200">
              <span className="font-bold text-slate-800">{s.name}</span>
              <p className="mt-1 text-[11px] text-slate-600 leading-relaxed font-mono">
                {s.detail}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
