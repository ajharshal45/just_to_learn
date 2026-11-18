import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { HealthResponse } from "../types/api";
import { Badge } from "../components/common/Badge";
import { LoadingSpinner } from "../components/common/LoadingSpinner";
import {
  Settings,
  Activity,
  Server,
  Code2,
  CheckCircle2,
  RefreshCw,
  Cpu,
  Layers,
} from "lucide-react";

export const SettingsPage: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [latency, setLatency] = useState<number | null>(null);

  const checkHealth = async () => {
    try {
      setLoading(true);
      const start = performance.now();
      const res = await api.getHealth();
      const end = performance.now();
      setLatency(Math.round(end - start));
      setHealth(res);
    } catch (err) {
      console.error("Health check failed:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  return (
    <div className="space-y-8 max-w-4xl">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
          <Settings className="h-6 w-6 text-blue-600" />
          System Settings & Environment Diagnostics
        </h1>
        <p className="mt-1 text-xs text-slate-500">
          Backend API connectivity, ML environment runtime status, and default parameter specifications.
        </p>
      </div>

      {loading ? (
        <LoadingSpinner message="Checking backend runtime status..." />
      ) : (
        <div className="space-y-6">
          {/* Connection & Latency Card */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                <Server className="h-4 w-4 text-blue-600" />
                Backend Service Connectivity
              </h3>
              <button
                onClick={checkHealth}
                className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700 font-medium"
              >
                <RefreshCw className="h-3 w-3" />
                <span>Re-ping</span>
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="rounded-lg bg-slate-50 p-3 border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase font-semibold">
                  API Status
                </span>
                <div className="mt-1 flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="font-bold text-xs text-slate-800">
                    {health?.status === "healthy" ? "Healthy (200 OK)" : "Disconnected"}
                  </span>
                </div>
              </div>

              <div className="rounded-lg bg-slate-50 p-3 border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase font-semibold">
                  Ping Round-Trip Latency
                </span>
                <div className="mt-1 font-mono font-bold text-xs text-blue-700">
                  {latency !== null ? `${latency} ms` : "—"}
                </div>
              </div>

              <div className="rounded-lg bg-slate-50 p-3 border border-slate-200">
                <span className="text-[10px] text-slate-400 uppercase font-semibold">
                  Python Runtime
                </span>
                <div className="mt-1 font-mono font-bold text-xs text-slate-800">
                  Python {health?.python_version || "3.10+"}
                </div>
              </div>
            </div>
          </div>

          {/* Supported Algorithms & Classifiers */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5 pb-3 border-b border-slate-100">
              <Cpu className="h-4 w-4 text-slate-400" />
              Verified Active ML Modules
            </h3>

            <div className="space-y-3 text-xs">
              <div>
                <span className="font-semibold text-slate-700 block mb-1">
                  Supported Oversampling Algorithms:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {(health?.available_oversamplers || []).map((o) => (
                    <Badge key={o} variant={o.includes("shap") ? "default" : "neutral"} size="sm">
                      {o}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="pt-2">
                <span className="font-semibold text-slate-700 block mb-1">
                  Available Classification Estimators:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {(health?.available_classifiers || []).map((c) => (
                    <Badge key={c} variant="purple" size="sm">
                      {c}
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="pt-2">
                <span className="font-semibold text-slate-700 block mb-1">
                  Registered Tabular Datasets:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {(health?.registered_datasets || []).map((d) => (
                    <Badge key={d} variant="info" size="sm">
                      {d}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Research Protocol Presets */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5 pb-2 border-b border-slate-100">
              <Layers className="h-4 w-4 text-slate-400" />
              Standard Research Default Constants
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="p-2.5 rounded bg-slate-50 border border-slate-200">
                <span className="text-[10px] text-slate-400 block font-sans">Stratified Splits</span>
                <span className="font-bold text-slate-800">5 Folds</span>
              </div>
              <div className="p-2.5 rounded bg-slate-50 border border-slate-200">
                <span className="text-[10px] text-slate-400 block font-sans">k-Neighbours</span>
                <span className="font-bold text-slate-800">k = 5</span>
              </div>
              <div className="p-2.5 rounded bg-slate-50 border border-slate-200">
                <span className="text-[10px] text-slate-400 block font-sans">Preliminary Base</span>
                <span className="font-bold text-slate-800">RF 50 trees, depth 5</span>
              </div>
              <div className="p-2.5 rounded bg-slate-50 border border-slate-200">
                <span className="text-[10px] text-slate-400 block font-sans">Reproducibility Seed</span>
                <span className="font-bold text-slate-800">Seed = 42</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
