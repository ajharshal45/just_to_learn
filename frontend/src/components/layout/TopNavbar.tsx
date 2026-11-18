import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../services/api";
import { HealthResponse } from "../../types/api";
import { Activity, ShieldCheck, GitBranch, FlaskConical, ExternalLink } from "lucide-react";

export const TopNavbar: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [activeRunsCount, setActiveRunsCount] = useState<number>(0);

  const checkStatus = async () => {
    try {
      const h = await api.getHealth();
      setHealth(h);
      setIsConnected(true);

      const runningExps = await api.getExperiments({ status: "running" });
      setActiveRunsCount(runningExps.length);
    } catch (_) {
      setIsConnected(false);
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 8000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-20 flex h-16 w-full items-center justify-between border-b border-slate-200 bg-white/95 px-6 backdrop-blur">
      <div className="flex items-center gap-4">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Research Environment
        </span>
        <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs">
          <span
            className={`h-2 w-2 rounded-full ${
              isConnected ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
            }`}
          />
          <span className="font-medium text-slate-700">
            {isConnected ? "Backend Active" : "Backend Offline"}
          </span>
          {health && (
            <span className="text-slate-400 font-mono text-[11px]">
              (v{health.version} | Py {health.python_version})
            </span>
          )}
        </div>

        {activeRunsCount > 0 && (
          <Link
            to="/history"
            className="flex items-center gap-1.5 rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700 hover:bg-blue-100 transition"
          >
            <Activity className="h-3.5 w-3.5 animate-spin" />
            <span>{activeRunsCount} Experiment(s) Running</span>
          </Link>
        )}
      </div>

      <div className="flex items-center gap-3">
        <Link
          to="/lab"
          className="flex items-center gap-1.5 rounded-lg bg-navy-900 px-3.5 py-1.5 text-xs font-medium text-white shadow-sm hover:bg-navy-800 transition"
        >
          <FlaskConical className="h-3.5 w-3.5 text-blue-400" />
          <span>New Experiment</span>
        </Link>

        <a
          href="http://127.0.0.1:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm hover:bg-slate-50 transition"
        >
          <span>API Docs</span>
          <ExternalLink className="h-3 w-3 text-slate-400" />
        </a>
      </div>
    </header>
  );
};
