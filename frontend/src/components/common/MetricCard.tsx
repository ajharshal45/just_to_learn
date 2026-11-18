import React from "react";
import { ArrowUpRight, ArrowDownRight, Minus, HelpCircle } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: number | string;
  std?: number;
  subtitle?: string;
  direction?: "higher" | "lower" | "neutral";
  tooltip?: string;
  icon?: React.ReactNode;
  badge?: string;
  highlight?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  std,
  subtitle,
  direction = "higher",
  tooltip,
  icon,
  badge,
  highlight = false,
}) => {
  const formattedValue = typeof value === "number" ? value.toFixed(4) : value;

  return (
    <div
      className={`rounded-xl border bg-white p-5 shadow-sm transition-all hover:shadow-md ${
        highlight ? "border-blue-300 ring-2 ring-blue-50" : "border-slate-200"
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            {title}
          </span>
          {tooltip && (
            <div className="group relative cursor-help">
              <HelpCircle className="h-3.5 w-3.5 text-slate-400" />
              <div className="pointer-events-none absolute bottom-full left-1/2 mb-1.5 hidden -translate-x-1/2 w-48 rounded bg-slate-900 px-2 py-1 text-xs text-white shadow-lg group-hover:block z-20">
                {tooltip}
              </div>
            </div>
          )}
        </div>
        {icon && <div className="text-slate-400">{icon}</div>}
      </div>

      <div className="mt-3 flex items-baseline gap-2">
        <span className="text-2xl font-bold tracking-tight text-slate-900 font-mono">
          {formattedValue}
        </span>
        {std !== undefined && std > 0 && (
          <span className="text-xs font-medium text-slate-500 font-mono">
            ±{std.toFixed(4)}
          </span>
        )}
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-slate-100 pt-2.5 text-xs text-slate-500">
        <span>{subtitle || "5-Fold Nested CV"}</span>
        <div className="flex items-center gap-1">
          {direction === "higher" && (
            <span className="inline-flex items-center text-emerald-600 font-medium">
              <ArrowUpRight className="h-3.5 w-3.5 mr-0.5" /> Higher better
            </span>
          )}
          {direction === "lower" && (
            <span className="inline-flex items-center text-blue-600 font-medium">
              <ArrowDownRight className="h-3.5 w-3.5 mr-0.5" /> Lower better
            </span>
          )}
          {direction === "neutral" && (
            <span className="inline-flex items-center text-slate-500 font-medium">
              <Minus className="h-3.5 w-3.5 mr-0.5" /> Stability [0-1]
            </span>
          )}
          {badge && (
            <span className="ml-1 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-600">
              {badge}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};
