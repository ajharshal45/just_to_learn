import React from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Database,
  FlaskConical,
  History,
  Sparkles,
  BarChart3,
  ShieldAlert,
  FileText,
  Settings,
  Layers,
  Sliders,
} from "lucide-react";

interface SidebarProps {
  collapsed?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ collapsed = false }) => {
  const navItems = [
    { to: "/", label: "Overview", icon: <LayoutDashboard className="h-4 w-4" /> },
    { to: "/datasets", label: "Dataset Explorer", icon: <Database className="h-4 w-4" /> },
    { to: "/lab", label: "Experiment Lab", icon: <FlaskConical className="h-4 w-4" /> },
    { to: "/history", label: "Runs & History", icon: <History className="h-4 w-4" /> },
    { to: "/shap", label: "SHAP Explainability", icon: <Sparkles className="h-4 w-4" /> },
    { to: "/simulation", label: "Manifold Simulation", icon: <Sliders className="h-4 w-4" /> },
    { to: "/comparison", label: "Results & Benchmarks", icon: <BarChart3 className="h-4 w-4" /> },
    { to: "/leakage", label: "Leakage Diagnostics", icon: <ShieldAlert className="h-4 w-4" /> },
    { to: "/reports", label: "Reports & Exports", icon: <FileText className="h-4 w-4" /> },
    { to: "/settings", label: "System & Info", icon: <Settings className="h-4 w-4" /> },
  ];

  return (
    <aside
      className={`fixed left-0 top-0 z-30 flex h-screen flex-col border-r border-slate-200 bg-white transition-all duration-200 ${
        collapsed ? "w-16" : "w-64"
      }`}
    >
      {/* Brand Header */}
      <div className="flex h-16 items-center gap-3 border-b border-slate-200 px-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-navy-900 text-white shadow-sm">
          <Layers className="h-5 w-5 text-blue-400" />
        </div>
        {!collapsed && (
          <div className="flex flex-col">
            <span className="font-bold text-sm tracking-tight text-navy-900">
              SHAP-SMOTE
            </span>
            <span className="text-[11px] font-medium text-slate-400">
              Research Platform
            </span>
          </div>
        )}
      </div>

      {/* Navigation List */}
      <nav className="flex-1 space-y-1 overflow-y-auto p-3">
        <div className="px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-slate-400">
          {!collapsed ? "Workspace" : "..."}
        </div>
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-xs font-medium transition-colors ${
                isActive
                  ? "bg-blue-50 text-blue-700 font-semibold"
                  : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
              }`
            }
          >
            <span className="text-slate-500">{item.icon}</span>
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Zero Leakage Indicator Footer */}
      {!collapsed && (
        <div className="border-t border-slate-200 p-4">
          <div className="rounded-lg bg-slate-50 p-3 border border-slate-200">
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-[11px] font-semibold text-slate-700">
                Zero-Leakage Guard
              </span>
            </div>
            <p className="mt-1 text-[10px] text-slate-500 leading-relaxed">
              Nested CV strictly isolates preprocessing & SHAP to training folds.
            </p>
          </div>
        </div>
      )}
    </aside>
  );
};
