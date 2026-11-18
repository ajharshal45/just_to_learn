import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Sidebar } from "./components/layout/Sidebar";
import { TopNavbar } from "./components/layout/TopNavbar";

// Pages
import { OverviewDashboard } from "./pages/OverviewDashboard";
import { DatasetExplorer } from "./pages/DatasetExplorer";
import { ExperimentLab } from "./pages/ExperimentLab";
import { ExperimentHistory } from "./pages/ExperimentHistory";
import { ShapExplorer } from "./pages/ShapExplorer";
import { ResultsComparison } from "./pages/ResultsComparison";
import { LeakageDiagnostics } from "./pages/LeakageDiagnostics";
import { ReportsAndExports } from "./pages/ReportsAndExports";
import { SettingsPage } from "./pages/SettingsPage";
import { SimulationPage } from "./pages/SimulationPage";

import { ErrorBoundary } from "./components/common/ErrorBoundary";

export const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <div className="flex min-h-screen bg-slate-50 text-slate-800 font-sans">
          {/* Persistent Left Sidebar */}
          <Sidebar />

          {/* Main Content Area */}
          <div className="flex-1 flex flex-col pl-64 transition-all duration-200">
            <TopNavbar />
            <main className="flex-1 p-8">
              <Routes>
                <Route path="/" element={<OverviewDashboard />} />
                <Route path="/datasets" element={<DatasetExplorer />} />
                <Route path="/lab" element={<ExperimentLab />} />
                <Route path="/history" element={<ExperimentHistory />} />
                <Route path="/shap" element={<ShapExplorer />} />
                <Route path="/simulation" element={<SimulationPage />} />
                <Route path="/comparison" element={<ResultsComparison />} />
                <Route path="/leakage" element={<LeakageDiagnostics />} />
                <Route path="/reports" element={<ReportsAndExports />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
    </ErrorBoundary>
  );
};

export default App;
