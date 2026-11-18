import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertOctagon, RotateCcw, Home } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.setState({ error, errorInfo });
  }

  public handleReload = () => {
    window.location.reload();
  };

  public handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    window.location.href = "/";
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50 p-6">
          <div className="max-w-xl w-full rounded-2xl border border-rose-200 bg-white p-8 shadow-sm">
            <div className="flex items-center gap-3 text-rose-600 mb-4">
              <AlertOctagon className="h-8 w-8" />
              <div>
                <h2 className="text-lg font-bold text-slate-900">
                  Application Rendering Exception
                </h2>
                <p className="text-xs text-rose-600 font-medium">
                  An unexpected error was intercepted by the research platform runtime.
                </p>
              </div>
            </div>

            <div className="rounded-lg bg-slate-900 p-4 font-mono text-xs text-rose-300 overflow-x-auto max-h-48 mb-6 border border-slate-800">
              {this.state.error?.message || "Unknown error occurred"}
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={this.handleReload}
                className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-blue-700 transition"
              >
                <RotateCcw className="h-4 w-4" />
                Reload Page
              </button>
              <button
                onClick={this.handleReset}
                className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition"
              >
                <Home className="h-4 w-4 text-slate-400" />
                Return to Overview
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
