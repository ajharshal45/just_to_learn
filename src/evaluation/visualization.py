"""
Visualization and Publication-Quality LaTeX Export Engine.

Generates:
- Benchmark multi-metric comparison bar charts
- Precision-Recall curves & ROC curves
- Calibration reliability diagrams
- Attribution stability heatmaps / comparisons
- Publication-ready LaTeX tables (booktabs formatted)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_benchmark_metric_comparison(
    df_results: pd.DataFrame,
    metric_col: str = "pr_auc_mean",
    metric_label: str = "PR-AUC (Average Precision)",
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plots grouped bar chart comparing oversamplers across classifiers or datasets.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    classifiers = df_results["classifier"].unique()
    oversamplers = df_results["oversampler"].unique()

    x = np.arange(len(oversamplers))
    width = 0.8 / len(classifiers)

    for i, clf in enumerate(classifiers):
        sub_df = df_results[df_results["classifier"] == clf]
        # Align with oversamplers order
        vals = []
        errors = []
        for o in oversamplers:
            row = sub_df[sub_df["oversampler"] == o]
            if len(row) > 0:
                vals.append(row[metric_col].values[0])
                std_col = metric_col.replace("_mean", "_std")
                errors.append(row[std_col].values[0] if std_col in row else 0.0)
            else:
                vals.append(0.0)
                errors.append(0.0)

        offset = (i - len(classifiers) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=errors,
            capsize=3,
            label=clf,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")
    ax.set_title(title or f"Empirical Benchmark Comparison: {metric_label}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(oversamplers, rotation=25, ha="right", fontsize=10)
    ax.legend(title="Classifier", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    return fig


def plot_multi_dimensional_radar(
    df_results: pd.DataFrame,
    oversamplers_to_compare: List[str],
    metrics: List[str] = None,
    metric_labels: List[str] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generates radar / polar chart comparing multi-dimensional trade-offs
    (Utility, Calibration, Stability).
    """
    if metrics is None:
        metrics = ["pr_auc_mean", "minority_f1_mean", "balanced_accuracy_mean", "attribution_kendall_tau_mean"]
    if metric_labels is None:
        metric_labels = ["PR-AUC", "Minority F1", "Balanced Acc", "Attribution Tau"]

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for o_name in oversamplers_to_compare:
        sub = df_results[df_results["oversampler"] == o_name]
        if len(sub) == 0:
            continue
        vals = [sub[m].mean() for m in metrics]
        # Invert ECE or Brier if present so outward is always better
        vals_plot = []
        for m, v in zip(metrics, vals):
            if "ece" in m or "brier" in m:
                vals_plot.append(max(0.0, 1.0 - v))
            else:
                vals_plot.append(v)
        vals_plot += vals_plot[:1]

        ax.plot(angles, vals_plot, linewidth=2, label=o_name)
        ax.fill(angles, vals_plot, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax.set_title("Multi-Dimensional Performance Trade-Offs", size=14, y=1.08, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    return fig


def generate_latex_table(
    df_results: pd.DataFrame,
    caption: str = "Empirical Benchmark Results across Imbalanced Oversampling Techniques.",
    label: str = "tab:benchmark_results",
    selected_cols: Optional[List[str]] = None,
) -> str:
    """
    Converts benchmark DataFrame into publication-ready LaTeX table using booktabs.
    """
    if selected_cols is None:
        selected_cols = [
            "oversampler",
            "classifier",
            "pr_auc_mean",
            "roc_auc_mean",
            "minority_f1_mean",
            "ece_mean",
            "attribution_kendall_tau_mean",
        ]

    df_sub = df_results[[c for c in selected_cols if c in df_results.columns]].copy()

    rename_map = {
        "oversampler": "Oversampler",
        "classifier": "Classifier",
        "pr_auc_mean": "PR-AUC",
        "roc_auc_mean": "ROC-AUC",
        "minority_f1_mean": "F1-Score",
        "ece_mean": "ECE $\\downarrow$",
        "attribution_kendall_tau_mean": "Kendall $\\tau$",
    }
    df_sub = df_sub.rename(columns=rename_map)

    # Format numeric floats to 4 decimal places
    for col in df_sub.columns:
        if df_sub[col].dtype in (np.float64, np.float32, float):
            df_sub[col] = df_sub[col].map(lambda x: f"{x:.4f}")

    col_fmt = "l" * len(df_sub.columns)
    headers = " & ".join(df_sub.columns) + " \\\\"
    
    rows = []
    for _, row in df_sub.iterrows():
        row_str = " & ".join(str(val) for val in row.values) + " \\\\"
        rows.append(row_str)
    
    body = "\n".join(rows)

    latex_code = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{" + col_fmt + "}\n"
        "\\toprule\n"
        f"{headers}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return latex_code
