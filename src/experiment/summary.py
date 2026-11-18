"""
Result summaries computed from the saved/in-memory results (never typed by hand):

    dataset_findings      per-dataset answers to the research questions, as a table + text lines
    load_all_results      read every dataset's metrics back from results/<dataset>/metrics
    cross-dataset tables  used by notebooks/05_cross_dataset_comparison.ipynb
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from src.experiment import config as C
from src.experiment.runner import METRIC_LABELS, CVResult, FinalResult

DATASET_TITLES = {
    "breast_cancer": "Breast Cancer",
    "credit_card": "Credit Card Fraud",
    "stroke": "Stroke",
    "nsl_kdd": "NSL-KDD",
}
KEY_METRICS = ["pr_auc", "f1", "recall", "precision", "roc_auc"]


def dataset_findings(cv: CVResult, tests: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    """One row per downstream model: best method and SHAP-vs-SMOTE differences (CV means)."""
    means = cv.mean_table(KEY_METRICS)
    rows = []
    for model in C.MODEL_ORDER:
        m = means.loc[model]
        row = {"model": model,
               "best PR-AUC": f"{m['pr_auc'].idxmax()} ({m['pr_auc'].max():.3f})",
               "best F1": f"{m['f1'].idxmax()} ({m['f1'].max():.3f})",
               "best recall": f"{m['recall'].idxmax()} ({m['recall'].max():.3f})"}
        for sm in C.PROPOSED_METHODS:
            short = "SHAP-F" if "Filtered" in sm else "SHAP-W"
            for metric in ["pr_auc", "f1", "recall", "precision"]:
                t = tests[(tests.model == model) & (tests.method == sm) & (tests.metric == METRIC_LABELS[metric])]
                p = t["p_value"].iloc[0]
                row[f"{short} - SMOTE {METRIC_LABELS[metric]}"] = (
                    f"{m.loc[sm, metric] - m.loc['SMOTE', metric]:+.3f} (p={p:.2f})")
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def findings_text(cv: CVResult, tests: pd.DataFrame, quality: pd.DataFrame, alpha: float = 0.05) -> List[str]:
    """Plain-language findings derived from the numbers (used as a checklist for the write-up)."""
    means = cv.mean_table(KEY_METRICS)
    out = []
    for model in C.MODEL_ORDER:
        m = means.loc[model]
        order = m["pr_auc"].sort_values(ascending=False)
        out.append(f"{model}: PR-AUC ranking = " + " > ".join(f"{k} {v:.3f}" for k, v in order.items()))
        for sm in C.PROPOSED_METHODS:
            parts = []
            for metric in ["pr_auc", "f1", "recall", "precision"]:
                t = tests[(tests.model == model) & (tests.method == sm) & (tests.metric == METRIC_LABELS[metric])].iloc[0]
                d = m.loc[sm, metric] - m.loc["SMOTE", metric]
                sig = "significant" if t.p_value < alpha else "not significant"
                parts.append(f"{METRIC_LABELS[metric]} {d:+.3f} ({t.folds_better}/{t.folds_better + t.folds_worse} "
                             f"folds better, p={t.p_value:.3f}, {sig})")
            out.append(f"  {sm} vs SMOTE: " + "; ".join(parts))
    q = quality
    if "majority_share_in_5nn" in q:
        out.append("Synthetic rows' majority share among 5 nearest real rows (lower = less overlap): "
                   + ", ".join(f"{k} {v:.3f}" for k, v in q["majority_share_in_5nn"].dropna().items()))
    if "mean_wasserstein_distance" in q:
        out.append("Mean marginal Wasserstein distance synthetic vs real minority (lower = closer): "
                   + ", ".join(f"{k} {v:.3f}" for k, v in q["mean_wasserstein_distance"].dropna().items()))
    if "correlation_matrix_similarity" in q:
        out.append("Correlation-structure similarity synthetic vs real minority (1 = identical): "
                   + ", ".join(f"{k} {v:.3f}" for k, v in q["correlation_matrix_similarity"].dropna().items()))
    if "detectability_auc_all" in q:
        out.append("Real-vs-synthetic detectability AUC, all columns / numeric only (0.5 = indistinguishable): "
                   + ", ".join(f"{k} {a:.3f} / {b:.3f}" for k, a, b in
                               q[["detectability_auc_all", "detectability_auc_numeric"]].dropna().itertuples()))
    return out


def comparison_table(final: FinalResult, model: str) -> pd.DataFrame:
    """The requested consolidated table for one downstream model (hold-out test set)."""
    t = final.test_metrics[final.test_metrics.model == model].set_index("method")
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity"]
    out = t[cols].rename(columns=METRIC_LABELS).reindex(C.METHOD_ORDER)
    out.index.name = "Sampling Method"
    return out


def cv_comparison_table(cv: CVResult, model: str) -> pd.DataFrame:
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity"]
    g = cv.fold_metrics[cv.fold_metrics.model == model].groupby("method")[cols]
    out = (g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)).reindex(C.METHOD_ORDER)
    out.index.name = "Sampling Method"
    return out.rename(columns=METRIC_LABELS)


# --------------------------------------------------------------------------------------------
# Cross-dataset
# --------------------------------------------------------------------------------------------
def load_all_results(slugs: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    slugs = slugs or list(DATASET_TITLES)
    out = {}
    for s in slugs:
        d = C.RESULTS_DIR / s / "metrics"
        if not (d / "cv_fold_metrics.csv").exists():
            raise FileNotFoundError(f"{d} has no results; run the {s} notebook first.")
        out[s] = {
            "cv": pd.read_csv(d / "cv_fold_metrics.csv"),
            "test": pd.read_csv(d / "test_metrics.csv"),
            "tests": pd.read_csv(d / "paired_tests_vs_smote.csv"),
            "quality": pd.read_csv(d / "synthetic_quality.csv"),
            "stability": pd.read_csv(d / "attribution_stability.csv"),
            "sampling": pd.read_csv(d / "cv_sampling_runtime.csv"),
        }
    return out


def cross_cv_means(results, metric: str) -> pd.DataFrame:
    """(dataset, model) x method table of CV means for one metric."""
    frames = []
    for s, r in results.items():
        t = r["cv"].groupby(["model", "method"])[metric].mean().unstack("method")[C.METHOD_ORDER]
        t.index = pd.MultiIndex.from_product([[DATASET_TITLES[s]], t.index], names=["dataset", "model"])
        frames.append(t.reindex(C.MODEL_ORDER, level="model"))
    return pd.concat(frames)


def relative_improvement(results, metric: str, reference: str = "SMOTE") -> pd.DataFrame:
    """Percentage change of each proposed method's CV mean vs the reference, per dataset and model."""
    t = cross_cv_means(results, metric)
    return pd.DataFrame({m: 100 * (t[m] - t[reference]) / t[reference].replace(0, np.nan) for m in C.PROPOSED_METHODS})


def best_methods(results, metric: str) -> pd.DataFrame:
    t = cross_cv_means(results, metric)
    return pd.DataFrame({"best method": t.idxmax(axis=1), "best mean": t.max(axis=1),
                         "SMOTE": t["SMOTE"], "No Sampling": t["No Sampling"]})


def average_ranks(results, metric: str):
    """Mean rank of each method over all (dataset, model) blocks (1 = best) + Friedman test."""
    t = cross_cv_means(results, metric)
    ranks = t.rank(axis=1, ascending=False)
    stat, p = friedmanchisquare(*[t[m] for m in C.METHOD_ORDER])
    return ranks.mean().reindex(C.METHOD_ORDER).sort_values(), stat, p


def significance_overview(results, alpha: float = 0.05) -> pd.DataFrame:
    """Count of (dataset, model) blocks where each proposed method is significantly better/worse than SMOTE."""
    rows = []
    for s, r in results.items():
        t = r["tests"]
        t = t[t.method.isin(C.PROPOSED_METHODS)]
        for _, x in t.iterrows():
            rows.append({"dataset": DATASET_TITLES[s], **x.to_dict()})
    t = pd.DataFrame(rows)
    t["outcome"] = np.where(t.p_value >= alpha, "no significant difference",
                            np.where(t.mean_difference > 0, "significantly better", "significantly worse"))
    return t.pivot_table(index=["method", "metric"], columns="outcome", values="model", aggfunc="count", fill_value=0)
