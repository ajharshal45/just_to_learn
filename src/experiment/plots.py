"""
Figures shared by the dataset notebooks. Every function draws one figure, saves it as PNG under
`results/<dataset>/plots/` (when `save_dir` is given) and returns the matplotlib Figure.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve

from src.experiment import config as C
from src.experiment.runner import METRIC_LABELS, CVResult, FinalResult

MAJ, MIN, SYN = C.CLASS_COLORS["majority"], C.CLASS_COLORS["minority"], C.CLASS_COLORS["synthetic"]


def set_style():
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 130, "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e6e6e6",
    })


def _save(fig, save_dir: Optional[Path], name: str):
    fig.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / f"{name}.png", bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------------------------
# Stage 2: EDA
# --------------------------------------------------------------------------------------------
def class_distribution(df: pd.DataFrame, title: str, save_dir=None):
    counts = df["target"].value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Majority (0)", "Minority (1)"], counts.values, color=[MAJ, MIN], width=0.5)
    for b, v in zip(bars, counts.values):
        ax.annotate(f"{v:,}\n({v / counts.sum():.2%})", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("Rows")
    ax.set_title(f"{title}: class distribution (imbalance {counts[0] / counts[1]:.1f} : 1)")
    return _save(fig, save_dir, "01_class_distribution")


def feature_distributions(df: pd.DataFrame, features: List[str], save_dir=None, log_skewed: bool = True):
    """Histograms of numeric features, majority vs minority (density-normalised per class)."""
    n = len(features)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows))
    for ax, f in zip(np.ravel(axes), features):
        x = df[f].astype(float)
        skewed = log_skewed and x.min() >= 0 and x.skew() > 5
        xx = np.log1p(x) if skewed else x
        bins = np.histogram_bin_edges(xx.dropna(), bins=40)
        for cls, color, lab in [(0, MAJ, "Majority"), (1, MIN, "Minority")]:
            ax.hist(xx[df["target"] == cls].dropna(), bins=bins, density=True, alpha=0.55, color=color, label=lab)
        ax.set_title(f"log1p({f})" if skewed else f, fontsize=10)
        ax.set_yticks([])
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    np.ravel(axes)[0].legend()
    fig.suptitle("Numeric feature distributions by class (density per class)", fontweight="bold")
    return _save(fig, save_dir, "02_feature_distributions")


def correlation_heatmap(df: pd.DataFrame, features: List[str], save_dir=None):
    corr = df[features + ["target"]].corr()
    size = max(6, 0.38 * len(corr))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Pearson r", "shrink": 0.7},
                annot=len(corr) <= 14, fmt=".2f", annot_kws={"fontsize": 7})
    ax.set_title("Correlation heatmap (features with the strongest class separation, plus target)")
    return _save(fig, save_dir, "03_correlation_heatmap")


def class_comparison(df: pd.DataFrame, features: List[str], ks: pd.Series, save_dir=None):
    """Box plots of the most class-separating features (KS statistic), majority vs minority."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.8))
    for ax, f in zip(np.atleast_1d(axes), features):
        data = [df.loc[df["target"] == 0, f].dropna(), df.loc[df["target"] == 1, f].dropna()]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], [MAJ, MIN]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks([1, 2], ["Maj.", "Min."])
        ax.set_title(f"{f}\nKS D = {ks[f]:.2f}", fontsize=10)
    fig.suptitle("Feature distributions compared between classes (outliers hidden)", fontweight="bold")
    return _save(fig, save_dir, "04_class_comparison")


# --------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------
def balanced_class_counts(final: FinalResult, save_dir=None):
    s = final.sampling
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(s))
    ax.bar(x - 0.2, s["final_majority"], width=0.38, color=MAJ, label="Majority (real)")
    ax.bar(x + 0.2, s["real_minority"], width=0.38, color=MIN, label="Minority (real)")
    ax.bar(x + 0.2, s["synthetic_minority"], bottom=s["real_minority"], width=0.38, color=SYN,
           label="Minority (synthetic)", edgecolor="white", linewidth=1.5)
    for i, (r, t) in enumerate(zip(s["real_minority"], s["final_minority"])):
        ax.annotate(f"{t:,}", (i + 0.2, t), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, s.index, rotation=12)
    ax.set_ylabel("Training rows")
    ax.set_title("Development set before (No Sampling) and after each oversampler")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, save_dir, "05_balanced_class_counts")


def masv_weights(weights: pd.DataFrame, save_dir=None, top: int = 20):
    t = weights.head(top)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [SYN if k else "#b5b4af" for k in t["kept_by_SHAP_Filtered"]]
    ax.bar(t.index, t["weight_w"], color=colors)
    ax.axhline(1 / len(weights), color="#52514e", linestyle="--", linewidth=1, label="Uniform weight (plain SMOTE)")
    ax.set_ylabel("Normalised MASV weight $w_j$")
    ax.tick_params(axis="x", rotation=70)
    ax.set_title(f"SHAP feature weights from the preliminary model (top {len(t)}; green = kept by SHAP-Filtered SMOTE)")
    ax.legend()
    return _save(fig, save_dir, "06_shap_feature_weights")


def shap_summary(final: FinalResult, save_dir=None, max_display: int = 15):
    """Beeswarm of the preliminary model's SHAP values: the guidance signal used by the SHAP samplers."""
    import shap

    s = final.samplers["SHAP-Weighted SMOTE"]
    explained = ~np.isnan(s.shap_values_).any(axis=1)
    shap.summary_plot(s.shap_values_[explained], final.X_dev[explained], feature_names=final.feature_names,
                      max_display=max_display, show=False, plot_size=(9, 6))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP summary of the preliminary model (development set) - the sampling guidance",
                          fontsize=11, fontweight="bold")
    return _save(fig, save_dir, "07_shap_summary_preliminary_model")


def pca_synthetic(final: FinalResult, save_dir=None, max_points: int = 3000, seed: int = C.SEED):
    """Real majority, real minority and synthetic minority projected on the PCA of the real dev data."""
    rng = np.random.default_rng(seed)
    pca = PCA(n_components=2, random_state=seed).fit(final.X_dev)
    maj = final.X_dev[final.y_dev == 0]
    maj = maj[rng.choice(len(maj), min(len(maj), max_points), replace=False)]
    P_maj, P_min = pca.transform(maj), pca.transform(final.X_dev[final.y_dev == 1])
    fig, axes = plt.subplots(1, len(C.OVERSAMPLERS), figsize=(4.2 * len(C.OVERSAMPLERS), 4.2), sharex=True, sharey=True)
    for ax, m in zip(axes, C.OVERSAMPLERS):
        syn = final.synthetic(m)
        ax.scatter(*P_maj.T, s=4, c=MAJ, alpha=0.15, label="Majority (real)")
        if len(syn):
            syn = syn[rng.choice(len(syn), min(len(syn), max_points), replace=False)]
            ax.scatter(*pca.transform(syn).T, s=5, c=SYN, alpha=0.35, label="Minority (synthetic)")
        ax.scatter(*P_min.T, s=12, c=MIN, edgecolors="white", linewidths=0.4, label="Minority (real)")
        ax.set_title(f"{m}\n({final.samplers[m].n_synthetic_:,} synthetic)", fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    axes[0].legend(loc="best", markerscale=2, fontsize=8)
    fig.suptitle("Original vs synthetic minority samples (PCA fitted on the real development data)", fontweight="bold")
    return _save(fig, save_dir, "08_pca_original_vs_synthetic")


def synthetic_feature_distributions(final: FinalResult, features: List[str], save_dir=None):
    """Real vs synthetic minority distribution on the top SHAP features, one row per feature."""
    idx = [final.feature_names.index(f) for f in features]
    real = final.X_dev[final.y_dev == 1]
    fig, axes = plt.subplots(len(idx), len(C.OVERSAMPLERS), figsize=(3.4 * len(C.OVERSAMPLERS), 2.4 * len(idx)),
                             sharex="row", squeeze=False)
    for r, (j, f) in enumerate(zip(idx, features)):
        lo, hi = np.percentile(final.X_dev[:, j], [0.5, 99.5])
        lo, hi = min(lo, real[:, j].min()), max(hi, real[:, j].max())
        bins = np.linspace(lo, hi, 40)
        for c, m in enumerate(C.OVERSAMPLERS):
            ax = axes[r, c]
            syn = final.synthetic(m)
            ax.hist(real[:, j], bins=bins, density=True, color=MIN, alpha=0.55, label="Real minority")
            if len(syn):
                ax.hist(syn[:, j], bins=bins, density=True, color=SYN, alpha=0.5, label="Synthetic")
            ax.set_yticks([])
            if r == 0:
                ax.set_title(m, fontsize=10)
            if c == 0:
                ax.set_ylabel(f, fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Real vs synthetic minority distributions on the top SHAP features (scaled units)", fontweight="bold")
    return _save(fig, save_dir, "09_synthetic_feature_distributions")


# --------------------------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------------------------
def performance_comparison(cv: CVResult, save_dir=None, metrics=("precision", "recall", "f1", "roc_auc", "pr_auc")):
    """Graph 1: grouped bars, CV mean +/- std, one panel per downstream model."""
    fm = cv.fold_metrics
    fig, axes = plt.subplots(1, len(C.MODEL_ORDER), figsize=(6 * len(C.MODEL_ORDER), 4.6), sharey=True)
    width = 0.8 / len(C.METHOD_ORDER)
    x = np.arange(len(metrics))
    for ax, model in zip(axes, C.MODEL_ORDER):
        sub = fm[fm.model == model]
        for i, m in enumerate(C.METHOD_ORDER):
            g = sub[sub.method == m][list(metrics)]
            ax.bar(x + (i - (len(C.METHOD_ORDER) - 1) / 2) * width, g.mean(), width=width * 0.92,
                   yerr=g.std(), capsize=2, color=C.METHOD_COLORS[m], label=m, error_kw={"elinewidth": 0.8})
        # mark the best method per metric
        means = sub.groupby("method")[list(metrics)].mean()
        for k, metric in enumerate(metrics):
            best = means[metric].idxmax()
            i = C.METHOD_ORDER.index(best)
            ax.annotate("▲", (k + (i - (len(C.METHOD_ORDER) - 1) / 2) * width, 1.02), ha="center", fontsize=8,
                        color="#0b0b0b", annotation_clip=False)
        ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
        ax.set_ylim(0, 1.08)
        ax.set_title(model)
    axes[0].set_ylabel(f"Mean over {C.CV_SPLITS}x{C.CV_REPEATS} CV folds (± 1 sd)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Graph 1 - Performance comparison (▲ = best mean per metric)", fontweight="bold")
    return _save(fig, save_dir, "10_graph1_performance_comparison")


def fold_distributions(cv: CVResult, save_dir=None, metrics=("pr_auc", "f1")):
    fm = cv.fold_metrics
    fig, axes = plt.subplots(len(metrics), len(C.MODEL_ORDER), figsize=(5.5 * len(C.MODEL_ORDER), 3.4 * len(metrics)),
                             sharey="row", squeeze=False)
    for r, metric in enumerate(metrics):
        for c, model in enumerate(C.MODEL_ORDER):
            ax = axes[r, c]
            sub = fm[fm.model == model]
            sns.boxplot(data=sub, x="method", y=metric, order=C.METHOD_ORDER, hue="method", hue_order=C.METHOD_ORDER,
                        palette=C.METHOD_COLORS, ax=ax, showfliers=False, legend=False, width=0.6)
            sns.stripplot(data=sub, x="method", y=metric, order=C.METHOD_ORDER, color="#0b0b0b", size=3, ax=ax, alpha=0.6)
            ax.set_xlabel("")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.tick_params(axis="x", rotation=35)
            for t in ax.get_xticklabels():
                t.set_ha("right")
            if r == 0:
                ax.set_title(model)
    fig.suptitle("Stability across cross-validation folds (each dot = one fold)", fontweight="bold")
    return _save(fig, save_dir, "11_cv_fold_stability")


def roc_pr_curves(final: FinalResult, save_dir=None):
    fig, axes = plt.subplots(2, len(C.MODEL_ORDER), figsize=(5.5 * len(C.MODEL_ORDER), 9.5))
    tm = final.test_metrics.set_index(["method", "model"])
    for c, model in enumerate(C.MODEL_ORDER):
        for m in C.METHOD_ORDER:
            p = final.test_prob[(m, model)]
            fpr, tpr, _ = roc_curve(final.y_test, p)
            prec, rec, _ = precision_recall_curve(final.y_test, p)
            kw = dict(color=C.METHOD_COLORS[m], linewidth=2 if m in C.PROPOSED_METHODS else 1.4)
            axes[0, c].plot(fpr, tpr, label=f"{m} ({tm.loc[(m, model), 'roc_auc']:.3f})", **kw)
            axes[1, c].plot(rec, prec, drawstyle="steps-post", label=f"{m} ({tm.loc[(m, model), 'pr_auc']:.3f})", **kw)
        axes[0, c].plot([0, 1], [0, 1], color="#b5b4af", linestyle="--", linewidth=1)
        axes[1, c].axhline(final.y_test.mean(), color="#b5b4af", linestyle="--", linewidth=1, label="No-skill")
        axes[0, c].set(title=f"ROC - {model}", xlabel="False positive rate", ylabel="True positive rate")
        axes[1, c].set(title=f"Precision-recall - {model}", xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
        axes[0, c].legend(fontsize=7.5, title="ROC-AUC", title_fontsize=8, loc="lower right")
        axes[1, c].legend(fontsize=7.5, title="PR-AUC", title_fontsize=8, loc="lower left")
    fig.suptitle("ROC and precision-recall curves on the untouched hold-out test set", fontweight="bold")
    return _save(fig, save_dir, "12_roc_pr_curves")


def confusion_matrices(final: FinalResult, save_dir=None):
    tm = final.test_metrics.set_index(["method", "model"])
    fig, axes = plt.subplots(len(C.MODEL_ORDER), len(C.METHOD_ORDER),
                             figsize=(2.6 * len(C.METHOD_ORDER), 2.5 * len(C.MODEL_ORDER)))
    for r, model in enumerate(C.MODEL_ORDER):
        for c, m in enumerate(C.METHOD_ORDER):
            t = tm.loc[(m, model)]
            cm = np.array([[t.tn, t.fp], [t.fn, t.tp]], dtype=int)
            ax = axes[r, c]
            sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
                        cbar=False, ax=ax, xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"],
                        annot_kws={"fontsize": 9})
            if r == 0:
                ax.set_title(m, fontsize=9)
            if c == 0:
                ax.set_ylabel(model, fontsize=9)
    fig.suptitle("Confusion matrices on the hold-out test set (shade = row-normalised rate)", fontweight="bold")
    return _save(fig, save_dir, "13_confusion_matrices")


def calibration_curves(final: FinalResult, save_dir=None, n_bins: int = 10):
    fig, axes = plt.subplots(1, len(C.MODEL_ORDER), figsize=(5.3 * len(C.MODEL_ORDER), 4.3), sharey=True)
    for ax, model in zip(axes, C.MODEL_ORDER):
        for m in C.METHOD_ORDER:
            frac, mean_p = calibration_curve(final.y_test, final.test_prob[(m, model)], n_bins=n_bins, strategy="quantile")
            ax.plot(mean_p, frac, marker="o", markersize=3, color=C.METHOD_COLORS[m], label=m, linewidth=1.4)
        ax.plot([0, 1], [0, 1], color="#b5b4af", linestyle="--", linewidth=1)
        ax.set(title=model, xlabel="Mean predicted probability", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("Observed minority fraction")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.suptitle("Probability calibration on the test set (quantile bins)", fontweight="bold")
    return _save(fig, save_dir, "14_calibration_curves")


# --------------------------------------------------------------------------------------------
# Explainability comparison
# --------------------------------------------------------------------------------------------
def shap_explainability_comparison(final: FinalResult, save_dir=None, top: int = 12):
    """
    Graph 2: for each downstream model, mean |SHAP| on the explained test rows, normalised per method
    to sum to 1, for the features most important on average.
    """
    fig, axes = plt.subplots(1, len(C.MODEL_ORDER), figsize=(6.2 * len(C.MODEL_ORDER), 0.42 * top + 2.6))
    for ax, model in zip(axes, C.MODEL_ORDER):
        imp = final.importance_table(model)
        imp = imp.div(imp.sum(axis=1).replace(0, 1), axis=0)
        cols = imp.mean().sort_values(ascending=False).index[:top]
        sns.heatmap(imp[cols].T, cmap="Blues", ax=ax, annot=True, fmt=".2f", annot_kws={"fontsize": 7},
                    cbar=False, vmin=0)
        ax.set_title(model)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        for t in ax.get_xticklabels():
            t.set_ha("right")
    fig.suptitle("Graph 2 - Share of mean |SHAP| per feature in the final models (test rows)", fontweight="bold")
    return _save(fig, save_dir, "15_graph2_shap_explainability_comparison")


def attribution_stability_plot(stab: pd.DataFrame, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    d = stab.reset_index()
    sns.barplot(data=d, x="model", y="kendall_tau_mean", hue="method", hue_order=C.METHOD_ORDER,
                palette=C.METHOD_COLORS, ax=ax)
    ax.set_ylabel("Mean pairwise Kendall τ across folds")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Stability of the downstream models' SHAP feature rankings across CV folds")
    return _save(fig, save_dir, "16_attribution_stability")
