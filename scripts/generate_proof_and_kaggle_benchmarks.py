"""
Comprehensive Empirical Proof and Kaggle Benchmark Engine.

Addresses all 4 Project Objectives:
  1. Objective 1 & 2 Proof: Geometric manifold comparison & alternative weighting schemes
     (Unweighted vs Gini vs Mutual Info vs TreeSHAP MASV).
  2. Objective 3 Proof: Empirical demonstration of Data Leakage Trap (Leaked vs Leak-Free).
  3. Objective 4 Proof: Multi-dimensional evaluation, PR/ROC curves, calibration diagrams,
     explanation stability across folds, and paired statistical significance tests.
  4. Real-world validation on the official Kaggle Stroke Prediction Dataset.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.datasets import make_classification, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.oversampling import (
    SHAPWeightedSMOTE,
    SHAPFilteredSMOTE,
    ClassicalSMOTE,
    BorderlineSMOTEWrapper,
    RawNoBalancing,
    compute_masv_weights,
)
from src.pipeline.datasets import load_benchmark_dataset
from src.pipeline.leak_free_cv import LeakFreePipeline, run_leak_free_benchmark
from src.evaluation.metrics import evaluate_multi_dimensional_metrics, compute_expected_calibration_error


def generate_feature_importance_weights(X, y, method="shap", random_state=42):
    """Computes normalized feature weights using SHAP, Gini, or Mutual Information."""
    p = X.shape[1]
    if method == "shap":
        weights, masv, _ = compute_masv_weights(X, y, random_state=random_state)
        return weights
    elif method == "gini":
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=random_state, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
        sum_imp = np.sum(importances)
        return importances / sum_imp if sum_imp > 0 else np.full(p, 1.0 / p)
    elif method == "mutual_info":
        mi = mutual_info_classif(X, y, random_state=random_state)
        sum_mi = np.sum(mi)
        return mi / sum_mi if sum_mi > 0 else np.full(p, 1.0 / p)
    elif method == "unweighted":
        return np.full(p, 1.0 / p)
    else:
        raise ValueError(f"Unknown weighting method: {method}")


class FeatureWeightedSMOTE:
    """Generalized Feature-Weighted SMOTE supporting SHAP, Gini, Mutual Info, or Unweighted."""
    def __init__(self, method="shap", k_neighbors=5, random_state=42):
        self.method = method
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        weights = generate_feature_importance_weights(X_arr, y_arr, method=self.method, random_state=self.random_state)
        self.feature_weights_ = weights

        classes, counts = np.unique(y_arr, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]

        minority_mask = y_arr == minority_class
        X_min = X_arr[minority_mask]
        n_min = len(X_min)
        n_to_gen = max(0, np.sum(y_arr == majority_class) - n_min)

        if n_to_gen == 0 or n_min < 2:
            return X_arr.copy(), y_arr.copy()

        # Weighted distance search via sqrt(w) projection
        sqrt_w = np.sqrt(weights)
        X_min_weighted = X_min * sqrt_w

        k = min(self.k_neighbors, n_min - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
        nn.fit(X_min_weighted)
        neighbors_idx = nn.kneighbors(X_min_weighted, return_distance=False)[:, 1:]

        synthetic = np.zeros((n_to_gen, X_arr.shape[1]), dtype=np.float64)
        for i in range(n_to_gen):
            base_idx = rng.integers(0, n_min)
            nn_choice = rng.integers(0, k)
            neighbor_idx = neighbors_idx[base_idx, nn_choice]
            lambd = rng.uniform(0.0, 1.0)
            synthetic[i] = X_min[base_idx] + lambd * (X_min[neighbor_idx] - X_min[base_idx])

        X_res = np.vstack([X_arr, synthetic])
        y_res = np.concatenate([y_arr, np.full(n_to_gen, minority_class, dtype=y_arr.dtype)])
        return X_res, y_res


# -----------------------------------------------------------------------------
# Proof 1: Geometric Decision Boundary & Synthetic Manifold Distortion
# -----------------------------------------------------------------------------
def run_geometric_proof(output_dir: Path):
    print("\n[+] Generating Proof 1: Geometric Manifold & Decision Boundary Comparison...")
    # Generate 2D non-linear crescent moon imbalanced distribution with subtle noise
    X_raw, y_raw = make_moons(n_samples=500, noise=0.22, random_state=42)
    # Impose 90:10 imbalance ratio
    maj_idx = np.where(y_raw == 0)[0]
    min_idx = np.where(y_raw == 1)[0][:35]  # 35 minority points vs 250 majority
    keep_idx = np.concatenate([maj_idx, min_idx])
    X = X_raw[keep_idx]
    y = y_raw[keep_idx]

    # Add 2 pure noise dimensions to evaluate feature agnosticism
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, size=(len(X), 2))
    X_high = np.hstack([X, noise])

    samplers = {
        "Raw (No Balancing)": RawNoBalancing(),
        "Standard SMOTE": ClassicalSMOTE(k_neighbors=4, random_state=42),
        "Borderline-SMOTE": BorderlineSMOTEWrapper(k_neighbors=4, m_neighbors=6, random_state=42),
        "SHAP-Weighted MASV-SMOTE": SHAPWeightedSMOTE(k_neighbors=4, random_state=42),
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    # Zero pad noise for grid evaluation
    grid_high = np.hstack([grid_2d, np.zeros((len(grid_2d), 2))])

    for ax, (title, sampler) in zip(axes, samplers.items()):
        X_res, y_res = sampler.fit_resample(X_high, y)
        n_synth = len(X_res) - len(X_high)

        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X_res, y_res)
        Z = clf.predict_proba(grid_high)[:, 1].reshape(xx.shape)

        # Plot decision contour
        ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5, linestyles="--")

        # Plot original points
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#2b5c8f", marker="o", s=25, label="Majority Class (Real)", alpha=0.7)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#d9381e", marker="^", s=50, edgecolors="black", label="Minority Class (Real)")

        # Plot synthetic points
        if n_synth > 0:
            X_synth = X_res[len(X_high):]
            ax.scatter(X_synth[:, 0], X_synth[:, 1], c="#ffc107", marker="*", s=60, edgecolors="black", label=f"Synthetic (n={n_synth})")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.suptitle("Proof of Objective 1 & 2: Synthetic Placement and Decision Manifold Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = output_dir / "fig1_geometric_decision_boundary_proof.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [->] Saved: {out_file}")


# -----------------------------------------------------------------------------
# Proof 2: The Data Leakage Trap (Objective 3 Proof)
# -----------------------------------------------------------------------------
def run_data_leakage_proof(output_dir: Path):
    print("\n[+] Generating Proof 2: The Data Leakage Trap (Leaked vs. Zero-Leakage)...")
    # Generate imbalanced dataset with 15 features (90:10 imbalance)
    X, y = make_classification(
        n_samples=600, n_features=15, n_informative=4, n_redundant=3,
        weights=[0.90, 0.10], flip_y=0.02, random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Flawed Leaked Pipeline: Over-sampling and scaling done globally BEFORE CV split
    scaler_leaked = StandardScaler()
    X_leaked_scaled = scaler_leaked.fit_transform(X)
    sampler_leaked = SHAPWeightedSMOTE(random_state=42)
    X_leaked_res, y_leaked_res = sampler_leaked.fit_resample(X_leaked_scaled, y)

    leaked_pr_aucs = []
    leaked_roc_aucs = []
    # Test evaluated on subsets of contaminated dataset
    skf_leaked = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf_leaked.split(X_leaked_res, y_leaked_res):
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X_leaked_res[tr_idx], y_leaked_res[tr_idx])
        probs = clf.predict_proba(X_leaked_res[te_idx])[:, 1]
        leaked_pr_aucs.append(average_precision_score(y_leaked_res[te_idx], probs))
        leaked_roc_aucs.append(roc_auc_score(y_leaked_res[te_idx], probs))

    # 2. Strict Leak-Free Pipeline: Everything inside inner CV training fold
    leak_free_pr_aucs = []
    leak_free_roc_aucs = []
    for tr_idx, te_idx in skf.split(X, y):
        pipeline = LeakFreePipeline(
            oversampler=SHAPWeightedSMOTE(random_state=42),
            classifier=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
            scale_features=True,
        )
        pipeline.fit(X[tr_idx], y[tr_idx])
        probs = pipeline.predict_proba(X[te_idx])[:, 1]
        leak_free_pr_aucs.append(average_precision_score(y[te_idx], probs))
        leak_free_roc_aucs.append(roc_auc_score(y[te_idx], probs))

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    metrics_names = ["Leaked Pipeline\n(Flawed Literature)", "Strict Leak-Free Pipeline\n(Our Protocol)"]
    pr_means = [np.mean(leaked_pr_aucs), np.mean(leak_free_pr_aucs)]
    pr_stds = [np.std(leaked_pr_aucs), np.std(leak_free_pr_aucs)]
    roc_means = [np.mean(leaked_roc_aucs), np.mean(leak_free_roc_aucs)]
    roc_stds = [np.std(leaked_roc_aucs), np.std(leak_free_roc_aucs)]

    colors = ["#e74c3c", "#27ae60"]

    bars1 = ax1.bar(metrics_names, pr_means, yerr=pr_stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax1.set_ylabel("PR-AUC (Average Precision)", fontsize=12, fontweight="bold")
    ax1.set_title("PR-AUC: Artificial Over-Optimism from Data Leakage", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.03, f"{yval:.4f}", ha="center", fontweight="bold")

    bars2 = ax2.bar(metrics_names, roc_means, yerr=roc_stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax2.set_ylabel("ROC-AUC", fontsize=12, fontweight="bold")
    ax2.set_title("ROC-AUC: Artificially Inflated Metric Distortion", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.03, f"{yval:.4f}", ha="center", fontweight="bold")

    plt.suptitle("Proof of Objective 3: The Data Leakage Trap (Validation of Zero-Leakage Architecture)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = output_dir / "fig2_data_leakage_trap_proof.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [->] Saved: {out_file}")

    return {
        "leaked_pr_auc": float(np.mean(leaked_pr_aucs)),
        "leak_free_pr_auc": float(np.mean(leak_free_pr_aucs)),
        "inflation_delta_pr_auc": float(np.mean(leaked_pr_aucs) - np.mean(leak_free_pr_aucs)),
    }


# -----------------------------------------------------------------------------
# Proof 3: Comparison with Existing SHAP and Feature Weighting Schemes
# -----------------------------------------------------------------------------
def run_weighting_schemes_comparison(X_real, y_real, output_dir: Path):
    print("\n[+] Generating Proof 3: Comparison with Existing SHAP & Weighting Paradigms...")
    # Evaluate across 4 weighting mechanisms under strict 5-fold CV
    methods = {
        "Standard Euclidean (Unweighted)": FeatureWeightedSMOTE(method="unweighted", random_state=42),
        "Random Forest Gini Impurity": FeatureWeightedSMOTE(method="gini", random_state=42),
        "Mutual Information (MI)": FeatureWeightedSMOTE(method="mutual_info", random_state=42),
        "TreeSHAP MASV (Our Approach)": FeatureWeightedSMOTE(method="shap", random_state=42),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {m: {"pr_auc": [], "f1": [], "ece": []} for m in methods}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_real, y_real)):
        X_tr, X_te = X_real[tr_idx], X_real[te_idx]
        y_tr, y_te = y_real[tr_idx], y_real[te_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        for m_name, sampler in methods.items():
            X_res, y_res = sampler.fit_resample(X_tr_sc, y_tr)
            clf = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1)
            clf.fit(X_res, y_res)
            probs = clf.predict_proba(X_te_sc)[:, 1]
            preds = clf.predict(X_te_sc)

            res = evaluate_multi_dimensional_metrics(clf, X_te_sc, y_te)
            results[m_name]["pr_auc"].append(res["pr_auc"])
            results[m_name]["f1"].append(res["minority_f1"])
            results[m_name]["ece"].append(res["ece"])

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = list(methods.keys())
    short_names = ["Euclidean\n(SMOTE)", "RF Gini\nImportance", "Mutual\nInformation", "TreeSHAP MASV\n(Ours)"]

    metrics = [("pr_auc", "PR-AUC (Average Precision)", axes[0], True),
               ("f1", "Minority F1-Score", axes[1], True),
               ("ece", "Expected Calibration Error (ECE)", axes[2], False)]

    colors = ["#95a5a6", "#3498db", "#9b59b6", "#e67e22"]

    for key, title, ax, higher_better in metrics:
        means = [np.mean(results[m][key]) for m in names]
        stds = [np.std(results[m][key]) for m in names]
        bars = ax.bar(short_names, means, yerr=stds, capsize=4, color=colors, alpha=0.88, edgecolor="black")
        ax.set_ylabel(title, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval * 1.02, f"{yval:.4f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Proof of Objective 2: Feature Weighting Paradigm Comparison on Kaggle Stroke Dataset", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = output_dir / "fig3_weighting_schemes_comparison.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [->] Saved: {out_file}")

    return results


# -----------------------------------------------------------------------------
# Proof 4: Full Kaggle Benchmark with PR Curves, Calibration, & Significance
# -----------------------------------------------------------------------------
def run_kaggle_stroke_benchmark(output_dir: Path):
    print("\n[+] Loading official Kaggle Stroke Prediction Dataset...")
    X, y, meta = load_benchmark_dataset("kaggle_stroke", subsample_n=3000, random_state=42)
    print(f"    Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, Minority ratio: {meta['minority_ratio']:.2%}")

    # Run Weighting Schemes comparison
    weight_res = run_weighting_schemes_comparison(X, y, output_dir)

    # Now evaluate Full Suite on Kaggle Stroke
    samplers = {
        "Raw (No Balancing)": RawNoBalancing(),
        "Standard SMOTE": ClassicalSMOTE(k_neighbors=5, random_state=42),
        "Borderline-SMOTE": BorderlineSMOTEWrapper(k_neighbors=5, random_state=42),
        "SHAP-Filtered SMOTE": SHAPFilteredSMOTE(cumulative_threshold=0.85, random_state=42),
        "SHAP-Weighted MASV-SMOTE": SHAPWeightedSMOTE(random_state=42),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = {s: [] for s in samplers}
    curve_data = {s: {"probs": [], "y_true": []} for s in samplers}
    tau_stability = {s: [] for s in samplers}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        for s_name, sampler in samplers.items():
            X_res, y_res = sampler.fit_resample(X_tr_sc, y_tr)
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            clf.fit(X_res, y_res)

            probs = clf.predict_proba(X_te_sc)[:, 1]
            m = evaluate_multi_dimensional_metrics(clf, X_te_sc, y_te)
            fold_results[s_name].append(m)

            curve_data[s_name]["probs"].extend(probs.tolist())
            curve_data[s_name]["y_true"].extend(y_te.tolist())

            # Local feature importances
            tau_stability[s_name].append(clf.feature_importances_)

    # 1. Precision-Recall Curves & Calibration Diagram
    print("\n[+] Generating Proof 4: Precision-Recall & Calibration Reliability Curves...")
    fig, (ax_pr, ax_cal) = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "Raw (No Balancing)": "#7f8c8d",
        "Standard SMOTE": "#e74c3c",
        "Borderline-SMOTE": "#9b59b6",
        "SHAP-Filtered SMOTE": "#2980b9",
        "SHAP-Weighted MASV-SMOTE": "#27ae60",
    }

    for s_name, data in curve_data.items():
        y_t = np.array(data["y_true"])
        p_t = np.array(data["probs"])

        # PR Curve
        prec, rec, _ = precision_recall_curve(y_t, p_t)
        ap = average_precision_score(y_t, p_t)
        ax_pr.plot(rec, prec, label=f"{s_name} (AP = {ap:.3f})", color=colors[s_name], linewidth=2.2)

        # Calibration Curve (Reliability Diagram)
        bins = np.linspace(0.0, 1.0, 11)
        bin_confs = []
        bin_accs = []
        for i in range(len(bins) - 1):
            mask = (p_t >= bins[i]) & (p_t < bins[i+1])
            if np.sum(mask) > 0:
                bin_confs.append(np.mean(p_t[mask]))
                bin_accs.append(np.mean(y_t[mask]))
        ax_cal.plot(bin_confs, bin_accs, marker="o", label=f"{s_name}", color=colors[s_name], linewidth=2)

    ax_pr.set_xlabel("Recall (Sensitivity to Stroke Events)", fontsize=12, fontweight="bold")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    ax_pr.set_title("Precision-Recall Curves on Kaggle Stroke Dataset", fontsize=13, fontweight="bold")
    ax_pr.legend(fontsize=10, loc="upper right")
    ax_pr.grid(True, linestyle=":", alpha=0.6)

    # Perfect calibration reference
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration")
    ax_cal.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold")
    ax_cal.set_ylabel("Empirical Fraction of Positives", fontsize=12, fontweight="bold")
    ax_cal.set_title("Probability Calibration Reliability Diagram", fontsize=13, fontweight="bold")
    ax_cal.legend(fontsize=10, loc="upper left")
    ax_cal.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Proof of Objective 4: Clinical Discrimination & Calibration on Kaggle Dataset", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = output_dir / "fig4_kaggle_stroke_pr_calibration_curves.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [->] Saved: {out_file}")

    # 2. Statistical Significance Tests & Attribution Stability
    print("\n[+] Calculating Paired Statistical Significance & Feature Stability...")
    baseline_pr = [m["pr_auc"] for m in fold_results["Standard SMOTE"]]
    shap_weighted_pr = [m["pr_auc"] for m in fold_results["SHAP-Weighted MASV-SMOTE"]]
    shap_filtered_pr = [m["pr_auc"] for m in fold_results["SHAP-Filtered SMOTE"]]

    t_stat, p_val_ttest = stats.ttest_rel(shap_weighted_pr, baseline_pr)
    wilc_stat, p_val_wilcoxon = stats.wilcoxon(shap_weighted_pr, baseline_pr)

    # Attribution stability (Kendall's tau pairwise across folds)
    taus_per_sampler = {}
    for s_name, rankings in tau_stability.items():
        t_vals = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                tau, _ = stats.kendalltau(rankings[i], rankings[j])
                if not np.isnan(tau):
                    t_vals.append(tau)
        taus_per_sampler[s_name] = t_vals

    fig, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot of PR-AUC across folds
    box_data = [[m["pr_auc"] for m in fold_results[s]] for s in samplers]
    bp = ax_box.boxplot(box_data, patch_artist=True, labels=[s.replace(" ", "\n") for s in samplers])
    palette = ["#7f8c8d", "#e74c3c", "#9b59b6", "#2980b9", "#27ae60"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax_box.set_ylabel("PR-AUC (5-Fold CV)", fontsize=11, fontweight="bold")
    ax_box.set_title("Cross-Fold PR-AUC Dispersion\n(Paired Wilcoxon p = {:.4f})".format(p_val_wilcoxon), fontsize=12, fontweight="bold")
    ax_box.grid(axis="y", linestyle=":", alpha=0.6)

    # Feature rank stability bar chart
    tau_means = [np.mean(taus_per_sampler[s]) for s in samplers]
    tau_stds = [np.std(taus_per_sampler[s]) for s in samplers]
    bars = ax_bar.bar([s.replace(" ", "\n") for s in samplers], tau_means, yerr=tau_stds, capsize=4, color=palette, alpha=0.85, edgecolor="black")
    ax_bar.set_ylabel("Pairwise Kendall's Tau Stability", fontsize=11, fontweight="bold")
    ax_bar.set_title("Downstream Explanation Stability Across CV Folds", fontsize=12, fontweight="bold")
    ax_bar.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", fontweight="bold")

    plt.suptitle("Proof of Objective 4: Statistical Significance and Feature Attribution Stability", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file2 = output_dir / "fig5_statistical_significance_and_stability.png"
    fig.savefig(out_file2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    [->] Saved: {out_file2}")

    return {
        "p_val_paired_ttest": float(p_val_ttest),
        "p_val_wilcoxon": float(p_val_wilcoxon),
        "fold_results": fold_results,
        "tau_stability": {s: float(np.mean(taus_per_sampler[s])) for s in samplers},
    }


def main():
    proof_dir = PROJECT_ROOT / "docs" / "images" / "proofs"
    proof_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OBJECTIVES SCIENTIFIC PROOF AND KAGGLE BENCHMARK SUITE")
    print(f"Output directory for proof figures: {proof_dir}")
    print("=" * 80)

    # Run Proofs 1, 2, and 4
    run_geometric_proof(proof_dir)
    leakage_stats = run_data_leakage_proof(proof_dir)
    kaggle_stats = run_kaggle_stroke_benchmark(proof_dir)

    print("\n" + "=" * 80)
    print("ALL PROOFS AND BENCHMARKS GENERATED SUCCESSFULLY!")
    print("Summary of Statistical Proofs:")
    print(f"  - Data Leakage Inflation Delta: +{leakage_stats['inflation_delta_pr_auc']:.4f} PR-AUC")
    print(f"  - Paired Wilcoxon p-value (SHAP-Weighted vs SMOTE): p = {kaggle_stats['p_val_wilcoxon']:.4f}")
    print(f"  - Explanation Stability (Kendall's Tau):")
    for s, tau in kaggle_stats["tau_stability"].items():
        print(f"      * {s:<26}: {tau:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
