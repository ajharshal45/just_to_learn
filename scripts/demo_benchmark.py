"""
Demonstration script executing zero-data-leakage benchmark comparison
between Baseline SMOTE, SHAP-Filtered SMOTE, and SHAP-Weighted MASV-SMOTE.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.oversampling.shap_filtered_smote import SHAPFilteredSMOTE
from src.oversampling.shap_weighted_smote import SHAPWeightedSMOTE
from src.oversampling.baselines import get_baseline_oversampler
from src.pipeline.leak_free_cv import run_leak_free_benchmark


def main():
    print("=" * 80)
    print("SHAP-SMOTE BENCHMARK DEMONSTRATION")
    print("Zero Data Leakage 5-Fold Stratified Cross-Validation")
    print("=" * 80)

    # Generate synthetic imbalanced tabular dataset (90:10 imbalance ratio)
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=5,
        n_redundant=4,
        n_repeated=0,
        weights=[0.90, 0.10],
        flip_y=0.01,
        random_state=42,
    )

    print(f"Dataset Shape: {X.shape}, Imbalance Ratio: {np.bincount(y)[0]}:{np.bincount(y)[1]}")
    print("-" * 80)

    oversamplers = {
        "Raw (No Balancing)": get_baseline_oversampler("raw"),
        "Standard SMOTE": get_baseline_oversampler("smote", random_state=42),
        "SHAP-Filtered SMOTE": SHAPFilteredSMOTE(cumulative_threshold=0.85, random_state=42),
        "SHAP-Weighted MASV-SMOTE": SHAPWeightedSMOTE(random_state=42),
    }

    classifiers = {
        "Logistic Regression": LogisticRegression(solver="liblinear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
    }

    header = f"{'Oversampler':<26} | {'Classifier':<20} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'F1-Score':<8} | {'ECE':<8} | {'Attribution Tau':<15}"
    print(header)
    print("-" * len(header))

    for o_name, oversampler in oversamplers.items():
        for c_name, classifier in classifiers.items():
            results = run_leak_free_benchmark(
                X,
                y,
                oversampler=oversampler,
                classifier=classifier,
                n_splits=5,
                n_repeats=1,
                random_state=42,
            )

            pr_auc = results["pr_auc_mean"]
            roc_auc = results["roc_auc_mean"]
            f1 = results["minority_f1_mean"]
            ece = results["ece_mean"]
            tau = results["attribution_kendall_tau_mean"]

            print(
                f"{o_name:<26} | {c_name:<20} | {pr_auc:<8.4f} | {roc_auc:<8.4f} | {f1:<8.4f} | {ece:<8.4f} | {tau:<15.4f}"
            )

    print("=" * 80)
    print("Benchmark execution completed successfully!")


if __name__ == "__main__":
    main()
