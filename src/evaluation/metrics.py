"""
Multi-dimensional evaluation metrics for imbalanced classification, probability calibration,
synthetic data statistical fidelity, and attribution stability.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
)
from scipy.stats import wasserstein_distance, ks_2samp


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE).

    ECE = sum_{b=1}^M (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Select samples within current bin boundary
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Geometric Mean (G-Mean) of recall across binary classes.
    G-Mean = sqrt(Recall_minority * Recall_majority)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    if len(classes) < 2:
        return 0.0

    recalls = []
    for cls in classes:
        rec = recall_score(y_true == cls, y_pred == cls, zero_division=0)
        recalls.append(rec)

    return float(np.sqrt(np.prod(recalls)))


def compute_distributional_fidelity(
    real_minority: np.ndarray,
    synthetic_minority: np.ndarray,
) -> Dict[str, float]:
    """
    Computes statistical divergence between real minority samples and synthetic instances.

    Returns mean Wasserstein distance and fraction of features passing Kolmogorov-Smirnov test (p >= 0.05).
    """
    real = np.asarray(real_minority, dtype=np.float64)
    synth = np.asarray(synthetic_minority, dtype=np.float64)

    n_features = real.shape[1]
    w_distances = []
    ks_p_values = []

    for j in range(n_features):
        wd = wasserstein_distance(real[:, j], synth[:, j])
        w_distances.append(wd)

        ks_res = ks_2samp(real[:, j], synth[:, j])
        ks_p_values.append(ks_res.pvalue)

    mean_wasserstein = float(np.mean(w_distances))
    ks_pass_ratio = float(np.mean(np.array(ks_p_values) >= 0.05))

    return {
        "mean_wasserstein_distance": mean_wasserstein,
        "ks_test_pass_ratio": ks_pass_ratio,
    }


def evaluate_multi_dimensional_metrics(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Computes multi-dimensional metric assessment matrix for a fitted classifier model on a test set.

    Metrics include:
      - PR-AUC (Average Precision)
      - ROC-AUC
      - Minority Recall
      - Minority Precision
      - F1 Score
      - Balanced Accuracy
      - G-Mean
      - Expected Calibration Error (ECE)
      - Brier Score
    """
    X_test_arr = np.asarray(X_test, dtype=np.float64)
    y_test_arr = np.asarray(y_test)

    classes, counts = np.unique(y_test_arr, return_counts=True)
    minority_class = classes[np.argmin(counts)]

    # Predictions
    y_pred = model.predict(X_test_arr)

    # Probabilities
    if hasattr(model, "predict_proba"):
        y_prob_raw = model.predict_proba(X_test_arr)
        if y_prob_raw.ndim == 2 and y_prob_raw.shape[1] == 2:
            # Get probability of minority class
            min_idx = list(classes).index(minority_class)
            y_prob = y_prob_raw[:, min_idx]
        else:
            y_prob = y_prob_raw.ravel()
    else:
        y_prob = y_pred.astype(float)

    y_binary = (y_test_arr == minority_class).astype(int)

    # Classification Metrics
    pr_auc = float(average_precision_score(y_binary, y_prob)) if len(np.unique(y_binary)) > 1 else 0.0
    roc_auc = float(roc_auc_score(y_binary, y_prob)) if len(np.unique(y_binary)) > 1 else 0.0
    rec = float(recall_score(y_binary, (y_pred == minority_class).astype(int), zero_division=0))
    prec = float(precision_score(y_binary, (y_pred == minority_class).astype(int), zero_division=0))
    f1 = float(f1_score(y_binary, (y_pred == minority_class).astype(int), zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_binary, (y_pred == minority_class).astype(int)))
    g_mean = compute_g_mean(y_test_arr, y_pred)

    # Probability Calibration
    ece = compute_expected_calibration_error(y_binary, y_prob)
    brier = float(brier_score_loss(y_binary, y_prob))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "minority_recall": rec,
        "minority_precision": prec,
        "minority_f1": f1,
        "balanced_accuracy": bal_acc,
        "g_mean": g_mean,
        "ece": ece,
        "brier_score": brier,
    }
