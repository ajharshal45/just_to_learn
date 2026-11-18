"""
Synthetic Data Statistical Fidelity Evaluation.

Quantifies distributional divergence between real minority samples and synthetic instances:
- Marginal Wasserstein Distance (mean & maximum)
- Marginal Kolmogorov-Smirnov (KS) two-sample tests
- Covariance & Correlation Matrix Preservation
"""

from typing import Dict, Any, List
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp, pearsonr


def compute_synthetic_fidelity(
    real_minority: np.ndarray,
    synthetic_minority: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluates statistical fidelity between real minority points and synthetic instances.

    Parameters
    ----------
    real_minority : np.ndarray of shape (N_real, p)
        Real minority samples from training fold.
    synthetic_minority : np.ndarray of shape (N_synth, p)
        Generated synthetic minority samples.

    Returns
    -------
    dict containing:
      - mean_wasserstein_distance
      - max_wasserstein_distance
      - ks_test_pass_ratio (fraction of features with p >= 0.05)
      - covariance_frobenius_dist
      - correlation_matrix_similarity (Pearson correlation between upper triangles)
    """
    real = np.asarray(real_minority, dtype=np.float64)
    synth = np.asarray(synthetic_minority, dtype=np.float64)

    if real.shape[0] < 2 or synth.shape[0] < 2:
        return {
            "mean_wasserstein_distance": 0.0,
            "max_wasserstein_distance": 0.0,
            "ks_test_pass_ratio": 1.0,
            "covariance_frobenius_dist": 0.0,
            "correlation_matrix_similarity": 1.0,
        }

    n_features = real.shape[1]
    w_distances = []
    ks_pvals = []

    for j in range(n_features):
        wd = wasserstein_distance(real[:, j], synth[:, j])
        w_distances.append(wd)

        ks_stat = ks_2samp(real[:, j], synth[:, j])
        ks_pvals.append(ks_stat.pvalue)

    mean_wd = float(np.mean(w_distances))
    max_wd = float(np.max(w_distances))
    ks_pass = float(np.mean(np.array(ks_pvals) >= 0.05))

    # Covariance preservation
    cov_real = np.cov(real, rowvar=False)
    cov_synth = np.cov(synth, rowvar=False)
    cov_frob = float(np.linalg.norm(cov_real - cov_synth, ord="fro"))

    # Correlation matrix preservation
    corr_real = np.corrcoef(real, rowvar=False)
    corr_synth = np.corrcoef(synth, rowvar=False)

    # Handle NaNs if zero variance
    corr_real = np.nan_to_num(corr_real, nan=0.0)
    corr_synth = np.nan_to_num(corr_synth, nan=0.0)

    triu_indices = np.triu_indices(n_features, k=1)
    if len(triu_indices[0]) > 0:
        tri_real = corr_real[triu_indices]
        tri_synth = corr_synth[triu_indices]
        # Pearson correlation between correlation matrix entries
        if np.std(tri_real) > 1e-8 and np.std(tri_synth) > 1e-8:
            corr_sim, _ = pearsonr(tri_real, tri_synth)
            corr_sim = float(corr_sim)
        else:
            corr_sim = 1.0 - min(1.0, float(np.mean(np.abs(tri_real - tri_synth))))
    else:
        corr_sim = 1.0

    return {
        "mean_wasserstein_distance": mean_wd,
        "max_wasserstein_distance": max_wd,
        "ks_test_pass_ratio": ks_pass,
        "covariance_frobenius_dist": cov_frob,
        "correlation_matrix_similarity": corr_sim,
    }
