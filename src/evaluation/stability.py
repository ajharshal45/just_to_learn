"""
Feature Attribution Stability Analysis.

Evaluates whether synthetic oversampling stabilizes or perturbs feature attributions
across cross-validation folds using rank correlation coefficients:
- Kendall's Tau (pairwise & mean)
- Spearman's Rho (pairwise & mean)
- Top-k Rank Overlap (Jaccard similarity of top-k features across folds)
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.stats import kendalltau, spearmanr


def compute_pairwise_rank_stability(
    rankings: List[np.ndarray],
) -> Dict[str, float]:
    """
    Computes pairwise Kendall's Tau and Spearman's Rho correlation across folds.

    Parameters
    ----------
    rankings : list of np.ndarray
        List of 1D arrays representing feature importance vectors per fold.

    Returns
    -------
    dict with mean and standard deviation of Kendall's tau and Spearman's rho.
    """
    if len(rankings) < 2:
        return {
            "kendall_tau_mean": 0.0,
            "kendall_tau_std": 0.0,
            "spearman_rho_mean": 0.0,
            "spearman_rho_std": 0.0,
        }

    kendall_taus = []
    spearman_rhos = []

    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            r_i, r_j = rankings[i], rankings[j]
            tau, _ = kendalltau(r_i, r_j)
            rho, _ = spearmanr(r_i, r_j)
            if not np.isnan(tau):
                kendall_taus.append(tau)
            if not np.isnan(rho):
                spearman_rhos.append(rho)

    tau_mean = float(np.mean(kendall_taus)) if kendall_taus else 0.0
    tau_std = float(np.std(kendall_taus)) if kendall_taus else 0.0
    rho_mean = float(np.mean(spearman_rhos)) if spearman_rhos else 0.0
    rho_std = float(np.std(spearman_rhos)) if spearman_rhos else 0.0

    return {
        "kendall_tau_mean": tau_mean,
        "kendall_tau_std": tau_std,
        "spearman_rho_mean": rho_mean,
        "spearman_rho_std": rho_std,
    }


def compute_top_k_rank_overlap(
    rankings: List[np.ndarray],
    k: int = 5,
) -> float:
    """
    Computes mean pairwise Jaccard index of top-k ranked features across folds.

    Parameters
    ----------
    rankings : list of np.ndarray
        Feature attribution vectors per fold.
    k : int, default=5
        Number of top features to compare.

    Returns
    -------
    mean_jaccard : float between 0.0 and 1.0.
    """
    if len(rankings) < 2:
        return 1.0

    top_k_sets = []
    for r in rankings:
        top_indices = set(np.argsort(r)[::-1][:k])
        top_k_sets.append(top_indices)

    jaccards = []
    for i in range(len(top_k_sets)):
        for j in range(i + 1, len(top_k_sets)):
            s1, s2 = top_k_sets[i], top_k_sets[j]
            inter = len(s1.intersection(s2))
            union = len(s1.union(s2))
            jaccard = inter / union if union > 0 else 0.0
            jaccards.append(jaccard)

    return float(np.mean(jaccards)) if jaccards else 0.0


def evaluate_explanation_stability(
    rankings: List[np.ndarray],
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Comprehensive explanation stability evaluation summary.
    """
    res = compute_pairwise_rank_stability(rankings)
    res["top_k_overlap"] = compute_top_k_rank_overlap(rankings, k=top_k)
    return res
