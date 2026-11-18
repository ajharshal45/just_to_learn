"""
Unit tests for Explanation Stability and Synthetic Statistical Fidelity.
"""

import pytest
import numpy as np

from src.evaluation.stability import (
    compute_pairwise_rank_stability,
    compute_top_k_rank_overlap,
    evaluate_explanation_stability,
)
from src.evaluation.fidelity import compute_synthetic_fidelity
from src.evaluation.visualization import generate_latex_table
import pandas as pd


def test_stability_evaluation():
    # Two identical rankings should give Kendall tau = 1.0 and Jaccard = 1.0
    r1 = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    r2 = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

    res = evaluate_explanation_stability([r1, r2], top_k=3)
    assert np.isclose(res["kendall_tau_mean"], 1.0)
    assert np.isclose(res["spearman_rho_mean"], 1.0)
    assert np.isclose(res["top_k_overlap"], 1.0)


def test_fidelity_evaluation():
    rng = np.random.default_rng(42)
    real = rng.normal(0, 1, size=(50, 4))
    synth = rng.normal(0, 1, size=(50, 4))

    fidelity = compute_synthetic_fidelity(real, synth)
    assert "mean_wasserstein_distance" in fidelity
    assert "ks_test_pass_ratio" in fidelity
    assert "correlation_matrix_similarity" in fidelity
    assert fidelity["mean_wasserstein_distance"] >= 0.0
    assert 0.0 <= fidelity["ks_test_pass_ratio"] <= 1.0


def test_latex_table_generation():
    df = pd.DataFrame([{
        "oversampler": "SHAP-Weighted MASV-SMOTE",
        "classifier": "Random Forest",
        "pr_auc_mean": 0.8523,
        "roc_auc_mean": 0.9124,
        "minority_f1_mean": 0.7845,
        "ece_mean": 0.0412,
        "attribution_kendall_tau_mean": 0.8241,
    }])
    latex_code = generate_latex_table(df)
    assert "\\begin{table}" in latex_code or "\\tabular" in latex_code or "Oversampler" in latex_code
    assert "SHAP-Weighted" in latex_code
