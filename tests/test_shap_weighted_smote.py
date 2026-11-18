"""
Unit tests for SHAP-Weighted SMOTE (MASV-SMOTE) algorithm.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification

from src.oversampling.shap_weighted_smote import SHAPWeightedSMOTE


def test_shap_weighted_smote_resampling():
    X, y = make_classification(
        n_samples=250,
        n_features=12,
        n_informative=4,
        weights=[0.88, 0.12],
        random_state=42,
    )

    n_minority_before = np.sum(y == 1)
    n_majority_before = np.sum(y == 0)

    sampler = SHAPWeightedSMOTE(
        k_neighbors=5,
        random_state=42,
    )

    X_res, y_res = sampler.fit_resample(X, y)

    n_minority_after = np.sum(y_res == 1)
    n_majority_after = np.sum(y_res == 0)

    assert n_minority_after == n_majority_after
    assert X_res.shape[0] == n_majority_after * 2
    assert X_res.shape[1] == 12
    assert hasattr(sampler, "feature_weights_")
    assert np.isclose(np.sum(sampler.feature_weights_), 1.0)


def test_shap_weighted_smote_custom_sampling_ratio():
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=3,
        weights=[0.85, 0.15],
        random_state=42,
    )

    # 50% ratio: target minority = 0.5 * majority
    sampler = SHAPWeightedSMOTE(
        sampling_strategy=0.5,
        random_state=42,
    )

    X_res, y_res = sampler.fit_resample(X, y)

    n_majority = np.sum(y == 0)
    n_minority_after = np.sum(y_res == 1)

    assert n_minority_after == int(n_majority * 0.5)
