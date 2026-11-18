"""
Unit tests for SHAP-Filtered SMOTE algorithm.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.oversampling.shap_filtered_smote import SHAPFilteredSMOTE
from src.oversampling.base import compute_masv_weights


def test_masv_weights_computation():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        n_repeated=0,
        weights=[0.85, 0.15],
        random_state=42,
    )

    weights, masv, fitted_est = compute_masv_weights(X, y, random_state=42)

    assert len(weights) == 10
    assert len(masv) == 10
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)


def test_shap_filtered_smote_resampling():
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=3,
        weights=[0.9, 0.1],
        random_state=42,
    )

    n_minority_before = np.sum(y == 1)
    n_majority_before = np.sum(y == 0)

    assert n_minority_before < n_majority_before

    sampler = SHAPFilteredSMOTE(
        cumulative_threshold=0.85,
        k_neighbors=3,
        random_state=42,
    )

    X_res, y_res = sampler.fit_resample(X, y)

    n_minority_after = np.sum(y_res == 1)
    n_majority_after = np.sum(y_res == 0)

    assert n_minority_after == n_majority_after
    assert X_res.shape[0] == n_majority_after * 2
    assert X_res.shape[1] == 8
    assert hasattr(sampler, "feature_weights_")
    assert hasattr(sampler, "selected_feature_indices_")
    assert len(sampler.selected_feature_indices_) > 0


def test_shap_filtered_smote_top_k():
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        n_informative=2,
        weights=[0.8, 0.2],
        random_state=42,
    )

    sampler = SHAPFilteredSMOTE(top_k=3, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)

    assert len(sampler.selected_feature_indices_) == 3
    assert X_res.shape[0] > X.shape[0]
