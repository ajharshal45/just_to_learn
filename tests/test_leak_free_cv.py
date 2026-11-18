"""
Unit tests for Leak-Free Cross-Validation pipeline and evaluation metrics.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.oversampling.shap_filtered_smote import SHAPFilteredSMOTE
from src.oversampling.shap_weighted_smote import SHAPWeightedSMOTE
from src.pipeline.leak_free_cv import LeakFreePipeline, run_leak_free_benchmark
from src.evaluation.metrics import evaluate_multi_dimensional_metrics, compute_expected_calibration_error


def test_leak_free_pipeline_fit_predict():
    X, y = make_classification(
        n_samples=150,
        n_features=6,
        weights=[0.8, 0.2],
        random_state=42,
    )

    oversampler = SHAPWeightedSMOTE(random_state=42)
    classifier = LogisticRegression(solver="liblinear", random_state=42)

    pipeline = LeakFreePipeline(oversampler=oversampler, classifier=classifier)
    pipeline.fit(X[:100], y[:100])

    preds = pipeline.predict(X[100:])
    probs = pipeline.predict_proba(X[100:])

    assert len(preds) == 50
    assert probs.shape == (50, 2)


def test_run_leak_free_benchmark():
    X, y = make_classification(
        n_samples=120,
        n_features=5,
        weights=[0.8, 0.2],
        random_state=42,
    )

    oversampler = SHAPFilteredSMOTE(top_k=3, random_state=42)
    classifier = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)

    results = run_leak_free_benchmark(
        X, y, oversampler, classifier, n_splits=3, n_repeats=1, random_state=42
    )

    assert "pr_auc_mean" in results
    assert "roc_auc_mean" in results
    assert "ece_mean" in results
    assert results["n_folds_evaluated"] == 3


def test_ece_computation():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.9, 0.9])

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0
