"""
Unit tests for benchmark dataset loaders.
"""

import pytest
import numpy as np

from src.pipeline.datasets import (
    generate_synthetic_imbalanced,
    load_breast_cancer_diagnostic,
    generate_credit_fraud_simulation,
    generate_nsl_kdd_simulation,
    load_benchmark_dataset,
    list_benchmark_datasets,
)


def test_list_benchmark_datasets():
    datasets = list_benchmark_datasets()
    assert "synthetic" in datasets
    assert "breast_cancer" in datasets
    assert "credit_fraud" in datasets
    assert "nsl_kdd" in datasets


def test_generate_synthetic_imbalanced():
    X, y, meta = generate_synthetic_imbalanced(n_samples=200, n_features=10, minority_ratio=0.1)
    assert X.shape == (200, 10)
    assert len(y) == 200
    assert np.sum(y == 1) < np.sum(y == 0)
    assert meta["domain"] == "Synthetic Simulation"


def test_load_breast_cancer_diagnostic():
    X, y, meta = load_breast_cancer_diagnostic(minority_ratio=0.15)
    assert X.shape[0] == len(y)
    assert X.shape[1] == 30
    assert np.sum(y == 1) < np.sum(y == 0)
    assert meta["domain"] == "Medical Diagnostics"


def test_generate_credit_fraud_simulation():
    X, y, meta = generate_credit_fraud_simulation(n_samples=300, n_features=15, minority_ratio=0.05)
    assert X.shape == (300, 15)
    assert len(y) == 300
    assert np.sum(y == 1) < np.sum(y == 0)
    assert meta["domain"] == "Financial Fraud Detection"


def test_unified_loader():
    X, y, meta = load_benchmark_dataset("synthetic", random_state=42)
    assert len(X) == len(y)
    with pytest.raises(ValueError):
        load_benchmark_dataset("non_existent_dataset")


def test_load_kaggle_stroke_dataset():
    from src.pipeline.datasets import load_kaggle_stroke_dataset
    X, y, meta = load_kaggle_stroke_dataset(subsample_n=200, random_state=42)
    assert X.shape[0] == 200
    assert X.shape[1] == 16
    assert np.sum(y == 1) < np.sum(y == 0)
    assert meta["source"] == "Kaggle (fedesoriano/stroke-prediction-dataset)"
