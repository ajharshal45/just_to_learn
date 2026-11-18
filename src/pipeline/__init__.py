"""
Leakage-Free Experimental Benchmarking Pipeline Architecture.
"""

from src.pipeline.datasets import (
    load_benchmark_dataset,
    list_benchmark_datasets,
    generate_synthetic_imbalanced,
    load_breast_cancer_diagnostic,
    generate_credit_fraud_simulation,
    generate_nsl_kdd_simulation,
)
from src.pipeline.models import get_classifier, list_available_classifiers
from src.pipeline.leak_free_cv import LeakFreePipeline, run_leak_free_benchmark
from src.pipeline.cross_validator import LeakFreeCrossValidator

__all__ = [
    "load_benchmark_dataset",
    "list_benchmark_datasets",
    "generate_synthetic_imbalanced",
    "load_breast_cancer_diagnostic",
    "generate_credit_fraud_simulation",
    "generate_nsl_kdd_simulation",
    "get_classifier",
    "list_available_classifiers",
    "LeakFreePipeline",
    "run_leak_free_benchmark",
    "LeakFreeCrossValidator",
]
