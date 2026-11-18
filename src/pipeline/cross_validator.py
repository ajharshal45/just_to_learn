"""
Cross-Validation Execution Engine (Zero Data Leakage).

Encapsulates repeated stratified nested cross-validation, timing profiling,
fold metric aggregation, and explanation stability tracking.
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
from sklearn.base import BaseEstimator
from src.pipeline.leak_free_cv import LeakFreePipeline, run_leak_free_benchmark


class LeakFreeCrossValidator:
    """
    Leak-Free Cross-Validation Runner for benchmark experiments.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 1,
        random_state: int = 42,
        scale_features: bool = True,
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scale_features = scale_features

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        oversampler: BaseEstimator,
        classifier: BaseEstimator,
    ) -> Dict[str, Any]:
        """
        Executes leak-free evaluation for a single (oversampler, classifier) pair.
        """
        start_time = time.perf_counter()
        results = run_leak_free_benchmark(
            X=X,
            y=y,
            oversampler=oversampler,
            classifier=classifier,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scale_features=self.scale_features,
        )
        wall_time = time.perf_counter() - start_time
        results["wall_clock_time_seconds"] = wall_time
        return results


__all__ = ["LeakFreeCrossValidator", "LeakFreePipeline", "run_leak_free_benchmark"]
