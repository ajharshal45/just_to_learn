"""
Classical and Baseline Synthetic Oversamplers.

Includes:
- RawNoBalancing (pass-through)
- RandomOversampler (ROS)
- Standard SMOTE
- Borderline-SMOTE
- ADASYN
"""

from typing import Tuple, Union, Optional
import numpy as np
from sklearn.base import BaseEstimator


class RawNoBalancing(BaseEstimator):
    """
    Pass-through baseline (No balancing applied).
    """

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(X, dtype=np.float64).copy(), np.asarray(y).copy()


class RandomOversampler(BaseEstimator):
    """
    Random Oversampling baseline (naive duplicate sampling).
    """

    def __init__(self, sampling_strategy: Union[float, str] = "auto", random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        try:
            from imblearn.over_sampling import RandomOverSampler as ImbROS
            ros = ImbROS(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            return ros.fit_resample(X_arr, y_arr)
        except ImportError:
            classes, counts = np.unique(y_arr, return_counts=True)
            min_class = classes[np.argmin(counts)]
            maj_count = np.max(counts)
            min_count = np.min(counts)

            n_to_sample = maj_count - min_count
            if n_to_sample <= 0:
                return X_arr.copy(), y_arr.copy()

            min_indices = np.where(y_arr == min_class)[0]
            rng = np.random.default_rng(self.random_state)
            sampled_indices = rng.choice(min_indices, size=n_to_sample, replace=True)

            X_res = np.vstack([X_arr, X_arr[sampled_indices]])
            y_res = np.concatenate([y_arr, y_arr[sampled_indices]])
            return X_res, y_res


class ClassicalSMOTE(BaseEstimator):
    """
    Standard Euclidean SMOTE wrapper.
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str] = "auto",
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        return smote.fit_resample(X, y)


class BorderlineSMOTEWrapper(BaseEstimator):
    """
    Borderline-SMOTE wrapper (synthesizes only near decision boundaries).
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str] = "auto",
        k_neighbors: int = 5,
        m_neighbors: int = 10,
        kind: str = "borderline-1",
        random_state: Optional[int] = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.kind = kind
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from imblearn.over_sampling import BorderlineSMOTE
        bsmote = BorderlineSMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            m_neighbors=self.m_neighbors,
            kind=self.kind,
            random_state=self.random_state,
        )
        return bsmote.fit_resample(X, y)


class ADASYNWrapper(BaseEstimator):
    """
    ADASYN (Adaptive Synthetic) wrapper.
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str] = "auto",
        n_neighbors: int = 5,
        random_state: Optional[int] = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from imblearn.over_sampling import ADASYN
        adasyn = ADASYN(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        return adasyn.fit_resample(X, y)
