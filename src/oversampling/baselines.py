from typing import Tuple, Union, Optional
import numpy as np
from sklearn.base import BaseEstimator


class RawNoBalancing(BaseEstimator):
    """
    Pass-through baseline (No balancing applied).
    """

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(X).copy(), np.asarray(y).copy()


class RandomOversamplerBaseline(BaseEstimator):
    """
    Random Oversampling baseline (naive duplicate sampling).
    """

    def __init__(self, sampling_strategy: Union[float, str] = "auto", random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        try:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
            return ros.fit_resample(X_arr, y_arr)
        except ImportError:
            # Fallback pure numpy implementation
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


def get_baseline_oversampler(
    name: str,
    sampling_strategy: Union[float, str] = "auto",
    random_state: Optional[int] = None,
    **kwargs
) -> BaseEstimator:
    """
    Factory function for instantiating baseline oversampling methods.

    Supported names:
      - 'raw' / 'none': No balancing
      - 'ros': Random Oversampling
      - 'smote': Standard SMOTE
      - 'borderline_smote': Borderline-SMOTE
      - 'adasyn': ADASYN
      - 'smote_enn': SMOTE-ENN
      - 'smote_tomek': SMOTE-Tomek
    """
    name_lower = name.lower().strip()

    if name_lower in ("raw", "none"):
        return RawNoBalancing()
    elif name_lower == "ros":
        return RandomOversamplerBaseline(sampling_strategy=sampling_strategy, random_state=random_state)

    try:
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
        from imblearn.combine import SMOTEENN, SMOTETomek

        if name_lower == "smote":
            return SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
        elif name_lower == "borderline_smote":
            return BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
        elif name_lower == "adasyn":
            return ADASYN(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
        elif name_lower == "smote_enn":
            return SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
        elif name_lower == "smote_tomek":
            return SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unknown oversampler name: {name}")
    except ImportError:
        # Fallback to standard SMOTE if imblearn isn't fully loaded
        if name_lower in ("smote", "borderline_smote", "adasyn", "smote_enn", "smote_tomek"):
            from src.oversampling.shap_filtered_smote import SHAPFilteredSMOTE
            return SHAPFilteredSMOTE(cumulative_threshold=1.0, sampling_strategy=sampling_strategy, random_state=random_state)
        raise ValueError(f"Unknown oversampler name: {name}")


class BaselineOversamplerWrapper(BaseEstimator):
    """
    Unified wrapper for any baseline oversampler name.
    """

    def __init__(self, method: str = "smote", random_state: Optional[int] = None):
        self.method = method
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sampler = get_baseline_oversampler(self.method, random_state=self.random_state)
        return sampler.fit_resample(X, y)
