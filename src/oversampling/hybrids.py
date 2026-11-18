"""
Hybrid Noise-Cleaned Oversamplers.

Combines synthetic oversampling with data cleaning techniques:
- SMOTE-ENN: SMOTE followed by Edited Nearest Neighbours (ENN)
- SMOTE-Tomek: SMOTE followed by Tomek Links boundary cleaning
"""

from typing import Tuple, Union, Optional
import numpy as np
from sklearn.base import BaseEstimator


class SMOTEENNWrapper(BaseEstimator):
    """
    SMOTE-ENN wrapper: SMOTE combined with Edited Nearest Neighbours.
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str] = "auto",
        random_state: Optional[int] = None,
        smote_k_neighbors: int = 5,
        enn_n_neighbors: int = 3,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote_k_neighbors = smote_k_neighbors
        self.enn_n_neighbors = enn_n_neighbors

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours

        smote = SMOTE(k_neighbors=self.smote_k_neighbors, random_state=self.random_state)
        enn = EditedNearestNeighbours(n_neighbors=self.enn_n_neighbors)

        sampler = SMOTEENN(
            sampling_strategy=self.sampling_strategy,
            smote=smote,
            enn=enn,
            random_state=self.random_state,
        )
        return sampler.fit_resample(X, y)


class SMOTETomekWrapper(BaseEstimator):
    """
    SMOTE-Tomek wrapper: SMOTE combined with Tomek Links removal.
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str] = "auto",
        random_state: Optional[int] = None,
        smote_k_neighbors: int = 5,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote_k_neighbors = smote_k_neighbors

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(k_neighbors=self.smote_k_neighbors, random_state=self.random_state)
        sampler = SMOTETomek(
            sampling_strategy=self.sampling_strategy,
            smote=smote,
            random_state=self.random_state,
        )
        return sampler.fit_resample(X, y)
