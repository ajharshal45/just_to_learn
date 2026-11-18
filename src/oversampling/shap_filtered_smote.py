"""
SHAP-Filtered SMOTE: Explainability-Aware Pre-Filtering Subspace Oversampler.
"""

from typing import Optional, Union, Tuple
import numpy as np

from src.oversampling.base import SHAPBaseOversampler, compute_masv_weights
from src.oversampling.provenance import (
    _ProvenanceSamplerMixin, empty_generation_log, interpolate_with_provenance, minority_knn,
)


class SHAPFilteredSMOTE(_ProvenanceSamplerMixin, SHAPBaseOversampler):
    """
    SHAP-Filtered SMOTE (Pre-Filtering Subspace Variant).

    Filters uninformative noise dimensions using Mean Absolute SHAP Values (MASV)
    calculated on the inner training set prior to nearest-neighbour graph construction
    and synthetic interpolation.

    Parameters
    ----------
    top_k : int, optional
        Exact number of top features to retain based on MASV rank.
    top_ratio : float, optional
        Fraction of top features (0.0 < top_ratio <= 1.0) to retain.
    cumulative_threshold : float, default=0.90
        Cumulative MASV threshold (0.0 < threshold <= 1.0) for feature selection.
        Features are included in descending order until cumulative attributions reach threshold.
    sampling_strategy : float or str, default="auto"
        Target ratio of minority to majority samples post-resampling.
    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE.
    base_estimator : BaseEstimator, optional
        Preliminary estimator for SHAP attribution.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    feature_weights_ : np.ndarray of shape (p,)
        Normalized MASV feature weights.
    masv_ : np.ndarray of shape (p,)
        Unnormalized Mean Absolute SHAP Values.
    selected_feature_indices_ : np.ndarray
        Indices of features retained for subspace SMOTE.
    shap_values_ : np.ndarray of shape (N, p)
        Per-row SHAP values of the preliminary model on the training data passed to fit_resample.
    generation_log_, neighbor_indices_, ... :
        Provenance attributes, see `src.oversampling.provenance`.
    """

    method_name = "SHAP-Filtered SMOTE"

    def __init__(
        self,
        top_k: Optional[int] = None,
        top_ratio: Optional[float] = None,
        cumulative_threshold: Optional[float] = 0.90,
        sampling_strategy: Union[float, str] = "auto",
        k_neighbors: int = 5,
        base_estimator: Optional[object] = None,
        random_state: Optional[int] = None,
        shap_max_rows: Optional[int] = None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            base_estimator=base_estimator,
            random_state=random_state,
            shap_max_rows=shap_max_rows,
        )
        self.top_k = top_k
        self.top_ratio = top_ratio
        self.cumulative_threshold = cumulative_threshold

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample the dataset using SHAP-Filtered SMOTE.

        Parameters
        ----------
        X : array-like of shape (N, p)
            Matrix of training features.
        y : array-like of shape (N,)
            Vector of target labels.

        Returns
        -------
        X_resampled : np.ndarray of shape (N_resampled, p)
            Resampled feature matrix containing original and synthetic samples.
        y_resampled : np.ndarray of shape (N_resampled,)
            Resampled target labels.
        """
        X_arr, y_arr, minority_class, majority_class = self._validate_input(X, y)
        n_samples, n_features = X_arr.shape

        rng = np.random.default_rng(self.random_state)

        # 1. Compute MASV weights on training set
        weights, masv, fitted_estimator, shap_values = compute_masv_weights(
            X_arr, y_arr, base_estimator=self.base_estimator, random_state=self.random_state,
            return_shap_values=True, shap_max_rows=self.shap_max_rows,
        )
        self.feature_weights_ = weights
        self.masv_ = masv
        self.shap_values_ = shap_values  # per-row SHAP of the preliminary model, training fold only
        self.shap_estimator_ = fitted_estimator

        # 2. Select top predictive feature subspace
        sorted_indices = np.argsort(masv)[::-1]  # descending order

        if self.top_k is not None:
            k = max(1, min(n_features, self.top_k))
            selected_indices = sorted_indices[:k]
        elif self.top_ratio is not None:
            k = max(1, min(n_features, int(np.ceil(n_features * self.top_ratio))))
            selected_indices = sorted_indices[:k]
        elif self.cumulative_threshold is not None:
            cum_weights = np.cumsum(weights[sorted_indices])
            cutoff = np.searchsorted(cum_weights, self.cumulative_threshold)
            k = max(1, min(n_features, cutoff + 1))
            selected_indices = sorted_indices[:k]
        else:
            selected_indices = np.arange(n_features)

        self.selected_feature_indices_ = selected_indices

        # 3. Filter data to selected feature subspace
        minority_positions = np.flatnonzero(y_arr == minority_class)
        X_minority = X_arr[minority_positions]
        n_minority = len(X_minority)
        subspace_mask = np.zeros(n_features, dtype=bool)
        subspace_mask[selected_indices] = True
        unit_scale = np.ones(n_features)

        n_to_generate = self._determine_n_samples_to_generate(y_arr, minority_class, majority_class)
        if n_to_generate <= 0 or n_minority < 2:
            self._store_graph(minority_positions, np.arange(n_minority), subspace_mask, unit_scale,
                              np.empty((n_minority, 0), int), np.empty((n_minority, 0)))
            return self._finish(X_arr, y_arr, np.empty((0, n_features)), empty_generation_log(), minority_class)

        # 4. Nearest Neighbors search in the SHAP subspace (Euclidean on the selected features only)
        neighbors_idx, neighbors_dist, _ = minority_knn(X_minority, unit_scale, subspace_mask, self.k_neighbors)
        self._store_graph(minority_positions, np.arange(n_minority), subspace_mask, unit_scale,
                          neighbors_idx, neighbors_dist)

        # 5. Interpolate in the FULL feature space using the neighbour identified in the subspace.
        #    Same draw order as before (anchor, neighbour slot, lambda), so the output is unchanged.
        synthetic_samples, log = interpolate_with_provenance(
            X_minority, neighbors_idx, neighbors_dist, n_to_generate, rng)

        # 6. Combine original and synthetic data (synthetic rows are appended after the originals)
        return self._finish(X_arr, y_arr, synthetic_samples, log, minority_class)
