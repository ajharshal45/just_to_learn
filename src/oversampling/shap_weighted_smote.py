from typing import Optional, Union, Tuple
import numpy as np

from src.oversampling.base import SHAPBaseOversampler, compute_masv_weights
from src.oversampling.provenance import (
    _ProvenanceSamplerMixin, empty_generation_log, interpolate_with_provenance, minority_knn,
)


class SHAPWeightedSMOTE(_ProvenanceSamplerMixin, SHAPBaseOversampler):
    """
    SHAP-Weighted SMOTE (MASV-SMOTE).

    Modifies nearest-neighbour search by weighting feature distances according to
    Mean Absolute SHAP Values (MASV). Features with high predictive attribution dictate
    neighbourhood selection, while noisy/uninformative features are downweighted.

    Distance Formula:
        d_W(x_a, x_b) = sqrt( sum_{j=1}^p  w_j * (x_{a,j} - x_{b,j})^2 )

    Parameters
    ----------
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
        Normalized MASV feature weights summing to 1.0.
    masv_ : np.ndarray of shape (p,)
        Unnormalized Mean Absolute SHAP Values.
    shap_values_ : np.ndarray of shape (N, p)
        Per-row SHAP values of the preliminary model on the training data passed to fit_resample.
    generation_log_, neighbor_indices_, ... :
        Provenance attributes, see `src.oversampling.provenance`.
    """

    method_name = "SHAP-Weighted SMOTE"

    def __init__(
        self,
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

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample dataset using SHAP-Weighted (MASV-SMOTE).

        Parameters
        ----------
        X : array-like of shape (N, p)
            Training feature matrix.
        y : array-like of shape (N,)
            Target labels.

        Returns
        -------
        X_resampled : np.ndarray of shape (N_resampled, p)
            Resampled feature matrix.
        y_resampled : np.ndarray of shape (N_resampled,)
            Resampled target labels.
        """
        X_arr, y_arr, minority_class, majority_class = self._validate_input(X, y)
        n_samples, n_features = X_arr.shape

        rng = np.random.default_rng(self.random_state)

        # 1. Compute MASV feature weights on training set
        weights, masv, fitted_estimator, shap_values = compute_masv_weights(
            X_arr, y_arr, base_estimator=self.base_estimator, random_state=self.random_state,
            return_shap_values=True, shap_max_rows=self.shap_max_rows,
        )
        self.feature_weights_ = weights
        self.masv_ = masv
        self.shap_values_ = shap_values  # per-row SHAP of the preliminary model, training fold only
        self.shap_estimator_ = fitted_estimator

        # 2. Identify minority samples
        minority_positions = np.flatnonzero(y_arr == minority_class)
        X_minority = X_arr[minority_positions]
        n_minority = len(X_minority)
        all_features = np.ones(n_features, dtype=bool)

        # 3. Scaling every column by sqrt(w_j) turns plain Euclidean distance into
        #    d_W(a, b) = sqrt(sum_j w_j (a_j - b_j)^2)
        sqrt_weights = np.sqrt(weights)

        n_to_generate = self._determine_n_samples_to_generate(y_arr, minority_class, majority_class)
        if n_to_generate <= 0 or n_minority < 2:
            self._store_graph(minority_positions, np.arange(n_minority), all_features, sqrt_weights,
                              np.empty((n_minority, 0), int), np.empty((n_minority, 0)))
            return self._finish(X_arr, y_arr, np.empty((0, n_features)), empty_generation_log(), minority_class)

        # 4. Nearest neighbours in the SHAP-weighted metric space
        neighbors_idx, neighbors_dist, _ = minority_knn(X_minority, sqrt_weights, all_features, self.k_neighbors)
        self._store_graph(minority_positions, np.arange(n_minority), all_features, sqrt_weights,
                          neighbors_idx, neighbors_dist)

        # 5. Interpolate in the unweighted (original) feature space. Same draw order as before
        #    (anchor, neighbour slot, lambda), so the output is unchanged.
        synthetic_samples, log = interpolate_with_provenance(
            X_minority, neighbors_idx, neighbors_dist, n_to_generate, rng)

        # 6. Combine original and synthetic data (synthetic rows are appended after the originals)
        return self._finish(X_arr, y_arr, synthetic_samples, log, minority_class)
