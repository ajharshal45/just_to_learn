from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import shap


def compute_masv_weights(
    X: np.ndarray,
    y: np.ndarray,
    base_estimator: Optional[BaseEstimator] = None,
    random_state: Optional[int] = None,
    return_shap_values: bool = False,
    shap_max_rows: Optional[int] = None,
):
    """
    Computes Mean Absolute SHAP Value (MASV) feature importance weights on the inner training set.

    Parameters
    ----------
    X : np.ndarray of shape (N, p)
        Training feature matrix.
    y : np.ndarray of shape (N,)
        Training target vector.
    base_estimator : BaseEstimator, optional
        Preliminary model used to compute SHAP attributions. Defaults to
        RandomForestClassifier(n_estimators=50, max_depth=5, random_state=random_state).
    random_state : int, optional
        Random seed for reproducibility.
    return_shap_values : bool, default=False
        Also return the per-row SHAP matrix phi (N, p) for the minority (positive) class, so callers
        can show the SHAP values of the parents of every synthetic sample.
    shap_max_rows : int, optional
        Efficiency option for large training sets. None (default) explains every row, exactly as
        before. Otherwise the preliminary model is still fitted on ALL rows, but SHAP is evaluated on
        every minority row plus a uniform random sample of majority rows so that at most
        `shap_max_rows` rows are explained. MASV is then the class-reweighted mean
            MASV_j = (1/N) * ( sum_{i in min} |phi_ij| + (N_maj / n_maj_sampled) * sum_{i in maj sample} |phi_ij| ),
        an unbiased estimate of the full-data MASV. Rows that were not explained get NaN in the
        returned SHAP matrix.

    Returns
    -------
    weights : np.ndarray of shape (p,)
        Normalized feature importance weights summing to 1.0.
    masv : np.ndarray of shape (p,)
        Unnormalized Mean Absolute SHAP Values per feature.
    estimator : BaseEstimator
        Fitted preliminary estimator.
    shap_values : np.ndarray of shape (N, p)
        Only when `return_shap_values=True`.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    n_samples, n_features = X.shape

    if base_estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        estimator = base_estimator

    estimator.fit(X, y)

    # Rows to explain: all of them (default), or every minority row + a uniform majority sample
    explain_idx = np.arange(n_samples)
    row_weight = np.ones(n_samples)
    if shap_max_rows is not None and n_samples > shap_max_rows:
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        min_idx = np.flatnonzero(y == minority_class)
        maj_idx = np.flatnonzero(y != minority_class)
        n_maj_sample = max(1, min(len(maj_idx), shap_max_rows - len(min_idx)))
        sample_rng = np.random.default_rng(random_state)
        maj_sample = np.sort(sample_rng.choice(maj_idx, size=n_maj_sample, replace=False))
        explain_idx = np.sort(np.concatenate([min_idx, maj_sample]))
        row_weight = np.where(y[explain_idx] == minority_class, 1.0, len(maj_idx) / n_maj_sample)
    X_explain = X[explain_idx]

    # Compute SHAP values
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_explain)
    except Exception:
        # Fallback to general Explainer if TreeExplainer fails or not supported
        explainer = shap.Explainer(estimator.predict_proba, X_explain)
        shap_values = explainer(X_explain).values

    # Handle multi-class / binary SHAP output structure
    if isinstance(shap_values, list):
        # List of arrays per class -> take minority class or sum over classes
        # For binary imbalanced data, class 1 (minority) SHAP values are standard
        classes = np.unique(y)
        if len(classes) == 2:
            minority_class = classes[1] if (y == classes[1]).sum() < (y == classes[0]).sum() else classes[0]
            class_idx = list(classes).index(minority_class)
            vals = shap_values[class_idx]
        else:
            vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:  # (n_samples, n_features, n_classes)
            vals = shap_values[:, :, 1]
        else:
            vals = shap_values
    else:
        # Explanation object from shap
        vals = np.asarray(shap_values.values)
        if vals.ndim == 3:
            vals = vals[:, :, 1]

    # MASV_j = (1/N) * sum_i |phi_j(x_i)|  (re-weighted when only a sample of majority rows is explained)
    if len(explain_idx) == n_samples:
        masv = np.mean(np.abs(vals), axis=0)
    else:
        masv = (row_weight[:, None] * np.abs(vals)).sum(axis=0) / n_samples

    # Normalize to form feature weights
    sum_masv = np.sum(masv)
    if sum_masv > 1e-12:
        weights = masv / sum_masv
    else:
        # Uniform fallback if attributions are zero
        weights = np.full(n_features, 1.0 / n_features)

    if return_shap_values:
        full = np.full((n_samples, n_features), np.nan)
        full[explain_idx] = np.asarray(vals, dtype=np.float64)
        return weights, masv, estimator, full
    return weights, masv, estimator


class SHAPBaseOversampler(BaseEstimator):
    """
    Abstract Base Class for SHAP-guided synthetic oversamplers.

    Parameters
    ----------
    sampling_strategy : float or str, default="auto"
        Desired ratio of minority to majority samples post-resampling.
    k_neighbors : int, default=5
        Number of nearest neighbors for synthetic sample generation.
    base_estimator : BaseEstimator, optional
        Preliminary estimator for SHAP attribution calculation.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str] = "auto",
        k_neighbors: int = 5,
        base_estimator: Optional[BaseEstimator] = None,
        random_state: Optional[int] = None,
        shap_max_rows: Optional[int] = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.shap_max_rows = shap_max_rows

    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) != 2:
            raise ValueError("SHAP-SMOTE oversamplers currently support binary classification targets.")

        minority_idx = np.argmin(counts)
        majority_idx = 1 - minority_idx

        minority_class = classes[minority_idx]
        majority_class = classes[majority_idx]

        return X_arr, y_arr, minority_class, majority_class

    def _determine_n_samples_to_generate(self, y: np.ndarray, minority_class: Any, majority_class: Any) -> int:
        n_minority = np.sum(y == minority_class)
        n_majority = np.sum(y == majority_class)

        if isinstance(self.sampling_strategy, str):
            if self.sampling_strategy in ("auto", "minority"):
                target_minority = n_majority
            else:
                raise ValueError(f"Unknown sampling strategy string: {self.sampling_strategy}")
        elif isinstance(self.sampling_strategy, (int, float)):
            target_minority = int(n_majority * float(self.sampling_strategy))
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")

        n_to_generate = max(0, target_minority - n_minority)
        return n_to_generate
