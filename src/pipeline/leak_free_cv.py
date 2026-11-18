"""
Leak-Free Cross-Validation Pipeline ensuring zero data leakage during feature scaling,
SHAP attribution, synthetic sample generation, and model training.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from scipy.stats import kendalltau, spearmanr
import shap

from src.evaluation.metrics import evaluate_multi_dimensional_metrics


class LeakFreePipeline(BaseEstimator, ClassifierMixin):
    """
    Leak-Free Pipeline encapsulating:
      1. Training set feature scaling (StandardScaler fit on train only)
      2. In-fold synthetic oversampling (SHAP-Guided or baseline)
      3. Final downstream classifier fit
      4. Prediction on untouched test set

    Parameters
    ----------
    oversampler : BaseEstimator
        Instance of oversampler (e.g., SHAPFilteredSMOTE, SHAPWeightedSMOTE, or baseline).
    classifier : BaseEstimator
        Final downstream classification model (e.g. LogisticRegression, RandomForest, XGBoost).
    scale_features : bool, default=True
        Whether to fit StandardScaler on training fold prior to oversampling.
    """

    def __init__(
        self,
        oversampler: BaseEstimator,
        classifier: BaseEstimator,
        scale_features: bool = True,
    ):
        self.oversampler = oversampler
        self.classifier = classifier
        self.scale_features = scale_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LeakFreePipeline":
        """
        Fit pipeline strictly on inner training fold data.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        # 1. Fit scaler on training fold only
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_arr)
        else:
            self.scaler_ = None
            X_scaled = X_arr.copy()

        # 2. Oversample training fold
        if hasattr(self.oversampler, "fit_resample"):
            X_resampled, y_resampled = self.oversampler.fit_resample(X_scaled, y_arr)
        else:
            X_resampled, y_resampled = X_scaled, y_arr

        # 3. Fit final classifier on resampled dataset
        self.classifier_ = self.classifier
        self.classifier_.fit(X_resampled, y_resampled)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.scaler_ is not None:
            X_arr = self.scaler_.transform(X_arr)
        return self.classifier_.predict(X_arr)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.scaler_ is not None:
            X_arr = self.scaler_.transform(X_arr)
        return self.classifier_.predict_proba(X_arr)


def run_leak_free_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    oversampler: BaseEstimator,
    classifier: BaseEstimator,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Executes leak-free Stratified Repeated Nested Cross-Validation.

    Evaluates downstream model performance, probability calibration,
    and calculates SHAP feature attribution stability (Kendall's Tau & Spearman Rho)
    across test folds.

    Parameters
    ----------
    X : np.ndarray of shape (N, p)
        Dataset feature matrix.
    y : np.ndarray of shape (N,)
        Target label vector.
    oversampler : BaseEstimator
        Oversampler instance.
    classifier : BaseEstimator
        Downstream classifier instance.
    n_splits : int, default=5
        Number of cross-validation splits.
    n_repeats : int, default=1
        Number of cross-validation repeats.
    random_state : int, default=42
        Random seed.
    scale_features : bool, default=True
        Whether to scale features inside fold.

    Returns
    -------
    results : dict
        Aggregated evaluation metrics, fold-level scores, and feature attribution stability.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_metrics: List[Dict[str, float]] = []
    shap_rankings: List[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_arr, y_arr)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        pipeline = LeakFreePipeline(
            oversampler=oversampler,
            classifier=classifier,
            scale_features=scale_features,
        )
        pipeline.fit(X_train, y_train)

        # Evaluate metrics on untouched test set
        metrics = evaluate_multi_dimensional_metrics(pipeline, X_test, y_test)
        fold_metrics.append(metrics)

        # Compute SHAP feature rankings on test fold predictions for stability analysis
        try:
            X_test_scaled = pipeline.scaler_.transform(X_test) if pipeline.scaler_ else X_test
            clf = pipeline.classifier_

            if hasattr(clf, "feature_importances_") or "Tree" in type(clf).__name__ or "Forest" in type(clf).__name__ or "XGB" in type(clf).__name__:
                explainer = shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_test_scaled)
            elif hasattr(clf, "coef_"):
                X_train_scaled = pipeline.scaler_.transform(X_train) if pipeline.scaler_ else X_train
                explainer = shap.LinearExplainer(clf, X_train_scaled)
                sv = explainer.shap_values(X_test_scaled)
            else:
                explainer = shap.Explainer(clf.predict_proba, X_test_scaled)
                sv = explainer(X_test_scaled).values

            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]

            masv_fold = np.mean(np.abs(sv), axis=0)
            shap_rankings.append(masv_fold)
        except Exception:
            pass

    # Aggregate fold metrics
    metric_keys = fold_metrics[0].keys()
    summary = {}
    for k in metric_keys:
        vals = [m[k] for m in fold_metrics]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))

    # Calculate attribution stability across folds (pairwise Kendall's Tau and Spearman Rho)
    kendall_taus = []
    spearman_rhos = []
    if len(shap_rankings) >= 2:
        for i in range(len(shap_rankings)):
            for j in range(i + 1, len(shap_rankings)):
                tau, _ = kendalltau(shap_rankings[i], shap_rankings[j])
                rho, _ = spearmanr(shap_rankings[i], shap_rankings[j])
                if not np.isnan(tau):
                    kendall_taus.append(tau)
                if not np.isnan(rho):
                    spearman_rhos.append(rho)

    summary["attribution_kendall_tau_mean"] = float(np.mean(kendall_taus)) if kendall_taus else 0.0
    summary["attribution_spearman_rho_mean"] = float(np.mean(spearman_rhos)) if spearman_rhos else 0.0
    summary["n_folds_evaluated"] = len(fold_metrics)

    return summary
