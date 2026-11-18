"""
Model Zoo and Classifiers Factory for Leak-Free Cross-Validation.

Supported classification families:
  1. Logistic Regression (Linear L2-regularized baseline)
  2. Random Forest (Bagging ensemble)
  3. XGBoost (Gradient boosting)
  4. LightGBM / Gradient Boosting (Histogram-based gradient boosting)
"""

from typing import Dict, Any, List, Optional
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_classifier(
    name: str,
    random_state: Optional[int] = 42,
    **kwargs
) -> BaseEstimator:
    """
    Factory function for downstream classification models.

    Parameters
    ----------
    name : str
        Classifier name: 'logistic_regression' (or 'lr'), 'random_forest' (or 'rf'),
        'xgboost' (or 'xgb'), 'gradient_boosting' (or 'gb'), 'lightgbm' (or 'lgb').
    random_state : int, optional
        Reproducibility seed.
    **kwargs : dict
        Additional estimator hyperparameter overrides.
    """
    key = name.lower().strip()

    if key in ("logistic_regression", "lr", "linear"):
        defaults = {
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 1000,
            "random_state": random_state,
        }
        defaults.update(kwargs)
        return LogisticRegression(**defaults)

    elif key in ("random_forest", "rf"):
        defaults = {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 4,
            "random_state": random_state,
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        return RandomForestClassifier(**defaults)

    elif key in ("xgboost", "xgb"):
        try:
            import xgboost as xgb
            defaults = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
                "random_state": random_state,
                "n_jobs": -1,
            }
            defaults.update(kwargs)
            return xgb.XGBClassifier(**defaults)
        except ImportError:
            # Fallback to GradientBoostingClassifier if xgboost unavailable
            defaults = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.08,
                "random_state": random_state,
            }
            defaults.update(kwargs)
            return GradientBoostingClassifier(**defaults)

    elif key in ("lightgbm", "lgb"):
        try:
            import lightgbm as lgb
            defaults = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.08,
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": -1,
            }
            defaults.update(kwargs)
            return lgb.LGBMClassifier(**defaults)
        except ImportError:
            defaults = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.08,
                "random_state": random_state,
            }
            defaults.update(kwargs)
            return GradientBoostingClassifier(**defaults)

    elif key in ("gradient_boosting", "gb"):
        defaults = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.08,
            "random_state": random_state,
        }
        defaults.update(kwargs)
        return GradientBoostingClassifier(**defaults)

    else:
        raise ValueError(f"Unknown classifier: '{name}'. Available: {list_available_classifiers()}")


def list_available_classifiers() -> List[str]:
    """Returns list of supported model family identifiers."""
    return ["logistic_regression", "random_forest", "xgboost", "gradient_boosting", "lightgbm"]
