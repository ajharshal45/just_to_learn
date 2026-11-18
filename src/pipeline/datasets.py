"""
Benchmark Dataset Loaders and Synthetic Tabular Generators.

Provides diverse benchmark datasets across imbalance ratios, dimensionalities,
and application domains according to Objective 3.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer


def generate_synthetic_imbalanced(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 6,
    n_redundant: int = 4,
    minority_ratio: float = 0.05,
    flip_y: float = 0.01,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates synthetic imbalanced tabular classification dataset with known signal/noise features.
    """
    weights = [1.0 - minority_ratio, minority_ratio]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        weights=weights,
        flip_y=flip_y,
        random_state=random_state,
    )
    meta = {
        "name": "synthetic_imbalanced",
        "domain": "Synthetic Simulation",
        "n_samples": len(y),
        "n_features": n_features,
        "n_informative": n_informative,
        "n_noise_features": n_features - n_informative - n_redundant,
        "minority_ratio": minority_ratio,
        "imbalance_ratio": f"{int((1.0 - minority_ratio)/minority_ratio)}:1",
    }
    return X, y, meta


def load_breast_cancer_diagnostic(
    minority_ratio: float = 0.10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads Breast Cancer Wisconsin (Diagnostic) tabular dataset, subsampling malignant cases
    to establish a realistic high-stakes medical imbalance.
    """
    data = load_breast_cancer()
    X_full, y_full = data.data, data.target
    # In sklearn WDBC: 0 = malignant (212), 1 = benign (357)
    # We treat malignant (0) as minority rare event (label 1 in binary convention)
    majority_mask = y_full == 1
    minority_mask = y_full == 0

    X_maj = X_full[majority_mask]
    y_maj = np.zeros(len(X_maj), dtype=int)

    X_min_all = X_full[minority_mask]
    n_maj = len(X_maj)
    n_target_min = max(10, int(n_maj * minority_ratio / (1.0 - minority_ratio)))

    rng = np.random.default_rng(random_state)
    min_indices = rng.choice(len(X_min_all), size=min(n_target_min, len(X_min_all)), replace=False)
    X_min = X_min_all[min_indices]
    y_min = np.ones(len(X_min), dtype=int)

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([y_maj, y_min])

    # Shuffle
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    actual_ratio = float(np.mean(y == 1))
    meta = {
        "name": "breast_cancer_diagnostic",
        "domain": "Medical Diagnostics",
        "n_samples": len(y),
        "n_features": X.shape[1],
        "feature_names": list(data.feature_names),
        "minority_ratio": actual_ratio,
        "imbalance_ratio": f"{int((1.0 - actual_ratio)/actual_ratio)}:1",
    }
    return X, y, meta


def generate_credit_fraud_simulation(
    n_samples: int = 2000,
    n_features: int = 28,
    minority_ratio: float = 0.02,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulates high-dimensional PCA-like transaction fraud data with extreme imbalance
    and subtle non-linear minority signatures.
    """
    rng = np.random.default_rng(random_state)
    n_minority = max(15, int(n_samples * minority_ratio))
    n_majority = n_samples - n_minority

    # Majority transactions: Gaussian standard PCA distribution
    X_maj = rng.normal(loc=0.0, scale=1.0, size=(n_majority, n_features))
    y_maj = np.zeros(n_majority, dtype=int)

    # Minority transactions: shifts along key fraudulent components (V14, V10, V12, V4, V11 style)
    X_min = rng.normal(loc=0.0, scale=1.2, size=(n_minority, n_features))
    candidate_dims = [2, 5, 9, 13, 17, 21, 25]
    fraud_dims = [d for d in candidate_dims if d < n_features]
    if not fraud_dims:
        fraud_dims = list(range(min(3, n_features)))
    for d in fraud_dims:
        X_min[:, d] += rng.uniform(2.5, 4.0, size=n_minority) * rng.choice([-1, 1], size=n_minority)
    y_min = np.ones(n_minority, dtype=int)

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([y_maj, y_min])

    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    meta = {
        "name": "credit_fraud_simulation",
        "domain": "Financial Fraud Detection",
        "n_samples": len(y),
        "n_features": n_features,
        "minority_ratio": float(np.mean(y == 1)),
        "imbalance_ratio": f"{int(n_majority / n_minority)}:1",
        "key_fraud_dimensions": fraud_dims,
    }
    return X, y, meta


def generate_nsl_kdd_simulation(
    n_samples: int = 1500,
    n_features: int = 35,
    minority_ratio: float = 0.08,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulates Network Intrusion Detection (NSL-KDD style) tabular data with multimodal
    distributions, boundary overlap, and noisy non-predictive variables.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=3,
        weights=[1.0 - minority_ratio, minority_ratio],
        flip_y=0.02,
        class_sep=0.75,  # Moderate boundary overlap
        random_state=random_state,
    )
    meta = {
        "name": "nsl_kdd_simulation",
        "domain": "Network Security / Intrusion Detection",
        "n_samples": len(y),
        "n_features": n_features,
        "minority_ratio": minority_ratio,
        "imbalance_ratio": f"{int((1.0 - minority_ratio)/minority_ratio)}:1",
    }
    return X, y, meta


def load_kaggle_stroke_dataset(
    csv_path: str = "data/kaggle_stroke/healthcare-dataset-stroke-data.csv",
    subsample_n: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads and pre-processes the official Kaggle Stroke Prediction Dataset
    (fedesoriano/stroke-prediction-dataset) containing clinical features for predicting rare stroke events.
    """
    from pathlib import Path
    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        # Search relative to repo root
        repo_root = Path(__file__).resolve().parent.parent.parent
        path = repo_root / csv_path

    if not path.exists():
        raise FileNotFoundError(f"Kaggle stroke dataset CSV not found at {path}")

    df = pd.read_csv(path)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Median impute bmi
    if "bmi" in df.columns:
        df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    cat_cols = [c for c in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"] if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    feature_cols = [c for c in df_encoded.columns if c != "stroke"]
    X = df_encoded[feature_cols].values.astype(float)
    y = df_encoded["stroke"].values.astype(int)

    if subsample_n is not None and subsample_n < len(y):
        rng = np.random.default_rng(random_state)
        # Stratified subsample
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        n_1 = int(subsample_n * len(idx_1) / len(y))
        n_0 = subsample_n - n_1
        sel_0 = rng.choice(idx_0, size=n_0, replace=False)
        sel_1 = rng.choice(idx_1, size=min(n_1, len(idx_1)), replace=False)
        all_idx = np.concatenate([sel_0, sel_1])
        rng.shuffle(all_idx)
        X = X[all_idx]
        y = y[all_idx]

    minority_ratio = float(np.mean(y == 1))
    meta = {
        "name": "kaggle_stroke_dataset",
        "domain": "Clinical Healthcare / Neurology",
        "source": "Kaggle (fedesoriano/stroke-prediction-dataset)",
        "n_samples": len(y),
        "n_features": X.shape[1],
        "feature_names": feature_cols,
        "minority_ratio": minority_ratio,
        "imbalance_ratio": f"{int((1.0 - minority_ratio) / minority_ratio)}:1",
    }
    return X, y, meta


DATASET_REGISTRY = {
    "synthetic": generate_synthetic_imbalanced,
    "breast_cancer": load_breast_cancer_diagnostic,
    "credit_fraud": generate_credit_fraud_simulation,
    "nsl_kdd": generate_nsl_kdd_simulation,
    "kaggle_stroke": load_kaggle_stroke_dataset,
    "stroke": load_kaggle_stroke_dataset,
}


def load_benchmark_dataset(
    dataset_name: str,
    random_state: int = 42,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unified loader for all benchmark datasets.
    """
    key = dataset_name.lower().strip()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown benchmark dataset: '{dataset_name}'. Available: {list_benchmark_datasets()}")
    loader = DATASET_REGISTRY[key]
    return loader(random_state=random_state, **kwargs)


def list_benchmark_datasets() -> List[str]:
    """Returns list of registered benchmark datasets."""
    return list(DATASET_REGISTRY.keys())
