"""
Leak-free preprocessing: every step that learns a parameter is fitted on the training rows only.

    numeric features      median imputation -> StandardScaler
    categorical features  most-frequent imputation -> one-hot encoding (unknown categories ignored)

One-hot columns are left as 0/1 (not standardised). Oversamplers then work in this transformed
space: SMOTE interpolation of one-hot columns gives fractional values, which is the standard
behaviour of SMOTE applied after one-hot encoding. It is identical for every method, and the
saved synthetic records also show the nearest valid category (largest one-hot value).
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.experiment.config import SEED, TEST_SIZE
from src.experiment.data import DatasetSpec


def split_dev_test(df: pd.DataFrame, seed: int = SEED, test_size: float = TEST_SIZE):
    """Stratified development / hold-out split. Returns (dev_df, test_df); row ids are preserved."""
    dev, test = train_test_split(df, test_size=test_size, stratify=df["target"], random_state=seed)
    return dev, test


class Preprocessor:
    """Thin wrapper around a ColumnTransformer that remembers output feature names and groups."""

    def __init__(self, spec: DatasetSpec):
        self.spec = spec

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        transformers = []
        if self.spec.numeric_features:
            transformers.append(("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), self.spec.numeric_features))
        if self.spec.categorical_features:
            transformers.append(("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), self.spec.categorical_features))
        self.ct_ = ColumnTransformer(transformers, remainder="drop").fit(df[self.spec.features])

        names: List[str] = list(self.spec.numeric_features)
        self.onehot_groups_ = {}
        if self.spec.categorical_features:
            enc = self.ct_.named_transformers_["cat"].named_steps["onehot"]
            for col, cats in zip(self.spec.categorical_features, enc.categories_):
                start = len(names)
                names += [f"{col}={c}" for c in cats]
                self.onehot_groups_[col] = (start, len(names), [str(c) for c in cats])
        self.feature_names_ = names
        self.n_numeric_ = len(self.spec.numeric_features)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.ct_.transform(df[self.spec.features]), dtype=np.float64)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def to_original_units(self, Z: np.ndarray) -> pd.DataFrame:
        """
        Map rows of the model space back to readable values: numeric features un-scaled, plus one
        column per categorical feature holding the category with the largest (possibly fractional)
        one-hot value. The raw one-hot values are kept too, so nothing is hidden.
        """
        Z = np.asarray(Z, dtype=np.float64)
        out = {}
        if self.n_numeric_:
            scaler = self.ct_.named_transformers_["num"].named_steps["scale"]
            num = scaler.inverse_transform(Z[:, :self.n_numeric_])
            for j, c in enumerate(self.spec.numeric_features):
                out[c] = num[:, j]
        for col, (a, b, cats) in self.onehot_groups_.items():
            block = Z[:, a:b]
            out[col] = np.asarray(cats, dtype=object)[block.argmax(axis=1)]
            for j, cat in enumerate(cats):
                out[f"{col}={cat}"] = block[:, j]
        return pd.DataFrame(out)

    def summary(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
        """What was learned from the training rows, and how it was applied to the test rows."""
        rows = []
        if self.spec.numeric_features:
            num = self.ct_.named_transformers_["num"]
            imp, sc = num.named_steps["impute"], num.named_steps["scale"]
            Ztr, Zte = self.transform(train_df), self.transform(test_df)
            for j, c in enumerate(self.spec.numeric_features):
                rows.append({
                    "Feature": c, "Kind": "numeric", "Missing (train)": int(train_df[c].isna().sum()),
                    "Imputation value (train median)": imp.statistics_[j],
                    "Scaler mean (train)": sc.mean_[j], "Scaler std (train)": sc.scale_[j],
                    "Train mean after": Ztr[:, j].mean(), "Test mean after": Zte[:, j].mean(),
                })
        for col, (a, b, cats) in self.onehot_groups_.items():
            rows.append({"Feature": col, "Kind": f"categorical -> {b - a} one-hot columns",
                         "Missing (train)": int(train_df[col].isna().sum()),
                         "Imputation value (train median)": None, "Scaler mean (train)": None,
                         "Scaler std (train)": None, "Train mean after": None, "Test mean after": None})
        return pd.DataFrame(rows).set_index("Feature")
