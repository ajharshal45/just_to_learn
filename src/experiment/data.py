"""
Raw loaders for the four benchmark datasets.

Each loader returns a DataFrame whose index is the ORIGINAL row number in the source file (kept
through de-duplication and subsampling, so every synthetic sample can be traced back to file rows),
the binary target in column `target` (1 = minority / positive class), and a `DatasetSpec` with
feature types and documentation.

Dataset-level steps done here (none of them learn from the data, so they cannot leak):
  * dropping identifier / non-feature columns,
  * dropping exact duplicate rows,
  * dropping columns that are constant in the whole file,
  * a ONE-OFF, seeded subsample of one class where the research design fixes the imbalance ratio
    (Breast Cancer 9:1, NSL-KDD 11:1), documented per dataset.
Everything that learns parameters (imputation, encoding, scaling) happens inside training folds,
see `src.experiment.preprocessing`.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.experiment.config import DATA_DIR, SEED


@dataclass
class DatasetSpec:
    slug: str
    title: str
    domain: str
    source: str
    url: str
    license: str
    file: str
    target_definition: str
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def features(self) -> List[str]:
        return self.numeric_features + self.categorical_features


def _finish(df: pd.DataFrame, target: pd.Series, drop: List[str], notes: List[str]) -> pd.DataFrame:
    """Attach the target, drop non-feature columns, exact duplicates and constant columns."""
    df = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    df["target"] = target.astype(int).to_numpy()
    n_before = len(df)
    df = df[~df.duplicated()]
    if n_before != len(df):
        notes.append(f"Dropped {n_before - len(df):,} exact duplicate rows (features + target).")
    constant = [c for c in df.columns if c != "target" and df[c].nunique(dropna=False) <= 1]
    if constant:
        df = df.drop(columns=constant)
        notes.append(f"Dropped constant column(s): {', '.join(constant)}.")
    df.index.name = "row_id"
    return df


def _subsample_class(df: pd.DataFrame, cls: int, n_keep: int, seed: int) -> pd.DataFrame:
    """Keep `n_keep` rows of class `cls` (seeded, without replacement); keep the other class whole."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(df["target"].to_numpy() == cls)
    keep = rng.choice(len(idx), size=n_keep, replace=False)
    mask = np.ones(len(df), dtype=bool)
    mask[idx] = False
    mask[idx[keep]] = True
    return df[mask]


# --------------------------------------------------------------------------------------------
# 1. Breast Cancer Wisconsin (Diagnostic)
# --------------------------------------------------------------------------------------------
def load_breast_cancer() -> Tuple[pd.DataFrame, DatasetSpec]:
    path = DATA_DIR / "breast_cancer_wisconsin" / "data.csv"
    raw = pd.read_csv(path)
    notes: List[str] = []
    df = _finish(raw, raw["diagnosis"].eq("M"), ["id", "diagnosis", "Unnamed: 32"], notes)

    # Same malignant subsample as src.pipeline.datasets.load_breast_cancer_diagnostic (minority 10%):
    # the file's row order equals sklearn's bundled copy, so the same seed selects the same 39 rows.
    n_benign = int((df["target"] == 0).sum())
    n_keep = max(10, int(n_benign * 0.10 / 0.90))
    rng = np.random.default_rng(SEED)
    malignant = np.flatnonzero(df["target"].to_numpy() == 1)
    chosen = malignant[rng.choice(len(malignant), size=n_keep, replace=False)]
    keep = np.zeros(len(df), dtype=bool)
    keep[df["target"].to_numpy() == 0] = True
    keep[chosen] = True
    notes.append(f"Malignant cases subsampled once (seed {SEED}) from {len(malignant)} to {n_keep} "
                 f"to create a 9:1 imbalance, as in the repository's existing loader.")
    df = df[keep]

    spec = DatasetSpec(
        slug="breast_cancer", title="Breast Cancer Wisconsin (Diagnostic)", domain="Medical diagnosis",
        source="UCI ML Repository (Wolberg, Street & Mangasarian, 1995), via Kaggle uciml/breast-cancer-wisconsin-data",
        url="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data",
        license="CC BY-NC-SA 4.0 (Kaggle listing); original UCI dataset CC BY 4.0",
        file="data/breast_cancer_wisconsin/data.csv",
        target_definition="diagnosis == 'M' (malignant) -> 1, 'B' (benign) -> 0",
        numeric_features=[c for c in df.columns if c != "target"], notes=notes,
    )
    return df, spec


# --------------------------------------------------------------------------------------------
# 2. Credit Card Fraud (ULB)
# --------------------------------------------------------------------------------------------
def load_credit_card() -> Tuple[pd.DataFrame, DatasetSpec]:
    path = DATA_DIR / "credit_card" / "creditcard.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download it with: kaggle datasets download -d mlg-ulb/creditcardfraud "
            "-p data/credit_card --unzip")
    raw = pd.read_csv(path)
    notes = ["`Time` dropped: seconds since the first transaction in the file, a row timestamp rather "
             "than a property of the transaction (same choice as the earlier credit-card notebook)."]
    df = _finish(raw, raw["Class"], ["Time", "Class"], notes)
    spec = DatasetSpec(
        slug="credit_card", title="Credit Card Fraud Detection (ULB)", domain="Financial fraud",
        source="Worldline & ULB Machine Learning Group (Dal Pozzolo et al., 2015), Kaggle mlg-ulb/creditcardfraud",
        url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        license="Open Database License (ODbL) 1.0",
        file="data/credit_card/creditcard.csv",
        target_definition="Class == 1 (fraud)",
        numeric_features=[c for c in df.columns if c != "target"], notes=notes,
    )
    return df, spec


# --------------------------------------------------------------------------------------------
# 3. Stroke Prediction
# --------------------------------------------------------------------------------------------
def load_stroke() -> Tuple[pd.DataFrame, DatasetSpec]:
    path = DATA_DIR / "kaggle_stroke" / "healthcare-dataset-stroke-data.csv"
    raw = pd.read_csv(path)
    notes = ["`bmi` is stored as text 'N/A' for some patients; read as missing and median-imputed "
             "inside each training fold."]
    raw["bmi"] = pd.to_numeric(raw["bmi"], errors="coerce")
    df = _finish(raw, raw["stroke"], ["id", "stroke"], notes)
    categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    spec = DatasetSpec(
        slug="stroke", title="Stroke Prediction", domain="Clinical risk prediction",
        source="Kaggle fedesoriano/stroke-prediction-dataset (confidential source, per the author)",
        url="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset",
        license="Data files (c) original authors (Kaggle listing); for research/educational use",
        file="data/kaggle_stroke/healthcare-dataset-stroke-data.csv",
        target_definition="stroke == 1",
        numeric_features=[c for c in df.columns if c not in categorical + ["target"]],
        categorical_features=categorical, notes=notes,
    )
    return df, spec


# --------------------------------------------------------------------------------------------
# 4. NSL-KDD network intrusion
# --------------------------------------------------------------------------------------------
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
NSL_KDD_ATTACK_SHARE = 0.08  # attack share after the one-off subsample (about 11.5:1)


def load_nsl_kdd() -> Tuple[pd.DataFrame, DatasetSpec]:
    path = DATA_DIR / "nsl_kdd" / "KDDTrain+.txt"
    raw = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
    notes = ["`difficulty` dropped: the NSL-KDD authors' count of how many reference classifiers got "
             "the record right, which is metadata, not a network feature.",
             "`label` (attack name) dropped after deriving the binary target."]
    df = _finish(raw, raw["label"].ne("normal"), ["label", "difficulty"], notes)
    n_normal = int((df["target"] == 0).sum())
    n_attack_all = int((df["target"] == 1).sum())
    n_keep = int(round(n_normal * NSL_KDD_ATTACK_SHARE / (1 - NSL_KDD_ATTACK_SHARE)))
    df = _subsample_class(df, 1, n_keep, SEED)
    notes.append(f"Attack records subsampled once (seed {SEED}) from {n_attack_all:,} to {n_keep:,} so "
                 f"attacks are {NSL_KDD_ATTACK_SHARE:.0%} of the data (about 11.5:1); the natural "
                 f"split is close to balanced (47% attacks). All normal records are kept.")
    categorical = ["protocol_type", "service", "flag"]
    spec = DatasetSpec(
        slug="nsl_kdd", title="NSL-KDD Network Intrusion (KDDTrain+)", domain="Network security",
        source="Canadian Institute for Cybersecurity, UNB (Tavallaee et al., 2009), via Kaggle hassan06/nslkdd",
        url="https://www.unb.ca/cic/datasets/nsl.html",
        license="Free for research use (CIC/UNB); Kaggle mirror lists no license",
        file="data/nsl_kdd/KDDTrain+.txt",
        target_definition="label != 'normal' (any attack) -> 1",
        numeric_features=[c for c in df.columns if c not in categorical + ["target"]],
        categorical_features=categorical, notes=notes,
    )
    return df, spec


DATASETS: Dict[str, Callable[[], Tuple[pd.DataFrame, DatasetSpec]]] = {
    "breast_cancer": load_breast_cancer,
    "credit_card": load_credit_card,
    "stroke": load_stroke,
    "nsl_kdd": load_nsl_kdd,
}


def load_dataset(slug: str) -> Tuple[pd.DataFrame, DatasetSpec]:
    if slug not in DATASETS:
        raise ValueError(f"Unknown dataset '{slug}'. Available: {list(DATASETS)}")
    return DATASETS[slug]()


# --------------------------------------------------------------------------------------------
# Ingestion reports (Stage 1)
# --------------------------------------------------------------------------------------------
def overview(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    y = df["target"]
    n_min, n_maj = int(y.sum()), int((y == 0).sum())
    return pd.DataFrame({
        "Dataset": [spec.title], "Domain": [spec.domain], "Rows": [len(df)],
        "Numeric features": [len(spec.numeric_features)],
        "Categorical features": [len(spec.categorical_features)],
        "Majority (0)": [n_maj], "Minority (1)": [n_min],
        "Minority %": [f"{n_min / len(df):.2%}"],
        "Imbalance ratio": [f"{n_maj / n_min:.1f} : 1"],
        "Minority:majority": [f"1 : {n_maj / n_min:.1f}"],
    }).set_index("Dataset")


def quality_report(raw_df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Per-column types, missing values, cardinality and range of the loaded data."""
    rows = []
    for c in spec.features:
        s = raw_df[c]
        numeric = c in spec.numeric_features
        rows.append({
            "Feature": c, "Type": "numeric" if numeric else "categorical", "dtype": str(s.dtype),
            "Missing": int(s.isna().sum()), "Unique": int(s.nunique()),
            "Min": s.min() if numeric else None, "Max": s.max() if numeric else None,
            "Binary 0/1": bool(numeric and set(s.dropna().unique()) <= {0, 1}),
        })
    return pd.DataFrame(rows).set_index("Feature")


def ks_ranking(df: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    """Two-sample KS statistic (majority vs minority) of every numeric feature, descending.
    Descriptive only: used to choose which features to plot, never for modelling."""
    from scipy.stats import ks_2samp

    y = df["target"]
    ks = {f: ks_2samp(df.loc[y == 0, f].dropna(), df.loc[y == 1, f].dropna()).statistic
          for f in spec.numeric_features if df[f].nunique() > 1}
    return pd.Series(ks, name="KS D").sort_values(ascending=False)


def quality_summary(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    X = df[spec.features]
    num = df[spec.numeric_features]
    return pd.DataFrame([
        ("Missing cells", int(X.isna().sum().sum()), "Median / most-frequent imputation fitted on each training fold"),
        ("Infinite cells", int(np.isinf(num.to_numpy(dtype=float, na_value=0)).sum()), "Would be set to missing"),
        ("Duplicate rows remaining", int(df.duplicated().sum()), "Exact duplicates already dropped at load time"),
        ("Rows with |z| > 5 on any numeric feature",
         int(((num - num.mean()) / num.std(ddof=0)).abs().gt(5).any(axis=1).sum()),
         "Flagged only: in imbalanced problems extreme rows are often the minority signal"),
        ("Categorical features", len(spec.categorical_features), "One-hot encoded inside each training fold"),
    ], columns=["Check", "Count", "Handling"]).set_index("Check")
