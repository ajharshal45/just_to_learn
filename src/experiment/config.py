"""
Experimental settings shared by all four dataset notebooks.

Nothing in here may differ between datasets: the notebooks only choose WHICH dataset to load. The
only dataset-specific choices (target definition, feature types, one-off subsampling to fix the
imbalance ratio) live in `src.experiment.data.DATASETS`.
"""

from pathlib import Path
from typing import Callable, Dict

from sklearn.base import BaseEstimator

from src.oversampling import (
    NoSampling,
    RandomOversamplerWithProvenance,
    SMOTEWithProvenance,
    BorderlineSMOTEWithProvenance,
    SHAPFilteredSMOTE,
    SHAPWeightedSMOTE,
)
from src.pipeline.models import get_classifier

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

# --- Reproducibility -------------------------------------------------------------------------
SEED = 42

# --- Evaluation protocol -----------------------------------------------------------------------
TEST_SIZE = 0.25           # stratified hold-out, never resampled, never used for fitting anything
CV_SPLITS = 5              # stratified k-fold on the development (non-test) part
CV_REPEATS = 2             # repeated with different shuffles -> 10 folds per (method, model)
DECISION_THRESHOLD = 0.5   # threshold for the label-based metrics (precision, recall, F1, ...)

# --- Oversampling hyper-parameters (the repository defaults, identical for every method) ------
SAMPLING_STRATEGY = "auto"     # balance the training data 1:1
K_NEIGHBORS = 5                # minority neighbours for interpolation (all SMOTE-family methods)
M_NEIGHBORS = 10               # Borderline-SMOTE: neighbours used to label DANGER rows
SHAP_CUMULATIVE_THRESHOLD = 0.90  # SHAP-Filtered SMOTE: keep features holding 90% of MASV
# SHAP guidance on large training sets: the preliminary model is fitted on every row, SHAP is
# evaluated on all minority rows + a uniform majority sample (<= 20,000 rows in total) and MASV is
# re-weighted to stay unbiased. Breast Cancer and Stroke are below the cap and are fully explained.
SHAP_MAX_ROWS = 20_000

# --- SHAP explanations of the downstream models ------------------------------------------------
SHAP_EXPLAIN_ROWS = 500        # rows of the evaluation fold explained per model (stratified)
SHAP_EXPLAIN_MAX_MINORITY = 250
SHAP_LINEAR_BACKGROUND = 200   # background rows for the logistic-regression LinearExplainer

# --- Synthetic-sample inspection ---------------------------------------------------------------
N_EXAMPLE_SYNTHETIC = 5        # synthetic rows walked through feature by feature in the notebooks
QUALITY_MAX_SYNTHETIC = 5000   # synthetic rows sub-sampled for the k-NN overlap check

# Methods in display order. "No Sampling" is the reference baseline; the next five are compared.
METHOD_ORDER = [
    "No Sampling",
    "Random Oversampling",
    "SMOTE",
    "Borderline-SMOTE",
    "SHAP-Filtered SMOTE",
    "SHAP-Weighted SMOTE",
]
PROPOSED_METHODS = ["SHAP-Filtered SMOTE", "SHAP-Weighted SMOTE"]
OVERSAMPLERS = METHOD_ORDER[1:]

# Fixed categorical order (validated for colour-vision deficiency, adjacent pairs); the neutral grey
# marks the no-sampling reference. Every chart also carries a legend or labels, never colour alone.
METHOD_COLORS = {
    "No Sampling": "#8a8985",
    "Random Oversampling": "#2a78d6",
    "SMOTE": "#eb6834",
    "Borderline-SMOTE": "#1baf7a",
    "SHAP-Filtered SMOTE": "#eda100",
    "SHAP-Weighted SMOTE": "#e87ba4",
}
CLASS_COLORS = {"majority": "#2a78d6", "minority": "#eb6834", "synthetic": "#1baf7a"}

MODEL_ORDER = ["Logistic Regression", "Random Forest", "XGBoost"]


def make_sampler(name: str, seed: int = SEED) -> BaseEstimator:
    """Fresh, unfitted sampler. Called once per fold so no fitted state crosses folds."""
    factories: Dict[str, Callable[[], BaseEstimator]] = {
        "No Sampling": lambda: NoSampling(),
        "Random Oversampling": lambda: RandomOversamplerWithProvenance(
            sampling_strategy=SAMPLING_STRATEGY, random_state=seed),
        "SMOTE": lambda: SMOTEWithProvenance(
            sampling_strategy=SAMPLING_STRATEGY, k_neighbors=K_NEIGHBORS, random_state=seed),
        "Borderline-SMOTE": lambda: BorderlineSMOTEWithProvenance(
            sampling_strategy=SAMPLING_STRATEGY, k_neighbors=K_NEIGHBORS, m_neighbors=M_NEIGHBORS,
            random_state=seed),
        "SHAP-Filtered SMOTE": lambda: SHAPFilteredSMOTE(
            cumulative_threshold=SHAP_CUMULATIVE_THRESHOLD, sampling_strategy=SAMPLING_STRATEGY,
            k_neighbors=K_NEIGHBORS, random_state=seed, shap_max_rows=SHAP_MAX_ROWS),
        "SHAP-Weighted SMOTE": lambda: SHAPWeightedSMOTE(
            sampling_strategy=SAMPLING_STRATEGY, k_neighbors=K_NEIGHBORS, random_state=seed,
            shap_max_rows=SHAP_MAX_ROWS),
    }
    return factories[name]()


def make_model(name: str, seed: int = SEED) -> BaseEstimator:
    """Fresh downstream classifier with the repository's default configuration."""
    key = {"Logistic Regression": "logistic_regression", "Random Forest": "random_forest",
           "XGBoost": "xgboost"}[name]
    return get_classifier(key, random_state=seed)


def settings_table():
    """The settings above as a two-column DataFrame, printed at the top of every notebook."""
    import pandas as pd

    rows = [
        ("Random seed", SEED),
        ("Hold-out test size", f"{TEST_SIZE:.0%} (stratified, untouched)"),
        ("Cross-validation", f"{CV_REPEATS} x {CV_SPLITS}-fold stratified on the development set"),
        ("Oversampling target", "1:1 minority:majority (sampling_strategy='auto')"),
        ("k neighbours (all SMOTE variants)", K_NEIGHBORS),
        ("Borderline-SMOTE m neighbours", M_NEIGHBORS),
        ("SHAP-Filtered cumulative threshold", SHAP_CUMULATIVE_THRESHOLD),
        ("SHAP preliminary model", "Random Forest, 50 trees, depth 5 (training fold only)"),
        ("SHAP rows explained", f"all rows up to {SHAP_MAX_ROWS:,}; above that all minority + "
                                "re-weighted uniform majority sample"),
        ("Downstream models", ", ".join(MODEL_ORDER)),
        ("Decision threshold", DECISION_THRESHOLD),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"]).set_index("Setting")
