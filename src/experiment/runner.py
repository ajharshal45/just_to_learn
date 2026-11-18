"""
Leak-free experiment runner shared by the four dataset notebooks.

Protocol (identical for every dataset):

    full data --stratified 25%--> TEST (untouched until the final evaluation)
        |
        +-- DEVELOPMENT (75%)
              |
              +-- repeated stratified k-fold CV. In each fold, on the training part only:
              |      preprocessing fit -> SHAP (inside the SHAP samplers) -> oversampling -> model fit
              |   then the untouched validation part is transformed and scored.
              |
              +-- final fit on the whole development set (same chain), scored once on TEST.

Oversampling, SHAP, imputation, encoding and scaling never see validation or test rows.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from src.evaluation.metrics import compute_expected_calibration_error
from src.evaluation.stability import evaluate_explanation_stability
from src.experiment import config as C
from src.experiment.data import DatasetSpec
from src.experiment.preprocessing import Preprocessor

METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "specificity": "Specificity",
    "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "balanced_accuracy": "Balanced accuracy",
    "g_mean": "G-mean", "brier": "Brier score", "ece": "ECE",
}
HEADLINE_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity"]


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = C.DECISION_THRESHOLD) -> Dict[str, float]:
    """All label- and probability-based metrics for the positive (minority) class."""
    y_true = np.asarray(y_true).astype(int)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = recall_score(y_true, pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "g_mean": float(np.sqrt(recall * specificity)),
        "brier": brier_score_loss(y_true, prob),
        "ece": compute_expected_calibration_error(y_true, prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# --------------------------------------------------------------------------------------------
# SHAP explanations of the downstream models
# --------------------------------------------------------------------------------------------
def explain_rows(y: np.ndarray, seed: int, n_rows: int = C.SHAP_EXPLAIN_ROWS,
                 max_minority: int = C.SHAP_EXPLAIN_MAX_MINORITY) -> np.ndarray:
    """Evaluation rows to explain: up to `max_minority` minority rows, majority rows for the rest."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = min(len(pos), max_minority, n_rows)
    n_neg = min(len(neg), n_rows - n_pos)
    return np.sort(np.r_[rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])


def downstream_shap(model, model_name: str, X_explain: np.ndarray, X_train: np.ndarray, seed: int) -> np.ndarray:
    """SHAP values (n_rows, p) of the fitted downstream model for the positive class."""
    if model_name == "Logistic Regression":
        rng = np.random.default_rng(seed)
        bg = X_train[rng.choice(len(X_train), min(len(X_train), C.SHAP_LINEAR_BACKGROUND), replace=False)]
        masker = shap.maskers.Independent(bg, max_samples=len(bg))
        sv = shap.LinearExplainer(model, masker).shap_values(X_explain)
    else:
        sv = shap.TreeExplainer(model).shap_values(X_explain)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


# --------------------------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------------------------
@dataclass
class CVResult:
    fold_metrics: pd.DataFrame          # one row per (fold, method, model)
    sampling: pd.DataFrame              # one row per (fold, method): synthetic counts, runtime
    importances: pd.DataFrame           # mean |SHAP| per (fold, method, model) x feature
    elapsed_s: float = 0.0

    def summary(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        metrics = metrics or list(METRIC_LABELS)
        g = self.fold_metrics.groupby(["model", "method"], sort=False)[metrics]
        out = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
        return out.rename(columns=METRIC_LABELS)

    def mean_table(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        metrics = metrics or list(METRIC_LABELS)
        return self.fold_metrics.groupby(["model", "method"], sort=False)[metrics].mean()


def run_cross_validation(dev_df: pd.DataFrame, spec: DatasetSpec, verbose: bool = True) -> CVResult:
    y_all = dev_df["target"].to_numpy()
    folds = RepeatedStratifiedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS, random_state=C.SEED)
    metric_rows, sampling_rows, importance_rows = [], [], []
    t_start = time.perf_counter()

    for fold, (tr, va) in enumerate(folds.split(dev_df, y_all)):
        t_fold = time.perf_counter()
        train_df, val_df = dev_df.iloc[tr], dev_df.iloc[va]
        pre = Preprocessor(spec).fit(train_df)                       # fitted on the training part only
        X_tr, y_tr = pre.transform(train_df), train_df["target"].to_numpy()
        X_va, y_va = pre.transform(val_df), val_df["target"].to_numpy()
        expl = explain_rows(y_va, seed=C.SEED + fold)

        for method in C.METHOD_ORDER:
            sampler = C.make_sampler(method)                          # fresh, unfitted sampler per fold
            t0 = time.perf_counter()
            X_rs, y_rs = sampler.fit_resample(X_tr, y_tr)             # SHAP + oversampling on training part
            t_sample = time.perf_counter() - t0
            sampling_rows.append({"fold": fold, "method": method, "train_rows": len(y_tr),
                                  "train_minority": int(y_tr.sum()), "synthetic": int(len(y_rs) - len(y_tr)),
                                  "resampled_rows": len(y_rs), "sampling_seconds": t_sample})
            for model_name in C.MODEL_ORDER:
                model = C.make_model(model_name)
                t0 = time.perf_counter()
                model.fit(X_rs, y_rs)
                t_fit = time.perf_counter() - t0
                prob = model.predict_proba(X_va)[:, 1]                # untouched validation part
                metric_rows.append({"fold": fold, "repeat": fold // C.CV_SPLITS, "method": method,
                                    "model": model_name, **classification_metrics(y_va, prob),
                                    "sampling_seconds": t_sample, "fit_seconds": t_fit})
                sv = downstream_shap(model, model_name, X_va[expl], X_rs, seed=C.SEED + fold)
                importance_rows.append({"fold": fold, "method": method, "model": model_name,
                                        **dict(zip(pre.feature_names_, np.abs(sv).mean(axis=0)))})
        if verbose:
            print(f"fold {fold + 1:>2}/{C.CV_SPLITS * C.CV_REPEATS} done in {time.perf_counter() - t_fold:6.1f} s")

    importances = pd.DataFrame(importance_rows).fillna(0.0)  # categories absent from a fold -> 0
    return CVResult(pd.DataFrame(metric_rows), pd.DataFrame(sampling_rows), importances,
                    elapsed_s=time.perf_counter() - t_start)


def attribution_stability(cv: CVResult, top_k: int = 5) -> pd.DataFrame:
    """Kendall tau / Spearman rho / top-k overlap of downstream SHAP rankings across CV folds."""
    meta = ["fold", "method", "model"]
    rows = []
    for (model, method), g in cv.importances.groupby(["model", "method"], sort=False):
        vectors = [r for r in g.drop(columns=meta).to_numpy()]
        rows.append({"model": model, "method": method, **evaluate_explanation_stability(vectors, top_k=top_k)})
    return pd.DataFrame(rows).set_index(["model", "method"])


def paired_tests(cv: CVResult, reference: str = "SMOTE", metrics=("pr_auc", "f1", "recall", "precision")) -> pd.DataFrame:
    """Two-sided Wilcoxon signed-rank tests, each method vs `reference`, paired by CV fold."""
    rows = []
    fm = cv.fold_metrics
    for model in C.MODEL_ORDER:
        for method in C.METHOD_ORDER:
            if method == reference:
                continue
            for metric in metrics:
                a = fm[(fm.model == model) & (fm.method == method)].sort_values("fold")[metric].to_numpy()
                b = fm[(fm.model == model) & (fm.method == reference)].sort_values("fold")[metric].to_numpy()
                diff = a - b
                p = wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
                rows.append({"model": model, "method": method, "vs": reference, "metric": METRIC_LABELS[metric],
                             "mean_difference": diff.mean(), "folds_better": int((diff > 0).sum()),
                             "folds_worse": int((diff < 0).sum()), "p_value": p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Final fit on the development set, single evaluation on the untouched test set
# --------------------------------------------------------------------------------------------
@dataclass
class FinalResult:
    preprocessor: Preprocessor
    X_dev: np.ndarray
    y_dev: np.ndarray
    dev_row_ids: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_row_ids: np.ndarray
    samplers: Dict[str, object] = field(default_factory=dict)
    models: Dict[Tuple[str, str], object] = field(default_factory=dict)
    test_prob: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    test_metrics: pd.DataFrame = None
    sampling: pd.DataFrame = None
    explain_idx: np.ndarray = None
    shap_values: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    @property
    def feature_names(self) -> List[str]:
        return self.preprocessor.feature_names_

    def synthetic(self, method: str) -> np.ndarray:
        """Synthetic rows (model space) rebuilt exactly from the sampler's generation log."""
        log = self.samplers[method].generation_log_
        if len(log) == 0:
            return np.empty((0, self.X_dev.shape[1]))
        anchor = self.X_dev[log["anchor_position"].to_numpy(dtype=int)]
        if log["lambda"].isna().all():
            return anchor.copy()
        nb = self.X_dev[log["neighbor_position"].to_numpy(dtype=int)]
        return anchor + log["lambda"].to_numpy()[:, None] * (nb - anchor)

    def importance_table(self, model: str) -> pd.DataFrame:
        """Mean |SHAP| of each method's final `model` on the explained test rows (methods x features)."""
        return pd.DataFrame({m: np.abs(self.shap_values[(m, model)]).mean(axis=0) for m in C.METHOD_ORDER},
                            index=self.feature_names).T


def run_final(dev_df: pd.DataFrame, test_df: pd.DataFrame, spec: DatasetSpec, verbose: bool = True) -> FinalResult:
    pre = Preprocessor(spec).fit(dev_df)
    res = FinalResult(pre, pre.transform(dev_df), dev_df["target"].to_numpy(), dev_df.index.to_numpy(),
                      pre.transform(test_df), test_df["target"].to_numpy(), test_df.index.to_numpy())
    res.explain_idx = explain_rows(res.y_test, seed=C.SEED)
    metric_rows, sampling_rows = [], []

    for method in C.METHOD_ORDER:
        sampler = C.make_sampler(method)
        t0 = time.perf_counter()
        X_rs, y_rs = sampler.fit_resample(res.X_dev, res.y_dev)
        t_sample = time.perf_counter() - t0
        assert np.array_equal(X_rs[:len(res.y_dev)], res.X_dev), "real rows must be preserved first"
        res.samplers[method] = sampler
        sampling_rows.append({"method": method, "real_majority": int((res.y_dev == 0).sum()),
                              "real_minority": int(res.y_dev.sum()), "synthetic_minority": int(len(y_rs) - len(res.y_dev)),
                              "final_majority": int((y_rs == 0).sum()), "final_minority": int(y_rs.sum()),
                              "sampling_seconds": t_sample})
        for model_name in C.MODEL_ORDER:
            model = C.make_model(model_name).fit(X_rs, y_rs)
            prob = model.predict_proba(res.X_test)[:, 1]
            res.models[(method, model_name)] = model
            res.test_prob[(method, model_name)] = prob
            metric_rows.append({"method": method, "model": model_name, **classification_metrics(res.y_test, prob)})
            res.shap_values[(method, model_name)] = downstream_shap(
                model, model_name, res.X_test[res.explain_idx], X_rs, seed=C.SEED)
        if verbose:
            print(f"{method:<22} sampled in {t_sample:6.1f} s, 3 models fitted and scored on the test set")

    res.test_metrics = pd.DataFrame(metric_rows)
    res.sampling = pd.DataFrame(sampling_rows).set_index("method")
    return res
