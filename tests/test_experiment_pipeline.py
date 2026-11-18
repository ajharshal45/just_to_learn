"""
Leakage and correctness tests for the four-dataset experiment pipeline (src/experiment).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.experiment import config as C
from src.experiment import runner as R
from src.experiment.data import DatasetSpec
from src.experiment.preprocessing import Preprocessor, split_dev_test
from src.oversampling.base import compute_masv_weights


@pytest.fixture()
def toy():
    X, y = make_classification(n_samples=400, n_features=6, n_informative=3, weights=[0.88, 0.12], random_state=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["cat"] = np.random.default_rng(0).choice(["a", "b", "c"], len(df))
    df.loc[::17, "f0"] = np.nan
    df["target"] = y
    spec = DatasetSpec("toy", "Toy", "-", "-", "-", "-", "-", "-",
                       numeric_features=[f"f{i}" for i in range(6)], categorical_features=["cat"])
    return df, spec


def test_preprocessor_learns_from_training_rows_only(toy):
    df, spec = toy
    train, test = split_dev_test(df)
    pre = Preprocessor(spec).fit(train)
    num = pre.ct_.named_transformers_["num"]
    assert np.allclose(num.named_steps["scale"].mean_, train[spec.numeric_features].fillna(
        train[spec.numeric_features].median()).mean().to_numpy())
    # categories absent from training are ignored, not learned
    test = test.copy()
    test.loc[test.index[0], "cat"] = "unseen"
    Z = pre.transform(test)
    assert Z.shape[1] == len(pre.feature_names_) and not np.isnan(Z).any()


def test_cv_samplers_only_see_training_rows(toy, monkeypatch):
    df, spec = toy
    dev, _ = split_dev_test(df)
    seen = []
    real_make = C.make_sampler

    def recording(name, seed=C.SEED):
        s = real_make(name, seed)
        orig = s.fit_resample

        def wrapped(X, y):
            seen.append((name, X.copy(), y.copy()))
            return orig(X, y)

        s.fit_resample = wrapped
        return s

    monkeypatch.setattr(C, "make_sampler", recording)
    monkeypatch.setattr(C, "CV_REPEATS", 1)
    monkeypatch.setattr(C, "CV_SPLITS", 3)
    cv = R.run_cross_validation(dev, spec, verbose=False)

    from sklearn.model_selection import RepeatedStratifiedKFold
    folds = list(RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=C.SEED).split(dev, dev["target"]))
    per_fold = len(C.METHOD_ORDER)
    for f, (tr, va) in enumerate(folds):
        pre = Preprocessor(spec).fit(dev.iloc[tr])
        X_tr, X_va = pre.transform(dev.iloc[tr]), pre.transform(dev.iloc[va])
        for name, X, y in seen[f * per_fold:(f + 1) * per_fold]:
            assert len(X) == len(tr), f"{name} received {len(X)} rows, training fold has {len(tr)}"
            assert np.allclose(X, X_tr)
            # no validation row reaches the sampler
            d = np.abs(X[:, None, :] - X_va[None, :, :]).sum(axis=2)
            assert (d > 1e-9).all()
    assert set(cv.fold_metrics.method) == set(C.METHOD_ORDER)
    assert len(cv.fold_metrics) == 3 * len(C.METHOD_ORDER) * len(C.MODEL_ORDER)


def test_classification_metrics_confusion_consistency():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.6, 0.2, 0.9, 0.4, 0.3, 0.8, 0.7])
    m = R.classification_metrics(y, p)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 2, 1, 2)
    assert np.isclose(m["specificity"], 3 / 5) and np.isclose(m["recall"], 2 / 3)
    assert np.isclose(m["precision"], 2 / 4)


def test_shap_row_cap_estimates_full_masv():
    X, y = make_classification(n_samples=6000, n_features=10, n_informative=4, weights=[0.95, 0.05], random_state=2)
    w_full, _, _ = compute_masv_weights(X, y, random_state=0)
    w_cap, _, _, sv = compute_masv_weights(X, y, random_state=0, shap_max_rows=2000, return_shap_values=True)
    assert np.corrcoef(w_full, w_cap)[0, 1] > 0.99
    assert np.isclose(w_cap.sum(), 1.0)
    # every minority row is explained; unexplained majority rows are NaN
    assert not np.isnan(sv[y == 1]).any()
    assert np.isnan(sv).any(axis=1).sum() == len(y) - 2000


def test_detectability_is_near_half_for_samples_from_the_real_distribution():
    from src.experiment.inspection import detectability_auc

    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 5))
    y = np.r_[np.ones(400, int), np.zeros(400, int)]
    pa, pb = rng.integers(0, 400, 1200), rng.integers(0, 400, 1200)
    same = rng.normal(size=(1200, 5))             # drawn from the same distribution as the real rows
    shifted = same + np.array([3, 0, 0, 0, 0])    # trivially separable on column 0
    assert 0.4 < detectability_auc(X, y, same, pa, pb) < 0.6
    assert detectability_auc(X, y, shifted, pa, pb) > 0.95
