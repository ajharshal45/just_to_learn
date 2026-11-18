"""
Tests for the provenance-tracking samplers: the recorded parents and lambdas must reproduce every
synthetic row exactly, and the re-implemented SMOTE / Borderline-SMOTE must agree with imbalanced-learn.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

from src.oversampling import (
    NoSampling,
    RandomOversamplerWithProvenance,
    SMOTEWithProvenance,
    BorderlineSMOTEWithProvenance,
    SHAPFilteredSMOTE,
    SHAPWeightedSMOTE,
)

ALL_SAMPLERS = [
    NoSampling(),
    RandomOversamplerWithProvenance(random_state=0),
    SMOTEWithProvenance(random_state=0),
    BorderlineSMOTEWithProvenance(random_state=0),
    SHAPFilteredSMOTE(random_state=0),
    SHAPWeightedSMOTE(random_state=0),
]


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=600, n_features=10, n_informative=4, weights=[0.9, 0.1],
                               class_sep=0.8, random_state=3)
    return X, y


@pytest.mark.parametrize("sampler", ALL_SAMPLERS, ids=lambda s: type(s).__name__)
def test_log_reproduces_every_synthetic_row(sampler, data):
    X, y = data
    X_res, y_res = sampler.fit_resample(X, y)
    log = sampler.generation_log_
    synthetic = X_res[len(y):]

    assert np.array_equal(X_res[:len(y)], X), "real rows must come first, unchanged"
    assert len(log) == len(synthetic) == sampler.n_synthetic_
    assert (y_res[len(y):] == 1).all()
    if len(log) == 0:
        return

    anchor = X[log["anchor_position"].to_numpy(dtype=int)]
    assert (y[log["anchor_position"].to_numpy(dtype=int)] == 1).all()
    if log["lambda"].isna().all():  # random oversampling: exact copies
        assert np.allclose(synthetic, anchor)
        return
    neighbour = X[log["neighbor_position"].to_numpy(dtype=int)]
    lam = log["lambda"].to_numpy()[:, None]
    assert np.allclose(synthetic, anchor + lam * (neighbour - anchor))
    assert ((lam >= 0) & (lam <= 1)).all()

    # the recorded neighbour really is the rank-r neighbour in the method's search space
    local_anchor = log["anchor_local"].to_numpy()
    rank = log["neighbor_rank"].to_numpy(dtype=int)
    assert np.array_equal(sampler.neighbor_indices_[local_anchor, rank - 1], log["neighbor_local"].to_numpy())


def test_smote_neighbour_graph_matches_imblearn(data):
    from imblearn.over_sampling import SMOTE

    X, y = data
    ours = SMOTEWithProvenance(k_neighbors=5, random_state=0)
    ours.fit_resample(X, y)
    ref = SMOTE(k_neighbors=5, random_state=0)
    X_ref, y_ref = ref.fit_resample(X, y)

    X_min = X[y == 1]
    ref_nn = ref.nn_k_.kneighbors(X_min, return_distance=False)[:, 1:]
    assert np.array_equal(np.sort(ours.neighbor_indices_, 1), np.sort(ref_nn, 1))
    assert (y_ref == 1).sum() == (y == 0).sum()


def test_borderline_danger_set_matches_imblearn(data):
    from imblearn.over_sampling import BorderlineSMOTE

    X, y = data
    ours = BorderlineSMOTEWithProvenance(k_neighbors=5, m_neighbors=10, random_state=0)
    ours.fit_resample(X, y)
    ref = BorderlineSMOTE(k_neighbors=5, m_neighbors=10, random_state=0)
    ref.fit_resample(X, y)
    ref_danger = ref._in_danger_noise(ref.nn_m_, X[y == 1], 1, y, kind="danger")
    assert np.array_equal(ours.danger_mask_, ref_danger)
    assert ours.danger_mask_.sum() > 0
    # every anchor is a DANGER row
    assert ours.danger_mask_[ours.generation_log_["anchor_local"].to_numpy()].all()


def test_shap_weighted_neighbours_use_weighted_metric(data):
    X, y = data
    s = SHAPWeightedSMOTE(random_state=0)
    s.fit_resample(X, y)
    Z = X[y == 1] * np.sqrt(s.feature_weights_)
    dist, idx = NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z)
    assert np.allclose(s.neighbor_distances_, dist[:, 1:])
    assert s.shap_values_.shape == X.shape


def test_shap_filtered_neighbours_use_subspace_only(data):
    X, y = data
    s = SHAPFilteredSMOTE(top_k=3, random_state=0)
    s.fit_resample(X, y)
    assert s.search_feature_mask_.sum() == 3
    Z = X[y == 1][:, s.search_feature_mask_]
    dist, _ = NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z)
    assert np.allclose(s.neighbor_distances_, dist[:, 1:])
