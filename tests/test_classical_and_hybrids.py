"""
Unit tests for classical and hybrid oversampling wrappers.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification

from src.oversampling.classical import (
    RawNoBalancing,
    RandomOversampler,
    ClassicalSMOTE,
    BorderlineSMOTEWrapper,
    ADASYNWrapper,
)
from src.oversampling.hybrids import SMOTEENNWrapper, SMOTETomekWrapper
from src.oversampling.baselines import get_baseline_oversampler


def test_classical_oversamplers():
    X, y = make_classification(n_samples=150, n_features=6, weights=[0.85, 0.15], random_state=42)
    n_maj = np.sum(y == 0)

    # Raw
    raw = RawNoBalancing()
    Xr, yr = raw.fit_resample(X, y)
    assert len(Xr) == len(X)

    # ROS
    ros = RandomOversampler(random_state=42)
    Xr, yr = ros.fit_resample(X, y)
    assert len(Xr) == 2 * n_maj

    # SMOTE
    smote = ClassicalSMOTE(k_neighbors=3, random_state=42)
    Xr, yr = smote.fit_resample(X, y)
    assert len(Xr) == 2 * n_maj

    # Borderline SMOTE
    bsmote = BorderlineSMOTEWrapper(k_neighbors=3, m_neighbors=5, random_state=42)
    Xr, yr = bsmote.fit_resample(X, y)
    assert len(Xr) == 2 * n_maj

    # ADASYN
    adasyn = ADASYNWrapper(n_neighbors=3, random_state=42)
    Xr, yr = adasyn.fit_resample(X, y)
    assert len(Xr) >= len(X)


def test_hybrid_oversamplers():
    X, y = make_classification(n_samples=150, n_features=6, weights=[0.85, 0.15], random_state=42)

    # SMOTE-ENN
    enn = SMOTEENNWrapper(smote_k_neighbors=3, enn_n_neighbors=3, random_state=42)
    Xr, yr = enn.fit_resample(X, y)
    assert len(Xr) > 0

    # SMOTE-Tomek
    tomek = SMOTETomekWrapper(smote_k_neighbors=3, random_state=42)
    Xr, yr = tomek.fit_resample(X, y)
    assert len(Xr) > 0
