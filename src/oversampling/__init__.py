"""
Explainability-Guided and Baseline Oversampling Algorithms.
"""

from src.oversampling.base import SHAPBaseOversampler, compute_masv_weights
from src.oversampling.shap_filtered_smote import SHAPFilteredSMOTE
from src.oversampling.shap_weighted_smote import SHAPWeightedSMOTE
from src.oversampling.classical import (
    RawNoBalancing,
    RandomOversampler,
    ClassicalSMOTE,
    BorderlineSMOTEWrapper,
    ADASYNWrapper,
)
from src.oversampling.hybrids import SMOTEENNWrapper, SMOTETomekWrapper
from src.oversampling.baselines import get_baseline_oversampler, BaselineOversamplerWrapper
from src.oversampling.provenance import (
    NoSampling,
    RandomOversamplerWithProvenance,
    SMOTEWithProvenance,
    BorderlineSMOTEWithProvenance,
    interpolate_with_provenance,
)

__all__ = [
    "SHAPBaseOversampler",
    "compute_masv_weights",
    "SHAPFilteredSMOTE",
    "SHAPWeightedSMOTE",
    "RawNoBalancing",
    "RandomOversampler",
    "ClassicalSMOTE",
    "BorderlineSMOTEWrapper",
    "ADASYNWrapper",
    "SMOTEENNWrapper",
    "SMOTETomekWrapper",
    "get_baseline_oversampler",
    "BaselineOversamplerWrapper",
    "NoSampling",
    "RandomOversamplerWithProvenance",
    "SMOTEWithProvenance",
    "BorderlineSMOTEWithProvenance",
    "interpolate_with_provenance",
]
