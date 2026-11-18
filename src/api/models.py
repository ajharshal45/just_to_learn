"""
Pydantic Request and Response Schemas for SHAP-SMOTE ML API.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    project: str = "Explainability-Aware Synthetic Oversampling (SHAP-SMOTE)"
    python_version: str
    available_classifiers: List[str]
    available_oversamplers: List[str]
    registered_datasets: List[str]


class DatasetMetadata(BaseModel):
    id: str
    name: str
    domain: str
    n_samples: int
    n_features: int
    minority_ratio: float
    imbalance_ratio: str
    feature_names: List[str] = Field(default_factory=list)
    is_custom: bool = False
    created_at: str
    description: Optional[str] = None


class DatasetSummary(BaseModel):
    id: str
    name: str
    domain: str
    n_samples: int
    n_features: int
    numeric_features: List[str]
    categorical_features: List[str]
    missing_values: Dict[str, int]
    duplicate_rows_count: int
    target_column: str
    target_classes: List[Union[str, int]]
    class_distribution: Dict[str, int]
    imbalance_ratio: str
    minority_class: Union[str, int]
    preprocessing_strategy: Dict[str, str]


class DatasetPreviewResponse(BaseModel):
    id: str
    name: str
    total_rows: int
    page: int
    page_size: int
    columns: List[str]
    rows: List[Dict[str, Any]]


class ExperimentConfigRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    oversampler: str = "shap_weighted"
    classifier: str = "random_forest"
    k_neighbors: int = 5
    sampling_strategy: Union[float, str] = "auto"
    random_state: int = 42
    n_splits: int = 5
    n_repeats: int = 1
    scale_features: bool = True
    cumulative_threshold: Optional[float] = 0.85
    top_k: Optional[int] = None
    top_ratio: Optional[float] = None


class ExperimentStatusResponse(BaseModel):
    id: str
    status: str  # "pending", "running", "completed", "failed"
    progress_percent: float = 0.0
    current_step: str = ""
    dataset_id: str
    dataset_name: str
    oversampler: str
    classifier: str
    created_at: str
    completed_at: Optional[str] = None
    wall_clock_seconds: Optional[float] = None
    error_message: Optional[str] = None
    is_benchmark: bool = False


class ExperimentMetrics(BaseModel):
    pr_auc_mean: float
    pr_auc_std: float = 0.0
    roc_auc_mean: float
    roc_auc_std: float = 0.0
    minority_recall_mean: float
    minority_recall_std: float = 0.0
    minority_precision_mean: float
    minority_precision_std: float = 0.0
    minority_f1_mean: float
    minority_f1_std: float = 0.0
    balanced_accuracy_mean: float
    balanced_accuracy_std: float = 0.0
    g_mean_mean: float
    g_mean_std: float = 0.0
    ece_mean: float
    ece_std: float = 0.0
    brier_score_mean: float
    brier_score_std: float = 0.0
    attribution_kendall_tau_mean: float = 0.0
    attribution_spearman_rho_mean: float = 0.0
    n_folds_evaluated: int = 5
    mean_wasserstein_distance: float = 0.0
    max_wasserstein_distance: Optional[float] = None
    ks_test_pass_ratio: float = 1.0
    covariance_frobenius_dist: Optional[float] = None
    correlation_matrix_similarity: Optional[float] = None


class ExperimentResultDetail(BaseModel):
    experiment: ExperimentStatusResponse
    config: ExperimentConfigRequest
    metrics: ExperimentMetrics


class SHAPFeatureAttribution(BaseModel):
    feature_name: str
    masv: float
    normalized_weight: float
    rank: int
    is_selected_in_subspace: Optional[bool] = None


class SHAPExplanationResponse(BaseModel):
    experiment_id: str
    method: str
    dataset_name: str
    features: List[SHAPFeatureAttribution]
    cumulative_threshold: Optional[float] = None
    retained_features_count: Optional[int] = None
    total_features_count: int
    explanation_type: str = "global_masv"
    summary_note: str


class LeakageDiagnosticData(BaseModel):
    protocol_steps: List[Dict[str, Any]]
    empirical_proof: Dict[str, Any]
    safeguards: List[Dict[str, str]]
