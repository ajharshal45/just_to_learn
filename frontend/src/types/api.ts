export interface HealthResponse {
  status: string;
  version: string;
  project: string;
  python_version: string;
  available_classifiers: string[];
  available_oversamplers: string[];
  registered_datasets: string[];
}

export interface DatasetMetadata {
  id: string;
  name: string;
  domain: string;
  n_samples: number;
  n_features: number;
  minority_ratio: number;
  imbalance_ratio: string;
  feature_names: string[];
  is_custom: boolean;
  created_at: string;
  description?: string;
}

export interface DatasetSummary {
  id: string;
  name: string;
  domain: string;
  n_samples: number;
  n_features: number;
  numeric_features: string[];
  categorical_features: string[];
  missing_values: Record<string, number>;
  duplicate_rows_count: number;
  target_column: string;
  target_classes: (string | number)[];
  class_distribution: Record<string, number>;
  imbalance_ratio: string;
  minority_class: string | number;
  preprocessing_strategy: Record<string, string>;
}

export interface DatasetPreviewResponse {
  id: string;
  name: string;
  total_rows: number;
  page: number;
  page_size: number;
  columns: string[];
  rows: Record<string, any>[];
}

export interface ExperimentConfigRequest {
  dataset_id: string;
  target_column?: string | null;
  oversampler: string;
  classifier: string;
  k_neighbors: number;
  sampling_strategy: number | string;
  random_state: number;
  n_splits: number;
  n_repeats: number;
  scale_features: boolean;
  cumulative_threshold?: number | null;
  top_k?: number | null;
  top_ratio?: number | null;
}

export interface ExperimentStatusResponse {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  progress_percent: number;
  current_step: string;
  dataset_id: string;
  dataset_name: string;
  oversampler: string;
  classifier: string;
  created_at: string;
  completed_at?: string | null;
  wall_clock_seconds?: number | null;
  error_message?: string | null;
  is_benchmark: boolean;
}

export interface ExperimentMetrics {
  pr_auc_mean: number;
  pr_auc_std: number;
  roc_auc_mean: number;
  roc_auc_std: number;
  minority_recall_mean: number;
  minority_recall_std: number;
  minority_precision_mean: number;
  minority_precision_std: number;
  minority_f1_mean: number;
  minority_f1_std: number;
  balanced_accuracy_mean: number;
  balanced_accuracy_std: number;
  g_mean_mean: number;
  g_mean_std: number;
  ece_mean: number;
  ece_std: number;
  brier_score_mean: number;
  brier_score_std: number;
  attribution_kendall_tau_mean: number;
  attribution_spearman_rho_mean: number;
  n_folds_evaluated: number;
  mean_wasserstein_distance: number;
  max_wasserstein_distance?: number | null;
  ks_test_pass_ratio: number;
  covariance_frobenius_dist?: number | null;
  correlation_matrix_similarity?: number | null;
}

export interface ExperimentResultDetail {
  experiment: ExperimentStatusResponse;
  config: ExperimentConfigRequest;
  metrics: ExperimentMetrics;
}

export interface SHAPFeatureAttribution {
  feature_name: string;
  masv: number;
  normalized_weight: number;
  rank: number;
  is_selected_in_subspace?: boolean;
}

export interface SHAPExplanationResponse {
  experiment_id: string;
  method: string;
  dataset_name: string;
  features: SHAPFeatureAttribution[];
  cumulative_threshold?: number | null;
  retained_features_count?: number | null;
  total_features_count: number;
  explanation_type: string;
  summary_note: string;
}

export interface LeakageDiagnosticData {
  protocol_steps: {
    step: number;
    title: string;
    description: string;
    leak_free: boolean;
    danger_in_flawed: string;
  }[];
  empirical_proof: {
    dataset: string;
    imbalance_ratio: string;
    leaked_pipeline: {
      pr_auc: number;
      minority_recall: number;
      minority_f1: number;
      description: string;
    };
    leak_free_pipeline: {
      pr_auc: number;
      minority_recall: number;
      minority_f1: number;
      description: string;
    };
    delta: {
      pr_auc_inflation: string;
      finding: string;
    };
  };
  safeguards: {
    name: string;
    detail: string;
  }[];
}

export interface BenchmarksSummary {
  total_experiments: number;
  precomputed_benchmarks: number;
  live_user_runs: number;
  distinct_datasets_count: number;
  datasets_used: string[];
  best_by_metric: {
    pr_auc: { value: number; oversampler: string; dataset: string; classifier: string };
    minority_f1: { value: number; oversampler: string; dataset: string; classifier: string };
    attribution_stability: { value: number; oversampler: string; dataset: string; classifier: string };
    lowest_ece: { value: number; oversampler: string; dataset: string; classifier: string };
  };
}
