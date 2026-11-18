"""
Multi-Dimensional Evaluation Framework.
"""

from src.evaluation.metrics import (
    evaluate_multi_dimensional_metrics,
    compute_expected_calibration_error,
    compute_g_mean,
    compute_distributional_fidelity,
)
from src.evaluation.stability import (
    compute_pairwise_rank_stability,
    compute_top_k_rank_overlap,
    evaluate_explanation_stability,
)
from src.evaluation.fidelity import compute_synthetic_fidelity
from src.evaluation.visualization import (
    plot_benchmark_metric_comparison,
    plot_multi_dimensional_radar,
    generate_latex_table,
)

__all__ = [
    "evaluate_multi_dimensional_metrics",
    "compute_expected_calibration_error",
    "compute_g_mean",
    "compute_distributional_fidelity",
    "compute_pairwise_rank_stability",
    "compute_top_k_rank_overlap",
    "evaluate_explanation_stability",
    "compute_synthetic_fidelity",
    "plot_benchmark_metric_comparison",
    "plot_multi_dimensional_radar",
    "generate_latex_table",
]
