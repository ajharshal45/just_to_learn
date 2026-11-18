"""
Four-dataset SHAP-guided SMOTE benchmark.

    config         single source of truth for every experimental setting (identical across datasets)
    data           raw loaders for the four datasets and dataset-level quality checks
    preprocessing  train-only imputation / encoding / scaling
    runner         leak-free cross-validation, final hold-out evaluation, SHAP of downstream models
    inspection     neighbour, generation and SHAP-rationale tables for every oversampler
    plots          figures shared by the four dataset notebooks and the cross-dataset notebook
    summary        cross-dataset consolidation
"""
