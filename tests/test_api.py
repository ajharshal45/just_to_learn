"""
Integration and Unit Tests for SHAP-SMOTE FastAPI Backend.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import io

from src.api.main import app
from src.api.database import init_db

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "available_classifiers" in data
    assert "available_oversamplers" in data
    assert "registered_datasets" in data
    assert "random_forest" in data["available_classifiers"]
    assert "shap_weighted" in data["available_oversamplers"]


def test_list_datasets():
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 5
    ids = [d["id"] for d in data]
    assert "synthetic" in ids
    assert "kaggle_stroke" in ids
    assert "breast_cancer" in ids


def test_dataset_summary_and_preview():
    # Summary
    summary_resp = client.get("/api/datasets/synthetic/summary")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["id"] == "synthetic"
    assert summary["n_samples"] > 0
    assert summary["n_features"] > 0
    assert "class_distribution" in summary
    assert "preprocessing_strategy" in summary

    # Preview
    preview_resp = client.get("/api/datasets/synthetic/preview?page=1&page_size=10")
    assert preview_resp.status_code == 200
    preview = preview_resp.json()
    assert len(preview["rows"]) == 10
    assert "columns" in preview
    assert "target" in preview["columns"]


def test_dataset_upload_valid_and_invalid():
    # Valid CSV
    df = pd.DataFrame({
        "f1": np.random.randn(50),
        "f2": np.random.randn(50),
        "f3": np.random.randn(50),
        "target": [0] * 40 + [1] * 10
    })
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    upload_resp = client.post(
        "/api/datasets/upload",
        files={"file": ("test_data.csv", csv_bytes.getvalue(), "text/csv")},
        data={"target_column": "target"}
    )
    assert upload_resp.status_code == 200
    custom_meta = upload_resp.json()
    assert custom_meta["is_custom"] is True
    assert custom_meta["n_samples"] == 50
    assert custom_meta["n_features"] == 3

    # Invalid: non-binary target (3 classes)
    df_invalid = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0] * 10,
        "f2": [1.0, 2.0, 3.0, 4.0] * 10,
        "target": [0, 1, 2, 0] * 10
    })
    bad_bytes = io.BytesIO()
    df_invalid.to_csv(bad_bytes, index=False)
    bad_bytes.seek(0)

    bad_resp = client.post(
        "/api/datasets/upload",
        files={"file": ("bad_target.csv", bad_bytes.getvalue(), "text/csv")},
        data={"target_column": "target"}
    )
    assert bad_resp.status_code == 400
    assert "binary classification" in bad_resp.json()["detail"].lower()


def test_experiment_creation_and_results():
    payload = {
        "dataset_id": "synthetic",
        "oversampler": "shap_weighted",
        "classifier": "logistic_regression",
        "k_neighbors": 3,
        "sampling_strategy": "auto",
        "random_state": 42,
        "n_splits": 2,
        "n_repeats": 1,
        "scale_features": True
    }
    create_resp = client.post("/api/experiments", json=payload)
    assert create_resp.status_code == 200
    exp = create_resp.json()
    exp_id = exp["id"]
    assert exp["status"] in ("pending", "running", "completed")

    # Poll until completed (short test run)
    import time
    for _ in range(30):
        status_resp = client.get(f"/api/experiments/{exp_id}")
        assert status_resp.status_code == 200
        st = status_resp.json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert st["status"] == "completed"

    # Verify Results
    results_resp = client.get(f"/api/experiments/{exp_id}/results")
    assert results_resp.status_code == 200
    res_data = results_resp.json()
    assert "metrics" in res_data
    assert "pr_auc_mean" in res_data["metrics"]
    assert "roc_auc_mean" in res_data["metrics"]
    assert "ece_mean" in res_data["metrics"]

    # Verify SHAP Explanations
    exp_resp = client.get(f"/api/experiments/{exp_id}/explanations")
    assert exp_resp.status_code == 200
    expl_data = exp_resp.json()
    assert "features" in expl_data
    assert len(expl_data["features"]) > 0
    assert "masv" in expl_data["features"][0]
    assert "normalized_weight" in expl_data["features"][0]

    # Export formats
    json_export = client.get(f"/api/experiments/{exp_id}/export?format=json")
    assert json_export.status_code == 200

    csv_export = client.get(f"/api/experiments/{exp_id}/export?format=csv")
    assert csv_export.status_code == 200

    md_export = client.get(f"/api/experiments/{exp_id}/export?format=markdown")
    assert md_export.status_code == 200
    assert "Empirical ML Experiment Report" in md_export.text


def test_leakage_diagnostics_endpoint():
    resp = client.get("/api/diagnostics/leakage")
    assert resp.status_code == 200
    data = resp.json()
    assert "protocol_steps" in data
    assert len(data["protocol_steps"]) == 7
    assert "empirical_proof" in data
    assert "leaked_pipeline" in data["empirical_proof"]
    assert "leak_free_pipeline" in data["empirical_proof"]


def test_benchmarks_summary_endpoint():
    resp = client.get("/api/benchmarks/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_experiments"] >= 108
    assert "best_by_metric" in data
    assert "pr_auc" in data["best_by_metric"]


def test_simulation_manifold_endpoint():
    resp = client.get("/api/simulation/manifold?dataset_id=synthetic&k_neighbors=5&sampling_ratio=1.0")
    assert resp.status_code == 200
    data = resp.json()
    assert "majority_points" in data
    assert "minority_points" in data
    assert "smote_synthetic" in data
    assert "shap_synthetic" in data
    assert "weights" in data
    assert len(data["majority_points"]) > 0
    assert len(data["minority_points"]) > 0

