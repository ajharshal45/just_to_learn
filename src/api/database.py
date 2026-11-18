"""
SQLite Persistence Layer for SHAP-SMOTE ML Benchmark Application.
Stores both pre-computed empirical benchmark records and user-executed live experiments.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "experiments.db"
BENCHMARK_JSON_PATH = Path(__file__).resolve().parent.parent.parent / "experiments" / "results" / "benchmark_results.json"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes tables and seeds benchmark data if fresh."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                oversampler TEXT NOT NULL,
                classifier TEXT NOT NULL,
                status TEXT NOT NULL, -- pending, running, completed, failed
                progress_percent REAL DEFAULT 0.0,
                current_step TEXT DEFAULT '',
                is_benchmark INTEGER DEFAULT 0,
                config_json TEXT NOT NULL,
                metrics_json TEXT,
                explanations_json TEXT,
                error_message TEXT,
                wall_clock_seconds REAL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)

        # Custom datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                domain TEXT NOT NULL,
                file_path TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                n_samples INTEGER NOT NULL,
                n_features INTEGER NOT NULL,
                target_column TEXT NOT NULL,
                minority_class TEXT NOT NULL,
                minority_ratio REAL NOT NULL,
                imbalance_ratio TEXT NOT NULL,
                feature_names_json TEXT NOT NULL,
                preprocessing_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()

        # Seed pre-computed benchmark records if table is empty
        cursor.execute("SELECT COUNT(*) FROM experiments WHERE is_benchmark = 1")
        count = cursor.fetchone()[0]
        if count == 0 and BENCHMARK_JSON_PATH.exists():
            try:
                with open(BENCHMARK_JSON_PATH, "r") as f:
                    benchmarks = json.load(f)
                
                for idx, record in enumerate(benchmarks):
                    exp_id = f"benchmark-{record.get('dataset', 'ds')}-{idx:03d}"
                    created_at = "2026-10-13T16:25:00"
                    
                    config = {
                        "dataset_id": record.get("dataset"),
                        "oversampler": record.get("oversampler"),
                        "classifier": record.get("classifier"),
                        "k_neighbors": 5,
                        "sampling_strategy": "auto",
                        "random_state": 42,
                        "n_splits": 5,
                        "n_repeats": 1,
                        "scale_features": True,
                        "cumulative_threshold": 0.85
                    }
                    
                    # Extract metrics
                    metrics = {
                        "pr_auc_mean": float(record.get("pr_auc_mean", 0.0)),
                        "pr_auc_std": float(record.get("pr_auc_std", 0.0)),
                        "roc_auc_mean": float(record.get("roc_auc_mean", 0.0)),
                        "roc_auc_std": float(record.get("roc_auc_std", 0.0)),
                        "minority_recall_mean": float(record.get("minority_recall_mean", 0.0)),
                        "minority_recall_std": float(record.get("minority_recall_std", 0.0)),
                        "minority_precision_mean": float(record.get("minority_precision_mean", 0.0)),
                        "minority_precision_std": float(record.get("minority_precision_std", 0.0)),
                        "minority_f1_mean": float(record.get("minority_f1_mean", 0.0)),
                        "minority_f1_std": float(record.get("minority_f1_std", 0.0)),
                        "balanced_accuracy_mean": float(record.get("balanced_accuracy_mean", 0.0)),
                        "balanced_accuracy_std": float(record.get("balanced_accuracy_std", 0.0)),
                        "g_mean_mean": float(record.get("g_mean_mean", 0.0)),
                        "g_mean_std": float(record.get("g_mean_std", 0.0)),
                        "ece_mean": float(record.get("ece_mean", 0.0)),
                        "ece_std": float(record.get("ece_std", 0.0)),
                        "brier_score_mean": float(record.get("brier_score_mean", 0.0)),
                        "brier_score_std": float(record.get("brier_score_std", 0.0)),
                        "attribution_kendall_tau_mean": float(record.get("attribution_kendall_tau_mean", 0.0)),
                        "attribution_spearman_rho_mean": float(record.get("attribution_spearman_rho_mean", 0.0)),
                        "n_folds_evaluated": int(record.get("n_folds_evaluated", 5)),
                        "mean_wasserstein_distance": float(record.get("mean_wasserstein_distance", 0.0)),
                        "max_wasserstein_distance": float(record.get("max_wasserstein_distance", 0.0)),
                        "ks_test_pass_ratio": float(record.get("ks_test_pass_ratio", 1.0)),
                        "covariance_frobenius_dist": float(record.get("covariance_frobenius_dist", 0.0)),
                        "correlation_matrix_similarity": float(record.get("correlation_matrix_similarity", 1.0)),
                    }
                    
                    cursor.execute("""
                        INSERT INTO experiments (
                            id, dataset_id, dataset_name, oversampler, classifier,
                            status, progress_percent, current_step, is_benchmark,
                            config_json, metrics_json, explanations_json,
                            wall_clock_seconds, created_at, completed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        exp_id,
                        record.get("dataset"),
                        record.get("dataset"),
                        record.get("oversampler"),
                        record.get("classifier"),
                        "completed",
                        100.0,
                        "Completed Empirical Benchmark Run",
                        1,
                        json.dumps(config),
                        json.dumps(metrics),
                        None,
                        float(record.get("wall_clock_seconds", 0.0)),
                        created_at,
                        created_at
                    ))
                conn.commit()
            except Exception as e:
                print(f"Error seeding benchmark results: {e}")


def insert_experiment(data: Dict[str, Any]):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO experiments (
                id, dataset_id, dataset_name, oversampler, classifier,
                status, progress_percent, current_step, is_benchmark,
                config_json, metrics_json, explanations_json, error_message,
                wall_clock_seconds, created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["id"],
            data["dataset_id"],
            data["dataset_name"],
            data["oversampler"],
            data["classifier"],
            data.get("status", "pending"),
            data.get("progress_percent", 0.0),
            data.get("current_step", "Queued"),
            1 if data.get("is_benchmark") else 0,
            json.dumps(data.get("config", {})),
            json.dumps(data.get("metrics")) if data.get("metrics") else None,
            json.dumps(data.get("explanations")) if data.get("explanations") else None,
            data.get("error_message"),
            data.get("wall_clock_seconds"),
            data.get("created_at", datetime.now().isoformat()),
            data.get("completed_at")
        ))
        conn.commit()


def update_experiment(exp_id: str, fields: Dict[str, Any]):
    with get_connection() as conn:
        cursor = conn.cursor()
        updates = []
        params = []
        for key, val in fields.items():
            if key in ("config_json", "metrics_json", "explanations_json") and isinstance(val, (dict, list)):
                updates.append(f"{key} = ?")
                params.append(json.dumps(val))
            elif key == "config":
                updates.append("config_json = ?")
                params.append(json.dumps(val))
            elif key == "metrics":
                updates.append("metrics_json = ?")
                params.append(json.dumps(val))
            elif key == "explanations":
                updates.append("explanations_json = ?")
                params.append(json.dumps(val))
            else:
                updates.append(f"{key} = ?")
                params.append(val)
        params.append(exp_id)
        sql = f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(sql, params)
        conn.commit()


def get_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        if res.get("config_json"):
            res["config"] = json.loads(res["config_json"])
        if res.get("metrics_json"):
            res["metrics"] = json.loads(res["metrics_json"])
        if res.get("explanations_json"):
            res["explanations"] = json.loads(res["explanations_json"])
        return res


def list_experiments(
    dataset_id: Optional[str] = None,
    oversampler: Optional[str] = None,
    classifier: Optional[str] = None,
    status: Optional[str] = None,
    is_benchmark: Optional[bool] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        if dataset_id:
            query += " AND dataset_id = ?"
            params.append(dataset_id)
        if oversampler:
            query += " AND oversampler = ?"
            params.append(oversampler)
        if classifier:
            query += " AND classifier = ?"
            params.append(classifier)
        if status:
            query += " AND status = ?"
            params.append(status)
        if is_benchmark is not None:
            query += " AND is_benchmark = ?"
            params.append(1 if is_benchmark else 0)
            
        query += " ORDER BY CASE status WHEN 'running' THEN 1 WHEN 'pending' THEN 2 ELSE 3 END, created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        out = []
        for r in rows:
            item = dict(r)
            if item.get("config_json"):
                item["config"] = json.loads(item["config_json"])
            if item.get("metrics_json"):
                item["metrics"] = json.loads(item["metrics_json"])
            if item.get("explanations_json"):
                item["explanations"] = json.loads(item["explanations_json"])
            out.append(item)
        return out


def insert_custom_dataset(data: Dict[str, Any]):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO custom_datasets (
                id, name, domain, file_path, original_filename,
                n_samples, n_features, target_column, minority_class,
                minority_ratio, imbalance_ratio, feature_names_json,
                preprocessing_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["id"],
            data["name"],
            data.get("domain", "User Uploaded"),
            data["file_path"],
            data["original_filename"],
            data["n_samples"],
            data["n_features"],
            data["target_column"],
            str(data["minority_class"]),
            data["minority_ratio"],
            data["imbalance_ratio"],
            json.dumps(data.get("feature_names", [])),
            json.dumps(data.get("preprocessing", {})),
            data.get("created_at", datetime.now().isoformat())
        ))
        conn.commit()


def get_custom_dataset(ds_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM custom_datasets WHERE id = ?", (ds_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        if res.get("feature_names_json"):
            res["feature_names"] = json.loads(res["feature_names_json"])
        if res.get("preprocessing_json"):
            res["preprocessing"] = json.loads(res["preprocessing_json"])
        return res


def list_custom_datasets() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM custom_datasets ORDER BY created_at DESC")
        rows = cursor.fetchall()
        out = []
        for r in rows:
            item = dict(r)
            if item.get("feature_names_json"):
                item["feature_names"] = json.loads(item["feature_names_json"])
            if item.get("preprocessing_json"):
                item["preprocessing"] = json.loads(item["preprocessing_json"])
            out.append(item)
        return out
