"""
Quick live experiment test script to verify end-to-end execution.
"""
import urllib.request
import json
import time

def test_live_experiment_end_to_end():
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/experiments",
        data=json.dumps({
            "dataset_id": "breast_cancer",
            "oversampler": "shap_weighted",
            "classifier": "random_forest",
            "k_neighbors": 5,
            "n_splits": 3,
            "random_state": 42
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    resp = json.loads(urllib.request.urlopen(req).read().decode())
    exp_id = resp["id"]
    assert exp_id.startswith("exp-")

    for _ in range(25):
        time.sleep(1)
        st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8000/api/experiments/{exp_id}").read().decode())
        if st["status"] in ("completed", "failed"):
            break

    assert st["status"] == "completed"
    assert st["progress_percent"] == 100.0

    res = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8000/api/experiments/{exp_id}/results").read().decode())
    assert "metrics" in res
    assert res["metrics"]["pr_auc_mean"] > 0

    expl = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8000/api/experiments/{exp_id}/explanations").read().decode())
    assert len(expl["features"]) > 0
    assert expl["features"][0]["normalized_weight"] > 0
