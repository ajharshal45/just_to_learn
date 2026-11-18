# 07 — Running the System

> **Use this for:** the live demo, and so anyone on the team can reproduce the numbers.
> Checked on 2026-11-17: Windows 11, Python 3.12.8, shap 0.54.1, imbalanced-learn 0.14.2,
> scikit-learn 1.9.1, xgboost 3.4.1. **34 out of 34 tests pass.**

---

## 1. Setup (once)

```bash
# from the repo root (SHAP+SMOTE/)
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Git Bash) / macOS / Linux
source .venv/Scripts/activate      # or: source .venv/bin/activate

pip install -r requirements.txt

# the credit-card file (151 MB) is not in git; download it once (needs a Kaggle API token)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/credit_card --unzip
```

`requirements.txt` installs numpy, scipy, pandas, scikit-learn, imbalanced-learn, shap, xgboost,
matplotlib, seaborn, pyyaml, pytest, Jupyter and the Kaggle tool. The `.venv/` folder is ignored by git.
The other three datasets are already in `data/` (see [data/README.md](../data/README.md)).

> ⚠️ If you see `ModuleNotFoundError: No module named 'shap'`, you forgot to activate the venv.

---

## 2. The demo commands

```bash
python -m pytest -q                                      # 1. prove it works: "34 passed" (about 2 min)
# 2. the main demo: open notebooks/01_breast_cancer_experiment.ipynb and scroll (outputs are saved)
#    or re-run it in about 4 minutes:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_breast_cancer_experiment.ipynb
python scripts/demo_benchmark.py                         # 3. small benchmark table on made-up data (about 1 min)
python scripts/generate_proof_and_kaggle_benchmarks.py   # 4. rebuilds the 5 proof figures (incl. the leakage trap)
```

The **four dataset notebooks are the main results**. The commands in the table below belong to the earlier,
simulated-data benchmark and are kept for the leakage proof and the quick demo.

| Command | What it does | Output |
|---|---|---|
| `demo_benchmark.py` | 4 methods × 2 models on 500 made-up rows, 5 folds | A table printed on screen (see [06](06_objective4_evaluation_stability.md) §6.3) |
| `run_benchmark.py` | Earlier benchmark: 9 methods × 3 models × 4 (partly simulated) datasets, using [benchmark_default.yaml](../experiments/configs/benchmark_default.yaml) | [experiments/results/](../experiments/results/): CSV, JSON, LaTeX, charts, report |
| `run_benchmark.py --quick` | Smaller, faster version | [experiments/results_quick/](../experiments/results_quick/) |
| `generate_proof_and_kaggle_benchmarks.py` | Leakage proof, weighting comparison, stroke benchmark | [docs/images/proofs/](../docs/images/proofs/) |

`run_benchmark.py` options: `--datasets`, `--oversamplers`, `--classifiers`, `--n_splits`,
`--n_repeats`, `--output_dir`, `--config`.

You may see harmless SHAP messages like *"Background dataset has … samples but max_samples=100"*.
You can ignore them.

---

## 3. The notebooks (best for a visual demo)

| Notebook | What it shows | Run time |
|---|---|---|
| [01_breast_cancer_experiment.ipynb](../notebooks/01_breast_cancer_experiment.ipynb) | The full benchmark on Breast Cancer | ~4 min |
| [02_credit_card_fraud_experiment.ipynb](../notebooks/02_credit_card_fraud_experiment.ipynb) | The full benchmark on real Credit Card Fraud | ~2.3 h |
| [03_stroke_prediction_experiment.ipynb](../notebooks/03_stroke_prediction_experiment.ipynb) | The full benchmark on Stroke | ~6 min |
| [04_nsl_kdd_intrusion_experiment.ipynb](../notebooks/04_nsl_kdd_intrusion_experiment.ipynb) | The full benchmark on NSL-KDD network attacks | ~1.5 h |
| [05_cross_dataset_comparison.ipynb](../notebooks/05_cross_dataset_comparison.ipynb) | Compares the four results (runs in seconds; run 01–04 first) | ~1 min |
| [archive/02_credit_fraud_shap_smote_walkthrough.ipynb](../notebooks/archive/02_credit_fraud_shap_smote_walkthrough.ipynb) | Earlier walkthrough on *simulated* fraud data with a known answer (source of the "7 of 7 features" result) | about 3 minutes |

Notebooks 01–04 are the same notebook run on different data; [11](11_notebook_walkthrough_cell_by_cell.md) explains
them cell by cell.

Open them in VS Code. If it asks you to **Select Kernel**, choose **Python Environments → .venv**.
The charts are already saved inside, so you can scroll through without running anything.

---

## 4. Using it from Python (for a live-coding moment)

```python
import numpy as np
from sklearn.datasets import make_classification
from src.oversampling import SHAPFilteredSMOTE, SMOTEWithProvenance

X, y = make_classification(n_samples=500, n_features=15, n_informative=5,
                           weights=[0.9, 0.1], random_state=42)

# See what SHAP "thinks" before any benchmarking
sm = SHAPFilteredSMOTE(cumulative_threshold=0.90, random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print(sm.feature_weights_.round(3))        # importance of each feature (adds up to 1)
print(sm.selected_feature_indices_)        # which features were kept for the neighbour search
print(len(y), "->", len(y_res))            # rows before and after balancing

# Every fake row's "birth certificate": its two parents and lambda
print(sm.generation_log_.head())           # anchor, neighbour, neighbour rank, lambda, distance

# Compare with plain SMOTE: which neighbours did SHAP change?
plain = SMOTEWithProvenance(random_state=42)
plain.fit_resample(X, y)
same = (np.sort(sm.neighbor_indices_, 1) == np.sort(plain.neighbor_indices_, 1)).all(axis=1).mean()
print(f"{same:.0%} of rare rows have exactly the same 5 neighbours as plain SMOTE")
```

To run a whole dataset from Python (what the notebooks do):

```python
from src.experiment import data as D, preprocessing as P, runner as R
df, spec = D.load_dataset("stroke")          # or breast_cancer, credit_card, nsl_kdd
dev, test = P.split_dev_test(df)
cv = R.run_cross_validation(dev, spec)       # 10 leak-free rounds, all methods and models (~3 min for stroke)
print(cv.mean_table(["precision", "recall", "f1", "pr_auc"]).round(3))
```

**The main settings** all live in [src/experiment/config.py](../src/experiment/config.py) and are the same for
every dataset:

| Setting | Default | What it does |
|---|---|---|
| `SAMPLING_STRATEGY` | `"auto"` | Make the rare class as big as the common one (1:1) |
| `K_NEIGHBORS` | 5 | How many neighbours SMOTE-type methods choose from |
| `M_NEIGHBORS` | 10 | How many neighbours Borderline-SMOTE checks to find "danger" rows |
| `SHAP_CUMULATIVE_THRESHOLD` | 0.90 | SHAP-Filtered keeps the features holding 90% of the importance |
| `SHAP_MAX_ROWS` | 20,000 | On big data, SHAP looks at all rare rows plus a sample of common rows (results stay unbiased) |
| `TEST_SIZE`, `CV_SPLITS`, `CV_REPEATS` | 25%, 5, 2 | The locked test set and the 10 cross-validation rounds |
| `SEED` | 42 | Same result every time |

---

## 5. Proof-of-trap script

This script produced the second table in [05](05_objective3_leak_free_benchmark_trap.md) §3. It's not
in the repo yet. To use it, save it as `scripts/proof_of_trap.py` and run
`python scripts/proof_of_trap.py .` from the repo root. (Figure 2, the first proof, comes from
`generate_proof_and_kaggle_benchmarks.py`, which is already in the repo.)

```python
"""Leaky (oversample-then-split) vs leak-free (split-then-oversample) comparison."""
import sys
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

sys.path.insert(0, sys.argv[1])
from src.pipeline.leak_free_cv import run_leak_free_benchmark
from src.oversampling.baselines import get_baseline_oversampler

X, y = make_classification(n_samples=2000, n_features=20, n_informative=4, n_redundant=2,
                           weights=[0.97, 0.03], flip_y=0.02, class_sep=0.8, random_state=7)
print("shape", X.shape, "counts", np.bincount(y))

for cname, mk in {"RF": lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
                  "LR": lambda: LogisticRegression(max_iter=2000)}.items():
    # LEAKY: SMOTE on full data, then CV on the resampled set (the trap)
    Xr, yr = SMOTE(random_state=0).fit_resample(X, y)
    pr, rec, f1 = [], [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xr, yr):
        m = mk().fit(Xr[tr], yr[tr])
        p = m.predict_proba(Xr[te])[:, 1]
        pr.append(average_precision_score(yr[te], p)); rec.append(recall_score(yr[te], p > .5)); f1.append(f1_score(yr[te], p > .5))
    print(f"{cname} LEAKY      PR-AUC={np.mean(pr):.3f} recall={np.mean(rec):.3f} F1={np.mean(f1):.3f}")

    res = run_leak_free_benchmark(X, y, get_baseline_oversampler("smote", random_state=0), mk(), n_splits=5)
    print(f"{cname} LEAK-FREE  PR-AUC={res['pr_auc_mean']:.3f} recall={res['minority_recall_mean']:.3f} F1={res['minority_f1_mean']:.3f}")
```

---

## 6. What the tests check

| File | Checks |
|---|---|
| `test_provenance_samplers.py` | Every fake row can be rebuilt exactly from its saved parents and λ; our SMOTE picks the same neighbours as the standard library's SMOTE; our Borderline-SMOTE finds the same "danger" rows; the SHAP methods search in the right space |
| `test_experiment_pipeline.py` | **No leakage:** in every round the samplers receive exactly the training rows and never a validation row; the preparation steps learn from training rows only; the metrics are computed correctly; the capped SHAP sample gives the same weights; the realism check gives about 0.5 for realistic data |
| `test_shap_filtered_smote.py` | Weights are not negative and add up to 1; classes are balanced afterwards; `top_k` keeps exactly k features |
| `test_shap_weighted_smote.py` | Classes are balanced; a custom balance ratio is respected |
| `test_classical_and_hybrids.py` | The SMOTE, Borderline, ADASYN, SMOTE-ENN and SMOTE-Tomek wrappers work |
| `test_leak_free_cv.py` | The earlier pipeline fits and predicts; ECE is between 0 and 1 |
| `test_datasets.py` | The earlier dataset loaders load with the right shapes |
| `test_stability_and_fidelity.py` | The stability and fidelity scores, and the LaTeX table output |

---

## 7. Five things to remember about the system

1. **Everything that learns lives inside the training part.** That includes the scaler, the SHAP
   scout model, the weights, the oversampler and the final model. This one decision is what makes
   the numbers trustworthy.
2. **SHAP only changes how neighbours are picked.** New rows are made exactly as plain SMOTE makes
   them, so any difference comes from the SHAP guidance.
3. **The test part keeps its real imbalance.** We never grade on resampled data.
4. **PR-AUC is the headline, not accuracy.** It's always shown together with calibration and stability.
5. **Everything is seeded.** The same command gives the same table.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'shap'` / `'imblearn'` | Activate `.venv` and run `pip install -r requirements.txt` |
| `No module named 'src'` | Run from the repo root |
| `ValueError: … binary classification` | The SHAP oversamplers only handle two classes. Convert other targets to "one class vs the rest" first. |
| Very slow on big data | Expected for Credit Card (~2.3 h) and NSL-KDD (~1.5 h). SHAP already uses a capped sample (`SHAP_MAX_ROWS`); most of the time is model training. Run them in the background with the `jupyter nbconvert` command above |
| Stability τ shows exactly 0.0 (earlier benchmark only) | SHAP failed quietly in every fold, a known issue of the old runner. The new notebooks stop with an error instead. |
| `FileNotFoundError: ... creditcard.csv` | Download it with the `kaggle` command in §1 |
| Notebook opens as raw text | Right-click the tab → **Reopen Editor With…** → **Jupyter Notebook** |
| Notebook charts don't appear when run | Keep the `%matplotlib inline` line in the first cell. `src.evaluation` switches matplotlib to a no-display mode. |
