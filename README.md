# Explainability-Aware Synthetic Oversampling for Imbalanced Tabular Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Does using SHAP feature attributions *inside* the oversampler produce better synthetic minority samples, and
> better minority-class classification, than conventional oversampling?**

## Research objective

SMOTE creates a synthetic minority sample by interpolating between a minority row and one of its nearest
minority neighbours. "Nearest" is plain Euclidean distance, so every feature counts equally: noisy or
uninformative features can decide which rows get paired. SHAP is normally used *after* training, to explain a
model. This project uses SHAP *before* training. A preliminary model trained on the training data tells the
oversampler which features matter, and the oversampler uses that to choose neighbours. The research question is
whether this produces better synthetic samples and better minority-class detection. Everything is measured
without data leakage, on four real datasets, against established oversamplers.

## The benchmark at a glance

| | |
|---|---|
| **Datasets (4)** | Breast Cancer Wisconsin (9.2:1), Credit Card Fraud (581.8:1), Stroke Prediction (19.5:1), NSL-KDD intrusion (11.5:1). See [data/README.md](data/README.md) |
| **Compared methods** | No Sampling (reference) · Random Oversampling · SMOTE · Borderline-SMOTE · **SHAP-Filtered SMOTE** · **SHAP-Weighted SMOTE** |
| **Downstream models** | Logistic Regression · Random Forest · XGBoost (identical settings for every method) |
| **Protocol** | Stratified 25% hold-out test set (never resampled) + 2 × 5-fold stratified CV on the rest; preprocessing, SHAP and oversampling fitted inside training folds only |
| **Metrics** | Accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, balanced accuracy, G-mean, Brier score, ECE, confusion matrices; synthetic-sample fidelity; SHAP attribution stability; Wilcoxon / Friedman tests |

## Methods

| Method | Status | How it creates minority samples |
|---|---|---|
| No Sampling | reference | none |
| Random Oversampling | established | duplicates random minority rows |
| SMOTE (Chawla et al., 2002) | established | $x_{new} = x_a + \lambda(x_{nn} - x_a)$, $\lambda\sim U(0,1)$, $x_{nn}$ one of the 5 nearest minority neighbours (Euclidean) |
| Borderline-SMOTE (Han et al., 2005) | established | SMOTE, anchors restricted to minority rows near the boundary (DANGER rows) |
| **SHAP-Filtered SMOTE** | **proposed** | SMOTE whose neighbours are searched only on the features holding 90% of the SHAP importance |
| **SHAP-Weighted SMOTE** | **proposed** | SMOTE whose neighbour distance is $d_W(a,b)=\sqrt{\sum_j w_j (a_j-b_j)^2}$, with $w_j$ the normalised mean absolute SHAP value |

### SHAP-guided SMOTE in one paragraph
Inside each training fold, a preliminary Random Forest (50 trees, depth 5) is fitted and explained with
TreeSHAP. Each feature's mean absolute SHAP value becomes a weight $w_j$, and the weights sum to 1.
**SHAP-Filtered SMOTE** keeps the top features that together hold 90% of the weight and searches neighbours on
those features only. **SHAP-Weighted SMOTE** keeps every feature but scales column $j$ by $\sqrt{w_j}$, so high-SHAP
features dominate the neighbour distance. Both then interpolate exactly like SMOTE, in the full, unweighted
space. Anchors, neighbour slots and $\lambda$ are drawn with the same seed for every SMOTE-family method, so the
only thing SHAP changes is *which neighbour is paired with which anchor*. The full mathematical formulation as
implemented, including the large-data SHAP option, is in
[docs/methodology_shap_guided_smote.md](docs/methodology_shap_guided_smote.md).

## Headline results

Full tables, figures and written interpretations are in the notebooks; `05_cross_dataset_comparison.ipynb`
consolidates them. In short, under the leak-free protocol:

* **No method dominates.** Over 4 datasets × 3 models, the Friedman test finds no significant difference between
  the six methods on PR-AUC (p = 0.056) or F1 (p = 0.107). *No Sampling* is the best single choice by F1 in 7 of 12
  dataset × model blocks. On Credit Card Fraud, every 1:1 oversampler is far behind it (XGBoost F1 0.86 vs ≤ 0.75).
* **SHAP-guided vs conventional SMOTE is situational.** With the tree models on Credit Card Fraud, both SHAP methods
  are better: XGBoost F1 +0.025 and precision +0.02 (p ≤ 0.014), with recall unchanged. On Breast Cancer and
  NSL-KDD they tie. On Stroke they cut tree-model recall by about half (p = 0.002, every fold).
* **Why Stroke fails:** SHAP down-weights one-hot categorical columns, so it pairs patients from different
  categories. Interpolation then produces fractional category values that make synthetic rows easy to recognise
  (detectability AUC 0.93–0.96 vs 0.81 for SMOTE), and the tree models learn that artefact.
* **SHAP-Filtered SMOTE is the more robust variant:** average rank 3.29 (PR-AUC) and 3.25 (F1), against 3.75 and
  4.33 for SHAP-Weighted SMOTE and 4.29 and 3.75 for SMOTE. The SHAP step makes oversampling 3–12× slower.

## Repository structure

```
SHAP+SMOTE/
├── data/                               # four datasets + data/README.md (sources, licences, checks)
├── notebooks/
│   ├── 01_breast_cancer_experiment.ipynb       ┐
│   ├── 02_credit_card_fraud_experiment.ipynb   │ the same experiment on each dataset,
│   ├── 03_stroke_prediction_experiment.ipynb   │ generated from one template
│   ├── 04_nsl_kdd_intrusion_experiment.ipynb   ┘
│   ├── 05_cross_dataset_comparison.ipynb       # consolidates the four
│   ├── interpretations/                        # written conclusions inserted into the notebooks
│   └── archive/02_credit_fraud_shap_smote_walkthrough.ipynb  # earlier walkthrough on simulated fraud data
├── results/<dataset>/                  # generated by the notebooks
│   ├── metrics/                        # CV fold metrics, test metrics, tests, stability, quality
│   ├── synthetic_samples/              # every synthetic row with its parents and lambda
│   ├── neighbour_analysis/             # neighbour tables, SHAP weights, SHAP rationale
│   └── plots/
├── src/
│   ├── oversampling/
│   │   ├── base.py                     # MASV SHAP weights (compute_masv_weights)
│   │   ├── shap_filtered_smote.py      # proposed: SHAP-Filtered SMOTE
│   │   ├── shap_weighted_smote.py      # proposed: SHAP-Weighted SMOTE
│   │   ├── provenance.py               # shared generation engine + SMOTE, Borderline-SMOTE, ROS, No Sampling
│   │   ├── classical.py, hybrids.py, baselines.py   # imbalanced-learn wrappers (earlier benchmark)
│   ├── experiment/                     # four-dataset benchmark
│   │   ├── config.py                   # every experimental setting (single source of truth)
│   │   ├── data.py                     # dataset loaders + ingestion reports
│   │   ├── preprocessing.py            # train-only imputation / encoding / scaling
│   │   ├── runner.py                   # leak-free CV, hold-out evaluation, downstream SHAP, tests
│   │   ├── inspection.py               # neighbour, generation and SHAP-rationale tables
│   │   ├── plots.py, outputs.py, summary.py
│   ├── pipeline/, evaluation/          # earlier benchmark pipeline and metrics (still used)
├── scripts/
│   ├── build_experiment_notebooks.py   # writes notebooks 01-05 from one template
│   └── insert_interpretations.py       # puts the written conclusions into the executed notebooks
├── experiments/                        # earlier YAML-driven benchmark (synthetic/simulated data)
├── docs/                               # literature review, methodology, proofs
├── presentation/
└── tests/                              # pytest suite (incl. provenance and leakage tests)
```

<<<<<<< HEAD
---

## Zero-Data-Leakage Protocol

```
                          FULL DATASET
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     TRAINING FOLDS                        HELD-OUT TEST FOLD
   (e.g., 80% of data)                    (e.g., 20% of data)
            │                                     │
            ├──> [1. Fit Scaler (MinMax/Standard)]───> Transform Test Fold
            │                                     │
            ├──> [2. Fit Base Estimator]          │
            │           │                         │
            │           ▼                         │
            ├──> [3. Calculate SHAP Values]       │
            │           │                         │
            │           ▼                         │
            ├──> [4. Compute MASV Weights]        │
            │           │                         │
            │           ▼                         │
            ├──> [5. Generate Synthetic Samples]  │
            │           │                         │
            │           ▼                         │
            └──> [6. Train Final Classifier] ─────┼──> Predict on Untouched
                                                  │    Test Fold
                                                  ▼
                                       [7. Unbiased Evaluation]
```

---

## Research Web Application Platform

A full-stack academic ML research web platform built with a **FastAPI** Python backend and a **React + TypeScript + Vite + Tailwind CSS** frontend.

### Architecture Overview

- **Backend (`src/api/`):** RESTful endpoints with Pydantic request/response validation, asynchronous nested cross-validation execution with thread pools, persistent SQLite experiment archiving (`data/experiments.db`), and secure dataset ingestion.
- **Frontend (`frontend/`):** Professional scientific dashboard featuring an Overview Dashboard, Dataset Explorer with CSV uploader & paginated preview, Experiment Lab with mathematical oversampling formulations, Experiment Runs & Live History, SHAP Explainability Explorer with interactive horizontal attributions, Multi-dimensional Results Comparison, Evidence-based Leakage Diagnostics, and Academic Report Generator with JSON/CSV/Markdown exports.

### Quick Startup Commands

#### 1. Start the Backend API Server
```bash
# From repository root
python run_backend.py
# Server runs on http://127.0.0.1:8000
# Interactive Swagger API Docs: http://127.0.0.1:8000/docs
```

#### 2. Start the Frontend Application
```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:5173
```

### Core API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Backend status, Python version, available ML classifiers & oversamplers |
| `GET` | `/api/datasets` | List registered benchmark datasets and uploaded custom datasets |
| `POST` | `/api/datasets/upload` | Secure CSV upload with binary target validation |
| `GET` | `/api/datasets/{id}/summary` | Class balance, missing values, and safe preprocessing strategy |
| `GET` | `/api/datasets/{id}/preview` | Paginated data preview prior to in-fold transformation |
| `GET` | `/api/experiments` | List stored experiments (filterable by dataset, method, status) |
| `POST` | `/api/experiments` | Launch live experiment under Zero-Leakage Nested CV |
| `GET` | `/api/experiments/{id}` | Live execution status, current step, and progress percentage |
| `GET` | `/api/experiments/{id}/results` | Full multi-dimensional evaluation metrics and fold results |
| `GET` | `/api/experiments/{id}/explanations` | Authentic Mean Absolute SHAP Values (MASV) & feature weights |
| `GET` | `/api/experiments/{id}/export` | Export experiment in JSON, CSV, or Markdown report format |
| `GET` | `/api/diagnostics/leakage` | Zero-Data-Leakage protocol verification & empirical proof data |
| `GET` | `/api/benchmarks/summary` | Top-level summary across all 108 empirical benchmark runs |

### Zero-Data-Leakage Protocol Workflow

1. **Stratified Split:** Split the dataset into outer training and held-out test partitions.
2. **Train Scaler:** Fit `StandardScaler` strictly on the training partition; test set is transformed independently.
3. **In-Fold Preliminary SHAP Model:** Train preliminary estimator (Random Forest) strictly on training data; compute Mean Absolute SHAP Values (MASV): $\text{MASV}_j = \frac{1}{N}\sum_i |\phi_j(x_i)|$.
4. **Weighting / Subspace Formulation:** Compute metric tensor weights $w_j = \text{MASV}_j / \sum \text{MASV}$ or select top predictive feature subspace.
5. **In-Fold Oversampling:** Synthesize minority instances strictly within the training fold.
6. **Classifier Training:** Train final classifier on resampled training data.
7. **Unbiased Evaluation:** Predict on untouched held-out test fold to calculate PR-AUC, ECE, F1, and attribution stability.

---

## Installation & Setup
=======
## Installation
>>>>>>> 34c50c8c2a9c6e4ae82122ec3a57e0f8209f88c5

```bash
git clone https://github.com/hasanr-26/SHAP-SMOTE.git
cd SHAP-SMOTE
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Three datasets are committed. The credit-card file (151 MB) is too large for GitHub:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/credit_card --unzip
```

(The Kaggle CLI needs an API token in `~/.kaggle/kaggle.json`.)

## Running the experiments

Open and run any notebook top to bottom in Jupyter, or execute them from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/01_breast_cancer_experiment.ipynb
```

Measured run times on a 16-thread desktop (10 CV folds × 6 methods × 3 models, plus the final fit and all
inspection tables): Breast Cancer ~4 min, Stroke ~6 min, NSL-KDD ~1.5 h, Credit Card ~2.3 h. Run `05_cross_dataset_comparison.ipynb`
after the four dataset notebooks; it reads `results/<dataset>/metrics/`.

To reproduce everything exactly:

```bash
python scripts/build_experiment_notebooks.py --cross   # regenerate notebooks 01-05 from the template
# execute 01-04, then 05 (commands above)
python scripts/insert_interpretations.py               # re-insert the written conclusions
pytest -q                                              # algorithm, provenance and leakage tests
```

All randomness is seeded (`SEED = 42` in `src/experiment/config.py`), so reruns produce the same numbers.
Large synthetic-sample files (Credit Card `*.csv.gz` and NSL-KDD `*.csv`, 15–40 MB each) are git-ignored and regenerated by the notebooks.

## Reproducibility and leakage safeguards

* One configuration file (`src/experiment/config.py`); the four notebooks differ only in the dataset key.
* Fresh preprocessor, sampler and models in every fold; nothing fitted is reused across folds.
* `tests/test_experiment_pipeline.py` checks that every sampler receives exactly the training rows of its fold
  and never a validation row, and that the preprocessor learns from training rows only.
* `tests/test_provenance_samplers.py` checks that each saved synthetic record (parents + $\lambda$) rebuilds the
  generated row exactly, and that SMOTE's neighbour graph and Borderline-SMOTE's DANGER set match imbalanced-learn.

## Earlier work in this repository

The literature review (`docs/literature/`), the leakage proof and stroke benchmark
(`docs/objectives_proof_and_kaggle_benchmark.md`) and the YAML benchmark on simulated data (`experiments/`) remain unchanged. The four-dataset benchmark above replaces the simulated credit-fraud and
NSL-KDD data with the real datasets. The earlier credit-fraud walkthrough is kept in `notebooks/archive/` because the
presentation quotes its simulation results (its known ground truth shows SHAP recovering all 7 hidden fraud features).

## References

* Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
* Han, H., Wang, W.-Y., & Mao, B.-H. (2005). Borderline-SMOTE. *ICIC 2005*, LNCS 3644, 878–887.
* Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 30*.
* Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE SSCI*.
* Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A Detailed Analysis of the KDD CUP 99 Data Set. *IEEE CISDA*.
* Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast Cancer Wisconsin (Diagnostic). UCI ML Repository.
