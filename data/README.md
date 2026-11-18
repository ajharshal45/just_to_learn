# Benchmark datasets

The four datasets used by `notebooks/01`–`04`. Each is loaded by `src/experiment/data.py`, which records every
dataset-level step (dropped columns, duplicates, one-off subsampling) in the notebook output.

| Key | Dataset | File | Rows (as used) | Minority | Ratio | Features (raw) |
|---|---|---|---:|---:|---:|---|
| `breast_cancer` | Breast Cancer Wisconsin (Diagnostic) | `breast_cancer_wisconsin/data.csv` | 396 | 39 malignant | 9.2 : 1 | 30 numeric |
| `credit_card` | Credit Card Fraud Detection (ULB) | `credit_card/creditcard.csv` | 275,663 | 473 fraud | 581.8 : 1 | 29 numeric |
| `stroke` | Stroke Prediction | `kaggle_stroke/healthcare-dataset-stroke-data.csv` | 5,110 | 249 stroke | 19.5 : 1 | 5 numeric + 5 categorical |
| `nsl_kdd` | NSL-KDD network intrusion (KDDTrain+) | `nsl_kdd/KDDTrain+.txt` | 73,199 | 5,856 attacks | 11.5 : 1 | 37 numeric + 3 categorical |

## Sources, licences and verification

| Key | Source | Licence | Download | Verification |
|---|---|---|---|---|
| `breast_cancer` | Wolberg, Street & Mangasarian (1995), UCI ML Repository | CC BY 4.0 (UCI); Kaggle listing CC BY-NC-SA 4.0 | `kaggle datasets download -d uciml/breast-cancer-wisconsin-data` | All 569 × 30 feature values and the labels are identical, in the same row order, to the UCI copy bundled with scikit-learn (`load_breast_cancer`) |
| `credit_card` | Worldline & ULB Machine Learning Group (Dal Pozzolo et al., 2015) | ODbL 1.0 | `kaggle datasets download -d mlg-ulb/creditcardfraud` (already in the repository, see `credit_card/README.md`) | 284,807 rows, 492 fraud, as published |
| `stroke` | Kaggle `fedesoriano/stroke-prediction-dataset` (confidential original source, per the author) | Data files © original authors | `kaggle datasets download -d fedesoriano/stroke-prediction-dataset` (already in the repository) | 5,110 rows, 249 strokes, as published |
| `nsl_kdd` | Tavallaee et al. (2009), Canadian Institute for Cybersecurity, UNB ([page](https://www.unb.ca/cic/datasets/nsl.html)) | Free for research use (CIC); the Kaggle mirror lists none | `kaggle datasets download -d hassan06/nslkdd` → `KDDTrain+.txt` | MD5 `f5592a95d1d1428348dfa6ca9652a800`, identical to the independent GitHub mirror `defcom17/NSL_KDD`; 125,973 rows, 67,343 normal / 58,630 attack, 22 attack types |

`credit_card/creditcard.csv` (151 MB) is not committed because it exceeds GitHub's 100 MB limit; download it
with the command above (or let the archived notebook `notebooks/archive/02_credit_fraud_shap_smote_walkthrough.ipynb` fetch it).

## Target definitions and dataset-level choices

- **Breast cancer:** `diagnosis == 'M'` → 1. The naturally mild imbalance (357 benign / 212 malignant) is turned
  into 9:1 by keeping a fixed random subset of 39 malignant cases (seed 42). This is the same subset as the
  repository's earlier loader (`src/pipeline/datasets.load_breast_cancer_diagnostic`).
- **Credit card:** `Class == 1` → 1. `Time` is dropped (a row timestamp, not a property of the transaction), and
  9,144 exact duplicate rows are dropped (the 492 → 473 fraud reduction comes from duplicates).
- **Stroke:** `stroke == 1` → 1. `id` is dropped. `bmi` 'N/A' becomes missing (201 rows) and is imputed inside
  each training fold.
- **NSL-KDD:** any attack label → 1. `difficulty` (metadata) and `label` are dropped, along with 9 duplicates and the
  constant `num_outbound_cmds` column. Attacks are 47% of the original file. To create the imbalanced task, a
  fixed random 5,856 of the attack records are kept (seed 42), so attacks make up 8% of rows (11.5:1). All
  normal traffic is kept.
