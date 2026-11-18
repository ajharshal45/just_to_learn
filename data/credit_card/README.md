# Credit Card Fraud Dataset (ULB / Kaggle)

| File | Contents |
|---|---|
| `creditcard.csv` | Raw data: 284,807 transactions, 492 fraud (0.17%). Columns `Time`, `V1`–`V28`, `Amount`, `Class` |
| `generated/*` | Files written by the **earlier** walkthrough notebook (`notebooks/archive/`), listed below |
| `generated/creditcard_train_balanced_classical_smote.csv` | Training fold + synthetic fraud from Classical SMOTE |
| `generated/creditcard_train_balanced_shap_filtered_smote.csv` | Training fold + synthetic fraud from SHAP-Filtered SMOTE |
| `generated/creditcard_train_balanced_shap_weighted_smote.csv` | Training fold + synthetic fraud from SHAP-Weighted SMOTE |
| `generated/creditcard_test_holdout.csv` | Untouched 25% hold-out test fold (never oversampled) |

## In the four-dataset benchmark

`notebooks/02_credit_card_fraud_experiment.ipynb` loads this file through `src/experiment/data.py`: it drops `Time`
and **9,144 exact duplicate rows**, leaving **275,663 transactions with 473 frauds** (582 : 1). It does not use the
`generated/` files. Its own outputs (metrics, every synthetic row with its two parent transactions, neighbour tables,
charts) go to `results/credit_card/`. The synthetic-sample files there are compressed (`.csv.gz`) and are not committed.

## The earlier `generated/` files

Generated files contain the feature columns (`V1`–`V28`, `Amount`), `Class`, `is_synthetic` (0 = real transaction,
1 = new synthetic row) and `method`.

## What the columns mean

The bank did not publish the raw transaction fields (merchant, location, card type, …) for confidentiality. It
published their **PCA projection** instead, so most columns are unnamed decimal numbers by design. They are not
corrupted. Every `V` column has mean 0, and the spread shrinks from `V1` (std 1.96) to `V28` (std 0.33), which is the
ordering PCA produces.

| Column | Meaning | Used as a model feature? |
|---|---|---|
| `Time` | Seconds since the first transaction in the file (0 to 172,792, i.e. 48 hours) | No: a row timestamp, not a property of the transaction |
| `V1`–`V28` | Principal components of the hidden raw fields | Yes |
| `Amount` | Transaction amount in euros | Yes |
| `Class` | `1` = fraud, `0` = legit | Target |

**Source:** [Kaggle `mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The copy
here is the original Kaggle file, downloaded from TensorFlow's public mirror
(`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`). An earlier copy came from OpenML
(data_id 1597), which leaves out `Time`; the other 30 columns are identical in both.

The CSVs are not committed (they exceed GitHub's 100 MB limit). Running
`notebooks/archive/02_credit_fraud_shap_smote_walkthrough.ipynb` (an earlier walkthrough, archived) downloads `creditcard.csv` if it is missing and writes the
`generated/` files (Sections 14–17). Section 18 walks through how individual synthetic rows are built.
