# 13 — Credit Card Fraud Notebook: What Each Cell Shows

> Notebook: `notebooks/02_credit_card_fraud_experiment.ipynb`. The cells are explained in
> [11](11_notebook_walkthrough_cell_by_cell.md); this file gives **this dataset's actual outputs**, cell by cell,
> in plain words. The rare class is a **fraudulent card transaction**.

---

## Setup and data (cells 1–5)

* **Cell 2:** `DATASET = "credit_card"`. Settings identical to the other notebooks.
* **Cell 4:** Real transactions from European cardholders over two days in 2013 (Worldline and the Université Libre
  de Bruxelles, via Kaggle). To protect privacy, the bank published most columns only as anonymous mathematical
  combinations named `V1`–`V28`, plus the transaction `Amount`. We dropped `Time` (just a timestamp) and **9,144
  exact duplicate rows**. Result: **275,663 rows × 29 features**.
* **Cell 5:** All features are numbers, nothing missing. 8,371 rows have an extreme value; they're kept, because
  fraud is often exactly the extreme.

## Looking at the data (cells 7–9)

* **Cell 7:** 275,190 normal and only **473 fraud (0.17%)**: an imbalance of **582 : 1**. A model that never says
  "fraud" is 99.8% accurate, which shows why accuracy is useless here.
* **Cell 8:** The most separating features are **`V14` (KS 0.84), `V10` (0.80), `V12` (0.78), `V4`, `V11`**.
* **Cell 9:** The `V` columns are built to be unrelated to each other, so the heatmap is mostly pale, except their
  links to the target.

## Preparing the data (cell 11)

* Development set: **206,747 rows, 355 fraud**. Test set: **68,916 rows, 118 fraud**.
* 29 numeric model columns, scaled.

## Making the fake rows (cells 13–24)

* **Cell 13:** Every method turns **355 real frauds into 206,392**, adding **206,037 fake fraud rows** (580 fakes
  per real fraud). Borderline-SMOTE starts only from **40 DANGER frauds**. SHAP-Filtered SMOTE judges similarity on
  **13 of the 29 features**.
* **Cell 15:** Borderline labels: **263 SAFE, 40 DANGER, 52 NOISE**. Neighbour overlap with SMOTE: **SHAP-Filtered
  0.56, SHAP-Weighted 0.52**, so SHAP changes about half of every fraud's neighbours.
* **Cell 18:** Fake row #0 of SMOTE, SHAP-Filtered and SHAP-Weighted all start from fraud **row 215984** with
  λ = 0.4389, and each picks a **different** partner: row 204503 (SMOTE), 77348 (SHAP-Filtered), 212516
  (SHAP-Weighted).
* **Cell 20:** SHAP's top features: **`V14` (weight 0.19), `V17` (0.13), `V12` (0.12), `V10` (0.10), `V16`**. These
  are the components the project's earlier credit-card work also found to carry the fraud signal.
* **Cell 21:** In the first 8 fake rows, **38%** (SHAP-Filtered) and **62%** (SHAP-Weighted) of parent pairs would
  **not** have been neighbours in plain SMOTE.
* **Cell 23 (quality):**

  | Method | Common-class neighbours (real frauds: 25%) | Realistic spread (KS pass) | Correlations kept | Detectability (0.5 = can't tell) |
  |---|---|---|---|---|
  | SMOTE | 18% | 97% | 0.997 | 0.56 |
  | Borderline-SMOTE | **35%** | **0%** | 0.93 | **0.78** |
  | SHAP-Filtered SMOTE | 18% | 97% | 0.997 | 0.58 |
  | SHAP-Weighted SMOTE | 19% | 97% | 0.997 | 0.58 |

  In plain words: SMOTE and the SHAP methods make realistic fake frauds. Borderline-SMOTE's fakes fail every
  realism check.
* **Cell 24:** All rows saved to `results/credit_card/synthetic_samples/*.csv.gz` (206,037 rows per method,
  compressed; these files are not in GitHub because of their size).

## The fair test (cells 26–29) — 10 rounds, 105 minutes

Average over the 10 rounds (cell 27):

| Model | Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | **No Sampling** | **0.88** | 0.62 | **0.72** | **0.75** |
| | SMOTE | 0.05 | 0.91 | 0.10 | 0.73 |
| | SHAP-Filtered / SHAP-Weighted | 0.06 / 0.05 | 0.91 / 0.91 | 0.10 / 0.10 | 0.72 / 0.73 |
| Random Forest | **No Sampling** | **0.93** | 0.75 | **0.83** | **0.82** |
| | SMOTE | 0.35 | 0.86 | 0.49 | 0.74 |
| | SHAP-Filtered | 0.37 | 0.86 | 0.52 | 0.76 |
| | SHAP-Weighted | 0.34 | 0.86 | 0.48 | 0.76 |
| XGBoost | **No Sampling** | **0.94** | 0.80 | **0.86** | **0.84** |
| | Borderline-SMOTE | 0.69 | 0.83 | 0.75 | 0.82 |
| | SMOTE | 0.25 | 0.87 | 0.39 | 0.81 |
| | SHAP-Filtered | 0.28 | 0.86 | 0.42 | 0.81 |
| | SHAP-Weighted | 0.28 | 0.86 | 0.42 | 0.81 |

* **Cell 28 (Graph 1):** The No Sampling bars win precision, F1 and PR-AUC for every model. The oversamplers only
  win recall.
* **Cell 29 (significance), SHAP methods vs SMOTE:** this dataset gives our methods their **best results**:
  * XGBoost: F1 **+0.025** for both SHAP methods (better in 8 and 9 of 10 rounds, p = 0.010 and 0.014) and
    precision **+0.02** (p = 0.010). **Significant.**
  * Random Forest: PR-AUC **+0.020** for SHAP-Filtered (7 of 10 rounds, p = 0.037). **Significant.**
  * Recall is unchanged everywhere. Logistic Regression shows no difference.

## The locked test set (cells 31–32) — 118 frauds among 68,916 transactions

| XGBoost | Frauds caught | Missed | False alarms |
|---|---|---|---|
| No Sampling | 84 | 34 | **8** |
| SMOTE | 99 | 19 | 315 |
| SHAP-Filtered SMOTE | 99 | 19 | 345 |
| SHAP-Weighted SMOTE | 98 | 20 | 327 |
| Borderline-SMOTE | 93 | 25 | 52 |

* Oversampling catches about 15 more frauds but raises false alarms from 8 to over 300. With Logistic Regression,
  oversampling causes **about 1,500–1,750 false alarms**.
* On this single test, the SHAP methods' XGBoost gain from the 10 rounds does **not** show up. Their Random Forest gain
  does: SHAP-Filtered F1 0.53 vs SMOTE 0.51.
* **Calibration (cell 32):** the No Sampling models' probabilities can be trusted almost perfectly (calibration error
  ECE ≤ 0.001). After oversampling, the models over-state the fraud risk (ECE up to 0.086).

## What the models pay attention to (cells 34–35)

* **Cell 34 (Graph 2):** All models lean mostly on `V14`, then `V10`, `V4`, `V12`, `V17`. Oversampling changes the order noticeably
  (ranking similarity to No Sampling 0.40–0.76).
* **Cell 35:** Random Forest's explanations are the most stable (Kendall τ 0.77–0.85). With the tree models the SHAP
  methods are slightly more stable than SMOTE (Random Forest 0.82 vs 0.80); with Logistic Regression slightly less (0.74 vs 0.77).

## Conclusion (cells 39–40), in plain words

* **Not oversampling wins clearly.** Turning 355 frauds into 206,392 by inventing 580 fakes per real one makes the
  models flag far too many normal transactions.
* **Compared with plain SMOTE, SHAP guidance helps the tree models**: fewer false alarms with the same number of
  frauds caught (XGBoost F1 +0.025, significant). These are the proposed methods' strongest results in the project.
* **But the gain is second-order:** the best SHAP result (XGBoost F1 0.42) is less than half of No Sampling's (0.86).
* SHAP guidance makes oversampling about **3× slower** (21 s vs 7 s per round).

**One sentence for the slide:** *"On real credit-card fraud, SHAP-guided SMOTE beats plain SMOTE with tree models
(fewer false alarms), but at 582 : 1 balancing the data 1:1 at all is the wrong move."*
