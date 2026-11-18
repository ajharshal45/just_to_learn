# 12 — Breast Cancer Notebook: What Each Cell Shows

> Notebook: `notebooks/01_breast_cancer_experiment.ipynb`. The cells are explained in
> [11](11_notebook_walkthrough_cell_by_cell.md); this file gives **this dataset's actual outputs**, cell by cell,
> in plain words. The rare class is a **malignant (cancerous) tumour**.

---

## Setup and data (cells 1–5)

* **Cell 2:** `DATASET = "breast_cancer"`. Settings table identical to the other three notebooks.
* **Cell 4:** The Breast Cancer Wisconsin data (University of Wisconsin, via Kaggle). Each row is one tumour, with
  30 measurements of its cells taken from a scan image (radius, texture, smoothness, and so on). The original file
  has 357 benign and 212 malignant tumours. **We kept a fixed random 39 of the malignant ones** to make the problem
  imbalanced (9 : 1), the same subset the project used before. Result: **396 rows × 30 features**.
* **Cell 5:** All 30 features are numbers. No missing values, no duplicates. 13 rows have an extreme value; they are
  flagged but kept.

## Looking at the data (cells 7–9)

* **Cell 7:** 357 benign, **39 malignant (9.85%)**, imbalance **9.2 : 1**.
* **Cell 8:** The features that best separate the classes are the "worst" (largest) measurements:
  `perimeter_worst` (KS 0.90), `radius_worst`, `area_worst` (0.86), `concave points_mean` (0.84). Malignant tumours
  are clearly bigger and more irregular, so this is an **easy** problem.
* **Cell 9:** Many features are almost copies of each other (radius, perimeter and area all measure size), so the
  heatmap has large dark-red blocks.

## Preparing the data (cell 11)

* Development set: **297 rows, 29 malignant**. Test set: **99 rows, 10 malignant**.
* 30 model columns, all numeric and scaled. Nothing to fill in.

## Making the fake rows (cells 13–24)

* **Cell 13:** Every method turns **29 real malignant rows into 268**, by adding **239 fake rows**. Borderline-SMOTE
  can only start from **7 "DANGER" rows** (the malignant tumours closest to benign ones). SHAP-Filtered SMOTE
  judges similarity on **17 of the 30 features**.
* **Cell 15:** Borderline labels: **21 SAFE, 7 DANGER, 1 NOISE**. Neighbour overlap with plain SMOTE
  (1.0 = identical): **SHAP-Filtered 0.69, SHAP-Weighted 0.61**, so SHAP changes about a third of the neighbours.
* **Cell 17:** The first fake rows. Example: SMOTE fake row `2-0` is built from tumour rows **370 and 330** with
  λ = 0.44; it has a radius of 16.2, between its parents' 16.35 and 16.03.
* **Cell 18:** Fake row #0 of SMOTE, SHAP-Filtered and SHAP-Weighted all start from **row 370** with **λ = 0.4389**.
  SMOTE pairs it with **row 330**, while both SHAP methods pair it with **row 408**, which plain SMOTE ranks only
  3rd-closest. Borderline-SMOTE starts from a DANGER row (row 39).
* **Cell 20:** SHAP's top features: **`concave points_worst` (weight 0.14), `perimeter_worst` (0.14),
  `area_worst` (0.09)**, `concave points_mean`, `radius_worst`. These match the features that separated the classes
  in cell 8.
* **Cell 21:** In the first 8 fake rows of each SHAP method, **25%** of parent pairs would **not** have been
  neighbours in plain SMOTE.
* **Cell 23 (quality):**

  | Method | Common-class neighbours (real rare rows: 22%) | Realistic spread (KS pass) | Correlations kept | Detectability (0.5 = can't tell) |
  |---|---|---|---|---|
  | SMOTE | 12% | 100% | 0.98 | 0.53 |
  | Borderline-SMOTE | **36%** | **43%** | **0.71** | **0.76** |
  | SHAP-Filtered SMOTE | 12% | 100% | 0.98 | 0.52 |
  | SHAP-Weighted SMOTE | 13% | 100% | 0.98 | 0.54 |

  In plain words: SMOTE and both SHAP methods make realistic fake tumours that a model can't tell from real ones.
  Borderline-SMOTE makes 239 rows from only 7 starting tumours, pushes them toward the benign side, and they look
  fake.
* **Cell 24:** All rows saved to `results/breast_cancer/synthetic_samples/` (239 rows per method, small files you
  can open in Excel).

## The fair test (cells 26–29) — 10 rounds, 1.3 minutes

Average over the 10 rounds (cell 27):

| Model | Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | No Sampling | **1.00** | 0.86 | **0.92** | 0.96 |
| | SMOTE | 0.88 | 0.89 | 0.88 | 0.96 |
| | SHAP-Filtered | 0.87 | 0.89 | 0.87 | 0.96 |
| | SHAP-Weighted | 0.85 | 0.89 | 0.86 | 0.96 |
| Random Forest | No Sampling | 0.96 | 0.81 | 0.87 | 0.92 |
| | SMOTE | 0.93 | 0.84 | 0.87 | 0.93 |
| | Borderline-SMOTE | 0.88 | 0.83 | 0.84 | **0.95** |
| | SHAP-Filtered | 0.94 | 0.84 | **0.88** | 0.92 |
| | SHAP-Weighted | 0.91 | 0.83 | 0.85 | 0.91 |
| XGBoost | No Sampling | 0.97 | 0.84 | **0.89** | 0.93 |
| | SMOTE | 0.84 | 0.84 | 0.83 | 0.92 |
| | SHAP-Filtered | 0.86 | 0.84 | 0.84 | 0.93 |
| | SHAP-Weighted | 0.84 | 0.84 | 0.83 | 0.92 |

* **Cell 28 (Graph 1):** All bars are close together. The ▲ for F1 goes to No Sampling for two of three models.
* **Cell 29:** **No SHAP-vs-SMOTE difference is significant** (all p ≥ 0.06). The closest is SHAP-Weighted with
  Random Forest, and it is *worse* (PR-AUC −0.02, p = 0.06).

## The locked test set (cells 31–32) — 10 malignant tumours

* Almost every method and model catches **9 of 10** malignant tumours with **no false alarms** (Random Forest catches
  8). The only false alarm in the whole table is XGBoost + Borderline-SMOTE (1).
* With so few cases, one tumour changes recall by 10 points, so the test set can't separate the methods.

## What the models pay attention to (cells 34–35)

* **Cell 34 (Graph 2):** All models lean on the same size and shape features (`area_worst`, `concave points_worst`,
  `perimeter_worst`) whatever the balancing. Feature rankings stay close to No Sampling's (similarity 0.71–0.92).
* **Cell 35:** Random Forest gives the most stable explanations (Kendall τ about 0.75 for most methods). Borderline-SMOTE
  makes explanations least stable (0.51–0.64).

## Conclusion (cells 39–40), in plain words

* **No method stands out.** The cancer class is easy to separate, so every method scores PR-AUC 0.91–0.96.
* **Not oversampling at all gives the best F1** for two of three models: with this much separation, adding fake rows
  mostly costs precision.
* **SHAP-guided SMOTE neither helped nor hurt in a reliable way.** It changed about a third of the neighbours, but the
  fake rows it made are statistically almost the same as SMOTE's.
* **Borderline-SMOTE made the least realistic fake rows** (built from only 7 tumours).

**One sentence for the slide:** *"On breast cancer, SHAP guidance changes which tumours get paired but not the
results: the problem is easy and every method does well."*
