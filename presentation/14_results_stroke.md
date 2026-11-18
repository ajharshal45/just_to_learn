# 14 — Stroke Notebook: What Each Cell Shows

> Notebook: `notebooks/03_stroke_prediction_experiment.ipynb`. The cells are explained in
> [11](11_notebook_walkthrough_cell_by_cell.md); this file gives **this dataset's actual outputs**, cell by cell,
> in plain words. The rare class is **a patient who had a stroke**. This dataset gives the project's clearest
> *negative* result, so read the "why" section carefully.

---

## Setup and data (cells 1–5)

* **Cell 2:** `DATASET = "stroke"`. Settings identical to the other notebooks.
* **Cell 4:** The Kaggle Stroke Prediction dataset: one row per patient with age, high blood pressure
  (`hypertension`), heart disease, average blood sugar (`avg_glucose_level`), BMI, and five **category** columns:
  gender, ever married, work type, home area (urban/rural) and smoking status. We dropped the patient `id`. Result:
  **5,110 rows × 10 features**.
* **Cell 5:** 5 numeric and 5 categorical features. **201 patients have no BMI recorded**; these are filled in with
  the training data's median BMI. No duplicates.

## Looking at the data (cells 7–9)

* **Cell 7:** 4,861 without a stroke, **249 with a stroke (4.87%)**, imbalance **19.5 : 1**.
* **Cell 8:** **Age** separates the classes by far the most (KS 0.54); then blood sugar (0.24), hypertension (0.18),
  BMI (0.17) and heart disease (0.14). The histograms show stroke patients are mostly older, but so are many
  patients without a stroke, so the classes **overlap heavily**. This is a **hard** problem.
* **Cell 9:** Age is linked to hypertension, heart disease and BMI; the links to the target are weak.

## Preparing the data (cell 11)

* Development set: **3,832 patients, 187 strokes**. Test set: **1,278 patients, 62 strokes**.
* The 5 category columns become 0/1 columns ("work_type=Private", "smoking_status=smokes", …), giving **21 model
  columns**. The BMI fill-in value is learned from the training data only.

## Making the fake rows (cells 13–24)

* **Cell 13:** Every method turns **187 real stroke patients into 3,645**, adding **3,458 fake patients**.
  Borderline-SMOTE starts from **108 DANGER patients**. SHAP-Filtered SMOTE judges similarity on **10 of the 21
  columns**.
* **Cell 15:** Borderline labels: **2 SAFE, 108 DANGER, 77 NOISE**. Almost no stroke patient sits in a "safe",
  stroke-only area, which shows how mixed the classes are. Neighbour overlap with SMOTE: **SHAP-Filtered 0.38,
  SHAP-Weighted 0.36**: SHAP changes **most** of the neighbours, more than on any other dataset.
* **Cell 15, a revealing detail:** under plain SMOTE, much of the distance between two patients comes from the
  **category columns** (e.g. "smoking_status=smokes (49%)"), so plain SMOTE mostly pairs patients **with the same
  smoking status, work type and home area**.
* **Cell 18:** Fake row #0 of SMOTE, SHAP-Filtered and SHAP-Weighted all start from patient **row 67** with
  λ = 0.4389. SMOTE pairs it with patient 200, SHAP-Filtered with patient 64, SHAP-Weighted with patient 127.
* **Cell 20:** SHAP's weights: **age 0.42** (almost half of all importance), blood sugar 0.13, hypertension 0.07,
  heart disease 0.07, BMI 0.05. Every category column gets a small weight (at most 0.05, most below 0.03).
* **Cell 21:** In the first 8 fake rows, **38%** of SHAP's parent pairs would **not** have been neighbours in
  plain SMOTE.
* **Cell 23 (quality):**

  | Method | Common-class neighbours (real stroke patients: 90%) | Correlations kept | Detectability (0.5 = can't tell) | Detectability, numbers only | Category columns stuck between 0 and 1 (per row) |
  |---|---|---|---|---|---|
  | SMOTE | 68% | 0.985 | 0.81 | 0.63 | 2.3 |
  | Borderline-SMOTE | 63% | 0.91 | 0.77 | 0.62 | 2.1 |
  | SHAP-Filtered SMOTE | 69% | 0.988 | **0.93** | **0.59** | **3.7** |
  | SHAP-Weighted SMOTE | 69% | 0.984 | **0.96** | **0.60** | **3.9** |

  Note that even *real* stroke patients have 90% non-stroke patients among their 5 closest neighbours: the classes
  are thoroughly mixed.
* **Cell 24:** All rows saved to `results/stroke/synthetic_samples/` (3,458 per method).

## The fair test (cells 26–29) — 10 rounds, 3.1 minutes

Average over the 10 rounds (cell 27):

| Model | Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | No Sampling | 0.00 | **0.00** | 0.00 | 0.19 |
| | Random Oversampling | 0.14 | 0.79 | 0.23 | 0.19 |
| | SMOTE | 0.13 | 0.75 | 0.23 | 0.19 |
| | SHAP-Filtered / SHAP-Weighted | 0.13 / 0.13 | 0.75 / 0.75 | 0.22 / 0.22 | 0.19 / 0.19 |
| Random Forest | No Sampling | 0.00 | **0.00** | 0.00 | 0.18 |
| | SMOTE | 0.12 | **0.62** | 0.21 | 0.15 |
| | SHAP-Filtered | 0.16 | **0.23** | 0.18 | 0.16 |
| | SHAP-Weighted | 0.14 | **0.26** | 0.18 | 0.15 |
| XGBoost | No Sampling | 0.25 | 0.04 | 0.06 | 0.18 |
| | Random Oversampling | 0.16 | 0.48 | 0.24 | 0.17 |
| | SMOTE | 0.15 | **0.20** | 0.17 | 0.15 |
| | SHAP-Filtered | 0.22 | **0.11** | 0.14 | 0.16 |
| | SHAP-Weighted | 0.21 | **0.08** | 0.11 | 0.16 |

* Without oversampling, the models **never predict a stroke** (recall 0): the rare class is too rare and too mixed.
* **Cell 28 (Graph 1):** all PR-AUC bars are low (0.15–0.19; random guessing would give 0.05).
* **Cell 29 (significance), SHAP methods vs SMOTE:**
  * Random Forest and XGBoost: recall **drops** sharply (Random Forest 0.62 → 0.23/0.26; XGBoost 0.20 → 0.11/0.08),
    worse in **all 10 rounds** (p = 0.002). **Significant, in the wrong direction.**
  * Precision rises a little (Random Forest +0.02 to +0.04, p = 0.037): the models just flag fewer people.
  * PR-AUC: no difference. Logistic Regression: no difference.

## The locked test set (cells 31–32) — 62 stroke patients among 1,278

| Random Forest | Strokes caught | Missed | False alarms |
|---|---|---|---|
| No Sampling | 0 | 62 | 0 |
| SMOTE | 39 | 23 | 319 |
| SHAP-Filtered SMOTE | **11** | 51 | 90 |
| SHAP-Weighted SMOTE | **21** | 41 | 143 |
| Random Oversampling | 46 | 16 | 347 |

* Same pattern as the 10 rounds: the SHAP methods catch far fewer strokes with the tree models.
* With Logistic Regression all oversamplers behave alike (48–50 of 62 caught, 277–330 false alarms).

## What the models pay attention to (cells 34–35)

* **Cell 34 (Graph 2):** every model relies mostly on **age** (about half of all attention). The models trained on
  SHAP-guided fake data reason **most differently** from the No Sampling model (ranking similarity only 0.27–0.29 for
  Random Forest), a first sign that they learned something different.
* **Cell 35:** explanations are moderately stable (Kendall τ 0.53–0.83); the SHAP methods show no consistent stability gain over SMOTE.

## Why did SHAP-guided SMOTE fail here? (the key explanation)

1. SHAP correctly finds that **age** matters most and that the category columns (work type, home area, smoking)
   matter little, so it gives them small weights.
2. As a result, SHAP pairs patients who are similar in **age and health**, but often from **different categories**
   (one rural, one urban; one smoker, one non-smoker).
3. A fake patient made between them ends up with values like **"0.4 rural, 0.6 urban"**: on average **3.7–3.9 such
   in-between category values per fake patient**, against 2.3 for plain SMOTE. No real patient looks like that.
4. A Random Forest can therefore tell SHAP's fake patients from real ones **93–96% of the time** (SMOTE's: 81%).
   On the number columns alone, SHAP's fakes are actually *more* realistic than SMOTE's (0.59–0.60 vs 0.63).
5. Tree models (Random Forest, XGBoost) learn the shortcut *"in-between category value = stroke"*. Real test patients
   never have in-between values, so the models rarely predict a stroke for them, and **recall collapses**.
   Logistic Regression can't exploit such a shortcut, so it is unaffected.

## Conclusion (cells 39–40), in plain words

* **This is a hard problem for everyone:** PR-AUC is 0.15–0.19 for all methods.
* **Some oversampling is essential** here: without it, the models never predict a stroke.
* **SHAP-guided SMOTE clearly hurt the tree models** (recall roughly halved, every round) because of the category
  columns, as explained above.
* **The fix is known but outside this study:** after interpolating, snap each category back to a real value, or use
  a category-aware variant of SMOTE.

**One sentence for the slide:** *"On stroke data, SHAP guidance pairs patients across categories, creating fake
patients with impossible half-categories; tree models learn that artefact and miss half the strokes. That's an
important limitation of the method on mixed data."*
