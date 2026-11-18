# 16 — The Cross-Dataset Notebook, Cell by Cell

> Notebook: `notebooks/05_cross_dataset_comparison.ipynb`. It **trains nothing**: it reads the results the four
> experiment notebooks saved in `results/<dataset>/metrics/` and puts them side by side, to answer one question:
> **does SHAP-guided SMOTE work on different kinds of data, or only on one?** Run it after notebooks 01–04.
> Unless a table says "test", every number is the average over the 10 cross-validation rounds.

---

### Cell 1 — Introduction *(text)*
States the goal and that notebooks 01–04 must have been run first.

### Cell 2 — Setup and loading *(code)*
Loads the same tools as the other notebooks, then reads the saved results of all four datasets.
**You see:** `loaded: ['breast_cancer', 'credit_card', 'stroke', 'nsl_kdd']`.

### Cells 3–4 — The four datasets *(text + code)*
**You see:** one overview table with all four datasets:

| Dataset | Rows | Rare cases | Rare % | Imbalance |
|---|---|---|---|---|
| Breast Cancer | 396 | 39 | 9.85% | 9.2 : 1 |
| Credit Card Fraud | 275,663 | 473 | 0.17% | 581.8 : 1 |
| Stroke | 5,110 | 249 | 4.87% | 19.5 : 1 |
| NSL-KDD | 73,199 | 5,856 | 8.00% | 11.5 : 1 |

**Say:** *"Four domains (medicine, banking, healthcare, cybersecurity), from mild to extreme imbalance."*

### Cells 5–6 — Big comparison tables *(text + code)*
**You see:** five tables (PR-AUC, F1, recall, precision, ROC-AUC). Each row is one dataset + model, each column is
one method, and the best value in each row is **bold**.
**Say:** *"This is the full scoreboard: 12 rows (4 datasets × 3 models) × 6 methods."*

### Cells 7–8 — Bar charts *(text + code)*
**You see:** a 3 × 3 grid of bar charts: rows are PR-AUC, F1 and recall; columns are the three models; each group
of bars is one dataset, one colour per method. Saved as `results/cross_dataset/plots/cross_dataset_f1_prauc_recall.png`.
**Say:** *"Within a dataset, the bars are usually close together. The big differences are between datasets, not
between methods."*

### Cells 9–10 — How much better or worse than plain SMOTE? *(text + code)*
**You see:** a colour-coded table and a bar chart of the **% change of each SHAP method vs plain SMOTE**, for
PR-AUC, F1, recall and precision (blue = better, red = worse). Saved as `relative_improvement_vs_smote.png`.
The pattern to point out:

| | SHAP-Filtered vs SMOTE | SHAP-Weighted vs SMOTE |
|---|---|---|
| Breast Cancer | within ±2.5% everywhere | within ±3.5% everywhere |
| Credit Card, tree models | F1 **+6%**, precision **+8 to +9%** | XGBoost F1 **+6%**, precision **+8%** |
| Stroke, tree models | recall **−47% to −62%** | recall **−58% to −63%** |
| NSL-KDD | within ±1% everywhere | within ±1% everywhere |

**Say:** *"Better on credit-card fraud, worse on stroke, no change elsewhere. It depends on the data."*

### Cells 11–12 — Who wins each dataset? *(text + code)*
**You see:** the winning method for every dataset + model, by PR-AUC and by F1, and a count of wins:

| Method | Wins by F1 (of 12) | Wins by PR-AUC (of 12) |
|---|---|---|
| No Sampling | **7** | **5** |
| Random Oversampling | 2 | 2 |
| Borderline-SMOTE | 2 | 2 |
| SHAP-Filtered SMOTE | 1 | 2 |
| SHAP-Weighted SMOTE | 0 | 1 |
| SMOTE | 0 | 0 |

**Say:** *"The most frequent winner is not oversampling at all. Plain SMOTE never wins outright."*

### Cells 13–14 — Is any method better overall? *(text + code)*
**You see:**
1. The **Friedman test** for each score: is there *any* real difference between the six methods across the 12
   dataset + model combinations?
   * PR-AUC: p = 0.056 and F1: p = 0.107, so **no significant difference**.
   * Recall: p < 0.001 and precision: p = 0.004, so **significant**. Oversampling catches more and cries wolf more;
     not oversampling is the reverse.
2. An **average rank** table and chart (1 = best, 6 = worst). Saved as `average_ranks.png`.

   | Method | PR-AUC rank | F1 rank |
   |---|---|---|
   | Random Oversampling | **2.50** | 2.75 |
   | No Sampling | 2.75 | **2.67** |
   | SHAP-Filtered SMOTE | 3.29 | 3.25 |
   | SHAP-Weighted SMOTE | 3.75 | 4.33 |
   | SMOTE | 4.29 | 3.75 |
   | Borderline-SMOTE | 4.42 | 4.25 |

3. A **significance count**: in how many of the 12 combinations each SHAP method was *significantly* better, worse,
   or no different from SMOTE. Most results are "no significant difference". SHAP-Filtered was significantly better
   on F1 once, on PR-AUC once and on precision twice, and significantly worse on recall twice (both on Stroke).
   SHAP-Weighted was significantly worse on F1 twice and better once.

**Say:** *"SHAP-Filtered SMOTE ranks above plain SMOTE on both PR-AUC and F1, so it is the better of our two
variants. But overall, no method is significantly better than the others."*

### Cell 15 — Fake-row quality, test scores and speed *(code)*
**You see:**
1. The **fake-row quality** table for all four datasets (the same measures as cell 23 of the experiment notebooks).
2. **Test-set PR-AUC** for every dataset, model and method (one single check, for reference).
3. **Time per round** to make the fake rows:

   | Method | Breast Cancer | Credit Card | Stroke | NSL-KDD |
   |---|---|---|---|---|
   | SMOTE | 0.04 s | 7.1 s | 0.16 s | 1.8 s |
   | SHAP-Filtered SMOTE | 0.40 s | 20.7 s | 2.0 s | 9.1 s |
   | SHAP-Weighted SMOTE | 0.31 s | 20.9 s | 1.9 s | 10.9 s |

**Say:** *"SHAP guidance costs 3 to 12 times more time, because it first trains a small model and runs SHAP."*

### Cell 16 — Interpretation *(text)*
The written conclusion. In plain words:

1. **SHAP-guided SMOTE is not better everywhere.** It beats plain SMOTE with tree models on credit-card fraud
   (fewer false alarms), ties on breast cancer and network attacks, and clearly loses on stroke.
2. **No method wins overall**, and the decision *whether* to oversample (and how much) matters more than *which*
   oversampler. Not oversampling at all is the most frequent winner.
3. **Why it fails on Stroke:** SHAP gives little weight to category columns (work type, smoking, …), so it pairs
   patients from different categories. The fake patients get impossible "half-categories", and tree models learn
   that artefact. It does no harm on NSL-KDD, where attacks are plentiful and easy to spot.
4. **SHAP-Filtered SMOTE is the safer variant** of the two.
5. **The fake rows are about as realistic as SMOTE's** on number columns. SHAP changes *which* rows get paired (often
   half of them), but the resulting data looks statistically similar, which is why most differences are small.

**One sentence for the slide:** *"Across four real datasets, SHAP-guided SMOTE is a situational improvement, not a
general one: it helps on all-numeric fraud data, does nothing on easy problems, and hurts on mixed numeric and
category data."*
