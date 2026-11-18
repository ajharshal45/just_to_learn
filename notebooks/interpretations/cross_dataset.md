## 7. Interpretation

**Does SHAP-guided SMOTE generalise?** Not as a consistent improvement. Across 4 datasets × 3 models (12 blocks),
the proposed methods beat conventional SMOTE in some settings, tie in most, and fail badly in one.

| Dataset | SHAP-guided vs SMOTE (CV, fold-paired Wilcoxon) |
|---|---|
| Breast Cancer (9.2:1) | Tie. All differences within ±0.02; none significant. |
| Credit Card Fraud (581.8:1) | **Better with tree models.** XGBoost F1 +0.025 and precision +0.02 for both SHAP methods (p ≤ 0.014); Random Forest PR-AUC +0.020 for SHAP-Filtered (p = 0.037). Recall unchanged. |
| Stroke (19.5:1) | **Worse with tree models.** Recall −47% to −63% relative (p = 0.002 in all four cases, every fold). PR-AUC unchanged. Caused by interpolated one-hot categories (see below). |
| NSL-KDD (11.5:1) | Tie in practice. Differences ≤ 0.007; two tiny significant effects in opposite directions. |

**Which method performed best overall?** No method dominates. The Friedman test over the 12 blocks does not find a
significant difference between the six methods on PR-AUC (p = 0.056) or F1 (p = 0.107). It does on recall
(p < 0.001) and precision (p = 0.004), which reflects the basic trade-off: oversampling buys recall and
pays in precision. By average rank (1 = best):
* **PR-AUC:** Random Oversampling 2.50, No Sampling 2.75, **SHAP-Filtered 3.29**, **SHAP-Weighted 3.75**, SMOTE 4.29, Borderline-SMOTE 4.42.
* **F1:** No Sampling 2.67, Random Oversampling 2.75, **SHAP-Filtered 3.25**, SMOTE 3.75, Borderline-SMOTE 4.25, **SHAP-Weighted 4.33**.

**No Sampling** is the best single choice by F1 in 7 of 12 blocks and by PR-AUC in 5. On Credit Card Fraud, every
oversampler is far behind it (XGBoost F1 0.86 vs ≤ 0.75), because full 1:1 balancing is a poor operating point.

**Did SHAP-guided SMOTE improve minority-class detection, PR-AUC and F1?** Compared with SMOTE specifically:
* **SHAP-Filtered SMOTE** has a higher mean PR-AUC in 8 of 12 blocks, but only 1 of those is significant. It ranks
  above SMOTE on both PR-AUC and F1, and it is the more robust of the two proposed variants.
* **SHAP-Weighted SMOTE** has a higher mean PR-AUC in 7 of 12 blocks (none significant) and ranks *below* SMOTE on F1:
  it is significantly worse on F1 in 2 blocks and better in 1.
* Recall is never improved in a practically meaningful way. The one significant recall gain is +0.0015 on
  NSL-KDD. The gains that do appear come from **precision** (Credit Card trees: +8–9% relative).

**Did the synthetic samples preserve meaningful feature relationships?** On numeric features, about as well as
SMOTE. Correlation-structure similarity and KS pass rates match SMOTE on every dataset. The share of majority
rows among a synthetic row's 5 nearest real neighbours is within about 0.01 of SMOTE's. Numeric-only
detectability is within ±0.05 of SMOTE's: slightly higher on Breast Cancer and Credit Card (+0.01 to +0.02),
lower on Stroke and NSL-KDD (−0.03 to −0.05). SHAP guidance changes a large part of the neighbour graph (mean
Jaccard with SMOTE's neighbour sets 0.36–0.38 on Stroke, 0.50–0.56 on NSL-KDD and Credit Card, 0.61–0.69 on Breast
Cancer). On three datasets it pairs parents with more similar explanations; on NSL-KDD all methods are already at
about 0.98 SHAP cosine. The change in *which* rows get paired is substantial, but the change in the *resulting
numeric distribution* is small, which is why most downstream effects are small.

**Where did it fail?** On mixed numeric/categorical data. SHAP gives one-hot columns low weight, so SHAP-guided
neighbours more often come from different categories, and interpolation leaves 1.6× (Stroke) to 7× (NSL-KDD) more
fractional category indicators than SMOTE. On Stroke, a small minority with weak numeric signal, this made the
synthetic rows easy to recognise (detectability 0.93–0.96 vs 0.81) and tree models learned the artefact, losing
half their recall. On NSL-KDD, a large minority with strong signal, the same artefact was harmless. A
categorical-aware generation step (snapping one-hot groups, or SMOTE-NC-style handling) is the obvious fix.
It is outside this benchmark.

**Cost.** The SHAP step makes the proposed oversamplers 3–12× slower than SMOTE (for example 21 s vs 7 s per Credit
Card fold), even with the large-data SHAP row cap.

**Bottom line.** Under a leak-free protocol on four real datasets, SHAP-guided SMOTE is a *situational*
improvement over conventional SMOTE, not a general one. It helps the tree models on the extremely imbalanced,
all-numeric fraud data (mainly through precision), is neutral where SMOTE's neighbours are already good, and is
harmful on mixed-type data with a weak signal. SHAP-Filtered SMOTE is the safer of the two variants. Neither variant
changes the larger conclusion that the decision to oversample at all, and how far to balance, matters more than
which oversampler is used.
