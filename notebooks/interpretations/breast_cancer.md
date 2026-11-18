### Interpretation: Breast Cancer Wisconsin (9.2 : 1)

**Which method performed best?** No method stands out. The minority class (39 malignant cases, 29 in the
development set) is well separated: ROC-AUC is ≥ 0.96 for every method and model in cross-validation. The best
CV PR-AUC is Random Oversampling with Logistic Regression (0.962, tied with SMOTE and both SHAP methods) and
Borderline-SMOTE with the tree models (Random Forest 0.946, XGBoost 0.929). The best CV F1 is **No Sampling**
with Logistic Regression (0.919) and XGBoost (0.892): at 9:1 with clean separation, oversampling mainly trades
precision for a little recall.

**Did SHAP-guided SMOTE improve minority-class detection?** No. Recall is identical to SMOTE for both SHAP methods
with Logistic Regression and XGBoost, and 0.017 lower for SHAP-Weighted SMOTE with Random Forest. On the hold-out
test set (10 malignant cases), every SMOTE-family method catches the same 8–9 cases.

**PR-AUC and F1?** The differences from SMOTE are within ±0.02 and none is statistically significant (Wilcoxon,
10 paired folds). The closest to significance is SHAP-Weighted SMOTE with Random Forest, which is *worse* than
SMOTE on PR-AUC (−0.019, worse in 5 of the 6 folds that differ, p = 0.06).

**Precision?** Mixed and small: SHAP-Filtered SMOTE is slightly more precise than SMOTE with the tree models
(+0.012 RF, +0.019 XGBoost); SHAP-Weighted SMOTE is slightly less precise with Logistic Regression and Random
Forest (−0.03 and −0.02). None of these is significant.

**Did the synthetic samples preserve feature relationships?** Yes, for SMOTE and both SHAP methods: correlation-matrix
similarity to the real minority is 0.98, every feature passes the KS test, and synthetic rows sit in minority
territory (12–13% majority among their 5 nearest real rows, below the 22% of the real minority rows themselves),
and a Random Forest can barely tell them from real malignant cases (detectability AUC 0.52–0.54, 0.5 = indistinguishable).
SHAP guidance changed the pairing (mean Jaccard similarity with SMOTE's neighbour sets only 0.61–0.69, and parent pairs with more similar
SHAP explanations: cosine 0.85–0.86 vs 0.84), but the resulting samples are statistically very close to SMOTE's.
Borderline-SMOTE is the outlier: all its samples come from the 7 DANGER rows, so they are pushed toward the
boundary (36% majority neighbours), distort the correlation structure (similarity 0.71, 43% of features pass
the KS test) and are easy to detect (AUC 0.76).

**Where did SHAP-guided SMOTE fail or underperform?** It brought no measurable gain on an easy, low-dimensional,
moderately imbalanced problem where plain Euclidean neighbours are already sensible. SHAP-Weighted SMOTE with
Random Forest is the weakest SMOTE variant here (CV PR-AUC 0.909 vs 0.927 for SMOTE). The small minority count
(10 test cases, about 6 per validation fold) also limits what any comparison can resolve.
