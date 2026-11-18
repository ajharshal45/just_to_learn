### Interpretation: Stroke Prediction (19.5 : 1)

**Which method performed best?** This is a hard problem for every method: CV PR-AUC is only 0.15–0.19 (the
no-skill level is 0.049) and ROC-AUC is 0.79–0.84. Ranked by threshold-free PR-AUC, the methods are almost tied
and Logistic Regression is the best model for all of them (0.189–0.192). At the 0.5 threshold, oversampling is
essential: without it the models predict almost no strokes (recall 0.00–0.04). Random Oversampling gives the best
CV F1 with Random Forest (0.208) and XGBoost (0.235), and Borderline-SMOTE the best with Logistic Regression (0.236).

**Did SHAP-guided SMOTE improve minority-class detection?** No, and for the tree models it clearly made it
worse. With Logistic Regression both SHAP methods match SMOTE (recall 0.75). With Random Forest, recall drops from
0.62 (SMOTE) to 0.23 (SHAP-Filtered) and 0.26 (SHAP-Weighted); with XGBoost, from 0.20 to 0.11 and 0.08. Each of
these four drops occurs in all 10 folds (Wilcoxon p = 0.002).

**PR-AUC and F1?** PR-AUC is unchanged: every difference from SMOTE is between −0.001 and +0.012, none
significant. F1 is lower for the tree models (−0.02 to −0.07). The −0.065 for SHAP-Weighted SMOTE with XGBoost is
significant (p = 0.037).

**Precision?** It improves where recall collapses: +0.035 (SHAP-Filtered, RF, p = 0.037), +0.019
(SHAP-Weighted, RF, p = 0.037), about +0.06 with XGBoost (not significant). The tree models trained on SHAP-guided
data become more conservative rather than better at ranking: PR-AUC, which does not depend on the threshold,
does not move.

**Did the synthetic samples preserve feature relationships?** On the numeric features, slightly better than
SMOTE. A Random Forest separates SHAP-guided synthetic rows from real stroke patients on the numeric features
with AUC 0.59–0.60, versus 0.63 for SMOTE (0.5 = indistinguishable). Correlation structure is preserved
(similarity 0.98–0.99), and paired parents have more similar explanations (SHAP cosine 0.81–0.88 vs 0.74).
**The categorical features are the failure point.** SHAP gives the one-hot columns (work type, residence,
smoking status) low weight, so it pairs patients from different categories. Interpolation then leaves 3.7–3.9
one-hot columns per synthetic row between 0 and 1, versus 2.3 for SMOTE. On all columns, the SHAP-guided rows
are detectable with AUC 0.93 (Filtered) and 0.96 (Weighted), versus 0.81 for SMOTE. The tree models can learn
"fractional category value → stroke", a pattern that never occurs in real test patients. This explains why their
recall collapses, while Logistic Regression, which can hardly exploit such artefacts, is unaffected.

**Where did SHAP-guided SMOTE fail?** On mixed-type data. Because SHAP down-weights low-importance categorical
columns, interpolation produces unrealistic category values more often than plain SMOTE does, and tree models
over-fit to them. Two fixes are implied but outside the compared methods: snap one-hot groups back to a valid
category, or use a categorical-aware variant (SMOTE-NC-style). This is the clearest negative result of the benchmark.
