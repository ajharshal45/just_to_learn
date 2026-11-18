### Interpretation: Credit Card Fraud (581.8 : 1)

**Which method performed best?** **No Sampling**, by a wide margin, on every metric except recall. XGBoost with no
sampling reaches CV F1 0.862 and PR-AUC 0.842; Random Forest reaches F1 0.831 and PR-AUC 0.824. Balancing 355
real frauds up to 206,392 rows (1:1) raises recall (XGBoost 0.80 → about 0.86, Random Forest 0.75 → about 0.86),
but it multiplies false alarms. Precision drops to 0.05–0.16 with Logistic Regression, 0.34–0.52 with Random Forest,
and 0.25–0.69 with XGBoost. Even the threshold-free PR-AUC is lower for every oversampler (best: Random Oversampling
with XGBoost, 0.826). Among the oversamplers, the best F1 is Borderline-SMOTE with XGBoost (0.747) and Random
Oversampling with Random Forest (0.643). Full 1:1 balancing is the wrong operating point at this imbalance.
That holds for every compared method, because they all share the same sampling target.

**Did SHAP-guided SMOTE improve minority-class detection?** Recall: no. It is the same as SMOTE (differences
≤ 0.006, none significant). Overall: yes, relative to SMOTE, through fewer false alarms at the same recall.
* **XGBoost:** F1 +0.025 for both SHAP methods (SHAP-Filtered better in 8/10 folds, p = 0.010; SHAP-Weighted 9/10,
  p = 0.014), precision +0.021–0.022 (p = 0.010 for both). These are the strongest positive results for the
  proposed methods in the whole benchmark.
* **Random Forest:** PR-AUC +0.020 (SHAP-Filtered, 7/10 folds, p = 0.037) and +0.024 (SHAP-Weighted, p = 0.065).
  SHAP-Filtered also improves F1 by +0.029 and precision by +0.027 (p = 0.08 and 0.06).
* **Logistic Regression:** no difference (all |Δ| ≤ 0.005).

**Did it improve or compromise precision?** It improved precision relative to SMOTE (by 8–9% relative with XGBoost
and 8% for SHAP-Filtered with Random Forest), but precision stays far below No Sampling. On the single hold-out
split (118 frauds) the XGBoost gain does not reproduce: SHAP-Filtered and SHAP-Weighted make 345 and 327 false
alarms against SMOTE's 315. The Random Forest gain does reproduce (SHAP-Filtered F1 0.525 vs 0.507).

**Did the synthetic samples preserve feature relationships?** Yes, for SMOTE and both SHAP methods. Correlation
similarity with the real frauds is 0.997, 97% of features pass the KS test, synthetic rows have fewer majority
neighbours than real frauds do (18% vs 25%), and they are hard to tell from real frauds (detectability AUC
0.56–0.58). SHAP guidance re-pairs about half of each fraud's neighbours (mean Jaccard 0.52–0.56 with SMOTE's).
SHAP-Filtered SMOTE searches on 13 of 29 features, led by V14, V17, V12 and V10, the components the earlier
credit-card notebook also identified as the fraud signal. Paired parents have slightly more similar explanations
(SHAP cosine 0.90–0.91 vs 0.90). Borderline-SMOTE fails every fidelity check (0% of features pass the KS
test, Wasserstein distance 10× SMOTE's, detectability 0.78).

**Where did SHAP-guided SMOTE fail or underperform?** It cannot fix the main problem on this dataset: any 1:1
oversampling of 355 frauds to 206,392 hurts more than it helps, so the best SHAP-guided result (XGBoost F1 0.417)
is less than half of No Sampling's (0.862). Its gains over SMOTE are real but second-order, and the
SHAP-Weighted F1 gain with Random Forest is slightly negative (−0.005). SHAP guidance also triples the oversampling
time (about 21 s vs 7 s per fold on 165,000-row training folds, even with the SHAP row cap).
