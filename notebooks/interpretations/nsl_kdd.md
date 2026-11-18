### Interpretation: NSL-KDD network intrusion (11.5 : 1)

**Which method performed best?** **No Sampling with XGBoost**: CV F1 0.993, PR-AUC 0.999, precision 0.998. The
task is close to saturated for gradient boosting: Random Oversampling, SMOTE and both SHAP methods with XGBoost are
within 0.001 of it on PR-AUC. Oversampling matters for the other two models, but in opposite directions:
* **Random Forest:** oversampling raises recall from 0.90 to about 0.97 and F1 from 0.946 to 0.969–0.974.
  Random Oversampling is best (F1 0.974) and SHAP-Filtered SMOTE second (0.972).
* **Logistic Regression:** oversampling raises recall from 0.93 to 0.96 but drops precision from 0.98 to about
  0.80, so F1 falls from 0.954 to about 0.875 for every oversampler.

**Did SHAP-guided SMOTE improve minority-class detection?** Not in any practical sense. Recall differences from
SMOTE are at most ±0.002. The one significant gain is SHAP-Weighted SMOTE with XGBoost (+0.0015 recall, better in
7 of 8 folds that differ, p = 0.016), a fraction of one attack per validation fold.

**PR-AUC and F1?** PR-AUC differences from SMOTE are below 0.001 everywhere (none significant). F1: SHAP-Filtered
SMOTE with Random Forest is the largest gain (+0.003, 8 of 10 folds, p = 0.19, not significant). SHAP-Weighted
SMOTE with Logistic Regression is significantly *worse* (−0.002, p = 0.049).

**Precision?** Unchanged to slightly lower. SHAP-Weighted SMOTE with Logistic Regression loses 0.004 (worse in 9 of
10 folds, p = 0.010). SHAP-Filtered SMOTE with Random Forest gains 0.007 in CV (not significant), but on the
single hold-out split it produced *more* false positives than SMOTE (55 vs 18).

**Did the synthetic samples preserve feature relationships?** SMOTE and the two SHAP methods produce statistically
similar samples: 95% of features pass the KS test, correlation-structure similarity is 0.90 (the same as random
oversampling), and only 1.2–1.5% majority rows sit among their 5 nearest real neighbours (the real attacks: 2.1%).
All three are fairly easy to tell from real attacks (detectability AUC 0.87–0.88). This is driven by the heavy-tailed
count and byte features, where interpolation creates values no real record has. On the numeric columns alone the
SHAP methods are slightly *more* realistic (0.85–0.86 vs 0.89). SHAP guidance changes about half of every seed's
neighbours (mean Jaccard 0.50–0.51 with SMOTE's), and paired parents already have very similar explanations under
every method (SHAP cosine 0.98). SHAP-Filtered SMOTE searches on 23 of 116 columns.

As on Stroke, SHAP pairing leaves far more fractional category indicators (0.84–0.92 per synthetic row vs 0.13 for
SMOTE: SHAP down-weights protocol/service/flag mismatches). Here it does not hurt the tree models, because
the attack signal in the numeric traffic features is strong and the minority class is large (about 3,500
attacks per training fold).

**Borderline-SMOTE** is clearly the worst method on this dataset. Its synthetic rows sit in normal-traffic territory
(32% majority neighbours), break the correlation structure (similarity 0.44), are almost perfectly detectable
(AUC 0.97), and cut Random Forest precision to 0.78.

**Where did SHAP-guided SMOTE fail or underperform?** It did not fail, but it had nothing to add. SHAP rewired half
of the neighbour graph and the downstream models barely noticed, because SMOTE's neighbours are already
in a dense, well-separated minority region. Its oversampling step is also about 5× slower than SMOTE's
(9–11 s vs 1.8 s per fold, mostly the preliminary model and SHAP).
