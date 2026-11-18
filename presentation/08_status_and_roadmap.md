# 08 — Where We Stand, Known Problems, and Next Steps

> **Why this file exists:** panels respect teams that know exactly where they stand. Show one slide
> of this, then move on. Never present unfinished work as finished.

---

## 1. Where each objective stands (as of 2026-11-17)

| # | Objective | Status | Evidence |
|---|---|---|---|
| 1 | Literature review and map of where SHAP can plug in | ✅ **Done** | [docs/literature/](../docs/literature/): review, 11-method comparison, 4-point map, reading notes |
| 2 | The two SHAP-guided methods | ✅ **Done** | [src/oversampling/](../src/oversampling/): Filtered and Weighted versions, plus a shared engine that records the two parents and λ of every fake row; the maths as implemented is in [docs/methodology_shap_guided_smote.md](../docs/methodology_shap_guided_smote.md) |
| 3 | Leak-free test setup and proof of the trap | ✅ **Done** | Four **real** datasets, one notebook per dataset ([11](11_notebook_walkthrough_cell_by_cell.md)), 25% locked test set + 2 × 5-fold cross-validation, fresh objects in every fold, an automatic leakage test; trap shown earlier (PR-AUC 0.954 leaky vs 0.469 fair) |
| 4 | Four-part scoreboard | ✅ **Done** | Catching rare cases, honest probabilities, stable explanations and realistic fake data, measured on all four datasets; Wilcoxon tests per dataset and a Friedman test across datasets ([16](16_cross_dataset_notebook.md)) |

**Tests:** 34 out of 34 pass. All five benchmark notebooks run from start to finish with no errors.

### What the results say (one line each)
- **No method wins overall.** Across 4 datasets × 3 models, the six methods are not significantly different on
  PR-AUC (p = 0.056) or F1 (p = 0.107). *Not oversampling at all* wins most often (7 of 12 by F1).
- **SHAP-guided vs plain SMOTE depends on the data:**
  - **Credit card fraud:** better with tree models. Fewer false alarms (XGBoost F1 +0.025, significant), same frauds caught.
  - **Breast cancer, network attacks:** no real difference.
  - **Stroke:** clearly worse with tree models (recall roughly halved, every round). See [14](14_results_stroke.md).
- **SHAP-Filtered SMOTE is the safer of our two methods.** It ranks above plain SMOTE on PR-AUC and F1;
  SHAP-Weighted ranks below SMOTE on F1.
- On the simulated fraud data with a known answer, SHAP finds the real signal features: **7 out of 7** (archived notebook).
- **Balancing to 1:1 is often the wrong move.** On credit-card fraud (582 : 1), every oversampler loses badly to no
  oversampling, because false alarms explode.
- Oversampling makes probabilities less honest on three of the four datasets (Breast Cancer is the exception).

---

## 2. Problems fixed since the last version

| Old problem | What we did |
|---|---|
| Credit fraud and network intrusion were **simulations** | Now the **real** Kaggle credit-card file (284,807 transactions) and the real NSL-KDD data |
| No automatic leakage test | `tests/test_experiment_pipeline.py` checks that the samplers only ever see training rows and the preparation steps learn from training rows only |
| The same model object was reused across folds | The new pipeline creates a fresh preprocessor, sampler and model in every round |
| SHAP errors in the stability step were silently ignored | The new pipeline doesn't hide errors; any failure stops the notebook |
| No significance tests across datasets | Wilcoxon per dataset + Friedman and average ranks across datasets |
| We couldn't show *what* the oversamplers generate | Every fake row is saved with its two parent rows and λ, and checked to rebuild exactly |

---

## 3. Known problems and limits (still open)

| # | Problem | Effect | Possible fix |
|---|---|---|---|
| 1 | **Category columns get interpolated** (e.g. "0.4 smoker"). SHAP guidance makes it worse because it pairs rows from different categories | The main reason SHAP-guided SMOTE fails on Stroke | After interpolating, snap each category back to a real value, or use a category-aware SMOTE (SMOTE-NC style) |
| 2 | **Every method balances to 1:1** | On very rare data (credit card) this causes huge numbers of false alarms | Try smaller targets (e.g. 1:10) or choose the target inside cross-validation |
| 3 | Settings (k = 5, the 90% cut-off, model settings) are fixed, not tuned; there is no inner tuning loop ("nested" CV) | Settings may not be the best for each dataset. Still leak-free, because nothing is tuned on test data | Add an inner loop that tunes them |
| 4 | Small numbers of rare cases in the test sets (10 malignant tumours, 62 strokes) | Single test-set scores are coarse | Rely on the 10 cross-validation rounds (already the main evidence) |
| 5 | Only 10 rounds, and the rounds share data | Significance tests have limited power and are a bit optimistic | More repeats (e.g. 5 × 5) if time allows |
| 6 | Two classes only | Multi-class data needs converting to "one vs the rest" | Document it, or extend later |
| 7 | The older pipeline (`src/pipeline/`, `experiments/run_benchmark.py`) still has the old issues (shared model objects, silent SHAP errors, quiet fallback if `imblearn` is missing) | Only affects the earlier simulated-data benchmark, not the four new notebooks | Fix or retire the old runner |
| 8 | The text in [docs/objectives_proof_and_kaggle_benchmark.md](../docs/objectives_proof_and_kaggle_benchmark.md) and the "Core Insights" in `BENCHMARK_REPORT.md` don't match their own figures | Someone could quote a number the charts contradict | Quote only the four new notebooks; that doc now carries a warning |
| 9 | SHAP guidance is 3–12× slower than SMOTE | Fine for these datasets (21 s per round at most) | Already uses a capped SHAP sample on large data |

---

## 4. Next steps

1. **Fix the category problem** (snap one-hot values back to a real category) and re-run Stroke and NSL-KDD. This
   is the single most promising improvement for the method.
2. **Try smaller balancing targets** (1:10, 1:4) on credit-card fraud, to see whether SHAP's precision gain survives
   at a sensible operating point.
3. Add a distance-based model (k-NN) to the four notebooks. The earlier simulation suggested SHAP helps it most.
4. A critical-difference chart (Nemenyi) for the cross-dataset ranks.
5. Write the final report from notebooks 01–05.

---

## 5. What would count as success?

We are **not** promising SHAP-SMOTE wins. Any **well-supported** answer is a good research outcome:
- ✅ *SHAP-guided wins on PR-AUC and stability, with significance.* That's a new-method result.
- ✅ *It wins in some situations but not others.* That's a useful "when to use it" result. **This is where the
  evidence points:** it helps on all-numeric, extremely rare fraud data with tree models, does nothing on easy
  problems, and hurts on mixed number + category data.
- ✅ *It doesn't beat SMOTE when tested without leakage.* That's an important negative result,
  especially if earlier positive claims came from leaky setups.

The leak-free test setup, the traceable fake rows and the four-part scoreboard are contributions **whichever way**
the results go.
