# 09 — Likely Questions and a Glossary

> Everyone should read this whole file, not just their own section. Questions rarely go to the
> person who presented that part.

---

## Part A: Likely questions and suggested answers

### The big picture

**Q: What's your project in one sentence?**
> We test whether using SHAP to tell SMOTE which features matter gives better fake rare-class data.
> We judge it on catching rare cases, honest probabilities, stable explanations and realistic data,
> in a setup where data leakage can't happen.

**Q: What's the gap you're filling?**
> SHAP is almost always used *after* training, to explain a model. Very little work uses it
> *before* training to shape the data. The one earlier SHAP-weighted SMOTE study used a single
> dataset, didn't check stability or calibration, and risked leakage.

**Q: Didn't someone already do SHAP-weighted SMOTE?**
> Yes, one network-attack paper did, and we cite it. What we add: (1) a map of where explainability
> can plug into balancing, (2) two versions (mute button vs volume dial) compared head to head,
> (3) SHAP computed *inside each training fold*, closing a leakage path others miss, and (4) the
> first multi-dataset test that includes **explanation stability** and **calibration**.

**Q: Why should industry care?**
> Rare-event models (fraud, sepsis, benefits fraud) fail expensively and publicly. Regulators
> increasingly require explainable decisions (EU AI Act, US credit-refusal rules). A balancing
> method driven by explanations, and tested on whether those explanations stay stable, addresses both.

### The methods (Objective 2)

**Q: Why does SMOTE's distance measure matter so much?**
> The neighbour choice decides every fake row. With 5 useful and 25 noise features, about 83% of the
> distance comes from noise, so the "neighbours" are mostly random, and so are the fake rows.

**Q: Filtered vs Weighted: which one wins?**
> On the four real datasets, **Filtered** is the safer choice: it ranks above plain SMOTE on both PR-AUC (3.29 vs
> 4.29) and F1 (3.25 vs 3.75), while Weighted ranks below SMOTE on F1 (4.33). Both did best on credit-card fraud
> and worst on stroke. On the earlier simulation, Filtered had the best k-NN PR-AUC at 1% fraud (0.751).
> Details are in [10](10_shap_filtered_smote_deep_dive.md) and [16](16_cross_dataset_notebook.md).

**Q: Why stretch the features by √w?**
> Because a normal distance on √w-stretched data is exactly the weighted distance. We get the
> correct weighted neighbours while using scikit-learn's fast, standard neighbour search.

**Q: Why build the new rows in the original space, not the filtered or stretched one?**
> So every fake row is a complete record with all its features, in real units. SHAP only decides
> *who pairs with whom*.

**Q: Does dropping features in the Filtered version lose information?**
> Only for choosing neighbours. Every new row still has all its columns. But if a dropped feature was
> actually useful, neighbours are chosen less well. On the stroke data the dropped columns were mostly the
> category columns, and ignoring them is exactly what caused the "half-category" problem (see the stroke question below).

**Q: Why a Random Forest as the scout model? Isn't that circular if the final model is also a Random Forest?**
> It's fast, handles complex patterns, and SHAP is exact for it. It's a separate, small model trained
> only on the training part, and the final model is trained afterwards on the balanced data. We also
> test other final models (Logistic Regression and XGBoost; k-NN in the earlier simulation). The scout model can be
> swapped through `base_estimator`.

**Q: Why not Mahalanobis distance?**
> It needs a matrix inversion that breaks when there are fewer rare rows than features, which is
> common with rare events. SHAP weights need no inversion.

**Q: Why not GANs or diffusion models?**
> They need far more rare examples than we usually have (often under 100). They break down, are hard
> to tune, and are black boxes, which defeats the point of explainability.

**Q: Why not just use class weights?**
> They change how much mistakes cost, but don't add any new rare examples. They're complementary,
> and could be added as extra baselines.

**Q: What does SHAP cost?**
> One small Random Forest and one SHAP pass per fold. On the four real datasets that's 3–12× the time of plain
> SMOTE, at most about 21 seconds per round (credit card, 165,000 training rows, vs 7 s for SMOTE). On large data
> SHAP is computed on all rare rows plus a sample of common rows, which keeps it fast without biasing the weights.

### Leakage (Objective 3)

**Q: How do you *know* there's no leakage?**
> Three ways. (1) **By design:** the preparation steps, SHAP and the oversampler are only fitted on training rows;
> the test part is only transformed and predicted on. (2) **By test:** `tests/test_experiment_pipeline.py` checks,
> round by round, that every sampler receives exactly the training rows and never a validation row. (3) **By
> tracing:** every fake row is saved with its two parent rows, and none of the 1.28 million saved fake rows has a
> parent from the test set.

**Q: How big is the leakage effect really?**
> In our own test, with the same data, method and model: PR-AUC **0.954 leaky vs 0.469 fair**
> (figure 2). In a more extreme run with a rarer class: **0.999 vs 0.207**. Vandewiele et al. (2021)
> found the same pattern across 11 published preterm-birth studies.

**Q: Some files say "nested CV", but there's no inner loop?**
> Correct. It's repeated stratified folds (2 × 5) with fixed settings, plus a locked test set. That's still
> leak-free, because nothing is tuned on test data. An inner tuning loop is on the list of next steps.

**Q: Why stratified folds?**
> With a rare class of 0.17%, a random fold could end up with zero frauds, and PR-AUC couldn't be
> computed. Stratifying keeps the real share in every fold.

**Q: Is your credit fraud data real?**
> Yes. The main benchmark uses the real Kaggle file (284,807 transactions, 492 frauds before duplicates are
> removed). We *also* ran a simulation earlier, because there we know which features truly carry the signal, so
> we could check that SHAP finds them (it found all 7 of 7). On the real data, SHAP's top features (V14, V17,
> V12, V10) largely match the ones that separate fraud best (V14, V10, V12, V4, V11).

**Q: Why did you subsample Breast Cancer and NSL-KDD?**
> In the original files the rare class isn't rare: 37% of tumours are malignant and 47% of connections are
> attacks. To study imbalance, we kept a fixed random subset of the rare class (once, with a fixed seed), giving
> 9 : 1 and 11.5 : 1. All common-class rows are kept. It's documented in [data/README.md](../data/README.md).

### Results (Objective 4)

**Q: Why PR-AUC instead of ROC-AUC?**
> ROC-AUC uses the false-alarm *rate*, which stays tiny when there are huge numbers of normal cases.
> That hides a flood of false alarms. PR-AUC uses precision, which shows them (Saito & Rehmsmeier
> 2015). The Epic sepsis model is the real-world example.

**Q: Why care about calibration?**
> Oversampling teaches the model that the rare class is about half of all cases, so its
> probabilities get inflated. On stroke, ECE goes from at most 0.017 with no balancing to as much as 0.28 with
> oversampling; on credit card from 0.001 to 0.086. Decisions are made on those probabilities.

**Q: What does explanation stability actually mean?**
> Retrain on different folds and check whether the model ranks the same features as important. We
> measure it with Kendall τ. If τ is low, any SHAP chart you show a regulator is partly noise.

**Q: So does SHAP-SMOTE win?**
> Not overall, and we don't claim it does. On four real datasets it **wins** against plain SMOTE with tree models
> on credit-card fraud (XGBoost F1 +0.025, fewer false alarms, significant), **ties** on breast cancer and network
> attacks, and **loses** on stroke (tree models catch about half as many strokes). Across all four, no method is
> significantly better than the others. A clear "sometimes, and here's when" is still a useful result.

**Q: Why does it fail on stroke?**
> Stroke data has category columns (work type, smoking, home area). SHAP correctly says they matter little, so
> it pairs patients who are similar in age and health but from *different* categories. A fake patient made
> between them gets values like "0.4 rural, 0.6 urban", which no real patient has. A model can spot SHAP's fake
> patients 93–96% of the time (plain SMOTE's: 81%), and the tree models learn "half-category = stroke". Real
> test patients never look like that, so the models stop predicting stroke. The fix: snap categories back to a
> real value after making each fake row. That is our top next step.

**Q: Then why not just skip balancing?**
> Often you should. No balancing wins 7 of 12 dataset + model combinations by F1, and on credit-card fraud it
> beats every oversampler by far (XGBoost F1 0.86 vs at most 0.75), because balancing 355 frauds up to 206,392 rows
> causes a flood of false alarms. But on stroke, without balancing the models almost never predict a stroke
> (recall 0 to 0.04). Balancing is a trade-off: more rare cases caught, more false alarms, less honest probabilities.
> That trade-off is exactly why we use several scores.

**Q: How will you decide a winner statistically?**
> Done: a Wilcoxon test for each SHAP version vs SMOTE on every dataset and model (paired by round), and a
> Friedman test across all 12 dataset + model combinations (PR-AUC p = 0.056, F1 p = 0.107: no significant overall
> difference). Still to add: the Nemenyi follow-up with a critical-difference chart (Demšar 2006).

---

## Part B: Glossary (plain English)

| Term | Meaning |
|---|---|
| **Imbalanced data** | One class (the rare one) is much less common than the other |
| **Minority / majority class** | The rare class (e.g. fraud) / the common class |
| **Imbalance ratio** | Common count ÷ rare count (real credit-card fraud is about 580) |
| **Oversampling** | Adding rare examples to the training data to balance it |
| **SMOTE** | Makes fake rare rows on the line between two similar rare rows |
| **Neighbours (k-NN)** | The k rows closest to a given row |
| **Euclidean distance** | Straight-line distance, with every feature counting equally |
| **SHAP** | A method that says how much each feature pushed a prediction, based on fair sharing from game theory |
| **TreeSHAP** | A fast, exact version of SHAP for tree models |
| **MASV** | Mean Absolute SHAP Value: a feature's average importance |
| **Scout model** | The small Random Forest used only to compute SHAP weights |
| **SHAP-Filtered SMOTE** | Finds neighbours using only the most important features ("mute button") |
| **SHAP-Weighted SMOTE** | Finds neighbours with each feature counted by its importance ("volume dial") |
| **Data leakage** | Test information sneaking into training, which makes scores look better than they are |
| **Fold / cross-validation** | Split the data into k parts and take turns using each part as the test set |
| **Stratified** | Each fold keeps the original class balance |
| **Detectability** | How easily a model can tell fake rows from real ones; 0.5 = not at all (realistic), 1.0 = always (they look fake) |
| **Parent rows** | The two real rows a fake row was made from; we save them for every fake row |
| **Locked test set** | The 25% of data set aside at the start and used only once, at the very end |
| **One-hot / category columns** | A category like smoking status turned into one 0/1 column per value; SMOTE can leave them at in-between values such as 0.4 |
| **Nested CV** | An extra inner loop of folds, used to tune settings inside each training part |
| **PR-AUC** | Area under the precision-recall curve: the main score for rare events |
| **Recall / Precision** | Share of real cases caught / share of alarms that were real |
| **F1** | A single number balancing recall and precision |
| **G-mean** | Combines how well each class is caught; 0 if either class is ignored |
| **Calibration / ECE** | Whether predicted probabilities match reality / the average gap between them |
| **Brier score** | The average squared error of predicted probabilities |
| **Kendall τ / Spearman ρ** | How similar two rankings are (1 = identical) |
| **Wasserstein distance** | How far apart two distributions are (the "earth-mover" cost) |
| **KS test** | A test of whether two samples could come from the same distribution |
| **Wilcoxon test** | A test of whether one method is consistently better than another across folds |
| **Outlier** | An unusual value. In fraud data, outliers are often the fraud itself, so we keep them |
| **Touchpoints T1–T4** | Where SHAP can plug into balancing: after the model / before balancing / during balancing / after making fake rows |
