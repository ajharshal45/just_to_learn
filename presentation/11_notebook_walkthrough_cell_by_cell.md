# 11 — The Four Experiment Notebooks, Cell by Cell

> **Use this for:** explaining any of the four experiment notebooks, from the first cell to the last, in plain
> words. **Read this first**, then the results file for your dataset:
> [12 Breast Cancer](12_results_breast_cancer.md) · [13 Credit Card](13_results_credit_card.md) ·
> [14 Stroke](14_results_stroke.md) · [15 NSL-KDD](15_results_nsl_kdd.md) · [16 Cross-dataset](16_cross_dataset_notebook.md)

---

## 0. One notebook, four datasets

There are four experiment notebooks:

| Notebook | Dataset | What the rare class is |
|---|---|---|
| `notebooks/01_breast_cancer_experiment.ipynb` | Breast Cancer Wisconsin | a malignant (cancerous) tumour |
| `notebooks/02_credit_card_fraud_experiment.ipynb` | Credit Card Fraud | a fraudulent transaction |
| `notebooks/03_stroke_prediction_experiment.ipynb` | Stroke Prediction | a patient who had a stroke |
| `notebooks/04_nsl_kdd_intrusion_experiment.ipynb` | NSL-KDD network traffic | a network attack |

**They are the same notebook.** All four are generated from one template, so they have the same 40 cells, the
same code and the same text. Only three cells differ:

* **Cell 1:** the title (the dataset name).
* **Cell 2:** one line, `DATASET = "breast_cancer"` (or `"credit_card"`, `"stroke"`, `"nsl_kdd"`).
* **Cell 40:** the written conclusion, which is about that dataset's results.

So if you can explain one notebook, you can explain all four. Only the numbers change.

### Words you'll hear (plain meanings)

| Word | Plain meaning |
|---|---|
| **Rare class / minority** | The uncommon thing we want to catch (fraud, cancer, stroke, attack). Labelled `1`. |
| **Common class / majority** | Everything else. Labelled `0`. |
| **Synthetic row / fake row** | A new rare-class row made up by an oversampler, so the model sees more rare examples. |
| **Oversampler / sampling method** | The recipe that makes the fake rows. We compare five, plus "no sampling". |
| **Neighbour** | Another rare row that looks similar. SMOTE builds a fake row *between* a rare row and one of its neighbours. |
| **λ (lambda)** | A random number between 0 and 1 that says *where* on the line between the two rows the fake row lands. 0 = on the first row, 1 = on the second, 0.5 = halfway. |
| **SHAP** | A tool that scores how much each column (feature) mattered to a model's decision. |
| **Leak / leakage** | When information from the test data sneaks into training, making results look better than they really are. We prevent it everywhere. |
| **Training data / test data** | The model learns from training data and is graded on test data it has never seen. |
| **Cross-validation (CV)** | Splitting the training data into 5 parts and taking turns: train on 4, check on 1. We do this twice (10 rounds). It shows how *stable* a result is. |
| **Recall** | Of all the real rare cases, how many did we catch? (Catching fraud.) |
| **Precision** | Of everything we flagged, how many were really rare cases? (Not crying wolf.) |
| **F1** | One number that balances recall and precision. |
| **PR-AUC** | How well the model *ranks* rare cases above common ones overall, without choosing a cut-off. Our main score for rare events. |
| **ROC-AUC** | A similar ranking score; it looks good too easily when the rare class is very rare, so we don't lead with it. |
| **Specificity** | Of all the normal cases, how many did we correctly leave alone? |
| **p-value** | How likely a difference this big is just luck. Below 0.05 we call it "significant". |

### The six methods compared

| Method | In plain words |
|---|---|
| **No Sampling** | Don't make any fake rows. Train on the data as it is. (The reference.) |
| **Random Oversampling** | Copy random rare rows until the classes are equal. No new information, just copies. |
| **SMOTE** | Make a new row on the line between a rare row and one of its 5 most similar rare rows. "Similar" counts every column equally. |
| **Borderline-SMOTE** | Like SMOTE, but only start from rare rows that sit close to the common class (the "danger zone"). |
| **SHAP-Filtered SMOTE** *(ours)* | Like SMOTE, but "similar" only looks at the columns SHAP says matter most (the top ones that make up 90% of the importance). |
| **SHAP-Weighted SMOTE** *(ours)* | Like SMOTE, but every column counts in proportion to its SHAP importance: important columns count a lot, unimportant ones barely. |

Every method that makes fake rows balances the training data to **1 : 1** (as many rare rows as common rows).

---

## The notebook, stage by stage

The notebook has **8 stages**. Below, every cell is listed in order with: **what it does**, **what you see**,
and **how to explain it**. Markdown cells are text only; code cells run something and show output.

---

## Title and setup (cells 1–2)

### Cell 1 — Title and introduction *(text)*
* **What it is:** The title, the research question, the list of six methods and the list of stages.
* **How to explain it:** *"The question is: if SHAP tells SMOTE which features matter, does SMOTE make better fake
  rare rows, and do models catch more rare cases?"*

### Cell 2 — Setup *(code)*
* **What it does:** Loads the Python tools and our project code, sets the chart style, picks the dataset with
  `DATASET = "..."`, and creates the folders where results get saved (`results/<dataset>/`).
* **What you see:** A **settings table**: random seed 42, 25% test set, 2 × 5-fold cross-validation, balance to
  1:1, 5 neighbours for SMOTE, 10 for Borderline-SMOTE, the 90% cut-off for SHAP-Filtered, the SHAP helper model
  (a small Random Forest), the three models, and the 0.5 decision threshold.
* **How to explain it:** *"Every setting is in one place and is the same for all four datasets. The only line
  that changes between notebooks is the dataset name."*

---

## Stage 1 — Loading the data (cells 3–5)

### Cell 3 — Stage 1 heading *(text)*
Explains that the loader removes ID columns, exact duplicate rows and columns that never change, and that every
row keeps its **original row number from the file**. That row number is how we later trace each fake row back to
the two real rows it came from.

### Cell 4 — Load the dataset *(code)*
* **What it does:** Reads the data file and applies the dataset's fixed rules (for example, keeping a fixed random
  subset of malignant cases in Breast Cancer, or dropping the `Time` column in Credit Card).
* **What you see:**
  1. A small table: source, web link, licence, file path and how the target (rare class) is defined.
  2. A bullet list of every change the loader made (dropped columns, duplicates removed, subsampling).
  3. The size, e.g. "396 rows × 30 features + target".
  4. The first 5 rows of the data.
* **How to explain it:** *"This is the raw data, and here is exactly what we changed before starting, and why."*

### Cell 5 — Data quality check *(code)*
* **What it does:** Sorts the columns into **numbers** and **categories** (like "smoking status"), and checks for
  problems.
* **What you see:**
  1. Which columns are numeric and which are categorical.
  2. A **quality summary**: missing cells, infinite values, duplicates left, extreme values, and how each is handled.
  3. A per-column table: type, missing values, number of different values, min and max.
* **How to explain it:** *"Missing values get filled in, but only using the training data. Extreme values are
  flagged, not deleted, because in rare-event data the extreme rows are often the rare cases themselves."*

---

## Stage 2 — Looking at the data (cells 6–9)

### Cell 6 — Stage 2 heading *(text)*
Says this stage only *describes* the data. Nothing here is used to train any model.

### Cell 7 — How imbalanced is it? *(code)*
* **What you see:** An **overview table** (rows, number of numeric and categorical features, how many common and
  rare rows, the rare-class percentage and the **imbalance ratio**, e.g. "19.5 : 1"), then a **bar chart** of the
  two classes.
* **How to explain it:** *"For every rare case there are 19 normal ones. A lazy model that always says 'normal'
  would be 95% accurate and completely useless. That's why accuracy alone is misleading."*

### Cell 8 — Which features separate the classes? *(code)*
* **What it does:** For each numeric feature, measures how different it looks for rare rows vs common rows (the
  "KS" score: 0 = identical, 1 = completely different).
* **What you see:** The top 10 features by that score, then **histograms** of the top 8 features with the two
  classes overlaid (skewed features like byte counts are shown on a log scale).
* **How to explain it:** *"Where the orange and blue shapes don't overlap, that feature helps tell the classes
  apart."*

### Cell 9 — Correlations and class comparison *(code)*
* **What you see:** A **correlation heatmap** (red = rise together, blue = one rises as the other falls) of the 12
  most separating features plus the target, then **box plots** comparing the two classes on the top 6 features.
* **How to explain it:** *"Strongly correlated features carry the same information twice. Plain SMOTE counts
  them twice when deciding what 'similar' means; SHAP can account for that."*

---

## Stage 3 — Preparing the data without leaking (cells 10–11)

### Cell 10 — Stage 3 heading *(text)*
Shows the plan as a small diagram: 25% of the data is locked away as the **test set**; the other 75% (the
**development set**) is used for training and for cross-validation. Every step that "learns" something (filling
missing values, turning categories into columns, scaling) learns it from training rows only.

### Cell 11 — Split and prepare *(code)*
* **What it does:** Splits the data 75/25, keeping the same rare-class percentage in both parts. Checks that no
  row is in both. Then fits the preparation steps on the development set.
* **What you see:**
  1. A table with the size of the development and test sets and how many rare rows each has.
  2. How many model columns there are after preparation (categories become one 0/1 column per category).
  3. A **preprocessing table**: for each feature, the value used to fill missing cells, the mean and spread used
     for scaling, and the mean *after* scaling in training vs test.
* **How to explain it:** *"Look at the last two columns: the training data is exactly 0 after scaling, the test
  data is only close to 0. That small difference is proof the test set was never used to fit anything."*

---

## Stage 4 — Making the fake rows and looking inside (cells 12–24)

This is the heart of the notebook: it shows **what each method actually generates**.

### Cell 12 — Stage 4 heading *(text)*
Explains that all SMOTE-type methods use the same recipe, *new row = row A + λ × (row B − row A)*, and only differ
in **which rows can be row A** and **how they decide which row B is "similar"**. Because all methods share the
same random seed, fake row #0 of SMOTE, SHAP-Filtered and SHAP-Weighted SMOTE starts from the **same row A** with
the **same λ**: any difference is purely the choice of neighbour.

### Cell 13 — Run every method *(code)* — **this is where the fake rows are created**
* **What it does:** Runs all six methods on the prepared development set, then trains the three models (Logistic
  Regression, Random Forest, XGBoost) on each result. *(Their test scores are only shown later, in Stage 6.)*
* **What you see:**
  1. A progress line per method.
  2. A **search-space table**: for each method, which rows can start a fake row, which columns it uses to judge
     "similar", and how many fake rows it made.
  3. A **counts table**: real common rows, real rare rows, fake rare rows, and the final totals.
  4. A **bar chart**: before and after balancing, with fake rows stacked on top of real rare rows.
* **How to explain it:** *"Each method takes the same small number of real rare rows and adds fake ones until
  there are as many rare as common rows."*

### Cell 14 — 4A heading: finding neighbours *(text)*
Explains the neighbour tables in the next cell.

### Cell 15 — Neighbour tables *(code)*
* **What you see:** For SMOTE, Borderline-SMOTE, SHAP-Filtered and SHAP-Weighted SMOTE, a table of the **5 nearest
  neighbours of the first two rare rows**, with:
  * the original row numbers of the row and of each neighbour;
  * the distance as this method measures it, and the plain straight-line distance;
  * **"rank under plain SMOTE"**: where that neighbour would rank if we ignored SHAP (rank 9 means plain SMOTE
    would not even have picked it);
  * the 3 features that make up most of the distance.

  Then a small table of Borderline-SMOTE's **SAFE / DANGER / NOISE** labels (only DANGER rows can start a fake
  row), and a table of **how much each method's neighbour list overlaps with plain SMOTE's** (1.0 = identical).
* **How to explain it:** *"This is where SHAP changes things: the SHAP methods pick different neighbours for many
  rows, because they judge similarity on the features that matter."*

### Cell 16 — 4B heading: making the fake rows *(text)*
Explains the next two cells.

### Cell 17 — The first fake rows of every method *(code)* — **see the new rows here**
* **What you see:** For each method, the **first 4 fake rows**, with:
  * an ID (`synthetic_id`), the method name;
  * `parent_1_row_id` and `parent_2_row_id`: the two real rows it was made from;
  * `neighbour_rank` (which of the 5 neighbours was used), `lambda`, and the distance between the parents;
  * `target = 1` (it's a rare-class row);
  * then the actual **feature values in their real units** (e.g. a real tumour size, a real transaction amount).

  For Random Oversampling there is no parent 2 or λ: the fake row is an exact copy of parent 1.
* **How to explain it:** *"Every fake row has a birth certificate: which two real rows it came from and how far
  along the line it sits."*

### Cell 18 — One fake row built step by step *(code)*
* **What you see:** For fake row #0 of each SMOTE-type method, a heading like
  *"SMOTE: synthetic #0 = parent_1 (row 370) + 0.4389 × (parent_2 (row 330) − parent_1)"*, and a table, feature by
  feature: parent 1's value, parent 2's value, the gap, λ × gap, and the **new value**. For the SHAP methods the
  features are sorted by SHAP importance and their weight is shown.
* **How to explain it:** *"The new value is always between the two parents. Same starting row, same λ, but a
  different neighbour: that's the whole difference between the methods."*

### Cell 19 — 4C heading: how SHAP guides the choice *(text)*
Explains the two SHAP methods in plain terms: a small helper model learns from the **training data only**, SHAP
scores each feature, and the scores become weights that add up to 1.

### Cell 20 — SHAP feature weights *(code)*
* **What you see:**
  1. A **weights table**: each feature's SHAP importance, its weight, the running total, how much SHAP-Weighted
     "stretches" it compared to plain SMOTE (above 1 = counts more), and whether SHAP-Filtered keeps it.
  2. A **bar chart** of the top 20 weights (green = kept by SHAP-Filtered; dashed line = what every feature would
     get in plain SMOTE).
  3. A **SHAP summary chart** (beeswarm): each dot is one training row; dots to the right push towards "rare
     class"; colour shows whether the feature value was high (red) or low (blue).
* **How to explain it:** *"SHAP tells us which features drive the rare class. These weights are what our methods
  feed into SMOTE."*

### Cell 21 — Why SHAP chose each neighbour *(code)*
* **What you see:** For each SHAP method, a **"rationale" table** for the first 8 fake rows: the parents, λ, the
  neighbour's rank in SHAP's view vs in plain SMOTE's view, how similar the two parents' SHAP explanations are
  (1.0 = they look alike to the model), which features made up the distance, and the parents' SHAP values on the
  top 3 features. Then a line like *"38% of these parent pairs would NOT be neighbours under plain SMOTE"*. Finally,
  a feature-level table for fake row #0: weight, share of the distance, both parents' SHAP values and values, and
  the new value.
* **How to explain it:** *"Here you can see SHAP's reasoning: it paired these two rows because they look alike on
  the features that matter, even when plain SMOTE wouldn't have paired them."*

### Cell 22 — 4D heading: are the fake rows any good? *(text)*
Explains the quality measures in the next cell.

### Cell 23 — Quality of the fake rows *(code)*
* **What you see:**
  1. A **quality table**, one row per method:
     * **majority_share_in_5nn**: for each fake row, what share of its 5 closest real rows are *common* class.
       High = fake rows land in the wrong territory. The first row ("Real minority") is the reference.
     * **Wasserstein distance** and **KS pass rate**: do the fake rows have the same spread of values as real rare
       rows, column by column? (Lower distance / higher pass rate = more realistic.)
     * **correlation similarity**: do columns that move together in real rare rows also move together in fake ones?
       (1.0 = identical.)
     * **parents' SHAP similarity** and **distinct parent pairs**.
     * **detectability**: we train a model to tell real rare rows from fake ones. 0.5 = it can't (good, realistic);
       1.0 = it always can (fake rows look fake).
     * **fractional one-hot columns per row** (datasets with categories only): how many category columns ended up
       between 0 and 1, e.g. "0.4 smoker", which no real person has.
  2. A **PCA picture**: all data squashed to 2-D. Blue = common, orange = real rare, green = fake rare, one panel
     per method.
  3. **Histograms** of real vs fake rare rows on the top 3 SHAP features.
* **How to explain it:** *"Good fake rows sit where real rare rows sit, have the same shape, and can't be told
  apart from real ones."*

### Cell 24 — Save all fake rows and neighbour tables *(code)* — **the final output of new rows**
* **What it does:** Writes **every** fake row of every method to a file, plus all neighbour tables and SHAP tables.
* **What you see:** A table listing each file, what it contains and how many rows:
  * `results/<dataset>/synthetic_samples/<method>_synthetic.csv` — **all generated rows**, each with its two parent
    row IDs, λ, neighbour rank, distance and all feature values in real units.
  * `results/<dataset>/neighbour_analysis/<method>_neighbours.csv` — every rare row's 5 neighbours.
  * `.../shap_weighted_smote_shap_rationale.csv` and `.../shap_filtered_smote_shap_rationale.csv` — SHAP's reasoning.
  * `.../shap_feature_weights.csv`, `.../minority_seed_shap_values.csv`, `.../neighbour_overlap_with_smote.csv`.
* **How to explain it:** *"Anyone can open these files and check any fake row against the two real rows it came
  from."*

> **Where exactly are the new synthetic rows?**
> * **Created** in **cell 13** (all methods are run there).
> * **First rows shown on screen** in **cell 17** (and one row explained feature by feature in **cell 18**).
> * **All rows saved** in **cell 24**, to `results/<dataset>/synthetic_samples/`. One file per method:
>   `random_oversampling_synthetic.csv`, `smote_synthetic.csv`, `borderline_smote_synthetic.csv`,
>   `shap_filtered_smote_synthetic.csv`, `shap_weighted_smote_synthetic.csv`.
>   (Credit Card files are compressed, `.csv.gz`. Credit Card and NSL-KDD files are too large for GitHub, so they
>   are only on the machine that ran the notebook.)

---

## Stage 5 — The fair test: cross-validation (cells 25–29)

### Cell 25 — Stage 5 heading *(text)*
Explains the 10-round test: in each round, *everything* (preparation, SHAP, fake rows, model) is redone from
scratch on 4/5 of the development data and checked on the remaining 1/5.

### Cell 26 — Run cross-validation *(code)*
* **What you see:** One line per round, e.g. *"fold 3/10 done in 7.1 s"*, and the total time.
* **How to explain it:** *"This is the slow part: 10 rounds × 6 methods × 3 models = 180 trained models."*

### Cell 27 — Results tables *(code)*
* **What you see:** One table per model (Logistic Regression, Random Forest, XGBoost). Rows = the six methods,
  columns = accuracy, precision, recall, F1, ROC-AUC, PR-AUC and specificity, each written as
  **average ± spread** over the 10 rounds.
* **How to explain it:** *"The spread matters: if two methods differ by less than the spread, the difference is
  probably luck."*

### Cell 28 — Graph 1 and the stability chart *(code)*
* **What you see:**
  1. **Graph 1: performance comparison.** Three panels (one per model), grouped bars for precision, recall, F1,
     ROC-AUC and PR-AUC, one colour per method, with error bars. A **▲** marks the best method for each score.
  2. **Box plots** of PR-AUC and F1 across the 10 rounds; each black dot is one round.
* **How to explain it:** *"Graph 1 shows who wins on average; the box plots show how consistent each method is."*

### Cell 29 — Is the difference real? *(code)*
* **What you see:** A table comparing each SHAP method with plain SMOTE for each model, on PR-AUC, F1, recall and
  precision: the average difference, in how many rounds it was better or worse, and the **p-value**. Rows with
  p < 0.05 are in **bold**.
* **How to explain it:** *"Bold means the difference is very unlikely to be luck. Most rows are not bold, which
  means most differences are too small to trust."*

---

## Stage 6 — The final exam: the locked test set (cells 30–32)

### Cell 30 — Stage 6 heading *(text)*
Explains that the models trained in cell 13 are now scored **once** on the 25% of data that was locked away, and
that this is a single check, so it is read together with the 10 rounds of Stage 5.

### Cell 31 — Test-set tables *(code)*
* **What you see:**
  1. One table per model in the format: **Sampling Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
     PR-AUC | Specificity**. The best value in each column is in bold.
  2. A table of **confusion-matrix counts**: correct normals (tn), false alarms (fp), missed rare cases (fn),
     caught rare cases (tp), plus two "honest probability" scores (Brier and ECE: lower = the model's percentages
     can be trusted more).
* **How to explain it:** *"fn is the number that hurts: each one is a missed fraud, missed tumour, missed stroke or
  missed attack. fp is the number of false alarms someone has to check."*

### Cell 32 — Test-set charts *(code)*
* **What you see:**
  1. **Confusion matrices**: an 18-square grid (3 models × 6 methods), each with the four counts.
  2. **ROC and precision–recall curves** for each model, one line per method, with each method's score in the
     legend. The dashed line is a model that guesses.
  3. **Calibration curves**: if a model says "30% chance", does it happen 30% of the time? The dashed diagonal is
     perfect honesty.
* **How to explain it:** *"Oversampling usually makes the model over-confident: it says 'high risk' more often than
  is true. You can see the lines drift away from the diagonal."*

---

## Stage 7 — Does the balancing change what the model looks at? (cells 33–35)

### Cell 33 — Stage 7 heading *(text)*
Explains that we run SHAP again, this time on the **final models**, to see which features they rely on, and check
whether that changes depending on how the data was balanced.

### Cell 34 — Graph 2 *(code)*
* **What you see:**
  1. **Graph 2: SHAP explainability comparison.** Three heatmaps (one per model): rows = the 12 most important
     features, columns = the six methods, each cell = that feature's share of the model's attention.
  2. A table: how similar each method's feature ranking is to the No Sampling model's (1.0 = identical ranking).
* **How to explain it:** *"If the colours stay the same across a row, the balancing method didn't change what the
  model pays attention to."*

### Cell 35 — Explanation stability *(code)*
* **What you see:** A table and bar chart of **Kendall τ**: how similar the model's feature ranking is from one
  round of cross-validation to the next (1.0 = always the same order).
* **How to explain it:** *"A trustworthy model should give the same explanation every time it's retrained. Higher
  is more stable."*

---

## Stage 8 — Saving and summing up (cells 36–40)

### Cell 36 — Stage 8 heading *(text)*
Lists the four results folders: `metrics/`, `synthetic_samples/`, `neighbour_analysis/`, `plots/`.

### Cell 37 — Save all results *(code)*
* **What you see:** A list of the 13 results files written to `results/<dataset>/metrics/` (cross-validation scores,
  test scores, significance tests, stability, fake-row quality, which rows went to training vs test, the
  settings), then the names of the 16 saved charts in `results/<dataset>/plots/`.

### Cell 38 — Final summary heading *(text)*
Says the numbers below are calculated by the code, not typed by hand.

### Cell 39 — Final summary *(code)*
* **What you see:**
  1. One line about the dataset, and the overview table again (size and imbalance).
  2. One line summing up the preprocessing.
  3. A table: for each method, real and fake row counts, time taken, and the main quality scores.
  4. A **SHAP vs SMOTE** table: for each model, the best method on PR-AUC, F1 and recall, and the difference between
     each SHAP method and plain SMOTE with its p-value.
  5. A bullet list of **findings written by the code from the numbers**: the PR-AUC ranking per model, each SHAP
     method vs SMOTE in words ("better in 7/10 rounds, p = 0.04, significant"), and the quality comparisons.

### Cell 40 — Interpretation *(text, written after the run)*
The conclusion for that dataset, answering the same seven questions in every notebook:
1. Which method performed best?
2. Did SHAP-guided SMOTE catch more rare cases?
3. Did it improve PR-AUC and F1?
4. Did it improve or hurt precision?
5. Did the fake rows keep realistic feature relationships?
6. (Across datasets: see notebook 05.)
7. Where did SHAP-guided SMOTE fail or underperform?

---

## Quick reference: which cell answers which question?

| If someone asks… | Show cell |
|---|---|
| "How imbalanced is the data?" | 7 |
| "Did you leak test data?" | 11 (training mean 0, test only close to 0) and 25–26 (everything refitted per round) |
| "Where are the fake rows?" | 13 (made), 17 (shown), **24 (all saved to `results/<dataset>/synthetic_samples/`)** |
| "How is one fake row built?" | 18 |
| "What does SHAP actually change?" | 15 (neighbour overlap), 20 (weights), 21 (rationale) |
| "Are the fake rows realistic?" | 23 |
| "Which method is best?" | 27 + 28 (Graph 1), then 31 (test set) |
| "Is the difference real or luck?" | 29 (p-values) |
| "How many cases did you miss?" | 31 (fn column) and 32 (confusion matrices) |
| "Does balancing change the model's reasoning?" | 34 (Graph 2) and 35 (stability) |
| "So what's the conclusion?" | 39 and 40 |
