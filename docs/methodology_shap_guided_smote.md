# Methodology: SHAP-Guided SMOTE as Implemented

This document defines every oversampling method in the benchmark **exactly as the code implements it**,
separates the established methods from the proposed contribution, and records the decisions and limitations
that affect how the results should be read.

Code: `src/oversampling/` (algorithms), `src/experiment/` (benchmark protocol).

---

## 1. Notation

* Training data (one training fold, after preprocessing): $\{(x_i, y_i)\}_{i=1}^{N}$, $x_i \in \mathbb{R}^p$, $y_i \in \{0, 1\}$, minority class $1$.
* $M = \{x_i : y_i = 1\}$, the $n_{min} = |M|$ minority rows; $n_{maj}$ majority rows.
* Number of synthetic rows: $G = n_{maj} - n_{min}$ (balance 1:1, `sampling_strategy="auto"`).
* $k = 5$ interpolation neighbours for every SMOTE-family method.

## 2. The shared generation step (all SMOTE-family methods)

Every SMOTE-family method generates synthetic row $g = 1, \dots, G$ by:

1. drawing an anchor $a_g$ uniformly from the method's **seed pool** $S \subseteq M$;
2. drawing a slot $r_g$ uniformly from $\{1, \dots, k\}$ and taking $x_{nn} = $ the $r_g$-th nearest minority neighbour of
   $x_{a_g}$ **in the method's search space**;
3. drawing $\lambda_g \sim U(0, 1)$ and returning

$$x^{new}_g = x_{a_g} + \lambda_g \,(x_{nn} - x_{a_g}),$$

computed in the **full, unweighted** feature space.

The draws happen in the order anchor → slot → $\lambda$, one synthetic row at a time, from
`numpy.random.default_rng(seed)`. All methods use the same seed (42), so for a given training set, synthetic row
$g$ of every method that shares the seed pool uses **the same anchor, slot and $\lambda$**. Differences between
SMOTE, SHAP-Filtered SMOTE and SHAP-Weighted SMOTE are therefore caused only by *which neighbour sits in each
slot*: a common-random-numbers design that isolates the effect of the neighbour metric.

The engine (`src/oversampling/provenance.py`) records $(a_g, r_g, nn_g, \lambda_g, d_g)$ for every synthetic row.
`tests/test_provenance_samplers.py` checks that the record rebuilds every synthetic row exactly.

| Method | Status | Seed pool $S$ | Neighbour search space | Implementation |
|---|---|---|---|---|
| No Sampling | reference | – | – | `NoSampling` |
| Random Oversampling | established | all of $M$ | none (exact copies of the anchor) | `RandomOversamplerWithProvenance` → imbalanced-learn `RandomOverSampler` |
| SMOTE (Chawla et al., 2002) | established | all of $M$ | Euclidean on all $p$ features | `SMOTEWithProvenance`; neighbour graph tested equal to imbalanced-learn's |
| Borderline-SMOTE-1 (Han et al., 2005) | established | DANGER rows only | Euclidean on all $p$ features | `BorderlineSMOTEWithProvenance`; DANGER set tested equal to imbalanced-learn's |
| **SHAP-Filtered SMOTE** | **proposed** | all of $M$ | Euclidean on the SHAP-selected feature subset $F$ | `SHAPFilteredSMOTE` |
| **SHAP-Weighted SMOTE** | **proposed** | all of $M$ | SHAP-weighted Euclidean $d_W$ on all features | `SHAPWeightedSMOTE` |

**Borderline-SMOTE DANGER rule.** For each $x \in M$, let $m'$ be the number of majority rows among its $m = 10$
nearest neighbours in the whole training fold. $x$ is DANGER if $m/2 \le m' < m$, NOISE if $m' = m$, and SAFE otherwise.
Anchors are drawn from DANGER rows only; their $k$ neighbours come from all of $M$. If no row is DANGER, nothing
is generated (same as imbalanced-learn).

**Random Oversampling** is delegated to imbalanced-learn (the implementation the repository already used); the
copied rows are read from its `sample_indices_`.

## 3. The proposed contribution: SHAP guidance

### 3.1 SHAP feature weights (MASV), training data only

Inside `fit_resample`, and therefore only on training rows:

1. Fit a preliminary model $f_0$ = `RandomForestClassifier(n_estimators=50, max_depth=5)` on the training fold.
2. Compute TreeSHAP values $\phi_{ij}$ (row $i$, feature $j$) of $f_0$ for the minority-class output.
3. Mean Absolute SHAP Value and normalised weight:

$$\text{MASV}_j = \frac{1}{N} \sum_{i=1}^{N} |\phi_{ij}|, \qquad w_j = \frac{\text{MASV}_j}{\sum_{l=1}^{p} \text{MASV}_l}, \quad \sum_j w_j = 1.$$

If all attributions are zero, $w_j = 1/p$ (the method reduces to SMOTE).

**Large training folds (efficiency option, `shap_max_rows`).** When $N > 20{,}000$ (Credit Card and NSL-KDD
only), $f_0$ is still fitted on all $N$ rows, but SHAP is evaluated on every minority row plus a uniform random
sample $B$ of $n_B = 20{,}000 - n_{min}$ majority rows, and

$$\text{MASV}_j = \frac{1}{N}\Big(\sum_{i \in M} |\phi_{ij}| + \frac{n_{maj}}{n_B} \sum_{i \in B} |\phi_{ij}|\Big),$$

an unbiased estimate of the full-data MASV. On a 40,000-row credit-card fold the capped weights correlated
0.9999 with the full ones (largest absolute weight difference 0.002), and `tests/test_experiment_pipeline.py`
checks the estimate. The default (`None`) explains every row, which is the original behaviour; Breast Cancer and
Stroke are always fully explained.

### 3.2 SHAP-Filtered SMOTE (subspace neighbour search)

Sort features by $\text{MASV}_j$ in descending order and keep the smallest prefix $F$ whose cumulative weight reaches
$\theta = 0.90$. Neighbours are found with

$$d_F(a, b) = \sqrt{\sum_{j \in F} (a_j - b_j)^2},$$

and interpolation (Section 2) still uses all $p$ features, so the synthetic row is complete.

### 3.3 SHAP-Weighted SMOTE (weighted neighbour search)

$$d_W(a, b) = \sqrt{\sum_{j=1}^{p} w_j\, (a_j - b_j)^2}.$$

This is implemented by multiplying column $j$ by $\sqrt{w_j}$ and running a standard Euclidean $k$-NN search, which
gives exactly $d_W$. Interpolation (Section 2) is unweighted.

### 3.4 What SHAP changes, and what it does not

SHAP changes **which minority rows are paired** (the neighbour graph). It does not change the anchor
distribution, the number of synthetic rows, or the interpolation formula. Each notebook shows this directly:
`neighbour_overlap_with_smote` (how many neighbours differ from SMOTE's), `rank_under_plain_SMOTE` (where the
SHAP-chosen neighbour would rank under Euclidean distance), and `parents_SHAP_cosine` (whether paired parents
have similar explanations).

This is the algorithm as it already existed in the repository (`compute_masv_weights`,
`SHAPFilteredSMOTE`, `SHAPWeightedSMOTE`). The benchmark work made two changes to it, neither of which changes its
output under the original settings:
1. generation was routed through the shared provenance engine with the same random-draw order. A fingerprint of
   the samplers' output before and after the refactor is bit-identical;
2. `shap_max_rows` was added (default off), together with `shap_values_` and the provenance attributes.

**Interpretation of the formulation as written in `Ideas.md`.** `Ideas.md` §4.3 says the weighted variant should
"interpolate along the weighted vector". The existing implementation interpolates in the *unweighted* space; only
the neighbour search is weighted. Interpolating in the weighted space and mapping back is algebraically identical
(the per-feature scaling cancels), so the two readings produce the same synthetic rows.

## 4. Benchmark protocol (identical for all four datasets)

* Stratified 75/25 development/test split (seed 42). The test set is never resampled and is used once.
* On the development set: 5-fold stratified cross-validation, repeated twice (10 folds). In every fold, a fresh
  preprocessor (median / most-frequent imputation, one-hot encoding, standard scaling of numeric columns), fresh
  sampler and fresh models are fitted on the training part only, and the validation part is scored.
* Final fit: the same chain on the whole development set, scored on the test set. The synthetic samples, neighbour
  tables and SHAP rationale shown in the notebooks come from this fit.
* Downstream models (repository defaults): Logistic Regression (L2, liblinear), Random Forest (100 trees, depth 6),
  XGBoost (100 trees, depth 5, learning rate 0.08).
* Metrics on the minority class at a 0.5 threshold: accuracy, precision, recall, specificity, F1, balanced accuracy,
  G-mean, plus the threshold-free ROC-AUC and PR-AUC (average precision), and calibration (Brier score, ECE).
* Statistics: two-sided Wilcoxon signed-rank tests of each method vs SMOTE, paired by fold; Friedman test over
  datasets × models in the cross-dataset notebook.
* Synthetic-sample quality (final development-set fit): marginal Wasserstein distance, KS pass rate and
  correlation-matrix similarity against the real minority; share of majority rows among each synthetic row's 5
  nearest real rows; and **detectability**, the ROC-AUC of a Random Forest that tries to tell real minority rows
  from synthetic ones (0.5 = indistinguishable). The detectability split is parent-aware: real minority rows are
  halved at random and a synthetic row is used only on the half that holds both of its parents. A naive
  random split puts a held-out real row next to its own synthetic children and biases the AUC below 0.5 (we
  observed 0.2–0.3 on Breast Cancer before correcting this).
* Explanations of the downstream models: SHAP on a fixed set of test (or validation) rows; the ranking stability
  across folds is measured with Kendall τ, Spearman ρ and top-5 overlap.

## 5. Limitations

1. **One-hot columns are interpolated, and this interacts with SHAP guidance.** SMOTE-family methods operate after
   one-hot encoding (Stroke, NSL-KDD), so synthetic rows can have fractional category indicators. The rule is the
   same for every method, but the effect is not: SHAP gives low-importance one-hot columns small weights (or drops
   them from the search), so SHAP-guided neighbours more often belong to different categories, and their
   children carry more fractional indicators. Tree models can learn that artefact (see the Stroke notebook).
   The saved records also show the nearest category. Snapping one-hot groups to a valid category, or a
   categorical-aware variant (SMOTE-NC), would address this. Neither is part of the compared methods.
2. **Fixed hyper-parameters.** $k$, $m$, $\theta$ and the model settings are the repository defaults, not tuned per
   dataset. Nothing is tuned on test data. Tuning could change the ranking of methods.
3. **Small minority counts.** The Breast Cancer test set has 10 malignant cases and the Credit Card test set
   about 118 frauds, so single-split test metrics are coarse. Repeated CV is the main evidence.
4. **One-off subsampling** fixes the imbalance ratio for Breast Cancer (9:1) and NSL-KDD (11.5:1). Results describe those
   tasks, not the natural class balance of the original files.
5. **10 folds limit statistical power.** Wilcoxon tests on 10 paired folds can only detect fairly consistent
   differences, and the CV folds overlap, which makes paired tests optimistic. Non-significance does not
   establish equivalence.
6. **Common random numbers.** Sharing the seed across methods reduces variance in the comparison of SMOTE with
   the SHAP variants. It also means each comparison reflects one realisation of the generation randomness per fold.
