# Explainability-Aware Synthetic Oversampling for Imbalanced Tabular Classification

## Final Project Title

**Explainability-Aware Synthetic Oversampling for Imbalanced Tabular Classification: A Systematic Literature Review and Benchmark of SHAP-Guided SMOTE Methods**

> **Status (2026-11-17).** This file is the original project charter and is kept as written. What was actually built
> differs in a few places:
> * **Datasets:** the benchmark uses four real datasets: Breast Cancer Wisconsin (subsampled to 9.2 : 1), Credit
>   Card Fraud (582 : 1), Stroke Prediction (19.5 : 1) and NSL-KDD (attacks subsampled to 11.5 : 1). German
>   Credit / Adult were not used. See [data/README.md](data/README.md).
> * **Methods compared:** No Sampling, Random Oversampling, SMOTE, Borderline-SMOTE, SHAP-Filtered SMOTE,
>   SHAP-Weighted SMOTE, each with Logistic Regression, Random Forest and XGBoost. (ADASYN, SMOTE-ENN, SMOTE-Tomek and
>   LightGBM remain in the earlier benchmark under `experiments/`.)
> * **Protocol:** a locked 25% test set plus 2 × 5 repeated stratified cross-validation, with all fitting inside the
>   training folds. Settings are fixed rather than tuned, so it is **not** nested cross-validation as described
>   below, although it is leak-free.
> * **Results:** SHAP-guided SMOTE is a situational improvement, not a general one. See the README "Headline
>   results" and `notebooks/05_cross_dataset_comparison.ipynb`. The algorithms as implemented are written up in
>   [docs/methodology_shap_guided_smote.md](docs/methodology_shap_guided_smote.md).

---

## Executive Summary

Imbalanced tabular data presents a persistent challenge in applied machine learning, particularly in high-stakes domains such as fraud detection, medical diagnostics, fault prediction, and network security. Conventional synthetic oversampling methods—most notably **SMOTE (Synthetic Minority Over-sampling Technique)** and its classical variants (Borderline-SMOTE, ADASYN)—generate synthetic minority instances by interpolating along Euclidean line segments between nearest neighbours. However, standard Euclidean distance treats all features uniformly, ignoring their true predictive contribution and allowing uninformative or noisy features to distort synthetic sample placement.

While **SHAP (SHapley Additive exPlanations)** is widely recognized as the gold-standard framework for post-hoc model explainability, existing research predominantly deploys SHAP **after** model training to interpret predictions. A critical research gap exists at the intersection of Explainable AI (XAI) and data balancing: **can SHAP-derived feature attributions be utilized directly during the data pre-processing and balancing phases to guide synthetic sample generation?**

This project establishes a systematic literature review, algorithmic formulation, and controlled empirical benchmark to investigate whether **SHAP-guided synthetic oversampling** provides statistically significant advantages over conventional oversampling in predictive minority utility, probabilistic calibration, synthetic sample fidelity, and explanation stability—while strictly eliminating data leakage via nested cross-validation.

---

## 1. Problem Statement & Research Gap

### 1.1 The Tabular Imbalance Problem
In many real-world tabular classification problems, the minority class represents critical rare events (e.g., fraud transactions at $<0.5\%$, malignant diagnoses, system failures). Classifiers trained on such data tend to default to majority-class bias, maximizing raw accuracy while failing to capture the minority class.

### 1.2 Limitations of Conventional SMOTE
Standard SMOTE generates synthetic samples according to:
$$x_{\text{new}} = x_i + \lambda \cdot (x_{zi} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$
where $x_i$ is a minority instance and $x_{zi}$ is one of its $k$-nearest minority neighbours identified via standard Euclidean distance.

This formulation suffers from several fundamental shortcomings:
1. **Feature Agnosticism**: Euclidean distance weights all dimensions equally ($\sum_j (x_{ij} - x_{zj})^2$), meaning noise features and uninformative variables influence neighbor selection as heavily as key predictive drivers.
2. **Boundary Distortion & Label Noise**: Interpolating uniformly across multi-dimensional feature space risks generating synthetic instances inside majority-class territories, creating artificial class overlap.
3. **Curse of Dimensionality**: In tabular data with mixed distributions and sparse signals, isotropic Euclidean neighbourhoods fail to reflect the true manifold of the minority distribution.

### 1.3 The Core Research Gap: Moving Beyond Post-Hoc XAI
Existing studies overwhelmingly follow a sequential pipeline:

```
[Imbalanced Data] ──> [Conventional SMOTE] ──> [Train Model] ──> [SHAP (Post-Hoc Diagnostic)]
```

Here, SHAP functions strictly as an after-the-fact explanation tool. This project shifts the paradigm:

```
[Imbalanced Data] ──> [Initial Feature Estimator] ──> [SHAP Feature Attributions]
                             │
                             ├──> (A) SHAP-Based Feature Selection (Pruning Noise)
                             └──> (B) SHAP-Weighted Interpolation (Geometry Modification)
                                         │
                                         ▼
                             [SHAP-Guided Synthetic Oversampling]
                                         │
                                         ▼
                           [Train Downstream Classifiers]
                                         │
                                         ▼
                     [Evaluate: Utility + Stability + Fidelity]
```

**Core Hypothesis:** Incorporating SHAP-derived feature importance into the synthetic sample generation process produces higher-quality minority samples, reduces boundary noise, and results in superior classification performance and attribution stability compared to conventional, feature-agnostic oversampling.

---

## 2. Project Aim & The 4 Core Objectives

### Project Aim
> **To systematically investigate, algorithmically formulate, and empirically benchmark explainability-aware synthetic oversampling techniques for imbalanced tabular classification—specifically evaluating whether SHAP-guided feature selection and SHAP-weighted minority generation outperform conventional SMOTE variants across predictive utility, model calibration, synthetic-data fidelity, and feature-attribution stability under strict leakage-free conditions.**

---

### The 4 Core Objectives

To ensure strict focus and disciplined execution throughout both research and future development, the project is structured around exactly **four core objectives**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            THE 4 CORE OBJECTIVES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Objective 1: Systematic Literature Review & Theoretical Taxonomy           │
│  Objective 2: Algorithmic Design & Formulation of SHAP-Guided Oversampling  │
│  Objective 3: Leakage-Free Experimental Benchmarking Architecture           │
│  Objective 4: Multi-Dimensional Evaluation & Explanation Stability Analysis │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Objective 1 — Systematic Literature Review & Theoretical Taxonomy
* **Formal Goal:** Conduct a systematic review of imbalanced learning and synthetic oversampling, and construct a formal taxonomy categorizing how explainability (XAI) can be integrated into data-level balancing.
* **Scope & Focus:**
  - Class imbalance taxonomy: between-class imbalance, within-class small disjuncts, and boundary overlap.
  - Systematic analysis of conventional, structure-aware, and cleaning-augmented oversampling (SMOTE, Borderline-SMOTE, ADASYN, K-Means-SMOTE, Geometric-SMOTE, SMOTE-ENN, SMOTE-Tomek).
  - Formal categorization of explainability touchpoints:
    1. *Post-hoc diagnostic interpretation* (baseline).
    2. *Pre-balancing feature filtering* (SHAP-pruned dimensional subspace).
    3. *In-process feature-weighted geometry* (SHAP-weighted distance metrics).
    4. *Post-generation synthetic fidelity validation*.
* **Implementation Anchor:** Anchors the literature synthesis, theoretical justifications, and taxonomy chapter in the project thesis/paper (`docs/literature/`).

#### Objective 2 — Algorithmic Design & Formulation of SHAP-Guided Oversampling
* **Formal Goal:** Mathematically formulate and implement modular, reproducible algorithms for SHAP-guided oversampling alongside standardized baseline oversamplers.
* **Scope & Focus:**
  - **SHAP-Filtered SMOTE:** Use tree-based Shapley values computed on the training set to identify and retain only top predictive features, filtering out uninformative noise dimensions prior to nearest-neighbour interpolation.
  - **SHAP-Weighted SMOTE (MASV-SMOTE):** Modify nearest-neighbour computation and sample synthesis using Mean Absolute SHAP Values (MASV). Compute feature weights $w_j = \frac{\overline{|\phi_j|}}{\sum_{m=1}^p \overline{|\phi_m|}}$ to define a weighted distance metric $d_W(x_a, x_b) = \sqrt{\sum_{j=1}^p w_j (x_{aj} - x_{bj})^2}$, focusing nearest-neighbour graphs on predictive dimensions.
  - Standardized benchmark implementations: Unbalanced Baseline, Random Oversampling (ROS), SMOTE, Borderline-SMOTE, ADASYN, SMOTE-ENN, and SMOTE-Tomek.
* **Implementation Anchor:** Directly guides the algorithmic code modules in `src/oversampling/`.

#### Objective 3 — Leakage-Free Experimental Benchmarking Architecture
* **Formal Goal:** Engineer a robust, reproducible, leakage-free nested cross-validation benchmarking pipeline across multiple diverse tabular imbalanced datasets and multiple classification algorithms.
* **Scope & Focus:**
  - **Zero Data Leakage Protocol:** Guarantee that SHAP attribution calculations, feature scaling, and oversampling operations occur strictly inside the inner training folds, never accessing validation or test splits.
  - Benchmark datasets: 3–5 well-characterized tabular datasets spanning distinct imbalance ratios (from moderate 10:1 to severe >100:1) and domains (e.g., Credit Card Fraud, German Credit/Adult, Disease Diagnostics, Network Intrusion).
  - Model Zoo:
    1. *Linear Baseline:* Logistic Regression (L2-regularized).
    2. *Bagging Ensemble:* Random Forest.
    3. *Gradient Boosting:* XGBoost and LightGBM.
  - Stratified Repeated Nested Cross-Validation ($5 \times 5$ or $5 \times 2$) with fixed random seeds for deterministic reproducibility.
* **Implementation Anchor:** Directly guides the pipeline and cross-validation execution engine in `src/pipeline/`.

#### Objective 4 — Multi-Dimensional Evaluation & Explanation Stability Analysis
* **Formal Goal:** Evaluate all benchmarked pipelines beyond conventional classification accuracy using a multi-dimensional assessment matrix encompassing minority predictive utility, probability calibration, synthetic-data statistical fidelity, and feature-attribution stability.
* **Scope & Focus:**
  - **Minority-Class Predictive Utility:** Area Under the Precision-Recall Curve (PR-AUC / Average Precision), Minority Recall, Minority Precision, F1-Score, Geometric Mean (G-Mean), and Balanced Accuracy.
  - **Attribution Stability:** Measure whether oversampling stabilizes or perturbs feature attributions across cross-validation folds using rank correlation coefficients (Kendall’s $\tau$, Spearman’s $\rho$) on SHAP value matrices.
  - **Probability Calibration:** Evaluate prediction probability reliability via Expected Calibration Error (ECE) and Brier Score.
  - **Synthetic Data Statistical Fidelity:** Quantify distributional divergence between real minority samples and synthetic instances using Wasserstein distance and Kolmogorov-Smirnov (KS) tests on marginal distributions.
  - **Computational Efficiency:** Profile runtime overhead of SHAP calculation and oversampling latency.
* **Implementation Anchor:** Directly guides the metrics calculation, statistical significance tests, and visualization engine in `src/evaluation/`.

---

## 3. Research Questions (Mapped 1-to-1 to Objectives)

| Research Question | Corresponding Objective | Focus |
|---|---|---|
| **RQ1 (Literature & Theory)**: *What are the theoretical and geometric limitations of Euclidean-based synthetic oversamplers, and how can XAI principles be structured into a formal data-balancing taxonomy?* | **Objective 1** | SLR, feature-agnostic limitations of SMOTE, taxonomic framework for explainability in balancing. |
| **RQ2 (Algorithmic Formulation)**: *How can SHAP feature attributions be mathematically formulated to guide synthetic generation (via feature pruning vs. weighted distance metrics), and what are their algorithmic properties?* | **Objective 2** | Formulation of SHAP-Filtered SMOTE and SHAP-Weighted (MASV) SMOTE algorithms. |
| **RQ3 (Empirical Benchmarking)**: *Does SHAP-guided synthetic oversampling yield statistically significant improvements in minority-class classification over conventional SMOTE variants without data leakage?* | **Objective 3** | Rigorous nested cross-validation benchmarking across multiple tabular datasets and model families. |
| **RQ4 (Multi-Dimensional Evaluation)**: *How does SHAP-guided oversampling impact downstream feature-attribution stability, probability calibration, and synthetic sample fidelity compared to standard baselines?* | **Objective 4** | Kendall's $\tau$ stability, ECE calibration, Wasserstein distributional fidelity, and runtime overhead. |

---

## 4. Algorithmic Formulation: SHAP-Guided Oversampling

### 4.1 Mean Absolute SHAP Value (MASV) Weighting
Let $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$ be the inner training fold with feature dimension $p$.
1. Train a preliminary base estimator $f_{\text{base}}$ (e.g., a shallow tree ensemble) on $\mathcal{D}_{\text{train}}$.
2. Compute local Shapley values $\phi_j(x_i)$ for each training instance $x_i$ and each feature $j \in \{1, \dots, p\}$.
3. Compute the **Mean Absolute SHAP Value (MASV)** for feature $j$:
   $$\text{MASV}_j = \frac{1}{N} \sum_{i=1}^N |\phi_j(x_i)|$$
4. Normalize the attributions to form feature importance weights $w = (w_1, \dots, w_p)^T$:
   $$w_j = \frac{\text{MASV}_j}{\sum_{k=1}^p \text{MASV}_k}, \quad \text{where } \sum_{j=1}^p w_j = 1, \; w_j \ge 0$$

### 4.2 Distance Modification in Nearest-Neighbour Search
Standard SMOTE identifies $k$-nearest minority neighbours using unweighted Euclidean distance:
$$d(x_a, x_b) = \sqrt{\sum_{j=1}^p (x_{aj} - x_{bj})^2}$$

In **SHAP-Weighted SMOTE (MASV-SMOTE)**, the distance is modulated by the feature weights:
$$d_W(x_a, x_b) = \sqrt{\sum_{j=1}^p w_j \cdot (x_{aj} - x_{bj})^2}$$

*Intuition:* Neighbours are selected based on similarity along the features that genuinely dictate minority-class separation, preventing noise features from falsely grouping distant samples.

### 4.3 Two Variants of Explainability-Aware Balancing
1. **SHAP Feature Selection + SMOTE (Pre-Filtering):**
   - Rank features by $\text{MASV}_j$.
   - Select top $k\%$ features (or features with cumulative MASV $\ge \theta$).
   - Execute standard SMOTE on the reduced, high-signal subspace.
2. **SHAP-Weighted SMOTE (In-Process Geometry Modification):**
   - Retain all features but calculate $k$-NN graphs using $d_W$.
   - Interpolate along the weighted vector to generate synthetic minority samples.

---

## 5. Strict Zero-Data-Leakage Protocol

A frequent flaw in published oversampling and XAI literature is applying balancing or feature selection across the entire dataset before splitting, which contaminates test distributions and yields artificially inflated, ungeneralizable performance.

### Protocol Specification
Every data-dependent transformation must occur strictly inside the training fold:

```
                          FULL DATASET
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     TRAINING FOLDS                        HELD-OUT TEST FOLD
   (e.g., 80% of data)                    (e.g., 20% of data)
            │                                     │
            ├──> [1. Fit Scaler (MinMax/Standard)]───> Transform Test Fold
            │                                     │
            ├──> [2. Fit Base Estimator]          │
            │           │                         │
            │           ▼                         │
            ├──> [3. Calculate SHAP Values]       │
            │           │                         │
            │           ▼                         │
            ├──> [4. Compute MASV Weights]        │
            │           │                         │
            │           ▼                         │
            ├──> [5. Generate Synthetic Samples]  │
            │           │                         │
            │           ▼                         │
            └──> [6. Train Final Classifier] ─────┼──> Predict on Untouched
                                                  │    Test Fold
                                                  ▼
                                       [7. Unbiased Evaluation]
```

* **Golden Rule:** The test/validation split is never seen by the scaler, the preliminary SHAP estimator, or the synthetic oversampler.

---

## 6. Proposed Experimental Benchmark Suite

### 6.1 Benchmark Methods Matrix

| Category | Method | Key Mechanism / Purpose |
|---|---|---|
| **Natural Baseline** | No Balancing (Raw) | Establish native performance on imbalanced distribution |
| **Simple Baseline** | Random Oversampling (ROS) | Naive duplication of minority instances (checks memorization) |
| **Classical Euclidean** | Standard SMOTE | Linear Euclidean interpolation between $k$-NN minority samples |
| **Boundary-Focused** | Borderline-SMOTE | Synthesize only near the decision boundary (DANGER zone) |
| **Density-Adaptive** | ADASYN | Adaptive generation proportional to difficulty level |
| **Hybrid Cleaned** | SMOTE-ENN | SMOTE followed by Edited Nearest Neighbours noise cleaning |
| **Hybrid Cleaned** | SMOTE-Tomek | SMOTE followed by Tomek Links boundary cleaning |
| **XAI-Guided (Method 1)** | SHAP Selection + SMOTE | Prune uninformative features via SHAP prior to SMOTE |
| **XAI-Guided (Method 2)** | SHAP-Weighted SMOTE | Distance metric weighted by Mean Absolute SHAP Values (MASV) |

### 6.2 Classification Model Suite

To ensure findings are robust and not artifacts of a single algorithm:
1. **Logistic Regression (L2 regularized):** Linear baseline; verifies linear separability and probability calibration.
2. **Random Forest:** Non-linear bagging ensemble; robust to outliers and variance.
3. **XGBoost:** High-performance gradient boosting; native handling of tabular sparsity and complex interactions.
4. **LightGBM (Optional/Secondary):** Fast histogram-based gradient boosting for scaling verification.

### 6.3 Candidate Benchmark Datasets

Selected based on varying degrees of imbalance, feature dimensionalities, and domain diversity:
* **Credit Card Fraud Detection (Kaggle/ULB):** Extreme imbalance ($\approx 0.17\%$ minority), PCA-transformed continuous features ($N \approx 284,807$, $p = 30$).
* **German Credit / Adult Census:** Moderate imbalance ($20\text{--}30\%$ minority), mixed numerical and categorical features.
* **Breast Cancer (WDBC) or Stroke Prediction:** High-stakes medical diagnostics with moderate-to-high imbalance ($1\text{--}5\%$ minority).
* **NSL-KDD / Intrusion Detection:** Network security domain with complex boundary overlap and high dimensional feature space.

---

## 7. Multi-Dimensional Evaluation Framework

Evaluating imbalanced classification requires looking well beyond traditional raw accuracy.

```
                         EVALUATION SUITE
                                │
        ┌───────────────┬───────┴───────┬───────────────┐
        ▼               ▼               ▼               ▼
 1. Minority Utility  2. Calibration  3. XAI Stability  4. Data Fidelity
 - PR-AUC (Avg Prec)  - Brier Score   - Kendall's τ     - Wasserstein Dist
 - Minority Recall    - ECE (Calib.)  - Spearman's ρ    - Marginal KS-tests
 - Minority Precision                 - Top-k Rank      - Covariance Corr
 - F1-Score & G-mean                    Overlap
```

### 7.1 Downstream Minority Utility
* **PR-AUC (Precision-Recall Area Under Curve):** Primary metric. Far more informative than ROC-AUC when positive instances are rare.
* **Minority Recall:** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (sensitivity to rare events).
* **Minority Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (false positive suppression).
* **F1-Score & G-Mean:** Harmonic and geometric balance between sensitivity and precision/specificity.
* **Balanced Accuracy:** Arithmetic mean of recall for each class.

### 7.2 Explanation & Attribution Stability
* **Hypothesis:** Does balancing stabilize or disrupt what the model learns as important?
* **Metric:** Extract the global SHAP feature ranking vector $R_{\text{fold}}$ across each cross-validation fold and compute pairwise rank correlation:
  $$\tau = \text{KendallTau}(R_{\text{fold}_a}, R_{\text{fold}_b}), \quad \rho = \text{SpearmanRho}(R_{\text{fold}_a}, R_{\text{fold}_b})$$
* A higher rank correlation indicates that the oversampling technique yields robust, reproducible feature attributions rather than fitting random synthetic artifacts.

### 7.3 Probabilistic Calibration
* Imbalanced sampling techniques frequently distort predicted probabilities, yielding poorly calibrated confidence scores.
* **Brier Score:** Mean squared difference between predicted probabilities and actual class labels.
* **Expected Calibration Error (ECE):** Discrepancy between predicted confidence and empirical accuracy across probability bins.

### 7.4 Synthetic Data Statistical Fidelity
* Evaluate whether synthetic samples preserve the underlying distribution of real minority instances or introduce artificial distortion:
  - **Wasserstein Distance:** Measures distributional distance between real and synthetic feature marginals.
  - **Kolmogorov-Smirnov (KS) Test:** Statistical test for identity of marginal distributions.
  - **Covariance Matrix Preservation:** Pearson correlation between real and synthetic feature correlation matrices.

---

## 8. Future Codebase Architecture (Mapped to Objectives)

To ensure that future development strictly adheres to the **4 Core Objectives**, the repository will be structured as follows:

```
SHAP+SMOTE/
├── docs/
│   └── literature/               <── [OBJECTIVE 1: SLR & Taxonomy Matrix]
│       ├── taxonomy_matrix.md
│       └── reading_notes.md
├── src/
│   ├── oversampling/             <── [OBJECTIVE 2: Algorithmic Formulations]
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract base sampler interface
│   │   ├── classical.py          # SMOTE, Borderline-SMOTE, ADASYN
│   │   ├── hybrids.py            # SMOTE-ENN, SMOTE-Tomek
│   │   ├── shap_selection.py     # SHAP-Filtered SMOTE
│   │   └── shap_weighted.py      # MASV-Weighted SMOTE
│   ├── pipeline/                 <── [OBJECTIVE 3: Leakage-Free Benchmarking]
│   │   ├── __init__.py
│   │   ├── datasets.py           # Dataset loaders and pre-processors
│   │   ├── cross_validator.py    # Nested CV engine (Zero Leakage)
│   │   └── models.py             # LR, Random Forest, XGBoost, LightGBM
│   └── evaluation/               <── [OBJECTIVE 4: Multi-Dimensional Evaluation]
│       ├── __init__.py
│       ├── metrics.py            # PR-AUC, Recall, F1, G-Mean, ECE
│       ├── stability.py          # Kendall's Tau / Spearman SHAP rank stability
│       ├── fidelity.py           # Wasserstein distance, KS tests, covariance
│       └── visualization.py      # Curves, plots, and summary LaTeX tables
├── experiments/
│   ├── configs/                  # YAML configurations for reproducible runs
│   └── run_benchmark.py          # Unified CLI entry point
├── tests/                        # Unit and leakage-prevention tests
├── Ideas.md                      # Project Charter & Specification (this file)
└── README.md
```

---

## 9. Recommended Literature Foundation

The project builds upon a foundational reading set structured across the 4 objectives:

### Foundational Imbalanced Learning & SMOTE
1. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research (JAIR), 16, 321–357.
2. **He, H., & Garcia, E. A. (2009).** *Learning from Imbalanced Data.* IEEE Transactions on Knowledge and Data Engineering (TKDE), 21(9), 1263–1284.
3. **Fernández, A., García, S., Herrera, F., & Chawla, N. V. (2018).** *SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-Year Anniversary.* Information Sciences, 460–461, 388–405.
4. **Bunkhumpornpat, C., Sinapiromsaran, K., & Lursinsap, C. (2009).** *Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-Sampling TEchnique for Handling the Class Imbalanced Problem.* PAKDD.
5. **Douzas, G., & Bacao, F. (2019).** *Geometric SMOTE: A novel synthetic oversampling method for imbalanced learning.* Information Sciences, 501, 118–135.

### Evaluation Metrics on Imbalanced Domains
6. **Saito, T., & Rehmsmeier, M. (2015).** *The Precision–Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLoS ONE, 10(3), e0118432.
7. **Branco, P., Torgo, L., & Ribeiro, R. P. (2016).** *A Survey of Predictive Modelling on Imbalanced Domains.* ACM Computing Surveys, 49(2), 1–44.
8. **Krawczyk, B. (2016).** *Learning from imbalanced data: open challenges and future directions.* Progress in Artificial Intelligence, 5(4), 221–232.

### Explainable AI & SHAP-Guided Balancing
9. **Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017.
10. **A Survey on Imbalanced Learning: Latest Research, Applications and Future Directions (2024).** Artificial Intelligence Review. [DOI: 10.1007/s10462-024-10759-6](https://doi.org/10.1007/s10462-024-10759-6).
11. **SHAP-Based Feature Selection and MASV-Weighted SMOTE for Enhanced Intrusion Detection.** [Institutional Publication](https://scholar.its.ac.id/en/publications/shap-based-feature-selection-and-masv-weighted-smote-for-enhanced/).

### Note on Academic Venue Classification
* In formal thesis and paper presentations, venue classifications (e.g., Q1/Q2) pertain to **journals indexed in Scopus/SCImago or Clarivate Web of Science (JCR)** for specific publication years and subject categories—not directly to individual papers or preprints.
* Leading venues in this corpus include *Information Sciences*, *IEEE TKDE*, *Artificial Intelligence Review*, and *ACM Computing Surveys*.
* Preprints (arXiv) and conference papers (e.g., NeurIPS, ICML) should be cited accurately by their respective peer-reviewed conference or preprint status.

---

## 10. Project Defense & Academic Q&A Guide

When presenting or defending this project to a professor, advisor, or reviewer:

### Q1: "What is your project in one sentence?"
> **"We are investigating whether incorporating SHAP-derived feature attributions into synthetic oversampling improves minority-class tabular classification and attribution stability compared to conventional, feature-agnostic SMOTE variants under strict leakage-free conditions."**

### Q2: "What is the specific research gap?"
> **"Most existing research treats SHAP as an after-the-fact explanation tool applied to a model trained on balanced data. Comparatively few studies investigate using SHAP upstream during the data balancing process to guide feature selection and nearest-neighbour distance weighting."**

### Q3: "What is your core technical novelty?"
> **"Rather than treating all features equally as standard Euclidean SMOTE does, we use Mean Absolute SHAP Values (MASV) from an initial training-fold estimator to weight the distance metric during neighbor selection and synthetic interpolation, preventing uninformative noise features from corrupting synthetic minority points."**

### Q4: "How do you ensure you don't leak information?"
> **"We enforce a strict nested cross-validation protocol: base estimator training, SHAP value extraction, MASV weight calculation, and synthetic sample generation occur exclusively within the inner training folds. Test and validation data are strictly isolated until final evaluation."**

### Q5: "Why not simply tune classifier class weights or focal loss?"
> **"Algorithm-level modifications such as cost-sensitive weights adjust error penalties, but do not address low minority density, sparse representation, or extreme feature space fragmentation. Data-level oversampling directly reshapes the minority representation manifold, and our benchmark will compare both data-level and algorithmic baselines."**

---

## 11. Milestone Execution Roadmap (Future Build Plan)

Every milestone in future implementation maps directly to the 4 objectives:

- [x] **Phase 1 (Objective 1):** Complete systematic literature matrix and formalize taxonomy of XAI integration points (`docs/literature/`).
- [x] **Phase 2 (Objective 2):** Implement `SHAPSelectionSMOTE` and `MASVWeightedSMOTE` alongside standard baseline oversamplers in `src/oversampling/` with thorough unit tests.
- [x] **Phase 3 (Objective 3):** Implement the dataset pre-processing loaders and the leakage-free nested cross-validation engine in `src/pipeline/`.
- [x] **Phase 4 (Objective 4):** Implement the multi-dimensional evaluation suite (PR-AUC, calibration ECE, Kendall's $\tau$ stability, and Wasserstein fidelity) in `src/evaluation/`.
- [x] **Phase 5 (Synthesis):** Run complete benchmarking suite across benchmark datasets and compile empirical results into formatted LaTeX tables and visualizations.
- [x] **Phase 6 (Real-data benchmark, 2026-11):** Four real datasets, one notebook each (`notebooks/01`–`04`), a cross-dataset comparison (`05`), traceable synthetic samples, an automatic leakage test, and written interpretations.
- [ ] **Next:** category-aware generation (snap one-hot groups), smaller balancing targets, an inner tuning loop (true nested CV), and a Nemenyi critical-difference chart.
