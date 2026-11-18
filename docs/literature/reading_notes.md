# Literature Reading Notes & Critical Syntheses

This document compiles structured reading notes, methodological critiques, and key takeaways for the foundational and state-of-the-art papers underpinning **Objective 1** of this project.

---

## 1. Foundational Imbalanced Learning & SMOTE

### Paper 1: SMOTE: Synthetic Minority Over-sampling Technique
* **Authors:** Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, W. Philip Kegelmeyer (2002)
* **Venue:** *Journal of Artificial Intelligence Research (JAIR)*, Vol. 16, pp. 321–357.
* **Core Contribution:**
  - First formal introduction of synthetic data generation for imbalanced classification, proposing linear interpolation between minority nearest neighbours rather than simple random oversampling with replacement.
* **Key Methodology:**
  - For each minority point, locate its $k$-nearest minority neighbours via Euclidean distance.
  - Interpolate: $x_{\text{new}} = x_i + \text{rand}(0, 1) \times (x_{zi} - x_i)$.
  - Combined with random undersampling of the majority class.
* **Critical Limitations:**
  - Blinds itself to majority-class distributions; easily generates synthetic points inside majority clusters when minority points are near the decision boundary.
  - Relies entirely on isotropic Euclidean distance, treating all features with equal importance regardless of their true predictive power.
* **Direct Relevance to Project:**
  - Serves as the primary classical baseline against which all SHAP-guided methods will be benchmarked.

---

### Paper 2: Learning from Imbalanced Data
* **Authors:** Haibo He, Edwardo A. Garcia (2009)
* **Venue:** *IEEE Transactions on Knowledge and Data Engineering (TKDE)*, Vol. 21(9), pp. 1263–1284.
* **Core Contribution:**
  - Comprehensive survey categorizing imbalanced learning into data-level techniques, algorithm-level approaches (cost-sensitive learning), and ensemble paradigms.
  - Formalizes the distinction between between-class imbalance and intrinsic data difficulties (small disjuncts, class overlap).
* **Key Insights:**
  - Demonstrates that the nominal imbalance ratio is often less detrimental to classifier performance than within-class fragmentation and boundary overlap.
  - Reviews ADASYN and early adaptive techniques that weight instances by difficulty.
* **Direct Relevance to Project:**
  - Provides the theoretical justification for evaluating data-level sampling alongside diverse classifier families (linear, bagging, boosting).

---

### Paper 3: SMOTE for Learning from Imbalanced Data: Progress and Challenges (15-Year Anniversary)
* **Authors:** Alberto Fernández, Salvador García, Francisco Herrera, Nitesh V. Chawla (2018)
* **Venue:** *Information Sciences*, Vol. 460–461, pp. 388–405.
* **Core Contribution:**
  - Landmark 15-year retrospective on SMOTE, categorizing over 100 extensions and establishing formal taxonomy of modifications.
* **Taxonomy of SMOTE Extensions Identified:**
  1. *Modified selection of minority instances* (Borderline-SMOTE, Safe-Level-SMOTE).
  2. *Integration with data cleaning* (SMOTE-ENN, SMOTE-Tomek).
  3. *Cluster-based generation* (K-Means-SMOTE, C-SMOTE).
  4. *Dimensionality reduction / subspace sampling*.
* **Identified Open Challenges:**
  - High-dimensional imbalanced spaces where Euclidean distance breaks down.
  - Overlap and noise preservation.
  - Lack of systematic evaluation across non-accuracy metrics and explanation stability.
* **Direct Relevance to Project:**
  - Validates our core research gap: while dozens of geometric variants exist, integrating model explainability (SHAP) directly into nearest-neighbour metric formulation remains an under-explored frontier.

---

## 2. Evaluation & Metric Foundations

### Paper 4: The Precision–Recall Plot Is More Informative than the ROC Plot on Imbalanced Datasets
* **Authors:** Takaya Saito, Marc Rehmsmeier (2015)
* **Venue:** *PLoS ONE*, Vol. 10(3), e0118432.
* **Core Contribution:**
  - Rigorous mathematical demonstration that ROC-AUC provides overly optimistic visual and numerical assessments under severe class imbalance due to the large true negative (TN) count dominating the False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$).
  - Precision-Recall curves (PR-AUC / Average Precision) evaluate the minority positive class without TN inflation ($\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$, $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$).
* **Direct Relevance to Project:**
  - Establishes **PR-AUC as the primary performance metric** in Objective 4.

---

### Paper 5: A Survey on Imbalanced Learning: Latest Research, Applications and Future Directions
* **Authors:** Recent Survey (2024)
* **Venue:** *Artificial Intelligence Review*, Springer. [DOI: 10.1007/s10462-024-10759-6](https://doi.org/10.1007/s10462-024-10759-6)
* **Core Contribution:**
  - Modern review connecting traditional resampling with contemporary deep learning, foundation models, and explainability.
  - Highlights the need for multi-criteria assessment (fairness, explanation stability, fidelity).
* **Direct Relevance to Project:**
  - Direct literature justification for our multi-dimensional evaluation suite (combining classification utility with attribution stability and data fidelity).

---

## 3. Explainability & SHAP-Guided Balancing

### Paper 6: A Unified Approach to Interpreting Model Predictions
* **Authors:** Scott M. Lundberg, Su-In Lee (2017)
* **Venue:** *Advances in Neural Information Processing Systems (NeurIPS 2017)*, Vol. 30.
* **Core Contribution:**
  - Unifies six existing feature attribution methods under the framework of Shapley values from cooperative game theory.
  - Formulates TreeSHAP for polynomial-time, exact Shapley computation on tree ensembles.
* **Key Properties:**
  - Proves that SHAP is the unique additive feature attribution method satisfying *Local Accuracy*, *Missingness*, and *Consistency*.
* **Direct Relevance to Project:**
  - TreeSHAP enables rapid, exact computation of feature importance inside the training folds, making SHAP-guided oversampling computationally feasible.

---

### Paper 7: SHAP-Based Feature Selection and MASV-Weighted SMOTE for Enhanced Intrusion Detection
* **Authors:** Recent Institutional Publication (2023)
* **Venue:** *Institut Teknologi Sepuluh Nopember (ITS)*
* **Core Contribution:**
  - Investigates using Mean Absolute SHAP Values (MASV) to weight features during the SMOTE process in network intrusion detection.
  - Reports improved detection rates on rare network attacks compared to standard unweighted SMOTE.
* **Methodological Strengths:**
  - Concrete validation that weighting the oversampling process by feature importance improves downstream minority detection.
* **Identified Gaps to Address in Our Project:**
  - Evaluated on only a single domain/dataset (intrusion detection).
  - Did not investigate explanation stability (Kendall's $\tau$), probability calibration (ECE), or synthetic data distribution divergence (Wasserstein distance).
  - High risk of data leakage if feature selection and SHAP calculations were performed prior to cross-validation splits.
* **Direct Relevance to Project:**
  - Serves as the primary algorithmic springboard for our formalization of **SHAP-Filtered SMOTE** and **MASV-Weighted SMOTE**, which we subject to rigorous multi-dataset, multi-model, leakage-free benchmarking.

---

## 4. Synthesis of Literature Gaps Addressed by This Project

| Literature Limitation | How This Project Resolves It |
|---|---|
| **Isotropic Euclidean Assumption in SMOTE** | Formulates MASV-weighted distance metrics ($d_W$) so predictive features dominate nearest-neighbour topology. |
| **Post-Hoc XAI Isolation** | Bridges the gap between explainability and pre-processing by using SHAP in-process during oversampling. |
| **Flawed Evaluation (Accuracy / ROC Bias)** | Adopts a multi-dimensional matrix: PR-AUC, G-Mean, ECE, Wasserstein fidelity, and Kendall's $\tau$ attribution stability. |
| **Data Leakage in Published Benchmarks** | Enforces a strict nested cross-validation protocol where all XAI estimators, scalers, and samplers operate purely on inner training folds. |
