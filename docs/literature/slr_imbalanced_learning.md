# Systematic Literature Review: Imbalanced Learning & Explainability-Aware Oversampling

This document fulfills **Objective 1** of the project:
> *Conduct a systematic review of existing approaches for handling class imbalance, analyze the geometric and theoretical limitations of conventional synthetic oversampling techniques on tabular data, and formalize a taxonomy categorizing how explainability (XAI) can be integrated into data-level balancing.*

---

## 1. Fundamentals of Imbalanced Learning

### 1.1 The Nature of Class Imbalance
Class imbalance occurs when the distribution of classes in a dataset is substantially skewed. In binary tabular classification, let the dataset be:
$$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N, \quad x_i \in \mathbb{R}^p, \quad y_i \in \{0, 1\}$$
where $y_i = 0$ denotes the majority (negative) class with $N_0$ instances, and $y_i = 1$ denotes the minority (positive) class with $N_1$ instances, such that $N_1 \ll N_0$.

The degree of imbalance is quantified by the **Imbalance Ratio (IR)**:
$$\text{IR} = \frac{N_0}{N_1}$$

In practical applications, imbalance spans several tiers:
- **Mild Imbalance:** $\text{IR} \in [2, 10]$ (e.g., customer churn, loan default).
- **Moderate Imbalance:** $\text{IR} \in (10, 100]$ (e.g., rare disease screening, defect detection).
- **Extreme Imbalance:** $\text{IR} > 100$ (e.g., credit card fraud where $\text{IR} \approx 500\text{--}1000$, network intrusion).

### 1.2 Why Standard Classifiers Fail Under Imbalance
Conventional machine learning algorithms (Logistic Regression, Support Vector Machines, Decision Trees, Gradient Boosted Trees) are formulated under the implicit assumption of balanced prior distributions and symmetric misclassification costs. Their standard objective function minimizes empirical classification error:
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f(x_i; \theta), y_i)$$

When $\text{IR} = 100$, a trivial classifier that predicts the majority class for all instances achieves an accuracy of $99.01\%$, despite having zero utility for detecting the rare class of interest. Specific mechanistic causes of failure include:
1. **Decision Boundary Shift:** The optimal Bayes decision boundary is pushed toward the minority class territory to maximize global margin or likelihood, causing minority instances near the boundary to be classified as negative.
2. **Gradient Starvation / Sub-gradient Dominance:** In gradient-based models (neural nets, gradient boosting), majority-class gradients overwhelm the loss surface, preventing the parameter updates necessary to separate minority clusters.
3. **Information Loss in Tree Induction:** Purity split criteria (Gini impurity, Shannon entropy) evaluate splits on overall node counts. Rare minority instances often lack the aggregate mass to justify specialized leaf splits, being absorbed into majority leaves as "noise."

### 1.3 Sources of Difficulty Beyond Class Ratios
Extensive empirical literature (He & Garcia, 2009; Fernández et al., 2018) demonstrates that the nominal imbalance ratio alone does not dictate learning difficulty. Rather, structural data factors exacerbate the problem:
- **Within-Class Small Disjuncts:** Rather than forming a single compact cluster, the minority class frequently fragments into small, isolated sub-clusters (disjuncts) scattered across the feature space. Classifiers struggle to generalize across sub-clusters containing only 2–5 instances.
- **Class Overlap & Ambiguity:** Regions in feature space where posterior probabilities $P(y=0|x)$ and $P(y=1|x)$ are close to equal.
- **Rare Borderline and Noisy Instances:** Minority points situated well inside majority regions (often due to sensor noise, recording errors, or atypical behavior) which, if oversampled, pollute the training signal.

---

## 2. Analysis of Synthetic Oversampling Techniques

Data-level oversampling seeks to rebalance the training distribution prior to model induction. Rather than simple random duplication (which causes severe overfitting without expanding boundary information), synthetic oversampling generates novel instances.

### 2.1 Standard SMOTE (Synthetic Minority Over-sampling Technique)
Proposed by Chawla et al. (2002), SMOTE forms synthetic instances along line segments connecting existing minority samples.

#### Algorithm:
For each minority instance $x_i \in \mathcal{S}_{\text{min}}$:
1. Find its $k$-nearest neighbours within $\mathcal{S}_{\text{min}}$ using Euclidean distance:
   $$d(x_i, x_j) = \|x_i - x_j\|_2 = \sqrt{\sum_{m=1}^p (x_{im} - x_{jm})^2}$$
2. Randomly select one neighbor $x_{zi}$ from the $k$-NN set.
3. Synthesize a new sample by linear interpolation:
   $$x_{\text{new}} = x_i + \lambda \cdot (x_{zi} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$

```
        o [Majority]
                           o [Majority]
    * [x_zi]
      \
       \   <-- x_new generated along linear segment
        \
         * [x_i]
                   o [Majority]
```

#### Theoretical Strengths:
- Expands the convex hull of the minority class.
- Forces decision trees and margin-based classifiers to create larger decision regions around minority pockets.

#### Critical Limitations:
- **Blind Linear Interpolation:** Ignores the presence of surrounding majority instances. If $x_i$ and $x_{zi}$ span an intrusion of majority points, $x_{\text{new}}$ is generated directly inside the majority territory, increasing label noise.
- **Isotropic Euclidean Assumption:** Treats all $p$ features as equally relevant to proximity. In real tabular datasets with high dimensions or uninformative variables, the Euclidean distance is dominated by noise dimensions, leading to nearest-neighbour graph corruption.

---

### 2.2 Difficulty-Aware and Boundary Oversamplers

#### 2.2.1 Borderline-SMOTE (Han et al., 2005)
Borderline-SMOTE categorizes each minority sample $x_i$ based on the composition of its $m$-nearest neighbours in the entire dataset (both classes):
- **Safe:** Majority neighbours $< m/2$. The instance is well-surrounded by minority points; oversampling here provides minimal boundary value.
- **Danger:** $m/2 \le$ Majority neighbours $< m$. The instance is near the decision boundary; oversampling here is prioritized.
- **Noise:** All $m$ neighbours are majority class. The instance is an isolated outlier; oversampling is suppressed to prevent noise propagation.

*Limitation:* Binary categorization can be brittle when boundaries are complex, and it still relies on isotropic Euclidean distance to evaluate both boundary status and interpolation.

#### 2.2.2 ADASYN (Adaptive Synthetic Sampling - He et al., 2008)
ADASYN automates density adaptation by weighting each minority instance $x_i$ according to its learning difficulty:
$$r_i = \frac{\Delta_i}{K}, \quad \hat{r}_i = \frac{r_i}{\sum_{j} r_j}$$
where $\Delta_i$ is the number of majority instances among the $K$-nearest neighbours of $x_i$. The number of synthetic samples generated around $x_i$ is directly proportional to $\hat{r}_i$.

*Limitation:* When minority instances are genuine mislabeled outliers (noise), ADASYN assigns them the highest difficulty weight, inadvertently generating large clouds of synthetic points around noise.

---

### 2.3 Hybrid Sampling (Oversampling + Cleaning)

To remedy boundary blurring and noise synthesis, hybrid strategies couple oversampling with post-hoc data cleaning:
- **SMOTE-ENN (Batista et al., 2004):** Performs standard SMOTE, then applies Wilson's Edited Nearest Neighbours (ENN). Any sample whose class differs from the majority of its 3 nearest neighbours is removed.
- **SMOTE-Tomek (Batista et al., 2004):** Performs SMOTE, then detects and removes Tomek Links (pairs of minimally distant instances from opposite classes), cleaning up ambiguous boundary regions.

*Advantage:* Substantially clears overlapping regions and sharpens class boundaries.
*Limitation:* Two-stage process with increased computational cost ($\mathcal{O}(N^2)$ for Tomek link searches), and still fails to incorporate feature-importance weighting during the initial generation phase.

---

### 2.4 Deep Generative Models for Tabular Data
Recent years have seen generative adversarial networks (CTGAN, MedGAN), variational autoencoders (TVAE), and tabular diffusion models (TabDDPM) adapted for minority oversampling.

While powerful for large-scale datasets, their application to imbalanced tabular problems is constrained by:
- **Data Hunger:** Deep generative networks require large sample volumes to reliably estimate continuous and categorical joint densities. Under severe class imbalance where minority samples number only $N_1 < 100$, GANs and diffusion models suffer mode collapse or severe overfitting.
- **Hyperparameter Sensitivity:** Adversarial training dynamics are notoriously sensitive and computationally intensive compared to geometric oversamplers.
- **Lack of Interpretability:** Deep generative pipelines function as opaque black boxes, preventing domain experts from verifying feature attribution constraints.

---

## 3. Explainability in Machine Learning & The Role of SHAP

### 3.1 Foundations of SHAP (SHapley Additive exPlanations)
Introduced by Lundberg & Lee (2017), SHAP grounds feature attribution in cooperative game theory (Shapley, 1953). In a game where $p$ players (features) collaborate to produce a payout (prediction $f(x)$), the Shapley value assigns a payout share $\phi_j$ to each player reflecting their marginal contribution across all possible feature coalitions.

For a model prediction $f(x)$, local explanations satisfy the additive property:
$$g(z') = \phi_0 + \sum_{j=1}^p \phi_j z_j'$$
where $\phi_0 = \mathbb{E}[f(X)]$ is the expected baseline prediction, and $z_j' \in \{0, 1\}$ indicates feature presence.

The classical Shapley value for feature $j$ is uniquely determined by four fundamental axioms:
1. **Efficiency:** $\sum_{j=1}^p \phi_j(x) = f(x) - \mathbb{E}[f(X)]$.
2. **Symmetry:** If features $j$ and $k$ contribute identically across all coalitions, $\phi_j = \phi_k$.
3. **Dummy (Null Player):** If feature $j$ does not alter prediction output for any coalition, $\phi_j = 0$.
4. **Additivity:** When combining two models, attributions sum directly: $\phi_j(f_1 + f_2) = \phi_j(f_1) + \phi_j(f_2)$.

### 3.2 Global Importance via Mean Absolute SHAP Values (MASV)
While individual $\phi_j(x_i)$ values explain local instance-level predictions, global feature attribution is derived by averaging absolute Shapley values over the dataset (or training fold):
$$\text{MASV}_j = \frac{1}{N} \sum_{i=1}^N |\phi_j(x_i)|$$

Unlike heuristic feature importance metrics (e.g., Gini importance in Random Forests, which biases toward high-cardinality features), MASV provides theoretically grounded, model-agnostic feature attributions.

---

## 4. Formal Taxonomy: Explainability Integration Points in Data Balancing

A primary contribution of this research is establishing a formal taxonomy classifying how explainability can be coupled with data balancing. Existing literature is structured into four distinct touchpoints:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       XAI INTEGRATION TAXONOMY MATRIX                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Touchpoint 1: Post-Hoc Diagnostic Interpretation (Status Quo)              │
│ Touchpoint 2: Pre-Balancing Feature Filtering (SHAP-Pruned Subspace)        │
│ Touchpoint 3: In-Process Feature-Weighted Geometry (SHAP Metric Shaping)    │
│ Touchpoint 4: Post-Generation Synthetic Validation & Constraint Verification│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Touchpoint 1: Post-Hoc Diagnostic Interpretation (Status Quo)
- **Workflow:** Balance Data $\to$ Train Classifier $\to$ Compute SHAP.
- **Role of XAI:** Evaluates the finished model to inspect which features drove predictions or verify that minority predictions are not based on spurious signals.
- **Limitation:** SHAP has zero active influence on the quality or placement of synthetic samples. If SMOTE generated corrupted samples, SHAP merely diagnoses the resulting degradation after the fact.

### Touchpoint 2: Pre-Balancing Feature Filtering (SHAP-Pruned Subspace)
- **Workflow:** Train Base Estimator $\to$ Compute MASV $\to$ Prune Low-Importance Features $\to$ Apply SMOTE $\to$ Train Final Classifier.
- **Role of XAI:** Eliminates noise and uninformative dimensions that degrade the $k$-NN distance calculation.
- **Mechanism:** Reduces dimensionality from $\mathbb{R}^p$ to $\mathbb{R}^{p'}$ ($p' < p$) retaining only features with $\text{MASV}_j \ge \tau$. SMOTE is executed strictly within this high-signal subspace.
- **Benefit:** Directly counters the curse of dimensionality and prevents irrelevant features from misguiding neighbour selection.

### Touchpoint 3: In-Process Feature-Weighted Geometry (SHAP-Weighted Metric)
- **Workflow:** Train Base Estimator $\to$ Compute MASV Weights $w_j \to$ Compute Weighted $k$-NN $\to$ Synthesize along Weighted Vector $\to$ Train Final Classifier.
- **Role of XAI:** Actively shapes the geometric metric of the oversampler.
- **Mechanism:** Replaces unweighted Euclidean distance with a feature-weighted distance:
  $$d_W(x_a, x_b) = \sqrt{\sum_{j=1}^p w_j \cdot (x_{aj} - x_{bj})^2}, \quad w_j = \frac{\text{MASV}_j}{\sum_m \text{MASV}_m}$$
- **Benefit:** Preserves all features while ensuring that proximity in feature space is determined by predictive influence rather than arbitrary scale or noise.

### Touchpoint 4: Post-Generation Synthetic Validation & Constraint Verification
- **Workflow:** Generate Synthetic Candidates $\to$ Explain Candidates via SHAP $\to$ Filter Out Inconsistent Samples $\to$ Train Final Classifier.
- **Role of XAI:** Functions as a quality gatekeeper. Synthetic minority samples whose SHAP attributions deviate drastically from the empirical distribution of real minority samples (e.g., attribution polarity inversions) are rejected as out-of-distribution artifacts.

---

## 5. Synthesis & Research Focus for this Project

The benchmark in this project directly implements and compares **Touchpoint 1 (Post-Hoc Baseline)**, **Touchpoint 2 (SHAP-Filtered SMOTE)**, and **Touchpoint 3 (SHAP-Weighted SMOTE)** against established classical and hybrid baselines.

By enforcing strict zero-data-leakage nested cross-validation, this study establishes the first comprehensive, multi-dimensional assessment of whether active explainability guidance (Touchpoints 2 and 3) provides verified empirical improvements over passive post-hoc analysis (Touchpoint 1).
