# Comparative Structural Taxonomy Matrix: Oversampling & Explainability

This document establishes the comparative taxonomy matrix fulfilling **Objective 1** of the project. It provides an analytical cross-comparison of conventional, boundary-adaptive, hybrid, and explainability-guided synthetic oversampling algorithms.

---

## 1. Algorithmic Oversampling Taxonomy Matrix

| Algorithm | Category | Geometric Synthesis Strategy | Feature-Aware? | Noise & Outlier Handling | Time Complexity (Sampling) | Primary Failure Mode / Vulnerability |
|---|---|---|:---:|---|:---:|---|
| **No Balancing** | Baseline | None (Natural Imbalance) | N/A | Ignored | $\mathcal{O}(1)$ | Extreme majority-class bias; decision boundary collapses |
| **Random Oversampling (ROS)** | Duplication | Exact sample duplication | No | Duplicates noise | $\mathcal{O}(N_{\text{new}})$ | Severe overfitting; zero boundary expansion |
| **Standard SMOTE** | Classical | Linear segment interpolation between $k$-NN | **No** (Uniform Euclidean) | Blindly interpolates through noise | $\mathcal{O}(N_{\text{min}} \log N_{\text{min}} + N_{\text{new}} p)$ | Generates synthetic points inside majority territory across uninformative dimensions |
| **Borderline-SMOTE 1 & 2** | Boundary-Focused | Interpolates only around minority instances in the "DANGER" zone | **No** (Uniform Euclidean) | Suppresses fully isolated noise | $\mathcal{O}(N_{\text{min}} \log N_{\text{all}} + N_{\text{new}} p)$ | Brittle $k$-NN majority threshold; Euclidean boundary definition degrades in high dimensions |
| **ADASYN** | Density-Adaptive | Density-weighted linear interpolation proportional to difficulty | **No** (Uniform Euclidean) | Highly vulnerable to noise | $\mathcal{O}(N_{\text{min}} \log N_{\text{all}} + N_{\text{new}} p)$ | Focuses high synthetic mass around genuine mislabeled outliers and noise points |
| **K-Means SMOTE** | Cluster-Based | K-Means clustering followed by intra-cluster SMOTE | **No** (Uniform Euclidean) | Filters out low-density / majority clusters | $\mathcal{O}(I \cdot K \cdot N_{\text{min}} p + N_{\text{new}} p)$ | Sensitive to initial $K$; spherical cluster assumption fails on non-convex manifolds |
| **Geometric SMOTE (G-SMOTE)** | Manifold/Geometric | Non-linear truncation within hyper-spheroids | **No** (Uniform Euclidean) | Parameter-controlled deformation | $\mathcal{O}(N_{\text{min}} \log N_{\text{min}} + N_{\text{new}} p)$ | Multiple geometric tuning hyperparameters; feature weights still uniform |
| **SMOTE-ENN** | Hybrid (Cleaned) | SMOTE followed by Edited Nearest Neighbours (ENN) | **No** (Uniform Euclidean) | Cleans misclassified points post-hoc | $\mathcal{O}(N_{\text{bal}} \log N_{\text{bal}})$ | High computational overhead; can excessively prune rare minority clusters |
| **SMOTE-Tomek** | Hybrid (Cleaned) | SMOTE followed by removal of Tomek link pairs | **No** (Uniform Euclidean) | Cleans ambiguous boundary pairs | $\mathcal{O}(N_{\text{bal}}^2)$ | $\mathcal{O}(N^2)$ pairwise distance checks; does not modify interpolation direction |
| **SHAP-Filtered SMOTE** | **XAI-Guided (Pre)** | SMOTE applied strictly within high-MASV predictive subspace | **Yes** (Subspace Selection) | Prunes uninformative dimensions before $k$-NN | $\mathcal{O}(T_{\text{SHAP}} + N_{\text{min}} \log N_{\text{min}} + N_{\text{new}} p')$ | Requires preliminary estimator to compute SHAP; threshold selection parameter |
| **SHAP-Weighted SMOTE (MASV)** | **XAI-Guided (In-Process)** | Linear interpolation along MASV-weighted Minkowski metric $d_W$ | **Yes** (Continuous MASV Weights) | Downweights influence of noise dimensions | $\mathcal{O}(T_{\text{SHAP}} + N_{\text{min}} \log N_{\text{min}} + N_{\text{new}} p)$ | Requires preliminary estimator; non-linear interactions across features partially averaged |

---

## 2. Taxonomy of Explainability (XAI) Integration Touchpoints

```
  Data Pipeline: [Raw Data] ──(1)──> [Balancing Engine] ──(2)──> [Downstream Model] ──(3)──> [Validation]
                                            │                                                │
                                    (Touchpoint 2 & 3)                                 (Touchpoint 1 & 4)
```

| Touchpoint Level | Pipeline Stage | Operational Mechanism | Theoretical Justification | Computational Overhead | Precedent / References |
|---|---|---|---|:---:|---|
| **Touchpoint 1: Post-Hoc Diagnostic** | Post-Training (Output) | Compute TreeSHAP or KernelSHAP on the final trained model | Verifies whether balancing altered top predictive drivers and diagnoses shortcut learning | $\mathcal{O}(M \cdot L \cdot D^2)$ (TreeSHAP) | Standard ML practice; widespread in applied medical/fraud studies |
| **Touchpoint 2: Pre-Balancing Feature Filtering** | Pre-Balancing (Input Space) | Train base estimator $\to$ compute global MASV $\to$ retain top $k$ features $\to$ apply SMOTE | Removing uninformative features concentrates nearest-neighbour searches on true discriminant manifolds | Low-to-Moderate (One preliminary tree fit + TreeSHAP) | Feature selection literature; MASV-based intrusion detection studies |
| **Touchpoint 3: In-Process Feature-Weighted Geometry** | In-Process (Metric Definition) | Weight distance metric $d_W(x_a, x_b) = \sqrt{\sum w_j (x_{aj} - x_{bj})^2}$ using normalized MASV $w_j$ | Prevents isotropic distortion where noise features exert equal pulling force in $k$-NN selection | Low-to-Moderate (Weights computed once per training fold) | Novel focus of this project; builds upon MASV-weighted SMOTE concepts |
| **Touchpoint 4: Post-Generation Synthetic Validation** | Post-Generation (Quality Gate) | Evaluate local SHAP attribution consistency of generated samples against empirical minority bounds | Rejects synthetic outliers that exhibit contradictory feature attributions prior to model induction | Moderate-to-High (SHAP evaluated for every synthetic sample) | Emerging constraint-verification literature |

---

## 3. Distance Metrics Comparison in Tabular Space

In tabular classification, the choice of distance metric fundamentally dictates nearest-neighbour topology:

### 3.1 Unweighted Euclidean Metric (Standard SMOTE)
$$d_{\text{Euclid}}(x_a, x_b) = \sqrt{\sum_{j=1}^p (x_{aj} - x_{bj})^2}$$
* **Hypothesis:** All $p$ features are orthogonal, equally scaled, and equally predictive.
* **Failure Condition:** If 5 features are highly predictive and 25 features are irrelevant noise, the noise dimensions contribute $\approx 83\%$ of the distance metric, rendering the $k$-NN graph completely spurious.

### 3.2 Mahalanobis Metric (Covariance Aware)
$$d_{\text{Mahal}}(x_a, x_b) = \sqrt{(x_a - x_b)^T \Sigma^{-1} (x_a - x_b)}$$
* **Hypothesis:** Accounts for linear feature covariance and variance scaling.
* **Failure Condition:** Inverting the empirical covariance matrix $\Sigma$ under extreme minority scarcity ($N_1 < p$) leads to ill-conditioned or singular matrices, causing numerical instability.

### 3.3 SHAP-Weighted Metric (MASV-Guided)
$$d_W(x_a, x_b) = \sqrt{\sum_{j=1}^p w_j \cdot (x_{aj} - x_{bj})^2}, \quad w_j = \frac{\text{MASV}_j}{\sum_{m=1}^p \text{MASV}_m}$$
* **Hypothesis:** Distance is proportional to empirical predictive influence learned by an ensemble model.
* **Advantage:** Unlike Mahalanobis, requires no matrix inversion (computationally stable even when $N_1 < p$). Unlike unweighted Euclidean, noise features receive near-zero weights ($w_{\text{noise}} \to 0$), suppressing their influence on neighbour graph topology.

---

## 4. Key Takeaways for Project Execution

1. **Baseline Suite Justification:** The benchmark must include standard SMOTE (unweighted Euclidean), Borderline-SMOTE (boundary heuristic), ADASYN (difficulty adaptive), and SMOTE-ENN (cleaning hybrid) to represent the key classical archetypes.
2. **XAI Innovation Focus:** The proposed *SHAP-Filtered SMOTE* (Touchpoint 2) and *SHAP-Weighted SMOTE* (Touchpoint 3) provide direct algorithmic answers to the isotropic Euclidean failure mode without the instability of covariance inversion or the data-hunger of deep generative models.
