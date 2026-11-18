# Objective 1: Systematic Literature Review & Theoretical Taxonomy

This directory contains the completed research deliverables for **Objective 1** of the project:
> *Conduct a systematic review of existing approaches for handling class imbalance, with emphasis on synthetic oversampling techniques for tabular datasets, analyze their theoretical and geometric assumptions, and formalize a taxonomy categorizing how explainability (XAI) can be integrated into data-level balancing.*

---

## Documents in this Module

1. **[Systematic Literature Review (`slr_imbalanced_learning.md`)](./slr_imbalanced_learning.md)**
   - Theoretical foundations of class imbalance and mathematical explanations for classifier failure.
   - Analysis of conventional and boundary-adaptive oversampling techniques (SMOTE, Borderline-SMOTE, ADASYN, SMOTE-ENN, SMOTE-Tomek).
   - The isotropic Euclidean assumption and its breakdown in tabular spaces.
   - Foundations of cooperative game theory, Shapley values, and TreeSHAP.
   - The 4-level taxonomy of explainability integration touchpoints.

2. **[Comparative Structural Taxonomy Matrix (`taxonomy_matrix.md`)](./taxonomy_matrix.md)**
   - Direct cross-comparison matrix of 11 oversampling algorithms across geometric strategy, feature awareness, noise resilience, and time complexity.
   - XAI integration touchpoint mapping table.
   - Mathematical comparison of Euclidean, Mahalanobis, and SHAP-weighted distance metrics.

3. **[Literature Reading Notes (`reading_notes.md`)](./reading_notes.md)**
   - In-depth critical notes on foundational papers (Chawla et al. 2002; He & Garcia 2009; Fernández et al. 2018; Saito & Rehmsmeier 2015; Lundberg & Lee 2017; etc.).
   - Synthesis of unresolved literature gaps and how this project directly addresses them.

---

## Next Development Phase
With the theoretical foundations and taxonomy of **Objective 1** completed, development proceeds to **Objective 2**: *Algorithmic Design & Formulation of SHAP-Guided Oversampling* in `src/oversampling/`.
