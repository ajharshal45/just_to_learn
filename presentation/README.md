# Presentation Pack: SHAP-Guided SMOTE

> Everything the team needs to explain the project, written in plain words.
> Each person owns one part, but **everyone should read 01, 02 and 09.**

## The project in 20 seconds

Rare events (fraud, disease, attacks) are hard for models to learn because there are so few
examples. **SMOTE** fixes this by making fake examples of the rare class. But SMOTE treats every
feature as equally important, so noise can make it build bad fake rows. **Our idea:** use
**SHAP** (a tool that measures how much each feature matters) to tell SMOTE which features to
pay attention to. Then we test this **fairly**, with no test data leaking into training, using
several scores, not just one.

## Files

| # | File | What it covers | Who |
|---|---|---|---|
| 01 | [01_problem_case_studies_novelty.md](01_problem_case_studies_novelty.md) | The problem, 6 real-world failures, and what's new about our approach | Opener |
| 02 | [02_how_the_system_works.md](02_how_the_system_works.md) | Every part of the system and how they fit together | Everyone |
| 03 | [03_objective1_literature_taxonomy.md](03_objective1_literature_taxonomy.md) | **Objective 1:** what other methods exist, where they fail, and where SHAP can plug in | Presenter 1 |
| 04 | [04_objective2_shap_guided_smote.md](04_objective2_shap_guided_smote.md) | **Objective 2:** our two methods (SHAP-Filtered and SHAP-Weighted), step by step | Presenter 2 |
| 05 | [05_objective3_leak_free_benchmark_trap.md](05_objective3_leak_free_benchmark_trap.md) | **Objective 3:** the data-leakage trap, our proof, and our leak-free test setup | Presenter 3 |
| 06 | [06_objective4_evaluation_stability.md](06_objective4_evaluation_stability.md) | **Objective 4:** the four scores we use and what the results show | Presenter 4 |
| 07 | [07_running_the_system.md](07_running_the_system.md) | Setup, demo commands, notebooks, and troubleshooting | Demo driver |
| 08 | [08_status_and_roadmap.md](08_status_and_roadmap.md) | What's done, what's not, and what comes next | Closer |
| 09 | [09_qa_defense_and_glossary.md](09_qa_defense_and_glossary.md) | Likely questions with answers, plus a glossary | Everyone |
| 10 | [10_shap_filtered_smote_deep_dive.md](10_shap_filtered_smote_deep_dive.md) | **Deep dive:** SHAP-Filtered SMOTE, what it is, how it works, and all its benchmark results | Presenter 2 |
| 11 | [11_notebook_walkthrough_cell_by_cell.md](11_notebook_walkthrough_cell_by_cell.md) | **The four experiment notebooks, cell 1 to cell 40**, in plain words (the four notebooks are identical except for the dataset); includes where the new synthetic rows are | Demo driver + everyone |
| 12 | [12_results_breast_cancer.md](12_results_breast_cancer.md) | What each cell shows on **Breast Cancer** (real outputs and numbers) | Presenter 4 |
| 13 | [13_results_credit_card.md](13_results_credit_card.md) | What each cell shows on **Credit Card Fraud** | Presenter 4 |
| 14 | [14_results_stroke.md](14_results_stroke.md) | What each cell shows on **Stroke**, including why SHAP-guided SMOTE fails there | Presenter 4 |
| 15 | [15_results_nsl_kdd.md](15_results_nsl_kdd.md) | What each cell shows on **NSL-KDD** network attacks | Presenter 4 |
| 16 | [16_cross_dataset_notebook.md](16_cross_dataset_notebook.md) | The cross-dataset notebook (05), cell by cell: does the method work everywhere? | Presenter 4 + Closer |

## Suggested running order (~25 min + Q&A)

| Min | What to cover | File |
|---|---|---|
| 0–4 | **Hook:** the problem plus 3 real failures (preterm-birth leakage, Epic sepsis, Dutch benefits) | 01 |
| 4–6 | Our idea: use SHAP *before* the model, not just after | 01 §4 |
| 6–9 | Objective 1: what exists, where it breaks, where SHAP can plug in | 03 |
| 9–13 | Objective 2: the two methods, with the credit-fraud pictures | 04 + 10 |
| 13–17 | Objective 3: **the leakage trap** (0.954 vs 0.469), then our four-dataset setup | 05 |
| 17–21 | Objective 4: the four scores and the honest results on four real datasets (credit card ✅, stroke ❌) | 06 + 16 (+ 14 for "why stroke fails") |
| 21–23 | Live demo: `pytest`, then scroll through a dataset notebook (outputs are saved) | 07 + 11 |
| 23–25 | Status, next steps, what success looks like | 08 |

## The most important pictures (use these on slides)

| Picture | What it shows | Use in |
|---|---|---|
| [fig1: where fake points land](../docs/images/proofs/fig1_geometric_decision_boundary_proof.png) | Plain SMOTE vs SHAP-Weighted SMOTE: where fake rows are placed | 04 |
| [SHAP finds the real features](../docs/images/credit_fraud/shap_weights_and_filtered_subspace.png) | SHAP picks out all 7 real fraud features out of 28 | 04, 10 |
| [fake fraud placement](../docs/images/credit_fraud/synthetic_placement_scatter.png) | Where each method puts its fake fraud rows | 10 |
| [fig2: the leakage trap](../docs/images/proofs/fig2_data_leakage_trap_proof.png) | Same data, same model: a leaky setup scores 0.954 PR-AUC, a fair one 0.469 | 05 |
| [Graph 1: credit card](../results/credit_card/plots/10_graph1_performance_comparison.png) | All six methods × three models on real fraud data | 06, 13 |
| [fake rows vs real rows (PCA)](../results/breast_cancer/plots/08_pca_original_vs_synthetic.png) | Where each method puts its fake rows on real data | 04, 12 |
| [relative change vs SMOTE](../results/cross_dataset/plots/relative_improvement_vs_smote.png) | Better on credit card, worse on stroke, no change elsewhere | 06, 16 |
| [average ranks](../results/cross_dataset/plots/average_ranks.png) | No overall winner; SHAP-Filtered above plain SMOTE | 16 |
| [stroke Graph 1](../results/stroke/plots/10_graph1_performance_comparison.png) | The stroke failure: recall drops for the tree models | 14 |

## Three sentences everyone must be able to say

1. **Problem:** *"SMOTE treats every feature as equally important, and it's easy to use in a way
   that leaks test data, so rare-event models look great on paper and fail in real use."*
2. **Solution:** *"We use SHAP, computed only on the training data, to decide which rare rows
   count as neighbours when SMOTE makes new fake rows."*
3. **How we prove it:** *"We test it with no leakage, on four kinds of score (catching rare
   cases, honest probabilities, stable explanations, realistic fake data), and we report the
   results honestly, even when plain SMOTE wins."*

## Ground rules for presenting

- **Don't claim SHAP-SMOTE clearly wins.** On the four real datasets it beats plain SMOTE with tree models on credit-card
  fraud, ties on breast cancer and network attacks, and clearly loses on stroke (mixed number + category data).
  See [16](16_cross_dataset_notebook.md) and [14](14_results_stroke.md). (The earlier simulated-data results are in
  [06](06_objective4_evaluation_stability.md) §7 and [10](10_shap_filtered_smote_deep_dive.md) §6.)
- **Only quote real-world numbers from the sources linked in 01.** Read the source first.
- **Be open about what's unfinished** ([08](08_status_and_roadmap.md)). Panels punish
  over-claiming, not honesty.
