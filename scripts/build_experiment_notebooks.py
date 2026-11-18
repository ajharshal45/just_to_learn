"""
Generates the four dataset notebooks from ONE template, so they are structurally identical:
the only per-dataset values are the dataset key and the names in the headings.

    python scripts/build_experiment_notebooks.py          # (re)write notebooks/01-04_*.ipynb
    python scripts/build_experiment_notebooks.py --cross  # also (re)write notebooks/05_cross_dataset_comparison.ipynb

After executing, the written interpretation cell (tagged `interpretation`) is filled in from
notebooks/interpretations/<dataset>.md by scripts/insert_interpretations.py, which does not
re-run the notebook.
"""

import argparse
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"

DATASETS = [
    ("breast_cancer", "01_breast_cancer_experiment.ipynb", "Breast Cancer Wisconsin (Diagnostic)"),
    ("credit_card", "02_credit_card_fraud_experiment.ipynb", "Credit Card Fraud Detection (ULB)"),
    ("stroke", "03_stroke_prediction_experiment.ipynb", "Stroke Prediction"),
    ("nsl_kdd", "04_nsl_kdd_intrusion_experiment.ipynb", "NSL-KDD Network Intrusion"),
]

SETUP = '''import sys, warnings
from pathlib import Path

# Locate the repository root (the folder that contains src/), wherever the notebook is run from
REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "experiment").is_dir())
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from src.experiment import config as C
from src.experiment import data as D, preprocessing as P, runner as R, inspection as I
from src.experiment import plots as PL, outputs as O, summary as S

%matplotlib inline
PL.set_style()
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

DATASET = "{slug}"   # <- the only line that differs between the four dataset notebooks
DIRS = O.results_dirs(DATASET)
PLOTS = DIRS["plots"]
display(C.settings_table())'''


def md(text: str, tags=None):
    c = nbf.v4.new_markdown_cell(text.strip())
    if tags:
        c.metadata["tags"] = tags
    return c


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def dataset_notebook(slug: str, title: str) -> nbf.NotebookNode:
    cells = [
        md(f"""
# SHAP-Guided SMOTE Benchmark: {title}

**Research question.** Does using SHAP feature attributions *inside* the oversampler produce better synthetic
minority samples, and better minority-class classification, than conventional oversampling?

This notebook is one of four that run **the same experiment** on different datasets
(`01` Breast Cancer, `02` Credit Card Fraud, `03` Stroke, `04` NSL-KDD). All four are generated from one template
(`scripts/build_experiment_notebooks.py`) and import every algorithm from `src/`, so the only difference between
them is the dataset loaded in the first code cell. `05_cross_dataset_comparison.ipynb` compares the four.

| | Method | Type | How it creates minority samples |
|---|---|---|---|
| 0 | **No Sampling** | reference | none, trains on the imbalanced data |
| 1 | **Random Oversampling** | established | duplicates random minority rows |
| 2 | **SMOTE** | established (Chawla et al., 2002) | interpolates between a minority row and one of its k nearest minority neighbours (Euclidean) |
| 3 | **Borderline-SMOTE** | established (Han et al., 2005) | SMOTE, but only from minority rows near the class boundary (DANGER rows) |
| 4 | **SHAP-Filtered SMOTE** | **proposed** | SMOTE whose neighbours are searched only on the features that hold 90% of the SHAP importance |
| 5 | **SHAP-Weighted SMOTE** | **proposed** | SMOTE whose neighbour distance weights every feature by its SHAP importance: $d_W=\\sqrt{{\\sum_j w_j (a_j-b_j)^2}}$ |

**Stages:** 1 ingestion → 2 EDA → 3 leak-free preprocessing → 4 oversampling and synthetic-sample inspection →
5 repeated cross-validation → 6 hold-out test evaluation → 7 SHAP explainability comparison → 8 saved outputs,
summary and interpretation.
"""),
        code(SETUP.replace("{slug}", slug)),

        # ---------------- Stage 1 ----------------
        md("""
## Stage 1: Data ingestion
The loader drops identifier and non-feature columns, exact duplicate rows and constant columns, and records any
one-off subsampling that fixes the imbalance ratio. None of these steps learns anything from the data. The
DataFrame index is the **original row number in the source file**, so every synthetic sample can later be traced
back to two real records.
"""),
        code('''df, spec = D.load_dataset(DATASET)
display(pd.Series({"Source": spec.source, "URL": spec.url, "License": spec.license, "File": spec.file,
                   "Target": spec.target_definition}, name=spec.title).to_frame())
display(Markdown("**Dataset-level steps applied by the loader:**\\n" + "\\n".join(f"- {n}" for n in spec.notes)))
print(f"Shape after loading: {df.shape[0]:,} rows x {df.shape[1] - 1} features + target")
display(df.head())'''),
        code('''quality = D.quality_report(df, spec)
print(f"Numerical features ({len(spec.numeric_features)}): {', '.join(spec.numeric_features[:12])}"
      + (" ..." if len(spec.numeric_features) > 12 else ""))
print(f"Categorical features ({len(spec.categorical_features)}): {', '.join(spec.categorical_features) or 'none'}")
display(D.quality_summary(df, spec))
display(quality.sort_values("Missing", ascending=False).head(12))'''),

        # ---------------- Stage 2 ----------------
        md("""
## Stage 2: Exploratory data analysis
EDA is descriptive only. The Kolmogorov-Smirnov ranking below only chooses which features to plot; it is never
used by any model or oversampler.
"""),
        code('''display(D.overview(df, spec))
PL.class_distribution(df, spec.title, PLOTS); plt.show()'''),
        code('''ks = D.ks_ranking(df, spec)
display(ks.head(10).to_frame("KS D (majority vs minority)"))
TOP_KS = list(ks.index[:8])
PL.feature_distributions(df, TOP_KS, PLOTS); plt.show()'''),
        code('''PL.correlation_heatmap(df, list(ks.index[:12]), PLOTS); plt.show()
PL.class_comparison(df, TOP_KS[:6], ks, PLOTS); plt.show()'''),

        # ---------------- Stage 3 ----------------
        md("""
## Stage 3: Leak-free preprocessing
```
full data --stratified 25%--> TEST (never resampled; used once, in Stage 6)
   └── DEVELOPMENT (75%) ── repeated stratified k-fold (Stage 5); in each fold, on the training part only:
                            impute + encode + scale  →  SHAP (inside the SHAP samplers)  →  oversample  →  fit model
```
Missing values: median (numeric) / most frequent (categorical) imputation. Categorical features are one-hot
encoded and numeric features are standardised. Every one of these steps is fitted on training rows only. The
table below shows the final fit on the whole development set: the scaler makes the development data exactly
mean 0, while the test set is only *close* to 0, because the test set was never used to fit anything.
"""),
        code('''dev_df, test_df = P.split_dev_test(df)
split = pd.DataFrame({"Rows": [len(dev_df), len(test_df)],
                      "Majority (0)": [int((dev_df.target == 0).sum()), int((test_df.target == 0).sum())],
                      "Minority (1)": [int(dev_df.target.sum()), int(test_df.target.sum())]},
                     index=["Development (75%)", "Hold-out test (25%)"])
split["Minority %"] = (split["Minority (1)"] / split["Rows"]).map("{:.2%}".format)
display(split)
assert not set(dev_df.index) & set(test_df.index), "development and test rows overlap"

pre = P.Preprocessor(spec).fit(dev_df)
print(f"Model space after encoding: {len(pre.feature_names_)} columns "
      f"({pre.n_numeric_} scaled numeric + {len(pre.feature_names_) - pre.n_numeric_} one-hot)")
display(pre.summary(dev_df, test_df).head(15))'''),

        # ---------------- Stage 4 ----------------
        md("""
## Stage 4: Oversampling and synthetic-sample inspection
The cell below fits **every sampler on the preprocessed development set** (the same chain is re-run inside every
CV fold in Stage 5) and fits the three downstream models on each resampled set. Test-set scores are only reported
in Stage 6.

All SMOTE-family methods share one generation engine (`src/oversampling/provenance.py`):
$x_{new} = x_a + \\lambda (x_{nn} - x_a)$ with $\\lambda \\sim U(0,1)$, interpolated in the full, unweighted feature space.
They differ only in **which rows can be anchors** $x_a$ and **in which space the neighbour $x_{nn}$ is searched**.
They also share one random seed. For SMOTE, SHAP-Filtered SMOTE and SHAP-Weighted SMOTE (which all draw anchors
from every minority row), synthetic row *i* therefore has the same anchor, the same neighbour rank and the same
$\\lambda$, so any difference between these three comes from neighbour selection alone. Borderline-SMOTE draws
from a smaller anchor pool, so its rows pair up differently. The engine records the parents, neighbour rank,
distance and $\\lambda$ of every synthetic row.
"""),
        code('''final = R.run_final(dev_df, test_df, spec)
display(I.search_space_description(final))
display(final.sampling)
PL.balanced_class_counts(final, PLOTS); plt.show()'''),

        md("""
### 4A. Neighbour identification
For each method: the k = 5 nearest minority neighbours of the first two minority seeds, the distance in the
method's search space, the plain Euclidean distance, the rank the same neighbour would have under plain SMOTE,
and the features that make up most of the search distance. For Borderline-SMOTE each seed is also labelled
SAFE / DANGER / NOISE from its 10 nearest neighbours in the whole training set; only DANGER rows can be anchors.
"""),
        code('''seed_ids = final.dev_row_ids[final.samplers["SMOTE"].minority_positions_[:2]]
for m in I.INTERPOLATING:
    t = I.neighbour_table(final, m, max_seeds=2)
    display(Markdown(f"**{m}**"))
    display(t.drop(columns="method"))
bl = final.samplers["Borderline-SMOTE"]
display(pd.Series({"SAFE": int(bl.safe_mask_.sum()), "DANGER (anchors)": int(bl.danger_mask_.sum()),
                   "NOISE": int(bl.noise_mask_.sum())}, name="Borderline-SMOTE seed status").to_frame())
display(I.neighbour_overlap_with_smote(final))'''),

        md("""
### 4B. Synthetic sample generation
Below are the first synthetic records of each method, with their two parents (original row ids), $\\lambda$ and
neighbour rank, followed by a feature-by-feature walk through synthetic row #0 of every interpolating method.
Because of the shared seed, row #0 of SMOTE, SHAP-Filtered and SHAP-Weighted SMOTE has the same anchor and the
same $\\lambda$, so you can see how the choice of neighbour changes the result (Borderline-SMOTE's anchor comes
from its DANGER pool). A Random Oversampling record is an exact copy of parent 1.
"""),
        code('''for m in C.OVERSAMPLERS:
    rec = I.generation_records(final, m)
    display(Markdown(f"**{m}**: {len(rec):,} synthetic rows"))
    display(rec.head(4).iloc[:, :14])'''),
        code('''for m in I.INTERPOLATING:
    if final.samplers[m].n_synthetic_ == 0:
        continue
    ex = I.example_walkthrough(final, m, 0, max_features=8)
    display(Markdown(f"**{ex.attrs['header']}**"))
    display(ex)'''),

        md("""
### 4C. SHAP guidance
Both proposed methods first fit a preliminary Random Forest (50 trees, depth 5) **on the training data only**,
explain it with `shap.TreeExplainer`, and turn the mean absolute SHAP value of each feature (MASV) into a weight
$w_j = \\text{MASV}_j / \\sum_k \\text{MASV}_k$.
* **SHAP-Filtered SMOTE** searches neighbours only on the top features whose cumulative weight reaches 90%.
* **SHAP-Weighted SMOTE** multiplies every column by $\\sqrt{w_j}$ before the Euclidean search, so the distance becomes
  $d_W(a,b)=\\sqrt{\\sum_j w_j (a_j-b_j)^2}$: high-SHAP features decide who counts as a neighbour.

The rationale tables show, for the first synthetic rows, the neighbour's rank in SHAP space and under plain SMOTE,
which features made up the SHAP distance, and how similar the two parents' SHAP explanations are (cosine).
"""),
        code('''weights = I.shap_weight_table(final)
display(weights.head(15))
PL.masv_weights(weights, PLOTS); plt.show()
PL.shap_summary(final, PLOTS); plt.show()'''),
        code('''for m in C.PROPOSED_METHODS:
    r = I.shap_rationale(final, m, n=8)
    display(Markdown(f"**{m}: neighbour-selection rationale (first 8 synthetic rows)**"))
    display(r)
    moved = (~r["would_plain_SMOTE_consider_it"]).mean()
    print(f"{moved:.0%} of these parent pairs would NOT be neighbours under plain SMOTE")
display(Markdown("**SHAP-Weighted SMOTE, synthetic row #0: parents' SHAP values on the highest-weight features**"))
display(I.parent_shap_detail(final, "SHAP-Weighted SMOTE", 0))'''),

        md("""
### 4D. Quality of the synthetic samples
* **majority_share_in_5nn**: share of majority rows among the 5 nearest real development rows of each synthetic
  row. Higher means more synthetic points sit in majority territory, a proxy for label noise. The real
  minority rows give the reference value.
* **Fidelity** of synthetic vs real minority rows: marginal Wasserstein distance, KS-test pass rate and
  correlation-matrix similarity (does the synthetic data keep the real feature relationships?).
* **parents_SHAP_cosine_mean**: how similar the explanations of the two parents are.
* **detectability_auc_all / _numeric**: cross-validated ROC-AUC of a Random Forest trained to tell real minority rows
  from synthetic ones (0.5 = indistinguishable, 1 = trivially separable), on all model columns and on the numeric
  columns only. **fractional_onehot_cols_per_row** counts one-hot columns that interpolation left between 0 and 1
  (datasets with categorical features only). A detectable synthetic class can teach a model an artefact that never
  appears in real test rows.
"""),
        code('''quality_tbl = I.synthetic_quality(final)
display(quality_tbl)
PL.pca_synthetic(final, PLOTS); plt.show()
PL.synthetic_feature_distributions(final, list(weights.index[:3]), PLOTS); plt.show()'''),
        code('''files = O.save_sampling_artifacts(DIRS, final)
display(files)'''),

        # ---------------- Stage 5 ----------------
        md("""
## Stage 5: Repeated stratified cross-validation (leak-free)
For each of the 2 × 5 folds of the development set, the full chain (preprocessing → SHAP → oversampling → model) is
re-fitted on the training part with fresh objects, and scored on the untouched validation part. This gives a
mean ± standard deviation for every (method, model) pair, plus paired per-fold comparisons.
"""),
        code('''cv = R.run_cross_validation(dev_df, spec)
print(f"Cross-validation finished in {cv.elapsed_s / 60:.1f} min")'''),
        code('''for model in C.MODEL_ORDER:
    display(Markdown(f"**{model}: cross-validation, mean ± sd over {C.CV_SPLITS * C.CV_REPEATS} folds**"))
    display(S.cv_comparison_table(cv, model))'''),
        code('''PL.performance_comparison(cv, PLOTS); plt.show()
PL.fold_distributions(cv, PLOTS); plt.show()'''),
        code('''tests = R.paired_tests(cv)
display(Markdown("**Wilcoxon signed-rank tests vs SMOTE, paired by fold** (two-sided; p < 0.05 marked)"))
display(tests[tests.method.isin(C.PROPOSED_METHODS)].style.format({"mean_difference": "{:+.4f}", "p_value": "{:.3f}"})
        .apply(lambda r: ["font-weight: bold" if r.p_value < 0.05 else "" for _ in r], axis=1))'''),

        # ---------------- Stage 6 ----------------
        md("""
## Stage 6: Hold-out test evaluation
The models fitted in Stage 4 on the (resampled) development set are scored **once** on the untouched 25% test
set, at the 0.5 decision threshold. Test-set rankings rest on a single split and few minority cases, so read them
together with the cross-validation results in Stage 5.
"""),
        code('''for model in C.MODEL_ORDER:
    display(Markdown(f"**{model}: hold-out test set**"))
    display(S.comparison_table(final, model).style.format("{:.4f}").highlight_max(axis=0, props="font-weight: bold"))
cm = final.test_metrics.set_index(["model", "method"])[["tn", "fp", "fn", "tp", "brier", "ece"]]
display(Markdown("**Confusion-matrix counts and calibration (Brier score, ECE) on the test set**"))
display(cm)'''),
        code('''PL.confusion_matrices(final, PLOTS); plt.show()
PL.roc_pr_curves(final, PLOTS); plt.show()
PL.calibration_curves(final, PLOTS); plt.show()'''),

        # ---------------- Stage 7 ----------------
        md("""
## Stage 7: SHAP explainability comparison
Do the final models rely on different features depending on how their training data was balanced? Each final
model is explained on the same test rows (all minority rows up to 250, plus majority rows up to 500 in total).
Logistic Regression uses `LinearExplainer` with a training background; the tree models use `TreeExplainer`.
Across CV folds, the stability of each model's SHAP feature ranking is measured with Kendall τ.
"""),
        code('''PL.shap_explainability_comparison(final, PLOTS); plt.show()
rank_corr = pd.DataFrame({model: final.importance_table(model).T.corr(method="spearman")["No Sampling"]
                          for model in C.MODEL_ORDER})
display(Markdown("**Spearman correlation of each final model's feature-importance vector with the No Sampling model**"))
display(rank_corr)'''),
        code('''stability = R.attribution_stability(cv)
display(stability)
PL.attribution_stability_plot(stability, PLOTS); plt.show()'''),

        # ---------------- Stage 8 ----------------
        md("""
## Stage 8: Saved outputs and summary
Every table and figure is saved under `results/<dataset>/` (same layout for all datasets):
`metrics/`, `synthetic_samples/`, `neighbour_analysis/` and `plots/`.
"""),
        code('''saved = O.save_metrics(DIRS, dev_df, test_df, cv, final, stability, tests, quality_tbl)
display(saved)
print("plots:", ", ".join(sorted(p.name for p in PLOTS.glob("*.png"))))'''),
        md("""
### Final summary
Numbers below are computed from this run, not typed by hand.
"""),
        code('''display(Markdown(f"**Dataset:** {spec.title}, {len(df):,} rows, {len(spec.features)} raw features "
                 f"({len(pre.feature_names_)} model columns after encoding)."))
display(D.overview(df, spec))
display(Markdown("**Preprocessing:** stratified 75/25 development/test split; median / most-frequent imputation, "
                 "one-hot encoding and standard scaling fitted on training rows only; oversampling only inside "
                 "training folds."))
display(Markdown("**Oversampling results (final development-set fit):**"))
display(final.sampling.join(quality_tbl.reindex(columns=["majority_share_in_5nn", "mean_wasserstein_distance",
                                          "correlation_matrix_similarity", "detectability_auc_all", "detectability_auc_numeric"]), how="left"))
display(Markdown("**SHAP-guided vs SMOTE (CV mean difference, Wilcoxon p-value):**"))
display(S.dataset_findings(cv, tests, quality_tbl).T)
display(Markdown("**Findings derived from the numbers:**\\n" + "\\n".join(f"- {l}" for l in S.findings_text(cv, tests, quality_tbl))))'''),
        md("""
### Interpretation
*(Written after the run; see `notebooks/interpretations/`.)*
""", tags=["interpretation"]),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def cross_notebook() -> nbf.NotebookNode:
    cells = [
        md("""
# Cross-Dataset Comparison: Does SHAP-Guided SMOTE Generalise?
This notebook reads the metrics saved by the four dataset notebooks (`results/<dataset>/metrics/`) and compares all
methods across Breast Cancer, Credit Card Fraud, Stroke and NSL-KDD. Every number is a cross-validation mean
(2 × 5 folds) unless labelled as hold-out test. Run notebooks 01-04 first.
"""),
        code(SETUP.replace('DATASET = "{slug}"   # <- the only line that differs between the four dataset notebooks\n'
                           'DIRS = O.results_dirs(DATASET)\nPLOTS = DIRS["plots"]\n', 'PLOTS = O.results_dirs("cross_dataset")["plots"]\n')
             .replace("display(C.settings_table())", "results = S.load_all_results()\nprint('loaded:', list(results))")),
        md("## 1. Datasets"),
        code('''rows = []
for slug in results:
    df_, spec_ = D.load_dataset(slug)
    rows.append(D.overview(df_, spec_).reset_index())
overview_all = pd.concat(rows).set_index("Dataset")
display(overview_all)'''),
        md("## 2. Cross-dataset performance tables (CV means)"),
        code('''for metric in ["pr_auc", "f1", "recall", "precision", "roc_auc"]:
    display(Markdown(f"**{R.METRIC_LABELS[metric]}**"))
    display(S.cross_cv_means(results, metric).style.format("{:.3f}").highlight_max(axis=1, props="font-weight: bold"))'''),
        md("## 3. F1, PR-AUC and recall across datasets"),
        code('''import seaborn as sns
fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharey="row")
for r, metric in enumerate(["pr_auc", "f1", "recall"]):
    t = S.cross_cv_means(results, metric).reset_index().melt(id_vars=["dataset", "model"], var_name="method", value_name=metric)
    for c, model in enumerate(C.MODEL_ORDER):
        ax = axes[r, c]
        sns.barplot(data=t[t.model == model], x="dataset", y=metric, hue="method", hue_order=C.METHOD_ORDER,
                    palette=C.METHOD_COLORS, ax=ax)
        ax.set(title=f"{R.METRIC_LABELS[metric]} - {model}", xlabel="", ylabel=R.METRIC_LABELS[metric], ylim=(0, 1.05))
        if (r, c) != (0, 2):
            ax.get_legend().remove()
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); fig.savefig(PLOTS / "cross_dataset_f1_prauc_recall.png", bbox_inches="tight"); plt.show()'''),
        md("## 4. Relative improvement of SHAP-guided SMOTE over conventional SMOTE"),
        code('''rel = pd.concat({R.METRIC_LABELS[m]: S.relative_improvement(results, m) for m in ["pr_auc", "f1", "recall", "precision"]}, axis=1)
display(rel.style.format("{:+.1f}%").background_gradient(cmap="RdBu", vmin=-20, vmax=20, axis=None))
fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
for ax, metric in zip(axes, ["pr_auc", "f1", "recall", "precision"]):
    t = S.relative_improvement(results, metric)
    t.index = [f"{d}\\n{m}" for d, m in t.index]
    t.plot.barh(ax=ax, color=[C.METHOD_COLORS[m] for m in t.columns], width=0.75, legend=False)
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.set(title=f"{R.METRIC_LABELS[metric]}: % change vs SMOTE", xlabel="%")
axes[-1].legend(t.columns, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); fig.savefig(PLOTS / "relative_improvement_vs_smote.png", bbox_inches="tight"); plt.show()'''),
        md("## 5. Best method per dataset and model"),
        code('''for metric in ["pr_auc", "f1"]:
    display(Markdown(f"**Best method by {R.METRIC_LABELS[metric]}**"))
    display(S.best_methods(results, metric))
    display(S.best_methods(results, metric)["best method"].value_counts().to_frame("wins (dataset x model blocks)"))'''),
        md("## 6. Overall consistency: average ranks, Friedman test and paired tests vs SMOTE"),
        code('''rank_tables = {}
for metric in ["pr_auc", "f1", "recall", "precision"]:
    ranks, stat, p = S.average_ranks(results, metric)
    rank_tables[R.METRIC_LABELS[metric]] = ranks
    print(f"{R.METRIC_LABELS[metric]:<10} Friedman chi2 = {stat:6.2f}, p = {p:.4f}  (12 dataset x model blocks, 6 methods)")
ranks_df = pd.DataFrame(rank_tables)
display(ranks_df.style.format("{:.2f}").highlight_min(axis=0, props="font-weight: bold"))
fig, ax = plt.subplots(figsize=(10, 4))
ranks_df.T.plot.bar(ax=ax, color=[C.METHOD_COLORS[m] for m in ranks_df.index], width=0.8)
ax.set(ylabel="Average rank (1 = best)", xlabel="", title="Average rank of each method over 4 datasets x 3 models")
ax.tick_params(axis="x", rotation=0); ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout(); fig.savefig(PLOTS / "average_ranks.png", bbox_inches="tight"); plt.show()
display(Markdown("**Fold-paired Wilcoxon tests of each proposed method vs SMOTE, counted over the 12 dataset x model blocks**"))
display(S.significance_overview(results))'''),
        code('''display(Markdown("**Synthetic-sample quality across datasets**"))
q = pd.concat({S.DATASET_TITLES[s]: r["quality"].set_index("method") for s, r in results.items()})
display(q.reindex(columns=["synthetic_rows", "majority_share_in_5nn", "mean_wasserstein_distance", "correlation_matrix_similarity",
                           "parents_SHAP_cosine_mean", "detectability_auc_all", "detectability_auc_numeric",
                           "fractional_onehot_cols_per_row"]))
display(Markdown("**Hold-out test PR-AUC (single split, for reference)**"))
test_pr = pd.concat({S.DATASET_TITLES[s]: r["test"].pivot(index="model", columns="method", values="pr_auc")[C.METHOD_ORDER]
                     for s, r in results.items()})
display(test_pr.style.format("{:.3f}").highlight_max(axis=1, props="font-weight: bold"))
rt = pd.concat({S.DATASET_TITLES[s]: r["sampling"].groupby("method")["sampling_seconds"].mean().reindex(C.METHOD_ORDER)
                for s, r in results.items()}, axis=1)
display(Markdown("**Mean oversampling time per CV fold (seconds; SHAP methods include the preliminary model and SHAP)**"))
display(rt)'''),
        md("## 7. Interpretation", tags=["interpretation"]),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--only", nargs="*", help="dataset keys to (re)build")
    args = ap.parse_args()
    for slug, fname, title in DATASETS:
        if args.only and slug not in args.only:
            continue
        nbf.write(dataset_notebook(slug, title), NB_DIR / fname)
        print("wrote", fname)
    if args.cross:
        nbf.write(cross_notebook(), NB_DIR / "05_cross_dataset_comparison.ipynb")
        print("wrote 05_cross_dataset_comparison.ipynb")
