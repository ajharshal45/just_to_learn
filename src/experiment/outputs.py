"""
Writes every experiment artefact to `results/<dataset>/`, with the same layout for all datasets:

    results/<dataset>/
        metrics/             CV fold metrics + summary, test metrics (incl. confusion matrices),
                             paired tests, attribution stability, synthetic quality, sampling counts,
                             downstream SHAP importances, dev/test split assignment, settings
        synthetic_samples/   one file per oversampler: every synthetic row, its two parents (row ids),
                             lambda, neighbour rank, search distance and values in original units
        neighbour_analysis/  one file per oversampler: every minority seed's k neighbours, distances,
                             and Borderline DANGER status; SHAP weights, minority SHAP values,
                             SHAP rationale for the SHAP methods; neighbour overlap with SMOTE
        plots/               every figure of the notebook (PNG)

Files larger than 50,000 rows are written gzip-compressed (`.csv.gz`, readable with pandas.read_csv).
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from src.experiment import config as C
from src.experiment import inspection as I
from src.experiment.runner import CVResult, FinalResult

SUBDIRS = ["metrics", "synthetic_samples", "neighbour_analysis", "plots"]
GZIP_ROWS = 50_000


def results_dirs(slug: str) -> Dict[str, Path]:
    root = C.RESULTS_DIR / slug
    dirs = {"root": root, **{d: root / d for d in SUBDIRS}}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def slugify(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    if len(df) > GZIP_ROWS:
        path = path.with_suffix(".csv.gz")
        df.to_csv(path, index=index, compression="gzip", float_format="%.6g")
    else:
        df.to_csv(path, index=index, float_format="%.6g")
    return path


def save_metrics(dirs, dev_df, test_df, cv: CVResult, final: FinalResult, stability, tests, quality) -> pd.DataFrame:
    d = dirs["metrics"]
    written = []
    split = pd.concat([pd.DataFrame({"row_id": dev_df.index, "split": "development"}),
                       pd.DataFrame({"row_id": test_df.index, "split": "test"})])
    written.append(write_csv(split, d / "split_assignment.csv"))
    written.append(write_csv(cv.fold_metrics, d / "cv_fold_metrics.csv"))
    written.append(write_csv(cv.mean_table().reset_index(), d / "cv_mean_metrics.csv"))
    written.append(write_csv(cv.summary().reset_index(), d / "cv_summary_mean_sd.csv"))
    written.append(write_csv(cv.sampling, d / "cv_sampling_runtime.csv"))
    written.append(write_csv(final.test_metrics, d / "test_metrics.csv"))
    written.append(write_csv(final.sampling.reset_index(), d / "final_sampling_counts.csv"))
    written.append(write_csv(stability.reset_index(), d / "attribution_stability.csv"))
    written.append(write_csv(tests, d / "paired_tests_vs_smote.csv"))
    written.append(write_csv(quality.reset_index(), d / "synthetic_quality.csv"))
    imp = pd.concat({model: final.importance_table(model) for model in C.MODEL_ORDER}, names=["model", "method"])
    written.append(write_csv(imp.reset_index(), d / "test_downstream_shap_importance.csv"))
    written.append(write_csv(cv.importances, d / "cv_downstream_shap_importance.csv"))
    written.append(write_csv(C.settings_table().reset_index(), d / "settings.csv"))
    return pd.DataFrame({"file": [p.relative_to(C.REPO_ROOT).as_posix() for p in written]})


def save_sampling_artifacts(dirs, final: FinalResult) -> pd.DataFrame:
    written = []
    for m in C.OVERSAMPLERS:
        rec = I.generation_records(final, m)
        if len(rec):
            written.append((m, "synthetic samples", len(rec),
                            write_csv(rec, dirs["synthetic_samples"] / f"{slugify(m)}_synthetic.csv")))
        nb = I.neighbour_table(final, m)
        if len(nb):
            written.append((m, "neighbours", len(nb),
                            write_csv(nb, dirs["neighbour_analysis"] / f"{slugify(m)}_neighbours.csv")))
    for m in C.PROPOSED_METHODS:
        r = I.shap_rationale(final, m, n=min(len(final.samplers[m].generation_log_), 2000))
        written.append((m, "SHAP rationale (first 2,000 rows)", len(r),
                        write_csv(r.reset_index(), dirs["neighbour_analysis"] / f"{slugify(m)}_shap_rationale.csv")))
    w = I.shap_weight_table(final)
    written.append(("SHAP methods", "SHAP feature weights", len(w),
                    write_csv(w.reset_index(names="feature"), dirs["neighbour_analysis"] / "shap_feature_weights.csv")))
    sv = I.minority_shap_values(final)
    written.append(("SHAP methods", "minority SHAP values", len(sv),
                    write_csv(sv.reset_index(), dirs["neighbour_analysis"] / "minority_seed_shap_values.csv")))
    ov = I.neighbour_overlap_with_smote(final)
    written.append(("all", "neighbour overlap with SMOTE", len(ov),
                    write_csv(ov.reset_index(), dirs["neighbour_analysis"] / "neighbour_overlap_with_smote.csv")))
    return pd.DataFrame(written, columns=["method", "content", "rows", "file"]).assign(
        file=lambda t: t["file"].map(lambda p: p.relative_to(C.REPO_ROOT).as_posix()))
