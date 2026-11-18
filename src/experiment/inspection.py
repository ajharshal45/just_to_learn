"""
Inspection tables: what every oversampler selected, which neighbours it used, and what it generated.

All tables are built from the provenance attributes of the fitted samplers (see
`src.oversampling.provenance`), on the final development-set fit. Row ids are the original row
numbers of the source file, so every synthetic record can be traced back to two real records.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.evaluation.fidelity import compute_synthetic_fidelity
from src.experiment import config as C
from src.experiment.runner import FinalResult

INTERPOLATING = ["SMOTE", "Borderline-SMOTE", "SHAP-Filtered SMOTE", "SHAP-Weighted SMOTE"]


def _search_space(sampler, X_min: np.ndarray) -> np.ndarray:
    return (X_min * sampler.search_scale_)[:, sampler.search_feature_mask_]


def search_space_description(final: FinalResult) -> pd.DataFrame:
    """How each method decides who is a neighbour."""
    names = np.array(final.feature_names)
    rows = []
    for m in C.METHOD_ORDER:
        s = final.samplers[m]
        mask = s.search_feature_mask_
        if m == "No Sampling":
            space, pool = "-", "-"
        elif m == "Random Oversampling":
            space, pool = "none (exact copies)", "all minority rows"
        elif m == "SHAP-Weighted SMOTE":
            space, pool = "all features, column j scaled by sqrt(w_j)", "all minority rows"
        elif m == "SHAP-Filtered SMOTE":
            space, pool = f"{mask.sum()} SHAP-selected features (Euclidean)", "all minority rows"
        elif m == "Borderline-SMOTE":
            space, pool = "all features (Euclidean)", f"DANGER rows only ({len(s.seed_pool_)} of {len(s.minority_positions_)})"
        else:
            space, pool = "all features (Euclidean)", "all minority rows"
        rows.append({"Method": m, "Anchor pool": pool, "Neighbour search space": space,
                     "Features in search": int(mask.sum()) if m not in ("No Sampling", "Random Oversampling") else 0,
                     "Search features (first 8)": ", ".join(names[mask][:8]) + (" ..." if mask.sum() > 8 else "")
                     if m not in ("No Sampling", "Random Oversampling") else "-",
                     "Synthetic rows": s.n_synthetic_})
    return pd.DataFrame(rows).set_index("Method")


def euclidean_rank_of(X_min: np.ndarray, anchors: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Rank (1 = nearest) the neighbour would have under plain Euclidean distance on all features."""
    anchors, neighbours = np.asarray(anchors, dtype=int), np.asarray(neighbours, dtype=int)
    ranks = np.empty(len(anchors), dtype=int)
    uniq, inv = np.unique(anchors, return_inverse=True)
    chunk = max(1, int(2e7 // max(1, X_min.size)))  # bound memory of the (chunk, n_min, p) block
    for start in range(0, len(uniq), chunk):
        u = uniq[start:start + chunk]
        d2 = ((X_min[None, :, :] - X_min[u][:, None, :]) ** 2).sum(axis=2)  # squared distances, same order
        d2[np.arange(len(u)), u] = np.inf                                   # exclude the anchor itself
        sel = np.flatnonzero((inv >= start) & (inv < start + len(u)))
        rows = inv[sel] - start
        ranks[sel] = (d2[rows] < d2[rows, neighbours[sel]][:, None]).sum(axis=1) + 1
    return ranks


def distance_contributions_many(sampler, A: np.ndarray, B: np.ndarray, names: List[str], top: int = 3) -> List[str]:
    """For each row pair (A[i], B[i]): the features that make up most of the squared search distance."""
    diff2 = ((A - B) * sampler.search_scale_) ** 2 * sampler.search_feature_mask_
    total = diff2.sum(axis=1)
    order = np.argsort(-diff2, axis=1, kind="stable")[:, :top]
    out = []
    for i in range(len(A)):
        if total[i] <= 0:
            out.append("-")
        else:
            out.append(", ".join(f"{names[j]} ({diff2[i, j] / total[i]:.0%})" for j in order[i]))
    return out


def distance_contributions(sampler, x_a: np.ndarray, x_b: np.ndarray, names: List[str], top: int = 3) -> str:
    """Features that make up most of the squared distance in the method's search space."""
    return distance_contributions_many(sampler, x_a[None, :], x_b[None, :], names, top)[0]


def neighbour_table(final: FinalResult, method: str, max_seeds: Optional[int] = None) -> pd.DataFrame:
    """
    One row per (minority seed, neighbour rank): ids, distance in the method's search space,
    plain Euclidean distance, the rank the neighbour would have under plain SMOTE, and the
    features that dominate the search distance.
    """
    s = final.samplers[method]
    if s.neighbor_indices_.shape[1] == 0:
        return pd.DataFrame()
    names = final.feature_names
    X_min = final.X_dev[s.minority_positions_]
    ids = final.dev_row_ids[s.minority_positions_]
    n_seeds = len(X_min) if max_seeds is None else min(max_seeds, len(X_min))
    k = s.neighbor_indices_.shape[1]
    a = np.repeat(np.arange(n_seeds), k)
    b = s.neighbor_indices_[:n_seeds].ravel()
    eucl = np.sqrt(((X_min[a] - X_min[b]) ** 2).sum(axis=1))
    out = pd.DataFrame({
        "seed_row_id": ids[a], "neighbour_rank": np.tile(np.arange(1, k + 1), n_seeds),
        "neighbour_row_id": ids[b], "search_distance": s.neighbor_distances_[:n_seeds].ravel(),
        "euclidean_distance": eucl, "rank_under_plain_smote": euclidean_rank_of(X_min, a, b),
        "top_features_in_search_distance": distance_contributions_many(s, X_min[a], X_min[b], names),
    })
    if method == "Borderline-SMOTE":
        status = np.where(s.danger_mask_, "DANGER", np.where(s.noise_mask_, "NOISE", "SAFE"))
        out.insert(1, "seed_status", status[a])
        out.insert(2, f"majority_in_{C.M_NEIGHBORS}_nn", s.n_majority_in_m_neighbours_[a])
    out.insert(0, "method", method)
    return out


def generation_records(final: FinalResult, method: str) -> pd.DataFrame:
    """Every synthetic row with its parents, lambda and values in original units."""
    s = final.samplers[method]
    log = s.generation_log_
    if len(log) == 0:
        return pd.DataFrame()
    syn = final.preprocessor.to_original_units(final.synthetic(method))
    nb_pos = log["neighbor_position"].to_numpy()
    has_nb = ~np.isnan(nb_pos)
    nb_ids = np.full(len(log), np.nan)
    nb_ids[has_nb] = final.dev_row_ids[nb_pos[has_nb].astype(int)]
    meta = pd.DataFrame({
        "synthetic_id": [f"{C.METHOD_ORDER.index(method)}-{i}" for i in log["synthetic_index"]],
        "method": method,
        "parent_1_row_id": final.dev_row_ids[log["anchor_position"].to_numpy(dtype=int)],
        "parent_2_row_id": nb_ids,
        "neighbour_rank": log["neighbor_rank"].to_numpy(),
        "lambda": log["lambda"].to_numpy(),
        "search_distance": log["search_distance"].to_numpy(),
        "target": 1,
    })
    return pd.concat([meta, syn], axis=1)


def example_walkthrough(final: FinalResult, method: str, i: int = 0, max_features: int = 12) -> pd.DataFrame:
    """Feature-by-feature construction of synthetic row `i`: parent 1, parent 2, gap, lambda x gap, result."""
    s = final.samplers[method]
    log = s.generation_log_.iloc[i]
    pre = final.preprocessor
    a = final.X_dev[int(log["anchor_position"])]
    b = final.X_dev[int(log["neighbor_position"])]
    lam = float(log["lambda"])
    syn = a + lam * (b - a)
    orig = pre.to_original_units(np.vstack([a, b, syn]))
    numeric = pre.spec.numeric_features
    cols = numeric + [c for c in orig.columns if c in pre.onehot_groups_]
    order = cols
    if hasattr(s, "feature_weights_"):  # put the features SHAP cares about first
        w = pd.Series(s.feature_weights_, index=final.feature_names)
        num_w = w.reindex(numeric).fillna(0)
        cat_w = pd.Series({c: w[[f for f in final.feature_names if f.startswith(c + "=")]].sum() for c in pre.onehot_groups_})
        order = list(pd.concat([num_w, cat_w]).sort_values(ascending=False).index)
    rows = []
    for c in order[:max_features]:
        p1, p2, new = orig.loc[0, c], orig.loc[1, c], orig.loc[2, c]
        num = c in numeric
        rows.append({"feature": c, "parent_1": p1, "parent_2": p2,
                     "gap (p2 - p1)": p2 - p1 if num else None,
                     "lambda x gap": lam * (p2 - p1) if num else None,
                     "synthetic": new,
                     "SHAP weight w_j": float(w.get(c, cat_w.get(c, np.nan))) if hasattr(s, "feature_weights_") else None})
    df = pd.DataFrame(rows).set_index("feature")
    df.attrs["header"] = (f"{method}: synthetic #{i} = parent_1 (row {final.dev_row_ids[int(log['anchor_position'])]}) "
                          f"+ {lam:.4f} x (parent_2 (row {final.dev_row_ids[int(log['neighbor_position'])]}) - parent_1); "
                          f"parent_2 is neighbour rank {int(log['neighbor_rank'])} in the search space")
    return df.dropna(axis=1, how="all")


# --------------------------------------------------------------------------------------------
# SHAP guidance
# --------------------------------------------------------------------------------------------
def shap_weight_table(final: FinalResult) -> pd.DataFrame:
    """MASV and normalised weights from the SHAP samplers' preliminary model (development set)."""
    w = final.samplers["SHAP-Weighted SMOTE"]
    f = final.samplers["SHAP-Filtered SMOTE"]
    t = pd.DataFrame({"MASV": w.masv_, "weight_w": w.feature_weights_,
                      "sqrt_w_stretch_vs_uniform": np.sqrt(w.feature_weights_) / np.sqrt(1 / len(w.masv_)),
                      "kept_by_SHAP_Filtered": f.search_feature_mask_}, index=final.feature_names)
    t = t.sort_values("MASV", ascending=False)
    t.insert(2, "cumulative_weight", t["weight_w"].cumsum())
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    return t


def minority_shap_values(final: FinalResult) -> pd.DataFrame:
    """Preliminary-model SHAP values of every real minority row in the development set."""
    s = final.samplers["SHAP-Weighted SMOTE"]
    sv = s.shap_values_[s.minority_positions_]
    return pd.DataFrame(sv, columns=final.feature_names,
                        index=pd.Index(final.dev_row_ids[s.minority_positions_], name="row_id"))


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def shap_rationale(final: FinalResult, method: str, n: int = 10, top_features: int = 3) -> pd.DataFrame:
    """
    For the first `n` synthetic rows of a SHAP method: which neighbour was used, its rank under the
    SHAP metric and under plain Euclidean SMOTE, how similar the parents' SHAP explanations are, and
    which features made up the SHAP-space distance.
    """
    s = final.samplers[method]
    log = s.generation_log_.head(n)
    names = final.feature_names
    X_min = final.X_dev[s.minority_positions_]
    sv = s.shap_values_[s.minority_positions_]
    a, b = log["anchor_local"].to_numpy(), log["neighbor_local"].to_numpy().astype(int)
    top_idx = np.argsort(s.feature_weights_)[::-1][:top_features]
    out = pd.DataFrame({
        "synthetic_index": log["synthetic_index"].to_numpy(),
        "parent_1_row_id": final.dev_row_ids[s.minority_positions_[a]],
        "parent_2_row_id": final.dev_row_ids[s.minority_positions_[b]],
        "lambda": log["lambda"].round(4).to_numpy(),
        "rank_in_SHAP_space": log["neighbor_rank"].to_numpy(),
        "rank_under_plain_SMOTE": euclidean_rank_of(X_min, a, b),
        "parents_SHAP_cosine": _cosine(sv[a], sv[b]).round(3),
        "distance_made_of": distance_contributions_many(s, X_min[a], X_min[b], names),
    })
    for j in top_idx:
        out[f"SHAP p1/p2 [{names[j]}]"] = [f"{sv[i, j]:+.3f} / {sv[k, j]:+.3f}" for i, k in zip(a, b)]
    out["would_plain_SMOTE_consider_it"] = out["rank_under_plain_SMOTE"] <= C.K_NEIGHBORS
    return out.set_index("synthetic_index")


def parent_shap_detail(final: FinalResult, method: str, i: int = 0, max_features: int = 10) -> pd.DataFrame:
    """Feature-level view of one SHAP-guided synthetic row: weights, parents' SHAP values and values."""
    s = final.samplers[method]
    log = s.generation_log_.iloc[i]
    pa, pb = int(log["anchor_position"]), int(log["neighbor_position"])
    lam = float(log["lambda"])
    sv = s.shap_values_
    order = np.argsort(s.feature_weights_)[::-1][:max_features]
    a, b = final.X_dev[pa], final.X_dev[pb]
    df = pd.DataFrame({
        "SHAP weight w_j": s.feature_weights_[order],
        "in search space": s.search_feature_mask_[order],
        "share of search distance": (((a - b) * s.search_scale_) ** 2 * s.search_feature_mask_)[order]
        / max((((a - b) * s.search_scale_) ** 2 * s.search_feature_mask_).sum(), 1e-12),
        "SHAP parent 1": sv[pa, order], "SHAP parent 2": sv[pb, order],
        "value parent 1 (scaled)": a[order], "value parent 2 (scaled)": b[order],
        "synthetic (scaled)": (a + lam * (b - a))[order],
    }, index=pd.Index(np.array(final.feature_names)[order], name="feature"))
    return df


def neighbour_overlap_with_smote(final: FinalResult) -> pd.DataFrame:
    """How different each method's neighbour graph is from plain SMOTE's (mean Jaccard per seed)."""
    ref = final.samplers["SMOTE"].neighbor_indices_
    rows = []
    for m in INTERPOLATING:
        nb = final.samplers[m].neighbor_indices_
        jac = [len(set(x) & set(r)) / len(set(x) | set(r)) for x, r in zip(nb, ref)]
        rows.append({"method": m, "mean_jaccard_with_SMOTE_neighbours": float(np.mean(jac)),
                     "seeds_with_identical_neighbours": float(np.mean(np.array(jac) == 1.0))})
    return pd.DataFrame(rows).set_index("method")


# --------------------------------------------------------------------------------------------
# Synthetic-sample quality
# --------------------------------------------------------------------------------------------
def synthetic_quality(final: FinalResult, seed: int = C.SEED) -> pd.DataFrame:
    """
    Per method, in the model (preprocessed) space of the development set:
      * fidelity of synthetic vs real minority rows (Wasserstein, KS pass rate, correlation similarity)
      * majority share among the 5 nearest REAL development rows of each synthetic row
        (higher = synthetic rows placed in majority territory, i.e. likely label noise);
        the same statistic for the real minority rows is given as a reference
      * mean cosine similarity of the two parents' preliminary-model SHAP explanations
      * detectability: cross-validated ROC-AUC of a Random Forest trained to tell real minority rows
        from synthetic ones (0.5 = indistinguishable), on all model columns and on the numeric
        columns only. Not computed for Random Oversampling, whose rows are exact copies of real rows.
    """
    X, y = final.X_dev, final.y_dev
    nn = NearestNeighbors(n_neighbors=6, n_jobs=-1).fit(X)
    rng = np.random.default_rng(seed)
    real_min = X[y == 1]
    ref_idx = nn.kneighbors(real_min, return_distance=False)[:, 1:]  # skip the row itself
    ref_share = (y[ref_idx] == 0).mean()
    sv_all = final.samplers["SHAP-Weighted SMOTE"].shap_values_
    rows = [{"method": "Real minority (reference)", "synthetic_rows": 0,
             "majority_share_in_5nn": ref_share, "synthetic_with_majority_nn1_%": np.nan}]
    for m in C.OVERSAMPLERS:
        syn = final.synthetic(m)
        if len(syn) == 0:
            rows.append({"method": m, "synthetic_rows": 0})
            continue
        pick = rng.choice(len(syn), min(len(syn), C.QUALITY_MAX_SYNTHETIC), replace=False)
        idx = nn.kneighbors(syn[pick], n_neighbors=5, return_distance=False)
        log = final.samplers[m].generation_log_
        row = {"method": m, "synthetic_rows": len(syn),
               "majority_share_in_5nn": (y[idx] == 0).mean(),
               "synthetic_with_majority_nn1_%": 100 * (y[idx[:, 0]] == 0).mean(),
               **compute_synthetic_fidelity(real_min, syn)}
        if log["neighbor_position"].notna().all():
            pa = log["anchor_position"].to_numpy(dtype=int)
            pb = log["neighbor_position"].to_numpy(dtype=int)
            row["parents_SHAP_cosine_mean"] = np.nanmean(_cosine(sv_all[pa], sv_all[pb]))
            row["distinct_parent_pairs"] = len(set(zip(pa, pb)))
            n_num = final.preprocessor.n_numeric_
            if n_num < X.shape[1]:
                onehot = syn[:, n_num:]
                row["fractional_onehot_cols_per_row"] = ((onehot > 1e-9) & (onehot < 1 - 1e-9)).sum(axis=1).mean()
            row["detectability_auc_all"] = detectability_auc(X, y, syn, pa, pb, None, seed)
            row["detectability_auc_numeric"] = (detectability_auc(X, y, syn, pa, pb, n_num, seed)
                                                if n_num < X.shape[1] else row["detectability_auc_all"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def detectability_auc(X: np.ndarray, y: np.ndarray, synthetic: np.ndarray, parent_a: np.ndarray,
                      parent_b: np.ndarray, n_cols=None, seed: int = C.SEED, repeats: int = 3,
                      max_rows: int = 2000) -> float:
    """
    ROC-AUC of a Random Forest separating real from synthetic minority rows (0.5 = indistinguishable).

    The split is parent-aware: the real minority rows are cut in two halves; a synthetic row is used only on
    the side that holds BOTH of its parents (rows with one parent on each side are dropped). Without this,
    a held-out real row would be surrounded by its own synthetic children in training, which biases the
    AUC below 0.5. Train on one half, score the other, swap, and repeat with `repeats` random halvings.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    cols = slice(None) if n_cols is None else slice(0, n_cols)
    real_pos = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(repeats):
        side = dict(zip(real_pos, rng.integers(0, 2, len(real_pos))))
        sa = np.array([side[p] for p in parent_a])
        sb = np.array([side[p] for p in parent_b])
        parts = []
        for h in (0, 1):
            r = X[real_pos[[side[p] == h for p in real_pos]]][:, cols]
            s_idx = np.flatnonzero((sa == h) & (sb == h))
            n = min(len(r), len(s_idx), max_rows)
            if n < 5:
                return np.nan
            r = r[rng.choice(len(r), n, replace=False)]
            s = synthetic[rng.choice(s_idx, n, replace=False)][:, cols]
            parts.append((np.vstack([r, s]), np.r_[np.zeros(n), np.ones(n)]))
        for tr, te in [(0, 1), (1, 0)]:
            clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(*parts[tr])
            aucs.append(roc_auc_score(parts[te][1], clf.predict_proba(parts[te][0])[:, 1]))
    return float(np.mean(aucs))
