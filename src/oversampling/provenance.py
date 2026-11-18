"""
Provenance-tracking SMOTE engine shared by every interpolating oversampler in the benchmark.

Every SMOTE-family method in this project generates a synthetic minority row the same way:

    1. pick an anchor x_a from a pool of eligible minority seeds,
    2. pick one of its k nearest minority neighbours x_nn (neighbours found in a method-specific
       *search space*),
    3. draw lambda ~ U(0, 1) and return x_new = x_a + lambda * (x_nn - x_a) in the ORIGINAL
       (unweighted, all-feature) space.

The methods differ only in step 1 (which seeds are eligible) and step 2 (the search space):

    | Method               | Seed pool               | Neighbour search space                        |
    |----------------------|-------------------------|-----------------------------------------------|
    | SMOTE                | all minority rows       | all features, Euclidean                       |
    | Borderline-SMOTE     | DANGER minority rows    | all features, Euclidean                       |
    | SHAP-Filtered SMOTE  | all minority rows       | top-MASV features only (subspace)             |
    | SHAP-Weighted SMOTE  | all minority rows       | all features scaled by sqrt(w_j) (weighted)   |

`interpolate_with_provenance` implements step 3 with exactly the random-draw order the original
SHAP samplers used (anchor, neighbour slot, lambda; one sample at a time), so routing the SHAP
samplers through it leaves their output bit-for-bit unchanged.

Every sampler that uses the engine exposes the same fitted attributes, so the notebooks can inspect
any method with the same code:

    minority_positions_   positions of the minority rows inside the X passed to fit_resample
    seed_pool_            local minority indices eligible as anchors
    search_feature_mask_  bool (p,), features that take part in the neighbour search
    search_scale_         float (p,), per-feature multiplier applied before the Euclidean search
    neighbor_indices_     (n_min, k) local minority indices of each row's k nearest neighbours
    neighbor_distances_   (n_min, k) distances in the search space
    generation_log_       DataFrame, one row per synthetic sample (anchor, neighbour, rank, lambda,
                          search-space distance)
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors


def minority_knn(
    X_minority: np.ndarray,
    search_scale: np.ndarray,
    search_feature_mask: np.ndarray,
    k_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    k nearest minority neighbours of every minority row, in a scaled / masked search space.

    Returns (neighbor_indices, neighbor_distances, effective_k); column 0 (the row itself) is dropped.
    """
    n_minority = len(X_minority)
    effective_k = min(k_neighbors, n_minority - 1)
    Z = (X_minority * search_scale)[:, search_feature_mask]
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean", n_jobs=-1).fit(Z)
    dist, idx = nn.kneighbors(Z)
    return idx[:, 1:], dist[:, 1:], effective_k


def interpolate_with_provenance(
    X_minority: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    n_to_generate: int,
    rng: np.random.Generator,
    seed_pool: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    SMOTE interpolation step that records where every synthetic row came from.

    Draw order per sample (identical to the original SHAP samplers): anchor position, neighbour
    slot, lambda. When `seed_pool` is None every minority row is eligible and the anchor draw is
    `rng.integers(0, n_minority)`, exactly as before.
    """
    n_minority, n_features = X_minority.shape
    effective_k = neighbor_indices.shape[1]
    pool = np.arange(n_minority) if seed_pool is None else np.asarray(seed_pool)

    synthetic = np.zeros((n_to_generate, n_features), dtype=np.float64)
    anchors = np.empty(n_to_generate, dtype=np.int64)
    slots = np.empty(n_to_generate, dtype=np.int64)
    lambdas = np.empty(n_to_generate, dtype=np.float64)

    for i in range(n_to_generate):
        base_idx = pool[rng.integers(0, len(pool))]
        nn_choice = rng.integers(0, effective_k)
        lambd = rng.uniform(0.0, 1.0)
        neighbor_idx = neighbor_indices[base_idx, nn_choice]

        synthetic[i] = X_minority[base_idx] + lambd * (X_minority[neighbor_idx] - X_minority[base_idx])
        anchors[i], slots[i], lambdas[i] = base_idx, nn_choice, lambd

    if n_to_generate == 0:
        return synthetic, empty_generation_log()
    log = pd.DataFrame({
        "anchor_local": anchors,
        "neighbor_local": neighbor_indices[anchors, slots],
        "neighbor_rank": slots + 1,
        "lambda": lambdas,
        "search_distance": neighbor_distances[anchors, slots],
    })
    return synthetic, log


LOG_COLUMNS = ["anchor_local", "neighbor_local", "neighbor_rank", "lambda", "search_distance"]


def empty_generation_log() -> pd.DataFrame:
    """Generation log for a sampler that created no synthetic rows."""
    log = pd.DataFrame({c: np.empty(0) for c in LOG_COLUMNS})
    return log.astype({"anchor_local": np.int64})


def n_to_generate_for(sampling_strategy, n_minority: int, n_majority: int) -> int:
    """Same rule as SHAPBaseOversampler: 'auto' balances 1:1, a float sets the minority:majority ratio."""
    if isinstance(sampling_strategy, str):
        if sampling_strategy not in ("auto", "minority"):
            raise ValueError(f"Unknown sampling strategy string: {sampling_strategy}")
        target = n_majority
    else:
        target = int(n_majority * float(sampling_strategy))
    return max(0, target - n_minority)


class _ProvenanceSamplerMixin:
    """Shared helpers for samplers that expose the provenance attributes listed in the module docstring."""

    method_name: str = "sampler"

    def _store_graph(self, minority_positions, seed_pool, mask, scale, nbr_idx, nbr_dist):
        self.minority_positions_ = minority_positions
        self.seed_pool_ = seed_pool
        self.search_feature_mask_ = mask
        self.search_scale_ = scale
        self.neighbor_indices_ = nbr_idx
        self.neighbor_distances_ = nbr_dist

    def _finish(self, X_arr, y_arr, synthetic, log, minority_class):
        log = log.copy()
        log.insert(0, "synthetic_index", np.arange(len(log)))
        log["anchor_position"] = self.minority_positions_[log["anchor_local"].to_numpy(dtype=np.int64)]
        has_neighbour = log["neighbor_local"].notna().to_numpy()
        neighbour_position = np.full(len(log), np.nan)
        neighbour_position[has_neighbour] = self.minority_positions_[
            log.loc[has_neighbour, "neighbor_local"].to_numpy(dtype=np.int64)]
        log["neighbor_position"] = neighbour_position
        self.generation_log_ = log
        self.n_synthetic_ = len(log)
        X_res = np.vstack([X_arr, synthetic]) if len(synthetic) else X_arr.copy()
        y_res = np.concatenate([y_arr, np.full(len(synthetic), minority_class, dtype=y_arr.dtype)])
        return X_res, y_res

    @staticmethod
    def _split_classes(X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) != 2:
            raise ValueError("Provenance samplers support binary targets only.")
        minority_class = classes[np.argmin(counts)]
        return X_arr, y_arr, minority_class, int(counts.min()), int(counts.max())


class SMOTEWithProvenance(_ProvenanceSamplerMixin, BaseEstimator):
    """
    Conventional SMOTE (Chawla et al., 2002): uniform anchor choice, Euclidean k-NN on all features.

    Algorithmically identical to `imblearn.over_sampling.SMOTE` (the same k-NN graph, uniform anchors
    with replacement, lambda ~ U(0,1)); it is re-implemented only so every synthetic row's parents
    can be recorded. `tests/test_provenance_samplers.py` checks the neighbour graph against imblearn.
    """

    method_name = "SMOTE"

    def __init__(self, sampling_strategy="auto", k_neighbors: int = 5, random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X_arr, y_arr, minority_class, n_min, n_maj = self._split_classes(X, y)
        rng = np.random.default_rng(self.random_state)
        positions = np.flatnonzero(y_arr == minority_class)
        X_min = X_arr[positions]
        p = X_arr.shape[1]
        mask, scale = np.ones(p, dtype=bool), np.ones(p)
        n_new = n_to_generate_for(self.sampling_strategy, n_min, n_maj)
        if n_new == 0 or n_min < 2:
            self._store_graph(positions, np.arange(n_min), mask, scale, np.empty((n_min, 0), int), np.empty((n_min, 0)))
            return self._finish(X_arr, y_arr, np.empty((0, p)), empty_generation_log(), minority_class)
        nbr_idx, nbr_dist, _ = minority_knn(X_min, scale, mask, self.k_neighbors)
        self._store_graph(positions, np.arange(n_min), mask, scale, nbr_idx, nbr_dist)
        synthetic, log = interpolate_with_provenance(X_min, nbr_idx, nbr_dist, n_new, rng)
        return self._finish(X_arr, y_arr, synthetic, log, minority_class)


class BorderlineSMOTEWithProvenance(_ProvenanceSamplerMixin, BaseEstimator):
    """
    Borderline-SMOTE-1 (Han, Wang & Mao, 2005), following imbalanced-learn's definition.

    1. For each minority row, find its m nearest neighbours in the WHOLE training set.
    2. m' = number of majority rows among them. The row is DANGER if m/2 <= m' < m
       (NOISE if m' = m, SAFE if m' < m/2).
    3. Only DANGER rows are used as anchors; their k neighbours come from all minority rows.
    If no row is in DANGER, nothing is generated (imbalanced-learn behaves the same way).
    """

    method_name = "Borderline-SMOTE"

    def __init__(self, sampling_strategy="auto", k_neighbors: int = 5, m_neighbors: int = 10,
                 random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X_arr, y_arr, minority_class, n_min, n_maj = self._split_classes(X, y)
        rng = np.random.default_rng(self.random_state)
        positions = np.flatnonzero(y_arr == minority_class)
        X_min = X_arr[positions]
        p = X_arr.shape[1]
        mask, scale = np.ones(p, dtype=bool), np.ones(p)

        # Step 1-2: DANGER set from the m-NN in the full training set (self excluded)
        m = min(self.m_neighbors, len(y_arr) - 1)
        nn_m = NearestNeighbors(n_neighbors=m + 1, n_jobs=-1).fit(X_arr)
        m_idx = nn_m.kneighbors(X_min, return_distance=False)[:, 1:]
        n_majority_in_m = (y_arr[m_idx] != minority_class).sum(axis=1)
        danger = (n_majority_in_m >= m / 2) & (n_majority_in_m < m)
        self.n_majority_in_m_neighbours_ = n_majority_in_m
        self.danger_mask_ = danger
        self.noise_mask_ = n_majority_in_m == m
        self.safe_mask_ = ~danger & ~self.noise_mask_

        n_new = n_to_generate_for(self.sampling_strategy, n_min, n_maj)
        seed_pool = np.flatnonzero(danger)
        if n_min < 2:
            n_new = 0
        nbr_idx, nbr_dist, _ = minority_knn(X_min, scale, mask, self.k_neighbors) if n_min >= 2 else (
            np.empty((n_min, 0), int), np.empty((n_min, 0)), 0)
        self._store_graph(positions, seed_pool, mask, scale, nbr_idx, nbr_dist)
        if n_new == 0 or len(seed_pool) == 0:
            return self._finish(X_arr, y_arr, np.empty((0, p)), empty_generation_log(), minority_class)
        synthetic, log = interpolate_with_provenance(X_min, nbr_idx, nbr_dist, n_new, rng, seed_pool=seed_pool)
        return self._finish(X_arr, y_arr, synthetic, log, minority_class)


class RandomOversamplerWithProvenance(_ProvenanceSamplerMixin, BaseEstimator):
    """
    Random Oversampling (ROS): duplicates minority rows chosen uniformly with replacement.

    Delegates to `imblearn.over_sampling.RandomOverSampler` (the implementation the repo already used)
    and reads which rows were copied from its `sample_indices_`. A copy has no neighbour and no lambda.
    """

    method_name = "Random Oversampling"

    def __init__(self, sampling_strategy="auto", random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        from imblearn.over_sampling import RandomOverSampler

        X_arr, y_arr, minority_class, n_min, n_maj = self._split_classes(X, y)
        positions = np.flatnonzero(y_arr == minority_class)
        p = X_arr.shape[1]
        ros = RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        X_res, _ = ros.fit_resample(X_arr, y_arr)
        copied_positions = ros.sample_indices_[len(y_arr):]
        local = np.searchsorted(positions, copied_positions)
        self._store_graph(positions, np.arange(n_min), np.zeros(p, dtype=bool), np.ones(p),
                          np.empty((n_min, 0), int), np.empty((n_min, 0)))
        log = pd.DataFrame({
            "anchor_local": local.astype(np.int64),
            "neighbor_local": np.full(len(local), np.nan),
            "neighbor_rank": np.full(len(local), np.nan),
            "lambda": np.full(len(local), np.nan),
            "search_distance": np.full(len(local), np.nan),
        })
        return self._finish(X_arr, y_arr, X_res[len(y_arr):], log, minority_class)


class NoSampling(_ProvenanceSamplerMixin, BaseEstimator):
    """Baseline: the training fold is used as-is."""

    method_name = "No Sampling"

    def fit_resample(self, X, y):
        X_arr, y_arr, minority_class, n_min, _ = self._split_classes(X, y)
        p = X_arr.shape[1]
        self._store_graph(np.flatnonzero(y_arr == minority_class), np.empty(0, int), np.zeros(p, dtype=bool),
                          np.ones(p), np.empty((n_min, 0), int), np.empty((n_min, 0)))
        return self._finish(X_arr, y_arr, np.empty((0, p)), empty_generation_log(), minority_class)
