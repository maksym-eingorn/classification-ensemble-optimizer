# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# ensemble/greedy_selection.py

import numpy as np

from tuning.validation import (
    validate_binary_target, validate_positive_integer, validate_random_seed
)


ModelMetadata = dict[str, object]
GreedyMetadata = dict[str, object]


def _validate_greedy_inputs(
    P,
    y,
    column_metadata: list[ModelMetadata],
    pool_size: int | None,
    ensemble_size: int,
    seed: int,
    allow_repeats: bool
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate greedy selection inputs and return the effective pool size."""
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if P.ndim != 2:
        raise ValueError(
            f"P must be 2D (n_samples_dev, n_models), got {P.ndim}D."
        )

    if y.ndim != 1:
        raise ValueError(f"y must be 1D (n_samples_dev,), got {y.ndim}D.")

    if P.shape[0] != y.shape[0]:
        raise ValueError(
            f"Numbers of samples in P and y must match, "
            f"got {P.shape[0]} != {y.shape[0]}."
        )

    if P.shape[1] < 1:
        raise ValueError("Greedy ensemble selection requires at least 1 model.")

    if not np.all(np.isfinite(P)):
        raise ValueError("P must contain only finite values.")

    if np.any(P < 0.0) or np.any(P > 1.0):
        raise ValueError(
            "P must contain only positive-class probabilities between 0 and 1."
        )

    validate_binary_target(y, "y")

    if len(column_metadata) != P.shape[1]:
        raise ValueError(
            "column_metadata must contain one entry per column of P, "
            f"got {len(column_metadata)} != {P.shape[1]}."
        )

    if pool_size is not None:
        validate_positive_integer(pool_size, "pool_size")
        effective_pool_size = min(pool_size, P.shape[1])
    else:
        effective_pool_size = P.shape[1]

    validate_positive_integer(ensemble_size, "ensemble_size")
    validate_random_seed(seed, "seed")

    if not isinstance(allow_repeats, bool):
        raise ValueError(
            f"allow_repeats must be a boolean, got {allow_repeats}."
        )

    return P, y, effective_pool_size


def precompute_dot_products(
    P: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute dot-product terms used for fast out-of-fold ensemble scoring.

    Parameters
    ----------
    P
        Unified out-of-fold positive-class probability matrix.
        Shape: (n_samples_dev, n_models).
    y
        Binary development target vector.
        Shape: (n_samples_dev,).

    Returns
    -------
    G
        Gram matrix of out-of-fold probability columns, equal to P.T @ P.
        Shape: (n_models, n_models).
    s
        Target projection vector, equal to P.T @ y.
        Shape: (n_models,).
    y2
        Squared target norm, equal to y @ y.
    """
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    G = P.T @ P
    s = P.T @ y
    y2 = float(y @ y)

    return G, s, y2


def _greedy_ensemble_selection_from_dots(
    G: np.ndarray,
    s: np.ndarray,
    y2: float,
    n_samples_dev: int,
    pool_size: int,
    ensemble_size: int,
    seed: int,
    allow_repeats: bool
) -> tuple[list[int], list[float], list[int], int, float]:
    """
    Run Caruana-style greedy ensemble selection from dot products.

    The algorithm first restricts the candidate set to the best individual
    models by OOF Brier score. It then builds an ensemble greedily by repeatedly
    adding the model that gives the lowest OOF Brier score after averaging the
    selected probability vectors. If repeats are allowed, the same model may be
    selected multiple times; final convex weights are selection counts divided
    by the number of completed selections.

    Parameters
    ----------
    G
        Gram matrix of out-of-fold probability columns.
        Shape: (n_models, n_models).
    s
        Target projection vector.
        Shape: (n_models,).
    y2
        Squared target norm.
    n_samples_dev
        Number of samples in the development set.
    pool_size
        Number of the best individual models considered by greedy selection.
    ensemble_size
        Maximum number of greedy selection steps.
    seed
        Random seed used for tie-breaking.
    allow_repeats
        Whether the same model may be selected more than once.

    Returns
    -------
    selected_columns
        Original unified matrix column indices of the selected unique models.
    weights
        Convex weights corresponding to selected_columns.
    selection_counts
        Number of times each selected model was chosen.
    actual_ensemble_size
        Number of greedy selection steps actually completed.
    oof_brier_score
        Full-vector OOF Brier score of the final greedy ensemble.
    """
    rng = np.random.RandomState(seed)

    diag_g = np.diag(G).astype(np.float64)
    s64 = s.astype(np.float64)

    individual_sse = y2 - 2.0 * s64 + diag_g
    individual_brier_scores = individual_sse / n_samples_dev

    pool_idx = np.argsort(individual_brier_scores)[:pool_size]

    G_pool = G[np.ix_(pool_idx, pool_idx)].astype(np.float64)
    s_pool = s64[pool_idx].astype(np.float64)

    m = len(pool_idx)

    counts = np.zeros(m, dtype=np.int64)
    v = np.zeros(m, dtype=np.float64)

    Sy = 0.0
    SS = 0.0
    k = 0
    tie_tolerance = 1e-12

    for _ in range(ensemble_size):
        best_sse = np.inf
        best_candidates: list[int] = []

        for j in range(m):
            if not allow_repeats and counts[j] > 0:
                continue

            Sy_new = Sy + s_pool[j]
            SS_new = SS + 2.0 * v[j] + G_pool[j, j]
            k_new = k + 1

            sse_new = y2 - 2.0 * (Sy_new / k_new) + SS_new / (k_new * k_new)

            if sse_new < best_sse - tie_tolerance:
                best_sse = sse_new
                best_candidates = [j]
            elif abs(sse_new - best_sse) <= tie_tolerance:
                best_candidates.append(j)

        if not best_candidates:
            break

        best_j = best_candidates[rng.randint(len(best_candidates))]

        previous_cross_sum = v[best_j]

        counts[best_j] += 1
        v += G_pool[best_j, :]
        Sy += s_pool[best_j]
        SS += 2.0 * previous_cross_sum + G_pool[best_j, best_j]
        k += 1

    if k == 0:
        raise RuntimeError(
            "Greedy ensemble selection produced an empty ensemble."
        )

    chosen_pool = np.flatnonzero(counts > 0)

    selected_columns = pool_idx[chosen_pool].astype(int).tolist()
    selected_counts = counts[chosen_pool].astype(int).tolist()
    weights = (counts[chosen_pool].astype(np.float64) / k).tolist()

    weights_array = np.asarray(weights, dtype=np.float64)
    s_sub = s64[selected_columns]
    G_sub = G[np.ix_(selected_columns, selected_columns)]

    sse_final = float(
        y2 - 2.0 * (s_sub @ weights_array)
        + weights_array @ (G_sub @ weights_array)
    )
    oof_brier_score = max(0.0, sse_final / n_samples_dev)

    return selected_columns, weights, selected_counts, k, oof_brier_score


def find_greedy_ensemble(
    P,
    y,
    column_metadata: list[ModelMetadata],
    pool_size: int | None = 100,
    ensemble_size: int = 50,
    seed: int = 42,
    allow_repeats: bool = True
) -> tuple[GreedyMetadata, np.ndarray]:
    """
    Build one Caruana-style greedy ensemble from saved out-of-fold
    probabilities.

    This function operates on a unified out-of-fold probability matrix. It
    selects a greedy convex ensemble, attaches the corresponding base model
    metadata, and returns both the ensemble metadata and its development set
    out-of-fold positive-class probability vector.

    Parameters
    ----------
    P
        Unified out-of-fold positive-class probability matrix.
        Shape: (n_samples_dev, n_models).
    y
        Binary development target vector.
        Shape: (n_samples_dev,).
    column_metadata
        Metadata list describing the saved model/trial represented by each
        column of P.
    pool_size
        Number of the best individual models considered by greedy selection.
        If None, the full model zoo is used.
    ensemble_size
        Maximum number of greedy selection steps.
    seed
        Random seed used for tie-breaking.
    allow_repeats
        Whether the same model may be selected more than once.

    Returns
    -------
    metadata
        Greedy ensemble metadata, including selected columns, convex weights,
        selection counts, selected model metadata, and OOF Brier score.
    oof_probabilities
        Development set out-of-fold positive-class probability vector for the
        final greedy ensemble.
        Shape: (n_samples_dev,).
    """
    P, y, effective_pool_size = _validate_greedy_inputs(
        P, y, column_metadata, pool_size, ensemble_size, seed, allow_repeats
    )

    G, s, y2 = precompute_dot_products(P, y)

    (
        selected_columns,
        weights,
        selection_counts,
        actual_ensemble_size,
        oof_brier_score
    ) = _greedy_ensemble_selection_from_dots(
        G,
        s,
        y2,
        n_samples_dev=P.shape[0],
        pool_size=effective_pool_size,
        ensemble_size=ensemble_size,
        seed=seed,
        allow_repeats=allow_repeats
    )

    selected_matrix = P[:, selected_columns]
    weights_array = np.asarray(weights, dtype=np.float64)
    oof_probabilities = selected_matrix @ weights_array

    metadata: GreedyMetadata = {
        "method": "caruana_style_greedy",
        "pool_size": pool_size,
        "effective_pool_size": effective_pool_size,
        "ensemble_size": ensemble_size,
        "actual_ensemble_size": actual_ensemble_size,
        "allow_repeats": allow_repeats,
        "selected_columns": selected_columns,
        "weights": weights,
        "selection_counts": selection_counts,
        "n_unique_models": len(selected_columns),
        "oof_brier_score": float(oof_brier_score),
        "models": [
            dict(column_metadata[column_index])
            for column_index in selected_columns
        ]
    }

    return metadata, oof_probabilities.astype(np.float64)
