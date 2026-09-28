# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# ensemble/triplet_search.py

import numpy as np

from tuning.validation import validate_binary_target, validate_positive_integer


def _load_native_triplet_search_module():
    """Load the compiled C++ triplet search extension."""
    try:
        from ensemble import _triplet_search
    except ImportError as exc:
        raise ImportError(
            "The compiled C++ triplet search extension was not found. "
            "Build it first with: "
            "python setup_triplet_search.py build_ext --inplace"
        ) from exc

    return _triplet_search


def _validate_triplet_search_inputs(
    P, y, top_n: int, n_threads: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs for native convex triplet search."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

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

    if P.shape[1] < 3:
        raise ValueError(
            f"Triplet search requires at least 3 model columns, "
            f"got {P.shape[1]}."
        )

    if not np.all(np.isfinite(P)):
        raise ValueError("P must contain only finite values.")

    if np.any(P < 0.0) or np.any(P > 1.0):
        raise ValueError(
            "P must contain only positive-class probabilities between 0 and 1."
        )

    validate_binary_target(y, "y")
    validate_positive_integer(top_n, "top_n")
    validate_positive_integer(n_threads, "n_threads")

    return P, y


def find_top_triplet_ensembles(
    P, y, top_n: int = 100, n_threads: int = 1
) -> tuple[list[dict[str, object]], np.ndarray]:
    """
    Run exact exhaustive Brier-optimal convex triplet search using the
    compiled C++ extension.

    Parameters
    ----------
    P
        Unified out-of-fold positive-class probability matrix.
        Shape: (n_samples_dev, n_models).
    y
        Binary development target vector.
        Shape: (n_samples_dev,).
    top_n : int, default=100
        Number of top OOF-ranked convex triplet ensembles retained.
    n_threads : int, default=1
        Number of OpenMP threads used by the native search.

    Returns
    -------
    triplet_metadata : list[dict[str, object]]
        Metadata for the retained convex triplet ensembles, ranked by
        increasing OOF Brier score.
    triplet_oof_probabilities : np.ndarray
        Out-of-fold positive-class probability matrix for the retained
        triplet ensembles.
        Shape: (n_samples_dev, n_kept_triplets).
    """
    P, y = _validate_triplet_search_inputs(P, y, top_n, n_threads)

    native = _load_native_triplet_search_module()

    raw_metadata, triplet_oof_probabilities = native.find_top_triplets(
        P, y, top_n, n_threads
    )

    triplet_oof_probabilities = np.asarray(
        triplet_oof_probabilities, dtype=np.float64
    )

    if triplet_oof_probabilities.ndim != 2:
        raise RuntimeError(
            "Native triplet search must return a 2D probability matrix."
        )

    if triplet_oof_probabilities.shape[0] != P.shape[0]:
        raise RuntimeError(
            "Native triplet search returned an unexpected number of rows."
        )

    if triplet_oof_probabilities.shape[1] != len(raw_metadata):
        raise RuntimeError(
            "Native triplet metadata and probability columns must have "
            "matching lengths."
        )

    if not np.all(np.isfinite(triplet_oof_probabilities)):
        raise RuntimeError(
            "Native triplet search returned non-finite probabilities."
        )

    probability_tolerance = 1e-12

    if (
        np.any(triplet_oof_probabilities < -probability_tolerance)
        or np.any(triplet_oof_probabilities > 1.0 + probability_tolerance)
    ):
        raise RuntimeError(
            "Native triplet search returned probabilities outside [0, 1]."
        )

    triplet_metadata: list[dict[str, object]] = []

    for rank, item in enumerate(raw_metadata, start=1):
        if len(item) != 7:
            raise RuntimeError(
                "Native triplet metadata entries must contain 7 values."
            )

        i, j, k, wi, wj, wk, brier_score = item

        i = int(i)
        j = int(j)
        k = int(k)

        if not (0 <= i < j < k < P.shape[1]):
            raise RuntimeError(
                "Native triplet search returned invalid model indices."
            )

        weights = np.asarray([wi, wj, wk], dtype=np.float64)

        if not np.all(np.isfinite(weights)):
            raise RuntimeError(
                "Native triplet search returned non-finite weights."
            )

        if np.any(weights < 0.0):
            raise RuntimeError(
                "Native triplet search returned negative convex weights."
            )

        weight_sum = float(np.sum(weights))

        if not np.isclose(weight_sum, 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"Native triplet weights must sum to 1, got {weight_sum}."
            )

        brier_score = float(brier_score)

        if not np.isfinite(brier_score) or brier_score < 0.0:
            raise RuntimeError(
                "Native triplet search returned an invalid OOF Brier score."
            )

        triplet_metadata.append({
            "rank": rank,
            "i": i,
            "j": j,
            "k": k,
            "wi": float(wi),
            "wj": float(wj),
            "wk": float(wk),
            "oof_brier_score": brier_score
        })

    return triplet_metadata, triplet_oof_probabilities
