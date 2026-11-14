# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# analysis/statistics.py

from collections.abc import Iterable

import numpy as np

from analysis.schemas import DistributionSummary
from analysis.validation import validate_tie_tolerance


def summarize_distribution(values: Iterable[float]) -> DistributionSummary:
    """Return compact descriptive statistics for a finite numeric sample."""
    array = np.asarray(list(values), dtype=np.float64)

    if array.ndim != 1 or len(array) == 0:
        raise ValueError(
            "values must contain at least one one-dimensional value."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite numbers.")

    std = (float(np.std(array, ddof=1)) if len(array) > 1 else 0.0)

    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": std,
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array))
    }


def count_lower_is_better_outcomes(
    differences: Iterable[float], tolerance: float
) -> tuple[int, int, int]:
    """
    Count wins, losses, and ties for paired differences where lower is better.
    """
    tolerance = validate_tie_tolerance(tolerance)

    array = np.asarray(list(differences), dtype=np.float64)

    if array.ndim != 1 or len(array) == 0:
        raise ValueError("differences must contain at least one value.")

    if not np.all(np.isfinite(array)):
        raise ValueError("differences must contain only finite values.")

    wins = int(np.sum(array < -tolerance))
    losses = int(np.sum(array > tolerance))
    ties = int(len(array) - wins - losses)

    return wins, losses, ties


def safe_pearson_correlation(
        x: Iterable[float], y: Iterable[float]
) -> float | None:
    """Return Pearson correlation, or None when it is not defined."""
    x_array = np.asarray(list(x), dtype=np.float64)
    y_array = np.asarray(list(y), dtype=np.float64)

    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    if len(x_array) != len(y_array):
        raise ValueError("x and y must have matching lengths.")

    if len(x_array) < 2:
        return None

    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("x and y must contain only finite values.")

    if np.ptp(x_array) == 0.0 or np.ptp(y_array) == 0.0:
        return None

    return float(np.corrcoef(x_array, y_array)[0, 1])
