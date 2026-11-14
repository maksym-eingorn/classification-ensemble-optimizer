# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# analysis/validation.py

import math
from collections.abc import Iterable

from storage import validate_data_split_seed
from tuning.validation import validate_positive_integer


def validate_analysis_data_split_seeds(
    seeds: Iterable[int]
) -> list[int]:
    """
    Validate a strictly increasing sequence of development/test split seeds.
    """
    values = list(seeds)

    if not values:
        raise ValueError("At least one analysis data-split seed is required.")

    for value in values:
        validate_data_split_seed(value)

    if len(set(values)) != len(values):
        raise ValueError("Analysis data-split seeds must be unique.")

    if values != sorted(values):
        raise ValueError(
            "Analysis data-split seeds must be in increasing order."
        )

    return values


def build_checkpoint_budgets(
    checkpoint_trials: int, max_trials: int
) -> list[int]:
    """Return the complete ordered checkpoint budget grid."""
    validate_positive_integer(checkpoint_trials, "checkpoint_trials")
    validate_positive_integer(max_trials, "max_trials")

    if max_trials % checkpoint_trials != 0:
        raise ValueError(
            "max_trials must be an exact multiple of checkpoint_trials."
        )

    return list(range(checkpoint_trials, max_trials + 1, checkpoint_trials))


def validate_tie_tolerance(tolerance: float) -> float:
    """Validate a finite nonnegative absolute tolerance."""
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or tolerance < 0.0
    ):
        raise ValueError(
            "tolerance must be a nonnegative finite number, "
            f"got {tolerance}."
        )

    value = float(tolerance)

    if not math.isfinite(value):
        raise ValueError(
            "tolerance must be a nonnegative finite number, "
            f"got {tolerance}."
        )

    return value
