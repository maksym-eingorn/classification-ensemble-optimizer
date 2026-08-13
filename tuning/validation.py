# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# tuning/validation.py

import numpy as np


def _to_numpy_array(array_like) -> np.ndarray:
    """Convert an array-like object to a NumPy array."""
    if isinstance(array_like, np.ndarray):
        return array_like

    if hasattr(array_like, "to_numpy"):
        return array_like.to_numpy()

    return np.asarray(array_like)


def validate_feature_target_arrays(
    X, y, X_name: str, y_name: str, sample_label: str = "n_samples"
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the feature matrix and the target vector."""
    X = _to_numpy_array(X)
    y = _to_numpy_array(y)

    if X.ndim != 2:
        raise ValueError(
            f"{X_name} must be 2D ({sample_label}, n_features), "
            f"got {X.ndim}D."
        )

    if y.ndim != 1:
        raise ValueError(
            f"{y_name} must be 1D ({sample_label},), got {y.ndim}D."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Numbers of samples in {X_name} and {y_name} must match, "
            f"got {X.shape[0]} != {y.shape[0]}."
        )

    if X.shape[0] < 1:
        raise ValueError(
            f"{X_name} and {y_name} must contain at least 1 sample."
        )

    if X.shape[1] < 1:
        raise ValueError(f"{X_name} must contain at least 1 feature.")

    if not np.all(np.isfinite(X)):
        raise ValueError(f"{X_name} must contain only finite values.")

    if not np.all(np.isfinite(y)):
        raise ValueError(f"{y_name} must contain only finite values.")

    return X, y


def validate_binary_target(y: np.ndarray, name: str) -> None:
    """Validate that a target contains exactly the binary classes 0 and 1."""
    classes = np.unique(y)
    expected_classes = np.array([0, 1])

    if not np.array_equal(classes, expected_classes):
        raise ValueError(
            f"{name} must contain exactly the binary classes 0 and 1, "
            f"got {classes.tolist()}."
        )


def validate_binary_classification_inputs(
    X_dev, y_dev
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return binary development data as NumPy arrays."""
    X_dev, y_dev = validate_feature_target_arrays(
        X_dev,
        y_dev,
        "X_dev",
        "y_dev",
        sample_label="n_samples_dev"
    )

    validate_binary_target(y_dev, "y_dev")

    return X_dev, y_dev


def validate_positive_integer(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}.")


def validate_random_seed(value: int, name: str) -> None:
    """Validate a NumPy-compatible random seed."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 2**32 - 1
    ):
        raise ValueError(
            f"{name} must be an integer between 0 and 2**32 - 1, "
            f"got {value}."
        )


def validate_integer_range_settings(
    min_value: int, max_value: int, step: int, name: str
) -> None:
    """Validate integer search-range settings."""
    validate_positive_integer(min_value, f"{name}_min")
    validate_positive_integer(max_value, f"{name}_max")
    validate_positive_integer(step, f"{name}_step")

    if max_value < min_value:
        raise ValueError(
            f"{name}_max must be greater than or equal to {name}_min, "
            f"got {max_value} < {min_value}."
        )

    if (max_value - min_value) % step != 0:
        raise ValueError(
            f"{name}_step must evenly divide the range "
            f"{name}_max - {name}_min, got "
            f"({max_value} - {min_value}) % {step} != 0."
        )


def validate_stratified_kfold_settings(
    n_splits: int, y: np.ndarray
) -> None:
    """Validate stratified K-fold cross-validation settings."""
    validate_positive_integer(n_splits, "n_splits")

    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}.")

    _, class_counts = np.unique(y, return_counts=True)
    minimum_class_count = int(class_counts.min())

    if n_splits > minimum_class_count:
        raise ValueError(
            "n_splits cannot exceed the number of samples in the smallest "
            f"class, got n_splits={n_splits}, "
            f"minimum_class_count={minimum_class_count}."
        )
