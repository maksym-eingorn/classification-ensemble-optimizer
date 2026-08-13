# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# tuning/feature_sets.py

import numpy as np


SUPPORTED_FEATURE_SETS = [
    "original",
    "original_scaled"
]

DEVELOPMENT_FEATURE_KEYS = {
    "original": "X_dev",
    "original_scaled": "X_dev_scaled"
}

TEST_FEATURE_KEYS = {
    "original": "X_test",
    "original_scaled": "X_test_scaled"
}


def validate_feature_set(feature_set: str) -> None:
    """Validate the selected feature set name."""
    if feature_set not in SUPPORTED_FEATURE_SETS:
        supported = ", ".join(SUPPORTED_FEATURE_SETS)
        raise ValueError(
            f"Unsupported feature set: {feature_set}. "
            f"Currently supported: {supported}."
        )


def validate_xgboost_feature_set(feature_set: str) -> None:
    """Validate the feature set supported by XGBoost workflows."""
    if feature_set != "original":
        raise ValueError(
            f"Unsupported XGBoost feature set: {feature_set}. "
            "Currently supported: original."
        )


def _get_array_by_key(
    arrays: dict[str, np.ndarray], key: str
) -> np.ndarray:
    """Return an array from prepared data by key."""
    if key not in arrays:
        raise KeyError(
            f"Required array '{key}' was not found in prepared data."
        )

    return arrays[key]


def get_development_data(
    arrays: dict[str, np.ndarray], feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select development features and target from prepared arrays.

    Parameters
    ----------
    arrays
        Dictionary of prepared NumPy arrays.
    feature_set
        Feature set name.

    Returns
    -------
    X_dev : np.ndarray
        Selected development feature matrix.
    y_dev : np.ndarray
        Development target vector.
    """
    validate_feature_set(feature_set)

    X_key = DEVELOPMENT_FEATURE_KEYS[feature_set]
    y_key = "y_dev"

    X_dev = _get_array_by_key(arrays, X_key)
    y_dev = _get_array_by_key(arrays, y_key)

    return X_dev, y_dev


def get_test_data(
    arrays: dict[str, np.ndarray], feature_set: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select test features and target from prepared arrays.

    Parameters
    ----------
    arrays
        Dictionary of prepared NumPy arrays.
    feature_set
        Feature set name.

    Returns
    -------
    X_test : np.ndarray
        Selected test feature matrix.
    y_test : np.ndarray
        Test target vector.
    """
    validate_feature_set(feature_set)

    X_key = TEST_FEATURE_KEYS[feature_set]
    y_key = "y_test"

    X_test = _get_array_by_key(arrays, X_key)
    y_test = _get_array_by_key(arrays, y_key)

    return X_test, y_test


def get_development_and_test_data(
    arrays: dict[str, np.ndarray], feature_set: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select development and test data for a configured feature set."""
    X_dev, y_dev = get_development_data(arrays, feature_set)
    X_test, y_test = get_test_data(arrays, feature_set)

    return X_dev, y_dev, X_test, y_test
