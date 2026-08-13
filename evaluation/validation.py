# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluation/validation.py

import numpy as np

from tuning.validation import (
    validate_binary_target, validate_feature_target_arrays
)


def validate_binary_classification_evaluation_inputs(
    X_dev, y_dev, X_test, y_test
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate development and test arrays for binary classification evaluation.
    """
    X_dev, y_dev = validate_feature_target_arrays(
        X_dev,
        y_dev,
        "X_dev",
        "y_dev",
        sample_label="n_samples_dev"
    )

    X_test, y_test = validate_feature_target_arrays(
        X_test,
        y_test,
        "X_test",
        "y_test",
        sample_label="n_samples_test"
    )

    validate_binary_target(y_dev, "y_dev")
    validate_binary_target(y_test, "y_test")

    if X_dev.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Numbers of features in X_dev and X_test must match, "
            f"got {X_dev.shape[1]} != {X_test.shape[1]}."
        )

    return X_dev, y_dev, X_test, y_test
