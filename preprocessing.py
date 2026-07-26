# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# preprocessing.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from storage import validate_data_split_seed


def split_dev_test(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split features and binary targets into stratified development and test sets.
    """
    validate_data_split_seed(config.DATA_SPLIT_SEED)

    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.DATA_SPLIT_SEED,
        stratify=y
    )


def scale_dev_test(
    X_dev: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on the development set and transform both
    development and test sets.
    """
    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    return X_dev_scaled, X_test_scaled, scaler
