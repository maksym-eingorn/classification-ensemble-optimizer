# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# datasets/breast_cancer.py

import numpy as np
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Breast Cancer Wisconsin classification dataset.

    Scikit-learn originally encodes malignant samples as 0 and benign samples
    as 1. This project reverses that encoding so that malignant is the positive
    class:

    0 = benign
    1 = malignant
    """
    data = load_breast_cancer()

    X = np.asarray(data.data, dtype=np.float64)
    y_original = np.asarray(data.target, dtype=np.int64)

    y = 1 - y_original

    return X, y
