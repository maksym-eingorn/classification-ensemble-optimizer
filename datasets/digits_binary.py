# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# datasets/digits_binary.py

import numpy as np
from sklearn.datasets import load_digits


def load_binary_digits_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Load a binary subset of the Digits classification dataset.

    The subset contains handwritten zeros and ones:

    0 = digit zero
    1 = digit one
    """
    data = load_digits()

    X_all = np.asarray(data.data, dtype=np.float64)
    y_all = np.asarray(data.target, dtype=np.int64)

    mask = (y_all == 0) | (y_all == 1)

    X = X_all[mask]
    y = y_all[mask]

    return X, y
