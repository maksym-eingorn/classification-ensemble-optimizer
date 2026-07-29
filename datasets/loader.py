# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# datasets/loader.py

import numpy as np

from datasets.breast_cancer import load_breast_cancer_dataset
from datasets.digits_binary import load_binary_digits_dataset


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a binary classification dataset by name."""
    if dataset_name == "breast_cancer":
        return load_breast_cancer_dataset()

    if dataset_name == "digits_binary":
        return load_binary_digits_dataset()

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. "
        "Currently supported: breast_cancer, digits_binary."
    )
