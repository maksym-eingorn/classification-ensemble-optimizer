# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# storage.py

from pathlib import Path

import joblib
import numpy as np


PREPARED_ARRAY_NAMES = [
    "X",
    "y",
    "X_dev",
    "y_dev",
    "X_test",
    "y_test",
    "X_dev_scaled",
    "X_test_scaled"
]

PREPARED_OBJECT_NAMES = [
    "scaler"
]


def validate_data_split_seed(data_split_seed: int) -> None:
    """Validate the development/test split seed."""
    if (
        isinstance(data_split_seed, bool)
        or not isinstance(data_split_seed, int)
        or data_split_seed < 0
    ):
        raise ValueError(
            "data_split_seed must be a non-negative integer, "
            f"got {data_split_seed}."
        )


def get_data_split_name(data_split_seed: int) -> str:
    """Return the folder name for one development/test split."""
    validate_data_split_seed(data_split_seed)

    return f"split_seed_{data_split_seed}"


def get_dataset_output_dir(
    root_dir: str | Path, dataset_name: str, data_split_seed: int
) -> Path:
    """Return the output directory for a specific dataset and data split."""
    return Path(root_dir) / dataset_name / get_data_split_name(data_split_seed)


def save_numpy_arrays(
    output_dir: str | Path, arrays: dict[str, np.ndarray]
) -> None:
    """Save multiple NumPy arrays into an output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, array in arrays.items():
        np.save(output_path / f"{name}.npy", array)


def load_numpy_arrays(
    output_dir: str | Path, names: list[str]
) -> dict[str, np.ndarray]:
    """Load multiple NumPy arrays from an output directory."""
    output_path = Path(output_dir)

    return {name: np.load(output_path / f"{name}.npy") for name in names}


def save_objects(
    output_dir: str | Path, objects: dict[str, object]
) -> None:
    """Save multiple Python objects with joblib."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, obj in objects.items():
        joblib.dump(obj, output_path / f"{name}.pkl", compress=3)


def load_objects(
    output_dir: str | Path, names: list[str]
) -> dict[str, object]:
    """Load multiple Python objects saved with joblib."""
    output_path = Path(output_dir)

    return {name: joblib.load(output_path / f"{name}.pkl") for name in names}


def load_prepared_data(
    output_dir: str | Path
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load prepared arrays and fitted preprocessing objects."""
    arrays = load_numpy_arrays(output_dir, PREPARED_ARRAY_NAMES)
    objects = load_objects(output_dir, PREPARED_OBJECT_NAMES)

    return arrays, objects
