# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# analysis/storage.py

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib

from storage import get_data_split_name


def get_stopping_analysis_output_dir(
    root_dir: str | Path,
    dataset_name: str,
    model_name: str,
    feature_set: str,
    ensemble_method: str
) -> Path:
    """Return the root directory for one stopping-analysis experiment."""
    return (
        Path(root_dir)
        / dataset_name
        / f"{model_name}-{feature_set}"
        / f"{ensemble_method}_stopping"
    )


def get_checkpoint_evaluation_seed_dir(
    analysis_dir: str | Path, data_split_seed: int
) -> Path:
    """Return the checkpoint-evaluation directory for one split seed."""
    return (
        Path(analysis_dir)
        / "checkpoint_evaluations"
        / get_data_split_name(data_split_seed)
    )


def get_aggregate_analysis_dir(analysis_dir: str | Path) -> Path:
    """Return the aggregate multi-seed analysis directory."""
    return Path(analysis_dir) / "aggregate"


def ensure_analysis_settings(
    analysis_dir: str | Path, settings: dict[str, object]
) -> None:
    """Create or validate the settings that define an analysis experiment."""
    output_path = Path(analysis_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    settings_path = output_path / "analysis_settings.pkl"

    if settings_path.exists():
        saved_settings = joblib.load(settings_path)

        if saved_settings != settings:
            raise RuntimeError(
                "Existing analysis results were created with different "
                "settings. Remove the analysis output directory or use a "
                "different analysis root."
            )
    else:
        joblib.dump(settings, settings_path, compress=3)


def save_records_csv(
    output_path: str | Path, records: Sequence[Mapping[str, object]]
) -> None:
    """Save scalar record dictionaries as a CSV file."""
    if not records:
        raise ValueError("records must contain at least one row.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(records[0].keys())

    for record in records:
        if list(record.keys()) != fieldnames:
            raise ValueError(
                "All CSV records must have identical keys in identical order."
            )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
