# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# run_greedy_ensemble_selection.py

from environment import configure_environment

configure_environment()

from pathlib import Path
from typing import cast

import joblib
import numpy as np

import config
from ensemble.greedy_selection import find_greedy_ensemble
from ensemble.oof_matrix import build_oof_matrix, load_oof_result_block
from storage import (
    get_data_split_name, get_dataset_output_dir, load_prepared_data
)
from tuning.feature_sets import (
    validate_lightgbm_feature_set,
    validate_logistic_regression_feature_set,
    validate_xgboost_feature_set
)


def _get_model_specs() -> list[tuple[str, str]]:
    """Return model names and configured feature sets for greedy selection."""
    return [
        ("xgboost", config.XGBOOST_FEATURE_SET),
        ("lightgbm", config.LIGHTGBM_FEATURE_SET),
        ("logistic_regression", config.LOGISTIC_REGRESSION_FEATURE_SET)
    ]


def _validate_model_feature_sets() -> None:
    """Validate configured feature sets for all greedy base model families."""
    validate_xgboost_feature_set(config.XGBOOST_FEATURE_SET)
    validate_lightgbm_feature_set(config.LIGHTGBM_FEATURE_SET)
    validate_logistic_regression_feature_set(
        config.LOGISTIC_REGRESSION_FEATURE_SET
    )


def _get_greedy_output_dir(
    root_dir: str | Path,
    dataset_name: str,
    data_split_seed: int,
    model_specs: list[tuple[str, str]]
) -> Path:
    """Return the output directory for greedy ensemble selection results."""
    spec_name = "__".join(
        f"{model_name}-{feature_set}"
        for model_name, feature_set in model_specs
    )

    return (
        Path(root_dir)
        / dataset_name
        / get_data_split_name(data_split_seed)
        / spec_name
        / "greedy_selection"
    )


def main() -> None:
    """Run Caruana-style greedy ensemble selection on saved out-of-fold
    probabilities."""
    _validate_model_feature_sets()
    model_specs = _get_model_specs()

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    y_dev = np.asarray(arrays["y_dev"], dtype=np.float64)

    blocks = [
        load_oof_result_block(
            config.OPTUNA_RESULTS_DIR,
            config.DATASET_NAME,
            config.DATA_SPLIT_SEED,
            feature_set,
            model_name
        )
        for model_name, feature_set in model_specs
    ]

    P, column_metadata = build_oof_matrix(blocks)

    greedy_metadata, greedy_oof_probabilities = find_greedy_ensemble(
        P,
        y_dev,
        column_metadata,
        pool_size=config.GREEDY_POOL_SIZE,
        ensemble_size=config.GREEDY_ENSEMBLE_SIZE,
        seed=config.RANDOM_SEED,
        allow_repeats=config.GREEDY_ALLOW_REPEATS
    )

    output_dir = _get_greedy_output_dir(
        config.ENSEMBLE_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        model_specs
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        column_metadata,
        output_dir / "base_model_column_metadata.pkl",
        compress=3
    )
    joblib.dump(
        greedy_metadata,
        output_dir / "greedy_oof_metadata.pkl",
        compress=3
    )
    joblib.dump(
        greedy_oof_probabilities.reshape(-1, 1).astype(np.float32),
        output_dir / "greedy_oof_probabilities.pkl",
        compress=3
    )

    weights = cast(list[float], greedy_metadata["weights"])
    selected_columns = cast(list[int], greedy_metadata["selected_columns"])

    print(f"Unified out-of-fold probability matrix shape: {P.shape}")
    print(
        "Greedy out-of-fold probability matrix shape: "
        f"{greedy_oof_probabilities.reshape(-1, 1).shape}"
    )

    print(f"Greedy ensemble selection results saved to: {output_dir}")
    print()

    oof_brier_score = cast(float, greedy_metadata["oof_brier_score"])

    print(
        "Greedy ensemble OOF Brier score = "
        f"{oof_brier_score:.5f}, "
        "selected columns "
        + "("
        + ", ".join(str(column) for column in selected_columns)
        + ")"
    )
    print(
        "Weights: "
        + "("
        + ", ".join(f"{weight:.5f}" for weight in weights)
        + ")"
    )


if __name__ == "__main__":
    main()
