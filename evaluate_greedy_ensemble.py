# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluate_greedy_ensemble.py

from environment import configure_environment

configure_environment()

from pathlib import Path
from typing import cast

import joblib

import config
from evaluation.ensemble_evaluation import evaluate_greedy_ensemble_on_test
from storage import (
    get_data_split_name, get_dataset_output_dir, load_prepared_data
)
from tuning.feature_sets import (
    validate_lightgbm_feature_set,
    validate_logistic_regression_feature_set,
    validate_xgboost_feature_set
)


def _get_model_specs() -> list[tuple[str, str]]:
    """Return model names and configured feature sets for greedy evaluation."""
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
    """Return the output directory for greedy ensemble results."""
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
    """Evaluate the saved greedy ensemble on the held-out test set."""
    _validate_model_feature_sets()
    model_specs = _get_model_specs()

    greedy_output_dir = _get_greedy_output_dir(
        config.ENSEMBLE_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        model_specs
    )

    greedy_metadata_path = greedy_output_dir / "greedy_oof_metadata.pkl"

    greedy_metadata = cast(dict[str, object], joblib.load(greedy_metadata_path))

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    evaluated_metadata, greedy_test_probabilities = (
        evaluate_greedy_ensemble_on_test(
            greedy_metadata,
            arrays,
            seed=config.RANDOM_SEED,
            logistic_regression_max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
            logistic_regression_tol=config.LOGISTIC_REGRESSION_TOL
        )
    )

    joblib.dump(
        evaluated_metadata,
        greedy_output_dir / "greedy_test_metadata.pkl",
        compress=3
    )
    joblib.dump(
        greedy_test_probabilities,
        greedy_output_dir / "greedy_test_probabilities.pkl",
        compress=3
    )

    oof_brier_score = cast(float, evaluated_metadata["oof_brier_score"])
    test_brier_score = cast(float, evaluated_metadata["test_brier_score"])
    test_log_loss = cast(float, evaluated_metadata["test_log_loss"])
    test_roc_auc = cast(float, evaluated_metadata["test_roc_auc"])
    test_accuracy = cast(float, evaluated_metadata["test_accuracy"])

    print("Greedy ensemble evaluated on test data.")
    print(
        "Greedy test probability matrix shape: "
        f"{greedy_test_probabilities.shape}"
    )
    print(f"Greedy test results saved to: {greedy_output_dir}")
    print()
    print(
        "Greedy ensemble: "
        f"OOF Brier score = {oof_brier_score:.5f}, "
        f"test Brier score = {test_brier_score:.5f}"
    )
    print(f"Test log loss = {test_log_loss:.5f}")
    print(f"Test ROC AUC = {test_roc_auc:.5f}")
    print(f"Test accuracy = {test_accuracy:.5f}")


if __name__ == "__main__":
    main()
