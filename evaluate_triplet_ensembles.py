# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluate_triplet_ensembles.py

from environment import configure_environment

configure_environment()

from pathlib import Path
from typing import cast

import joblib

import config
from evaluation.ensemble_evaluation import evaluate_triplet_ensembles_on_test
from storage import (
    get_data_split_name, get_dataset_output_dir, load_prepared_data
)
from tuning.feature_sets import (
    validate_lightgbm_feature_set,
    validate_logistic_regression_feature_set,
    validate_xgboost_feature_set
)


def _get_model_specs() -> list[tuple[str, str]]:
    """Return model names and configured feature sets for triplet evaluation."""
    return [
        ("xgboost", config.XGBOOST_FEATURE_SET),
        ("lightgbm", config.LIGHTGBM_FEATURE_SET),
        ("logistic_regression", config.LOGISTIC_REGRESSION_FEATURE_SET)
    ]


def _validate_model_feature_sets() -> None:
    """Validate configured feature sets for all triplet base model families."""
    validate_xgboost_feature_set(config.XGBOOST_FEATURE_SET)
    validate_lightgbm_feature_set(config.LIGHTGBM_FEATURE_SET)
    validate_logistic_regression_feature_set(
        config.LOGISTIC_REGRESSION_FEATURE_SET
    )


def _get_triplet_output_dir(
    root_dir: str | Path,
    dataset_name: str,
    data_split_seed: int,
    model_specs: list[tuple[str, str]]
) -> Path:
    """Return the output directory for convex triplet search results."""
    spec_name = "__".join(
        f"{model_name}-{feature_set}" for model_name, feature_set in model_specs
    )

    return (
        Path(root_dir)
        / dataset_name
        / get_data_split_name(data_split_seed)
        / spec_name
        / "triplet_convex"
    )


def main() -> None:
    """Evaluate saved top convex triplet ensembles on the held-out test set."""
    _validate_model_feature_sets()
    model_specs = _get_model_specs()

    triplet_output_dir = _get_triplet_output_dir(
        config.ENSEMBLE_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        model_specs
    )

    triplet_metadata_path = triplet_output_dir / "triplet_oof_metadata.pkl"

    triplet_metadata = cast(
        list[dict[str, object]], joblib.load(triplet_metadata_path)
    )

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    evaluated_metadata, triplet_test_probabilities = (
        evaluate_triplet_ensembles_on_test(
            triplet_metadata,
            arrays,
            seed=config.RANDOM_SEED,
            logistic_regression_max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
            logistic_regression_tol=config.LOGISTIC_REGRESSION_TOL
        )
    )

    joblib.dump(
        evaluated_metadata,
        triplet_output_dir / "triplet_test_metadata.pkl",
        compress=3
    )
    joblib.dump(
        triplet_test_probabilities,
        triplet_output_dir / "triplet_test_probabilities.pkl",
        compress=3
    )

    best_by_oof = evaluated_metadata[0]
    best_by_test = min(
        evaluated_metadata,
        key=lambda item: cast(float, item["test_brier_score"])
    )

    print(
        f"{len(evaluated_metadata)} triplet ensembles evaluated on test data."
    )
    print(
        "Triplet test probability matrix shape: "
        f"{triplet_test_probabilities.shape}"
    )
    print(f"Triplet test results saved to: {triplet_output_dir}")
    print()

    print(
        "Best OOF triplet: "
        f"OOF Brier score = "
        f"{cast(float, best_by_oof['oof_brier_score']):.5f}, "
        f"test Brier score = "
        f"{cast(float, best_by_oof['test_brier_score']):.5f}, "
        f"columns "
        f"({cast(int, best_by_oof['i'])}, "
        f"{cast(int, best_by_oof['j'])}, "
        f"{cast(int, best_by_oof['k'])})"
    )
    print(f"Test log loss = {cast(float, best_by_oof['test_log_loss']):.5f}")
    print(f"Test ROC AUC = {cast(float, best_by_oof['test_roc_auc']):.5f}")
    print(f"Test accuracy = {cast(float, best_by_oof['test_accuracy']):.5f}")
    print()

    print(
        f"Best test triplet among saved {len(evaluated_metadata)} "
        f"ensembles (diagnostic only): "
        f"OOF Brier score = "
        f"{cast(float, best_by_test['oof_brier_score']):.5f}, "
        f"test Brier score = "
        f"{cast(float, best_by_test['test_brier_score']):.5f}, "
        f"columns "
        f"({cast(int, best_by_test['i'])}, "
        f"{cast(int, best_by_test['j'])}, "
        f"{cast(int, best_by_test['k'])})"
    )
    print(f"Test log loss = {cast(float, best_by_test['test_log_loss']):.5f}")
    print(f"Test ROC AUC = {cast(float, best_by_test['test_roc_auc']):.5f}")
    print(f"Test accuracy = {cast(float, best_by_test['test_accuracy']):.5f}")


if __name__ == "__main__":
    main()
