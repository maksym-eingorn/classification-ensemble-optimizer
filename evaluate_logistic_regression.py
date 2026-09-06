# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluate_logistic_regression.py

from environment import configure_environment

configure_environment()

from typing import cast

import optuna

import config
from evaluation.logistic_regression_evaluation import (
    evaluate_best_logistic_regression_on_test
)
from storage import get_dataset_output_dir, load_prepared_data
from tuning.feature_sets import (
    get_development_and_test_data, validate_logistic_regression_feature_set
)
from tuning.result_storage import get_optuna_result_dir, load_optuna_results


def main() -> None:
    """Evaluate the best tuned Logistic Regression model on the test set."""
    validate_logistic_regression_feature_set(
        config.LOGISTIC_REGRESSION_FEATURE_SET
    )

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    X_dev, y_dev, X_test, y_test = get_development_and_test_data(
        arrays, config.LOGISTIC_REGRESSION_FEATURE_SET
    )

    result_dir = get_optuna_result_dir(
        config.OPTUNA_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        config.LOGISTIC_REGRESSION_FEATURE_SET,
        "logistic_regression"
    )

    optuna_results = load_optuna_results(result_dir)
    study_lr = cast(optuna.study.Study, optuna_results["study"])

    test_metrics_lr = evaluate_best_logistic_regression_on_test(
        study_lr,
        X_dev,
        y_dev,
        X_test,
        y_test,
        max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
        tol=config.LOGISTIC_REGRESSION_TOL,
        seed=config.RANDOM_SEED
    )

    print(
        "Best Logistic Regression model: "
        f"test Brier score = {test_metrics_lr['brier_score']:.5f}"
    )
    print(f"Test log loss = {test_metrics_lr['log_loss']:.5f}")
    print(f"Test ROC AUC = {test_metrics_lr['roc_auc']:.5f}")
    print(f"Test accuracy = {test_metrics_lr['accuracy']:.5f}")


if __name__ == "__main__":
    main()
