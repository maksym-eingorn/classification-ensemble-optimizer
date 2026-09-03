# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# run_logistic_regression_optuna.py

from environment import configure_environment

configure_environment()

import config
from storage import get_dataset_output_dir, load_prepared_data
from tuning.feature_sets import (
    get_development_data, validate_logistic_regression_feature_set
)
from tuning.result_storage import get_optuna_result_dir, save_optuna_results
from tuning.logistic_regression_optuna import (
    run_optuna_kfold_logistic_regression
)


def main() -> None:
    """
    Run Logistic Regression Optuna tuning on the selected prepared dataset.
    """
    validate_logistic_regression_feature_set(
        config.LOGISTIC_REGRESSION_FEATURE_SET
    )

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    X_dev, y_dev = get_development_data(
        arrays, config.LOGISTIC_REGRESSION_FEATURE_SET
    )

    (
        study_lr,
        trial_numbers_lr,
        oof_probabilities_lr,
        oof_brier_scores_lr,
        fold_brier_scores_lr,
        hyperparams_lr,
    ) = run_optuna_kfold_logistic_regression(
        X_dev,
        y_dev,
        n_trials=config.LOGISTIC_REGRESSION_N_TRIALS,
        n_jobs=config.LOGISTIC_REGRESSION_N_JOBS,
        n_splits=config.LOGISTIC_REGRESSION_N_SPLITS,
        max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
        tol=config.LOGISTIC_REGRESSION_TOL,
        seed=config.RANDOM_SEED,
        verbose=config.LOGISTIC_REGRESSION_VERBOSE
    )

    print(
        f"\nBest OOF Brier score = {study_lr.best_value:.5f} "
        f"(trial {study_lr.best_trial.number})"
    )
    print(f"Best hyperparameters:\n{study_lr.best_trial.params}")

    result_dir = get_optuna_result_dir(
        config.OPTUNA_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        config.LOGISTIC_REGRESSION_FEATURE_SET,
        "logistic_regression"
    )

    save_optuna_results(
        result_dir,
        study_lr,
        trial_numbers_lr,
        oof_probabilities_lr,
        oof_brier_scores_lr,
        fold_brier_scores_lr,
        hyperparams_lr
    )

    print(f"\nLogistic Regression Optuna results saved to: {result_dir}")


if __name__ == "__main__":
    main()
