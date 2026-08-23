# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# run_lightgbm_optuna.py

from environment import configure_environment

configure_environment()

import config
from storage import get_dataset_output_dir, load_prepared_data
from tuning.feature_sets import (
    get_development_data, validate_lightgbm_feature_set
)
from tuning.result_storage import get_optuna_result_dir, save_optuna_results
from tuning.lightgbm_optuna import run_optuna_kfold_lightgbm


def main() -> None:
    """Run LightGBM Optuna tuning on the selected prepared dataset."""
    validate_lightgbm_feature_set(config.LIGHTGBM_FEATURE_SET)

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    X_dev, y_dev = get_development_data(arrays, config.LIGHTGBM_FEATURE_SET)

    (
        study_lgb,
        trial_numbers_lgb,
        oof_probabilities_lgb,
        oof_brier_scores_lgb,
        fold_brier_scores_lgb,
        hyperparams_lgb,
    ) = run_optuna_kfold_lightgbm(
        X_dev,
        y_dev,
        n_trials=config.LIGHTGBM_N_TRIALS,
        n_jobs=config.LIGHTGBM_N_JOBS,
        n_splits=config.LIGHTGBM_N_SPLITS,
        n_estimators_min=config.LIGHTGBM_N_ESTIMATORS_MIN,
        n_estimators_max=config.LIGHTGBM_N_ESTIMATORS_MAX,
        n_estimators_step=config.LIGHTGBM_N_ESTIMATORS_STEP,
        seed=config.RANDOM_SEED,
        verbose=config.LIGHTGBM_VERBOSE
    )

    print(
        f"\nBest OOF Brier score = {study_lgb.best_value:.5f} "
        f"(trial {study_lgb.best_trial.number})"
    )
    print(f"Best hyperparameters:\n{study_lgb.best_trial.params}")

    result_dir = get_optuna_result_dir(
        config.OPTUNA_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        config.LIGHTGBM_FEATURE_SET,
        "lightgbm"
    )

    save_optuna_results(
        result_dir,
        study_lgb,
        trial_numbers_lgb,
        oof_probabilities_lgb,
        oof_brier_scores_lgb,
        fold_brier_scores_lgb,
        hyperparams_lgb
    )

    print(f"\nLightGBM Optuna results saved to: {result_dir}")


if __name__ == "__main__":
    main()
