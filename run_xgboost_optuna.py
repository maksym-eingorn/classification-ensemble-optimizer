# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# run_xgboost_optuna.py

from environment import configure_environment

configure_environment()

import config
from storage import get_dataset_output_dir, load_prepared_data
from tuning.feature_sets import get_development_data
from tuning.result_storage import get_optuna_result_dir, save_optuna_results
from tuning.xgboost_optuna import run_optuna_kfold_xgboost


def _validate_xgboost_feature_set(feature_set: str) -> None:
    """Validate the feature set supported by XGBoost tuning."""
    if feature_set != "original":
        raise ValueError(
            f"Unsupported XGBoost feature set: {feature_set}. "
            "Currently supported: original."
        )


def main() -> None:
    """Run XGBoost Optuna tuning on the selected prepared dataset."""
    _validate_xgboost_feature_set(config.XGBOOST_FEATURE_SET)

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    X_dev, y_dev = get_development_data(arrays, config.XGBOOST_FEATURE_SET)

    (
        study_xgb,
        trial_numbers_xgb,
        oof_probabilities_xgb,
        oof_brier_scores_xgb,
        fold_brier_scores_xgb,
        hyperparams_xgb,
    ) = run_optuna_kfold_xgboost(
        X_dev,
        y_dev,
        n_trials=config.XGBOOST_N_TRIALS,
        n_jobs=config.XGBOOST_N_JOBS,
        n_splits=config.XGBOOST_N_SPLITS,
        n_estimators_min=config.XGBOOST_N_ESTIMATORS_MIN,
        n_estimators_max=config.XGBOOST_N_ESTIMATORS_MAX,
        n_estimators_step=config.XGBOOST_N_ESTIMATORS_STEP,
        seed=config.RANDOM_SEED,
        verbose=config.XGBOOST_VERBOSE
    )

    print(
        f"\nBest OOF Brier score = {study_xgb.best_value:.5f} "
        f"(trial {study_xgb.best_trial.number})"
    )
    print(f"Best hyperparameters:\n{study_xgb.best_trial.params}")

    result_dir = get_optuna_result_dir(
        config.OPTUNA_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        config.XGBOOST_FEATURE_SET,
        "xgboost"
    )

    save_optuna_results(
        result_dir,
        study_xgb,
        trial_numbers_xgb,
        oof_probabilities_xgb,
        oof_brier_scores_xgb,
        fold_brier_scores_xgb,
        hyperparams_xgb
    )

    print(f"\nXGBoost Optuna results saved to: {result_dir}")


if __name__ == "__main__":
    main()
