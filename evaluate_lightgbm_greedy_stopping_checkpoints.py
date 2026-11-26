# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluate_lightgbm_greedy_stopping_checkpoints.py

from environment import configure_environment

configure_environment()

import config
from analysis.greedy.checkpoint_evaluation import (
    evaluate_greedy_checkpoints_for_seed
)
from analysis.storage import (
    ensure_analysis_settings,
    get_checkpoint_evaluation_seed_dir,
    get_stopping_analysis_output_dir
)
from analysis.validation import (
    build_checkpoint_budgets, validate_analysis_data_split_seeds
)
from storage import get_dataset_output_dir, load_prepared_data
from tuning.feature_sets import validate_lightgbm_feature_set
from tuning.result_storage import get_optuna_result_dir, load_optuna_results


def main() -> None:
    """
    Evaluate LightGBM-only greedy ensembles at every stopping checkpoint.
    """
    validate_lightgbm_feature_set(config.LIGHTGBM_FEATURE_SET)

    data_split_seeds = (
        validate_analysis_data_split_seeds(config.ANALYSIS_DATA_SPLIT_SEEDS)
    )

    budgets = build_checkpoint_budgets(
        config.LIGHTGBM_STOPPING_CHECKPOINT_TRIALS,
        config.LIGHTGBM_STOPPING_MAX_TRIALS
    )

    analysis_dir = (
        get_stopping_analysis_output_dir(
            config.ANALYSIS_RESULTS_DIR,
            config.DATASET_NAME,
            "lightgbm",
            config.LIGHTGBM_FEATURE_SET,
            "greedy"
        )
    )

    analysis_settings: dict[str, object] = {
        "dataset_name": config.DATASET_NAME,
        "model_name": "lightgbm",
        "feature_set": config.LIGHTGBM_FEATURE_SET,
        "ensemble_method": "greedy",
        "data_split_seeds": tuple(data_split_seeds),
        "budgets": tuple(budgets),
        "random_seed": config.RANDOM_SEED,
        "greedy_pool_size": config.GREEDY_POOL_SIZE,
        "greedy_ensemble_size": config.GREEDY_ENSEMBLE_SIZE,
        "greedy_allow_repeats": config.GREEDY_ALLOW_REPEATS,
        "logistic_regression_max_iter": config.LOGISTIC_REGRESSION_MAX_ITER,
        "logistic_regression_tol": config.LOGISTIC_REGRESSION_TOL
    }

    ensure_analysis_settings(analysis_dir, analysis_settings)

    for data_split_seed in data_split_seeds:
        prepared_data_dir = (
            get_dataset_output_dir(
                config.PREPARED_DATA_DIR, config.DATASET_NAME, data_split_seed
            )
        )

        arrays, _ = load_prepared_data(prepared_data_dir)

        optuna_dir = get_optuna_result_dir(
            config.OPTUNA_RESULTS_DIR,
            config.DATASET_NAME,
            data_split_seed,
            config.LIGHTGBM_FEATURE_SET,
            "lightgbm"
        )

        optuna_results = load_optuna_results(optuna_dir)

        seed_output_dir = (
            get_checkpoint_evaluation_seed_dir(analysis_dir, data_split_seed)
        )

        evaluate_greedy_checkpoints_for_seed(
            arrays,
            optuna_results,
            output_dir=seed_output_dir,
            data_split_seed=data_split_seed,
            model_name="lightgbm",
            feature_set=config.LIGHTGBM_FEATURE_SET,
            budgets=budgets,
            pool_size=config.GREEDY_POOL_SIZE,
            ensemble_size=config.GREEDY_ENSEMBLE_SIZE,
            allow_repeats=config.GREEDY_ALLOW_REPEATS,
            seed=config.RANDOM_SEED,
            logistic_regression_max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
            logistic_regression_tol=config.LOGISTIC_REGRESSION_TOL
        )

    print(f"Checkpoint evaluations saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
