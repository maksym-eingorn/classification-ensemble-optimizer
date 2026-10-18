# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# run_lightgbm_ensemble_stopping.py

from environment import configure_environment

configure_environment()

from typing import cast

import config
from storage import get_dataset_output_dir, load_prepared_data
from stopping.lightgbm_ensemble_stopping import (
    get_lightgbm_stopping_output_dir, run_lightgbm_ensemble_aware_stopping
)
from tuning.feature_sets import (
    get_development_data, validate_lightgbm_feature_set
)


def main() -> None:
    """Run cross-fitted LightGBM ensemble-aware Optuna stopping."""
    validate_lightgbm_feature_set(config.LIGHTGBM_FEATURE_SET)

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    X_dev, y_dev = get_development_data(arrays, config.LIGHTGBM_FEATURE_SET)

    output_dir = get_lightgbm_stopping_output_dir(
        config.STOPPING_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        config.LIGHTGBM_FEATURE_SET
    )

    summary = run_lightgbm_ensemble_aware_stopping(
        X_dev,
        y_dev,
        output_dir=output_dir,
        feature_set=config.LIGHTGBM_FEATURE_SET,
        meta_n_splits=config.LIGHTGBM_STOPPING_META_N_SPLITS,
        checkpoint_trials=config.LIGHTGBM_STOPPING_CHECKPOINT_TRIALS,
        patience=config.LIGHTGBM_STOPPING_PATIENCE,
        max_trials=config.LIGHTGBM_STOPPING_MAX_TRIALS,
        max_workers=config.LIGHTGBM_STOPPING_MAX_WORKERS,
        inner_n_splits=config.LIGHTGBM_N_SPLITS,
        n_estimators_min=config.LIGHTGBM_N_ESTIMATORS_MIN,
        n_estimators_max=config.LIGHTGBM_N_ESTIMATORS_MAX,
        n_estimators_step=config.LIGHTGBM_N_ESTIMATORS_STEP,
        greedy_pool_size=config.GREEDY_POOL_SIZE,
        greedy_ensemble_size=config.GREEDY_ENSEMBLE_SIZE,
        greedy_allow_repeats=config.GREEDY_ALLOW_REPEATS,
        seed=config.RANDOM_SEED,
        verbose=False
    )

    selected_budget = cast(int, summary["selected_budget"])

    selected_brier_score = cast(
        float, summary["selected_cross_fitted_brier_score"]
    )

    stopping_checkpoint = cast(int, summary["stopping_checkpoint"])

    stop_reason = cast(str, summary["stop_reason"])

    print("LightGBM ensemble-aware HPO stopping completed.")
    print(f"Selected Optuna budget: {selected_budget} trials")
    print(
        f"Selected cross-fitted ensemble Brier score: "
        f"{selected_brier_score:.5f}"
    )
    print(f"Stopping checkpoint: {stopping_checkpoint} trials")
    print(f"Stop reason: {stop_reason}")
    print(f"Stopping results saved to: {output_dir}")


if __name__ == "__main__":
    main()
