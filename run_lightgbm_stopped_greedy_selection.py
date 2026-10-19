# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# run_lightgbm_stopped_greedy_selection.py

from environment import configure_environment

configure_environment()

from pathlib import Path
from typing import cast

import joblib
import numpy as np

import config
from ensemble.greedy_selection import find_greedy_ensemble
from ensemble.oof_matrix import OOFResultBlock, build_oof_matrix
from storage import get_dataset_output_dir, load_prepared_data
from stopping.lightgbm_ensemble_stopping import (
    get_lightgbm_stopping_comparison_output_dir,
    get_lightgbm_stopping_output_dir
)
from tuning.feature_sets import validate_lightgbm_feature_set
from tuning.result_storage import get_optuna_result_dir, load_optuna_results


def _validate_full_trial_sequence(
    trial_numbers: np.ndarray
) -> None:
    """Require stored full-development trials 0, 1, ..., N-1."""
    expected = np.arange(len(trial_numbers), dtype=int)

    if not np.array_equal(trial_numbers, expected):
        raise RuntimeError(
            "Stored full-development LightGBM trials are not a complete "
            "contiguous sequence starting at trial 0."
        )


def _build_and_save_greedy_ensemble(
    output_dir: Path,
    y_dev: np.ndarray,
    feature_set: str,
    trial_numbers: np.ndarray,
    oof_probabilities: np.ndarray,
    oof_brier_scores: np.ndarray,
    fold_brier_scores: np.ndarray,
    hyperparams: list[dict[str, object]],
    budget: int
) -> dict[str, object]:
    """
    Build and save a LightGBM-only greedy ensemble from the first `budget`
    Optuna trials.
    """
    block = OOFResultBlock(
        model_name="lightgbm",
        feature_set=feature_set,
        trial_numbers=trial_numbers[:budget],
        oof_probabilities=oof_probabilities[:, :budget],
        oof_brier_scores=oof_brier_scores[:budget],
        fold_brier_scores=fold_brier_scores[:budget],
        hyperparams=hyperparams[:budget]
    )

    P, column_metadata = build_oof_matrix([block])

    greedy_metadata, greedy_oof_probabilities = (
        find_greedy_ensemble(
            P,
            y_dev,
            column_metadata,
            pool_size=config.GREEDY_POOL_SIZE,
            ensemble_size=config.GREEDY_ENSEMBLE_SIZE,
            seed=config.RANDOM_SEED,
            allow_repeats=config.GREEDY_ALLOW_REPEATS
        )
    )

    greedy_metadata["candidate_budget"] = budget

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

    return greedy_metadata


def main() -> None:
    """
    Build LightGBM greedy ensembles from the selected trial budget and
    the full stored trial zoo.
    """
    validate_lightgbm_feature_set(config.LIGHTGBM_FEATURE_SET)

    stopping_dir = get_lightgbm_stopping_output_dir(
        config.STOPPING_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        config.LIGHTGBM_FEATURE_SET
    )

    stopping_summary = cast(
        dict[str, object], joblib.load(stopping_dir / "stopping_summary.pkl")
    )

    selected_budget = cast(int, stopping_summary["selected_budget"])

    if selected_budget < 1:
        raise RuntimeError(
            f"Invalid selected budget: {selected_budget}."
        )

    result_dir = get_optuna_result_dir(
        config.OPTUNA_RESULTS_DIR,
        config.DATASET_NAME,
        config.DATA_SPLIT_SEED,
        config.LIGHTGBM_FEATURE_SET,
        "lightgbm"
    )

    optuna_results = load_optuna_results(result_dir)

    trial_numbers = np.asarray(optuna_results["trial_numbers"], dtype=int)
    oof_probabilities = np.asarray(
        optuna_results["oof_probabilities"], dtype=np.float64
    )
    oof_brier_scores = np.asarray(
        optuna_results["oof_brier_scores"], dtype=np.float64
    )
    fold_brier_scores = np.asarray(
        optuna_results["fold_brier_scores"], dtype=np.float64
    )
    hyperparams = cast(list[dict[str, object]], optuna_results["hyperparams"])

    _validate_full_trial_sequence(trial_numbers)

    full_budget = len(trial_numbers)

    if selected_budget > full_budget:
        raise RuntimeError(
            f"Stopping selected {selected_budget} trials, but only "
            f"{full_budget} full-development LightGBM trials are stored."
        )

    prepared_data_dir = get_dataset_output_dir(
        config.PREPARED_DATA_DIR, config.DATASET_NAME, config.DATA_SPLIT_SEED
    )

    arrays, _ = load_prepared_data(prepared_data_dir)

    y_dev = np.asarray(arrays["y_dev"], dtype=np.float64)

    comparison_dir = (
        get_lightgbm_stopping_comparison_output_dir(
            config.ENSEMBLE_RESULTS_DIR,
            config.DATASET_NAME,
            config.DATA_SPLIT_SEED,
            config.LIGHTGBM_FEATURE_SET
        )
    )

    selected_metadata = _build_and_save_greedy_ensemble(
        comparison_dir / "selected_budget",
        y_dev,
        config.LIGHTGBM_FEATURE_SET,
        trial_numbers,
        oof_probabilities,
        oof_brier_scores,
        fold_brier_scores,
        hyperparams,
        selected_budget
    )

    full_metadata = _build_and_save_greedy_ensemble(
        comparison_dir / "full_budget",
        y_dev,
        config.LIGHTGBM_FEATURE_SET,
        trial_numbers,
        oof_probabilities,
        oof_brier_scores,
        fold_brier_scores,
        hyperparams,
        full_budget
    )

    selected_oof_brier = cast(float, selected_metadata["oof_brier_score"])

    full_oof_brier = cast(float, full_metadata["oof_brier_score"])

    print(
        f"Selected-budget LightGBM ensemble: "
        f"{selected_budget} candidates, "
        f"OOF Brier = {selected_oof_brier:.5f}"
    )

    print(
        f"Full-budget LightGBM ensemble: "
        f"{full_budget} candidates, "
        f"OOF Brier = {full_oof_brier:.5f}"
    )

    print(f"Comparison ensembles saved to: {comparison_dir}")


if __name__ == "__main__":
    main()
