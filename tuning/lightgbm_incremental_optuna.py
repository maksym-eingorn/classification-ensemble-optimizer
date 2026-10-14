# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# tuning/lightgbm_incremental_optuna.py

import numpy as np
import optuna

from tuning.lightgbm_optuna import (
    LightGBMHyperparams,
    evaluate_lightgbm_hyperparams_oof,
    suggest_lightgbm_hyperparams
)
from tuning.validation import (
    validate_binary_classification_inputs,
    validate_integer_range_settings,
    validate_positive_integer,
    validate_random_seed,
    validate_stratified_kfold_settings
)


optuna.logging.set_verbosity(optuna.logging.WARNING)

LightGBMTrialResult = tuple[
    int, np.ndarray, float, np.ndarray, LightGBMHyperparams
]


def run_optuna_kfold_lightgbm_block(
    X_dev,
    y_dev,
    study: optuna.study.Study | None = None,
    n_trials: int = 100,
    n_splits: int = 5,
    n_estimators_min: int = 100,
    n_estimators_max: int = 1000,
    n_estimators_step: int = 50,
    seed: int = 42,
    verbose: bool = False
) -> tuple[
    optuna.study.Study,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]]
]:
    """
    Run one additional sequential block of LightGBM Optuna trials.

    If study is None, create a new seeded TPE study. Otherwise, append trials
    to the supplied study while preserving its existing optimization history.

    Returned trial artifacts contain only the newly completed block.
    """
    X_dev, y_dev = validate_binary_classification_inputs(X_dev, y_dev)

    validate_positive_integer(n_trials, "n_trials")
    validate_stratified_kfold_settings(n_splits, y_dev)
    validate_integer_range_settings(
        n_estimators_min,
        n_estimators_max,
        n_estimators_step,
        "n_estimators"
    )
    validate_random_seed(seed, "seed")

    if not isinstance(verbose, bool):
        raise ValueError(f"verbose must be a boolean, got {verbose}.")

    if study is None:
        # noinspection PyTypeChecker
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

    results: list[LightGBMTrialResult] = []

    def objective_lgb(trial: optuna.trial.Trial) -> float:
        params = suggest_lightgbm_hyperparams(
            trial, n_estimators_min, n_estimators_max, n_estimators_step
        )

        (
            oof_probability, oof_brier_score, fold_brier_scores_array
        ) = evaluate_lightgbm_hyperparams_oof(
            X_dev, y_dev, params, n_splits=n_splits, seed=seed
        )

        results.append((
            trial.number,
            oof_probability.copy(),
            oof_brier_score,
            fold_brier_scores_array,
            params
        ))

        if verbose:
            print(
                f"Trial {trial.number} | "
                f"OOF Brier score = {oof_brier_score:.5f}"
            )

        return oof_brier_score

    # Intentionally sequential. Parallelism occurs across independent
    # meta-fold studies, not within one TPE trajectory.
    study.optimize(objective_lgb, n_trials=n_trials, n_jobs=1)

    results.sort(key=lambda item: item[0])

    trial_numbers: np.ndarray = np.array(
        [item[0] for item in results], dtype=int
    )

    oof_probabilities: np.ndarray = np.column_stack(
        [item[1] for item in results]
    ).astype(np.float32)

    oof_brier_scores: np.ndarray = np.array(
        [item[2] for item in results], dtype=float
    )

    fold_brier_scores: np.ndarray = np.vstack(
        [item[3] for item in results]
    ).astype(float)

    hyperparams: list[dict[str, object]] = [item[4].copy() for item in results]

    return (
        study,
        trial_numbers,
        oof_probabilities,
        oof_brier_scores,
        fold_brier_scores,
        hyperparams
    )
