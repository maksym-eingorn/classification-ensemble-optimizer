# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# tuning/logistic_regression_optuna.py

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from tuning.validation import (
    validate_binary_classification_inputs,
    validate_random_seed,
    validate_positive_integer,
    validate_stratified_kfold_settings
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_kfold_logistic_regression(
    X_dev,
    y_dev,
    n_trials: int = 100,
    n_jobs: int = 1,
    n_splits: int = 5,
    max_iter: int = 20000,
    tol: float = 1e-4,
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
    Run Optuna to tune a Logistic Regression classifier with stratified K-fold
    cross-validation.

    Out-of-fold probabilities are stored for future ensemble search. Each
    Optuna trial trains one Logistic Regression configuration across all K folds
    and produces one full out-of-fold positive-class probability vector for the
    development set.

    Parameters
    ----------
    X_dev
        Development feature matrix used for model training and cross-validation.
        Expected shape: (n_samples_dev, n_features).
    y_dev
        Binary development target vector corresponding to X_dev.
        Expected shape: (n_samples_dev,), with classes 0 and 1.
    n_trials : int, default=100
        Number of Optuna trials.
    n_jobs : int, default=1
        Number of parallel Optuna workers. Sequential execution is recommended
        for maximum reproducibility.
    n_splits : int, default=5
        Number of stratified K-fold splits used for cross-validation.
    max_iter : int, default=20000
        Maximum number of Logistic Regression optimization iterations.
    tol : float, default=1e-4
        Optimization tolerance for Logistic Regression.
    seed : int, default=42
        Random seed used by the Optuna sampler, stratified K-fold splitting,
        and Logistic Regression.
    verbose : bool, default=False
        Whether to print the OOF Brier score for each trial.

    Returns
    -------
    study : optuna.study.Study
        Completed Optuna study.
    trial_numbers : np.ndarray
        Trial numbers sorted in ascending order.
        Shape: (n_trials,).
    oof_probabilities : np.ndarray
        Out-of-fold positive-class probability matrix.
        Shape: (n_samples_dev, n_trials). Each column corresponds to one trial.
    oof_brier_scores : np.ndarray
        OOF Brier scores for each trial.
        Shape: (n_trials,).
    fold_brier_scores : np.ndarray
        Per-fold Brier scores for each trial.
        Shape: (n_trials, n_splits).
    hyperparams : list[dict[str, object]]
        Hyperparameter dictionary for each trial, ordered by trial number.
    """
    X_dev, y_dev = validate_binary_classification_inputs(X_dev, y_dev)

    validate_positive_integer(n_trials, "n_trials")
    validate_positive_integer(n_jobs, "n_jobs")
    validate_stratified_kfold_settings(n_splits, y_dev)
    validate_positive_integer(max_iter, "max_iter")
    validate_random_seed(seed, "seed")

    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError(f"tol must be a positive finite number, got {tol}.")

    if not isinstance(verbose, bool):
        raise ValueError(f"verbose must be a boolean, got {verbose}.")

    # noinspection PyTypeChecker
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    results = []

    def objective_lr(trial: optuna.trial.Trial) -> float:
        params = {
            "C": trial.suggest_float(
                "C", 1e-4, 1e4, log=True
            ),
            "l1_ratio": trial.suggest_float(
                "l1_ratio", 0.0, 1.0
            )
        }

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )

        oof_probability = np.zeros(len(y_dev), dtype=np.float32)
        per_fold_brier_scores = []

        for train_idx, valid_idx in skf.split(X_dev, y_dev):
            X_train, X_valid = X_dev[train_idx], X_dev[valid_idx]
            y_train, y_valid = y_dev[train_idx], y_dev[valid_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_valid_scaled = scaler.transform(X_valid)

            model = LogisticRegression(
                solver="saga",
                max_iter=max_iter,
                tol=tol,
                random_state=seed,
                **params
            )

            model.fit(X_train_scaled, y_train)
            y_probability = model.predict_proba(X_valid_scaled)[:, 1]

            oof_probability[valid_idx] = y_probability

            fold_brier_score = brier_score_loss(
                y_valid, y_probability, pos_label=1, scale_by_half=True
            )
            per_fold_brier_scores.append(fold_brier_score)

        fold_brier_scores_array = np.array(per_fold_brier_scores, dtype=float)
        oof_brier_score = brier_score_loss(
            y_dev, oof_probability, pos_label=1, scale_by_half=True
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

    study.optimize(objective_lr, n_trials=n_trials, n_jobs=n_jobs)

    results.sort(key=lambda item: item[0])

    trial_numbers = np.array([item[0] for item in results], dtype=int)

    oof_probabilities = np.column_stack(
        [item[1] for item in results]
    ).astype(np.float32)

    oof_brier_scores = np.array([item[2] for item in results], dtype=float)

    fold_brier_scores = np.vstack([item[3] for item in results]).astype(float)

    hyperparams = [item[4] for item in results]

    return (
        study,
        trial_numbers,
        oof_probabilities,
        oof_brier_scores,
        fold_brier_scores,
        hyperparams
    )
