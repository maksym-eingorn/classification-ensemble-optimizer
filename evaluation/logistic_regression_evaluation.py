# Copyright 2026 Maksym Eingorn
# SPDX-License-Identifier: Apache-2.0

# evaluation/logistic_regression_evaluation.py

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

from evaluation.validation import (
    validate_binary_classification_evaluation_inputs
)
from tuning.validation import validate_positive_integer, validate_random_seed


def evaluate_best_logistic_regression_on_test(
    study_lr: optuna.study.Study,
    X_dev,
    y_dev,
    X_test,
    y_test,
    max_iter: int = 20000,
    tol: float = 1e-4,
    seed: int = 42
) -> dict[str, float]:
    """
    Retrain the best Logistic Regression model from Optuna and evaluate it on
    the test set.

    Parameters
    ----------
    study_lr : optuna.study.Study
        Completed Optuna study for Logistic Regression.
    X_dev
        Development feature matrix used for final model retraining.
        Expected shape: (n_samples_dev, n_features).
    y_dev
        Binary development target vector corresponding to X_dev.
        Expected shape: (n_samples_dev,), with classes 0 and 1.
    X_test
        Test feature matrix used for evaluation.
        Expected shape: (n_samples_test, n_features).
    y_test
        Binary test target vector corresponding to X_test.
        Expected shape: (n_samples_test,), with classes 0 and 1.
    max_iter : int, default=20000
        Maximum number of Logistic Regression optimization iterations.
    tol : float, default=1e-4
        Optimization tolerance for Logistic Regression.
    seed : int, default=42
        Random seed used by the final Logistic Regression model.

    Returns
    -------
    test_metrics : dict[str, float]
        Test Brier score, log loss, ROC AUC, and accuracy of the best
        Logistic Regression model.
        Accuracy uses a positive-class probability threshold of 0.5.
    """
    X_dev, y_dev, X_test, y_test = (
        validate_binary_classification_evaluation_inputs(
            X_dev, y_dev, X_test, y_test
        )
    )
    validate_positive_integer(max_iter, "max_iter")
    validate_random_seed(seed, "seed")

    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError(f"tol must be a positive finite number, got {tol}.")

    best_params = study_lr.best_params.copy()

    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver="saga",
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        **best_params
    )

    model.fit(X_dev_scaled, y_dev)

    y_probability_test = model.predict_proba(X_test_scaled)[:, 1]
    y_prediction_test = np.asarray(
        y_probability_test >= 0.5, dtype=np.int64
    )

    return {
        "brier_score": float(brier_score_loss(
            y_test, y_probability_test, pos_label=1, scale_by_half=True
        )),
        "log_loss": float(log_loss(y_test, y_probability_test, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_test, y_probability_test)),
        "accuracy": float(accuracy_score(y_test, y_prediction_test))
    }
